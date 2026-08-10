"""Phase 6 — Direct Preference Optimization. sft.pt -> dpo.pt.

DPO's objective, for a pair (x, y_w, y_l):

    L = -log sigmoid( beta * [ (logp_th(y_w|x) - logp_ref(y_w|x))
                             - (logp_th(y_l|x) - logp_ref(y_l|x)) ] )

The reference model is a frozen copy of `sft.pt`. The policy starts from the same
weights, so at step 0 both bracketed terms are zero, the argument to sigmoid is
zero, and the loss is exactly `-log(0.5) = 0.6931`. **That is a checkable
starting value, not an approximate one**, and this trainer asserts it before step
1 - the DPO equivalent of the mask assertion in Phase 5. A DPO run that does not
start at 0.6931 has a reference/policy mismatch, and like a bad loss mask it
would still train to a plausible-looking curve.

**Why the reference logprobs are precomputed.** The reference never updates, so
its logprobs are constants. Computing them once turns four forward passes per
pair into two, and - more usefully - it makes them *inspectable*: they get
written to disk, so "did the reference drift" is answerable rather than assumed.

**What this stage can degrade.** Everything SFT bought. DPO sharpens toward the
preference signal, and past its useful point it trades every other property
against that signal. The floors registered in ADR-037 and the read order in
ADR-039 exist for exactly that, and the read order is not optional: the
side-condition and the floors are read *before* the delta, because an amber delta
with a floor breach is not "DPO underdelivered", it is "DPO started eating what
SFT bought", and those have opposite responses.

    python -m src.dpo train
"""

from __future__ import annotations

import argparse
import json
import math
import time

import numpy as np
import torch
import torch.nn.functional as F

from config import REPO_ROOT, Config
from src.model import GPT
from src.sft_data import IGNORE, max_seq, wire_format
from src.train import build_optimizer
from utils.device import probe
from utils.io import write_json, write_text
from utils.seed import set_seed


def encode_side(cfg, tok, eot: int, prompt: str, response: str):
    """One (prompt, response) into ids + the index where the response starts."""
    p_ids = tok.encode(wire_format(cfg, prompt)).ids
    r_ids = tok.encode(response).ids + [eot]
    return p_ids, r_ids


def pack_pairs(cfg: Config) -> dict:
    """Tensorise the pairs, dropping any side that overflows the window.

    A pair is only usable if BOTH sides fit: dropping one side would leave a
    comparison against nothing. The count dropped is reported rather than
    absorbed, same rule as ADR-008 and the Phase 5 budget census.
    """
    from src.tokenizer import load as load_tokenizer

    tok = load_tokenizer(cfg)
    eot = tok.token_to_id(cfg.doc_separator)
    pairs = json.loads((REPO_ROOT / cfg.data_dir / "pref_pairs.json")
                       .read_text(encoding="utf-8"))
    L, limit = cfg.context_len, max_seq(cfg)
    keep, dropped = [], 0
    for pr in pairs:
        sides = {}
        for side in ("chosen", "rejected"):
            p_ids, r_ids = encode_side(cfg, tok, eot, pr["prompt"], pr[side])
            if len(p_ids) + len(r_ids) > limit:
                sides = None
                break
            sides[side] = (p_ids, r_ids)
        if sides is None:
            dropped += 1
            continue
        keep.append(sides)

    n = len(keep)
    out = {}
    for side in ("chosen", "rejected"):
        x = np.zeros((n, L), dtype=np.int32)
        y = np.full((n, L), IGNORE, dtype=np.int32)
        for i, s in enumerate(keep):
            p_ids, r_ids = s[side]
            seq = p_ids + r_ids
            xs, ys = seq[:-1], seq[1:]
            x[i, :len(xs)] = xs
            for t in range(len(p_ids) - 1, len(ys)):
                y[i, t] = ys[t]
        out[side] = (x, y)
    print(f"packed {n:,} pairs ({dropped:,} dropped - a side exceeded "
          f"{limit} tokens; both sides must fit or the comparison is empty)")
    return {"n": n, "dropped": dropped, **out}


def seq_logprob(model: GPT, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Sum of log p(token) over response positions only. One value per row."""
    logits, _ = model(x)
    logp = F.log_softmax(logits.float(), dim=-1)
    mask = y != IGNORE
    safe = y.clone()
    safe[~mask] = 0
    tok_lp = logp.gather(-1, safe.unsqueeze(-1)).squeeze(-1)
    return (tok_lp * mask).sum(dim=-1)


@torch.no_grad()
def reference_logprobs(model: GPT, packed: dict, batch: int,
                       device: torch.device) -> dict:
    """Precompute the frozen reference's logprobs once. They are constants."""
    model.eval()
    out = {}
    for side in ("chosen", "rejected"):
        x, y = packed[side]
        vals = []
        for i in range(0, len(x), batch):
            xb = torch.from_numpy(x[i:i + batch].astype(np.int64)).to(device)
            yb = torch.from_numpy(y[i:i + batch].astype(np.int64)).to(device)
            vals.append(seq_logprob(model, xb, yb).cpu())
        out[side] = torch.cat(vals)
    return out


def dpo_loss(pi_c, pi_r, ref_c, ref_r, beta: float):
    """Returns (loss, accuracy, mean margin). Accuracy = fraction of pairs the
    policy already ranks correctly, which is the interpretable training signal -
    the loss value alone says little."""
    logits = beta * ((pi_c - ref_c) - (pi_r - ref_r))
    loss = -F.logsigmoid(logits).mean()
    return loss, (logits > 0).float().mean(), logits.mean()


def train(cfg: Config) -> dict:
    stats = REPO_ROOT / cfg.results_dir / "pref_pairs_stats.json"
    if not stats.exists():
        raise SystemExit("run `python -m src.prefs stats` first - the pair-set "
                         "pre-flight is a gate, not a report.")
    st = json.loads(stats.read_text(encoding="utf-8"))
    if not st.get("preflight_pass"):
        raise SystemExit(f"pair-set pre-flight did not pass: {st}")
    print(f"pair-set pre-flight on record: length parity p={st['welch_p']:.4f}, "
          f"chosen-rejected {st['length_diff']:+.2f} words")

    set_seed(cfg.seed, strict=cfg.strict_determinism)
    dev = probe()
    packed = pack_pairs(cfg)
    n = packed["n"]

    ck = torch.load(REPO_ROOT / cfg.ckpt_dir / "sft.pt", map_location="cpu",
                    weights_only=False)
    policy = GPT(cfg).to(dev.device)
    policy.load_state_dict(ck["model"])
    ref = GPT(cfg).to(dev.device)
    ref.load_state_dict(ck["model"])
    for p in ref.parameters():
        p.requires_grad_(False)
    ref.eval()
    print(f"loaded sft.pt at step {ck['step']} into policy and frozen reference")

    print("precomputing reference logprobs (constants - the reference never moves)...")
    refs = reference_logprobs(ref, packed, cfg.dpo_micro_batch, dev.device)

    # DROPOUT IS OFF FOR THE WHOLE DPO RUN, and this is a correctness
    # requirement rather than a tuning choice (ADR-041).
    #
    # The reference logprobs above were computed once, in eval mode. If the
    # policy then ran in train mode, its logprobs would carry dropout noise the
    # reference's do not, so the bracketed term would be noise-against-a-constant
    # rather than a comparison of two evaluations of the same function. Measured
    # on this model: per-pair logit sd 0.67 at initialisation, where the whole
    # preference signal DPO is trying to learn is smaller than that.
    #
    # The model has no batchnorm, so eval mode disables dropout and changes
    # nothing else. Gradients still flow - eval mode is not no_grad.
    policy.eval()

    # The startup assertion, in the mode training actually uses (ADR-041). It has
    # two halves and the second is the one that would have caught the bug:
    #   value       - policy == reference, so the loss must be exactly -log(0.5)
    #   determinism - the same batch twice must give the SAME loss
    # A value check alone passes happily while the objective is stochastic,
    # because dropout noise is zero-mean in the bracket. Repeatability is what
    # distinguishes "policy equals reference" from "policy equals reference on
    # average".
    xb = torch.from_numpy(packed["chosen"][0][:cfg.dpo_micro_batch].astype(np.int64)).to(dev.device)
    yb = torch.from_numpy(packed["chosen"][1][:cfg.dpo_micro_batch].astype(np.int64)).to(dev.device)
    xr = torch.from_numpy(packed["rejected"][0][:cfg.dpo_micro_batch].astype(np.int64)).to(dev.device)
    yr = torch.from_numpy(packed["rejected"][1][:cfg.dpo_micro_batch].astype(np.int64)).to(dev.device)

    def _step0():
        with torch.no_grad():
            return dpo_loss(seq_logprob(policy, xb, yb),
                            seq_logprob(policy, xr, yr),
                            refs["chosen"][:cfg.dpo_micro_batch].to(dev.device),
                            refs["rejected"][:cfg.dpo_micro_batch].to(dev.device),
                            cfg.dpo_beta)

    l0, acc0, m0 = _step0()
    l0b, _, _ = _step0()
    expected = math.log(2)
    print(f"  step-0 loss {l0.item():.6f}  expected {expected:.6f} "
          f"(-log 0.5, because policy == reference)")
    if abs(l0.item() - expected) > 1e-3:
        raise SystemExit(
            f"DPO did not start at -log(0.5). Policy and reference disagree at "
            f"step 0, which means one of them is not sft.pt. Loss {l0.item():.6f}.")
    if l0.item() != l0b.item():
        raise SystemExit(
            f"DPO step-0 loss is not repeatable: {l0.item():.9f} then "
            f"{l0b.item():.9f} on the SAME batch with the SAME weights. The "
            f"objective is stochastic, so the policy is being compared against a "
            f"reference computed under different conditions. Dropout left on in "
            f"the policy is the usual cause (ADR-041).")
    print(f"  reference/policy identity assertion PASSED "
          f"(value exact to {abs(l0.item() - expected):.1e}, repeatable bit-for-bit)")

    opt = build_optimizer(policy, cfg)
    for g in opt.param_groups:
        g["lr"] = cfg.dpo_lr
    per_epoch = max(n // (cfg.dpo_micro_batch * cfg.dpo_grad_accum_steps), 1)
    total = per_epoch * cfg.dpo_epochs
    print(f"  {per_epoch:,} steps/epoch x {cfg.dpo_epochs} = {total:,} steps, "
          f"beta {cfg.dpo_beta}, lr {cfg.dpo_lr:g}")

    rng = np.random.default_rng(cfg.seed)
    order = rng.permutation(n)
    pos = 0
    rows = ["step,loss,accuracy,margin,lr,elapsed_s"]
    started = time.time()
    amp = (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
           if dev.device.type == "cuda" else torch.autocast("cpu", enabled=False))

    for step in range(total):
        lr = cfg.dpo_lr * min(1.0, (step + 1) / max(cfg.dpo_warmup_steps, 1))
        for g in opt.param_groups:
            g["lr"] = lr
        opt.zero_grad(set_to_none=True)
        agg_l = agg_a = agg_m = 0.0
        for _ in range(cfg.dpo_grad_accum_steps):
            if pos + cfg.dpo_micro_batch > n:
                order, pos = rng.permutation(n), 0
            idx = order[pos:pos + cfg.dpo_micro_batch]
            pos += cfg.dpo_micro_batch
            t = lambda a: torch.from_numpy(a[idx].astype(np.int64)).to(dev.device)
            with amp:
                pc = seq_logprob(policy, t(packed["chosen"][0]), t(packed["chosen"][1]))
                pr = seq_logprob(policy, t(packed["rejected"][0]), t(packed["rejected"][1]))
                loss, acc, marg = dpo_loss(
                    pc, pr, refs["chosen"][idx].to(dev.device),
                    refs["rejected"][idx].to(dev.device), cfg.dpo_beta)
            (loss / cfg.dpo_grad_accum_steps).backward()
            agg_l += loss.item(); agg_a += acc.item(); agg_m += marg.item()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.grad_clip)
        opt.step()
        k = cfg.dpo_grad_accum_steps
        if (step + 1) % cfg.log_interval == 0 or step == 0:
            rows.append(f"{step+1},{agg_l/k:.6f},{agg_a/k:.6f},{agg_m/k:.6f},"
                        f"{lr:.6e},{time.time()-started:.1f}")
            print(f"  step {step+1:>5}/{total}  loss {agg_l/k:.4f}  "
                  f"acc {agg_a/k:.3f}  margin {agg_m/k:+.3f}")
            write_text(REPO_ROOT / cfg.results_dir / "dpo_curve.csv",
                       "\n".join(rows) + "\n")

    torch.save({"model": policy.state_dict(), "step": total,
                "base": "sft.pt", "base_step": ck["step"],
                "beta": cfg.dpo_beta, "config": cfg.as_dict(),
                "config_hash": cfg.hash()},
               REPO_ROOT / cfg.ckpt_dir / "dpo.pt")
    write_text(REPO_ROOT / cfg.results_dir / "dpo_curve.csv", "\n".join(rows) + "\n")
    summary = {"pairs": n, "dropped": packed["dropped"], "steps": total,
               "beta": cfg.dpo_beta, "lr": cfg.dpo_lr,
               "step0_loss": round(l0.item(), 6),
               "final_loss": round(agg_l / k, 6),
               "final_accuracy": round(agg_a / k, 6),
               "elapsed_s": round(time.time() - started, 1), "device": dev.name}
    write_json(REPO_ROOT / cfg.results_dir / "dpo_train_summary.json", summary)
    print(f"\ndpo.pt written. final loss {agg_l/k:.4f}, "
          f"ranking accuracy {agg_a/k:.1%}, {(time.time()-started)/60:.1f} min")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="DPO (Phase 6).")
    ap.add_argument("command", choices=("train",))
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    train(Config.load(args.config))


if __name__ == "__main__":
    main()
