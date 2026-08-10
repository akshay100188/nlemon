"""Phase 7 — the eval harness. A scorecard that certifies rather than adjudicates.

    python -m src.eval scorecard

Phase 7 gates the **instrument**, not the model (ADR-049). Every earlier phase
gated a capability being acquired, so a bar on the model was the natural object.
This phase acquires nothing; it certifies. Inventing new model bars after the
model exists would be fitting bars to a known result, which is the move ADR-017's
family exists to refuse.

So the gates ask whether the scorecard is trustworthy:

  G1  eval-set integrity   both frozen sets match their registered hashes
  G2  environment          device asserted, torch build recorded
  G3  determinism          two full passes bit-identical
  G4  recompute vs record  recomputed values agree with what was published
  G5  precision            measured deff-adjusted relative SE <= 1.0%

G4 is the gate that can fail *because of the model*, and G5 is the only measured
bar - deliberately about precision rather than performance, and registered from
the eval set's SIZE, a design fact that existed before any model number.

**G5 passes on the MEASURED design-effect-adjusted SE, never the iid one.** The
iid figure over 4.68M tokens is ~0.09%, which is a lower bound and a lie if
published as the precision: tokens cluster inside documents. If the measured
figure exceeds 1.0% the bar has done its job and the response is pre-committed -
report the true interval, do not relax the threshold and do not republish at a
friendlier number.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time

import numpy as np
import torch
import torch.nn.functional as F

from config import REPO_ROOT, Config
from src.evalset import assert_frozen
from src.sample import load_checkpoint
from utils.io import write_json, write_text
from utils.seed import set_seed

# Result-invariant performance knob, like log_interval. Windows are independent,
# so how many are batched together cannot change a single per-token NLL - which is
# exactly why it is not a registered field.
EVAL_BATCH = 32


def icc_continuous(values: np.ndarray, groups: np.ndarray) -> dict[str, float]:
    """One-way random-effects ICC and design effect for a continuous measure.

    The discrete version in `checker.effective_n` prices bars on clustered
    Bernoulli outcomes. This is the same correction for per-token NLL clustered
    inside documents: a document that is easy to predict makes all ~213 of its
    tokens easy together, so 4.68M tokens are worth far fewer independent
    observations (ADR-033).
    """
    k = int(groups.max()) + 1
    n = values.size
    counts = np.bincount(groups, minlength=k).astype(np.float64)
    sums = np.bincount(groups, weights=values, minlength=k)
    keep = counts > 0
    counts, sums = counts[keep], sums[keep]
    k = counts.size
    means = sums / counts
    grand = values.mean()

    ssb = float((counts * (means - grand) ** 2).sum())
    sst = float(((values - grand) ** 2).sum())
    ssw = sst - ssb
    if k < 2 or n <= k:
        return {"k": k, "icc": 0.0, "deff": 1.0, "n_eff": float(n)}
    msb, msw = ssb / (k - 1), ssw / (n - k)
    m0 = (n - (counts ** 2).sum() / n) / (k - 1)
    den = msb + (m0 - 1) * msw
    icc = max(0.0, min(1.0, (msb - msw) / den)) if den > 0 else 0.0
    m_bar = n / k
    deff = 1.0 + (m_bar - 1.0) * icc
    return {"k": k, "mean_group_size": round(m_bar, 2), "icc": round(icc, 6),
            "deff": round(deff, 4), "n_eff": round(n / deff, 1)}


@torch.no_grad()
def token_nll(model, tokens: np.ndarray, ctx: int, device, dtype) -> np.ndarray:
    """Per-token NLL over the whole set, in non-overlapping windows.

    Non-overlapping rather than strided: every token is scored exactly once, so
    the mean is a clean average over the set with no token double-counted and no
    choice of stride to tune. The cost is that the first token of each window is
    predicted from little context, which inflates the mean slightly and
    identically on every run and every checkpoint - so it cannot flatter one
    model over another, which is what matters for a comparison.

    Precision matches the training-time estimator (autocast bf16 on cuda) so the
    G4 recompute compares like with like rather than measuring the dtype.
    """
    n_win = (tokens.size - 1) // ctx
    out = np.empty(n_win * ctx, dtype=np.float32)
    model.eval()
    for start in range(0, n_win, EVAL_BATCH):
        stop = min(start + EVAL_BATCH, n_win)
        idx = [slice(w * ctx, w * ctx + ctx + 1) for w in range(start, stop)]
        chunk = np.stack([tokens[s] for s in idx]).astype(np.int64)
        x = torch.from_numpy(chunk[:, :-1]).to(device)
        y = torch.from_numpy(chunk[:, 1:]).to(device)
        with torch.autocast(device_type=device.type, dtype=dtype,
                            enabled=device.type == "cuda"):
            logits, _ = model(x)
        loss = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)),
                               y.reshape(-1), reduction="none")
        out[start * ctx:stop * ctx] = loss.float().cpu().numpy()
    return out


@torch.no_grad()
def training_estimator(model, tokens: np.ndarray, cfg: Config, device, dtype,
                       nll_sd: float) -> dict:
    """Reproduce train.py's val estimator, and measure its own sampling error.

    `eval_iters` random-offset batches, mean of batch means - the procedure that
    produced the recorded number. The SE comes from the spread of the batch losses
    it actually drew, so it needs no independence assumption: batches are drawn
    independently even though tokens inside one are not.
    """
    ctx, B = cfg.context_len, cfg.micro_batch
    rng = np.random.default_rng(cfg.seed)
    losses = np.empty(cfg.eval_iters)
    model.eval()
    for i in range(cfg.eval_iters):
        off = rng.integers(0, tokens.size - ctx - 1, size=B)
        chunk = np.stack([np.asarray(tokens[o:o + ctx + 1]) for o in off]).astype(np.int64)
        x = torch.from_numpy(chunk[:, :-1]).to(device)
        y = torch.from_numpy(chunk[:, 1:]).to(device)
        with torch.autocast(device_type=device.type, dtype=dtype,
                            enabled=device.type == "cuda"):
            _, loss = model(x, y)
        losses[i] = loss.item()
    se = float(losses.std(ddof=1) / math.sqrt(cfg.eval_iters))
    # What the naive route would have predicted, using the REAL per-token sd
    # measured on the full set rather than one derived from these batches (which
    # would be circular). The gap between the two is the whole point.
    se_iid = nll_sd / math.sqrt(cfg.eval_iters * B * ctx)
    return {"mean_nll": float(losses.mean()), "se_measured": se,
            "se_iid_formula": round(se_iid, 8),
            "batches": int(cfg.eval_iters), "batch_seqs": int(B),
            "inflation_vs_iid": round(se / max(se_iid, 1e-12), 2)}


def perplexity_block(model, tokens, doc_id, cfg, device, dtype) -> dict:
    nll = token_nll(model, tokens, cfg.context_len, device, dtype)
    groups = doc_id[1:nll.size + 1]          # token i is predicted from i-1
    mean = float(nll.mean())
    sd = float(nll.std(ddof=1))
    clus = icc_continuous(nll, groups)

    se_iid = sd / math.sqrt(nll.size)
    se_clustered = sd / math.sqrt(clus["n_eff"])
    # ppl = exp(mean NLL), so d(ppl)/ppl = d(mean NLL): the relative SE on
    # perplexity IS the absolute SE on mean NLL, to first order.
    return {
        "tokens_scored": int(nll.size),
        "mean_nll": round(mean, 6),
        "perplexity": round(math.exp(mean), 4),
        "nll_sd": round(sd, 6),
        "clustering": clus,
        "se_iid_nats": round(se_iid, 8),
        "relative_se_iid": round(se_iid, 8),
        "se_measured_nats": round(se_clustered, 8),
        "relative_se_measured": round(se_clustered, 8),
        "inflation_vs_iid": round(se_clustered / se_iid, 2),
        "nll_digest": hashlib.sha256(nll.tobytes()).hexdigest()[:16],
    }


def cmd_scorecard(cfg: Config, require_device: str, allow_cpu: bool) -> None:
    from src.checker import assert_device
    from utils.device import probe

    t0 = time.time()
    print("Phase 7 scorecard — gates the instrument, not the model (ADR-049)\n")

    print("G1  eval-set integrity")
    hashes = assert_frozen(cfg)

    print("\nG2  environment")
    assert_device(require_device, allow_cpu)
    dev = probe()
    dtype = torch.bfloat16 if dev.device.type == "cuda" else torch.float32
    env = {"type": dev.device.type, "name": dev.name, "torch": torch.__version__,
           "cuda_available": torch.cuda.is_available(),
           "autocast_dtype": str(dtype).replace("torch.", "")}

    set_seed(cfg.seed, strict=cfg.strict_determinism)
    tokens = np.memmap(REPO_ROOT / cfg.data_dir / "val.bin", dtype=np.uint16, mode="r")
    from src.tokenizer import load as load_tokenizer
    eot = load_tokenizer(cfg).token_to_id(cfg.doc_separator)
    # Document index per token: a new document starts after each separator.
    doc_id = np.cumsum(np.asarray(tokens) == eot, dtype=np.int32)
    print(f"  val.bin {tokens.size:,} tokens, {int(doc_id.max()) + 1:,} documents")

    print("\nG3  determinism — two full passes over the frozen set")
    blocks, ckpt_meta, models = {}, {}, {}
    for label, ck in (("base", "checkpoints/base.pt"), ("sft", "checkpoints/sft.pt")):
        model, state = load_checkpoint(REPO_ROOT / ck, dev.device)
        first = perplexity_block(model, tokens, doc_id, cfg, dev.device, dtype)
        blocks[label] = first
        models[label] = model
        ckpt_meta[label] = {"checkpoint": ck, "step": state.get("step")}
        print(f"  {label:<5} ppl {first['perplexity']:.4f}  digest {first['nll_digest']}")
    second = perplexity_block(model, tokens, doc_id, cfg, dev.device, dtype)
    g3 = second["nll_digest"] == blocks["sft"]["nll_digest"]
    print(f"  repeat pass on sft: digest {second['nll_digest']}  "
          f"-> {'BIT-IDENTICAL' if g3 else 'NON-DETERMINISTIC'}")

    print("\nG4  recompute vs recorded")
    rec = json.loads((REPO_ROOT / cfg.results_dir / "train_summary.json")
                     .read_text(encoding="utf-8"))
    rec_ppl = rec["best_val_perplexity"]
    got = blocks["base"]["perplexity"]
    # WHAT G4 COMPARES, and the first version of this got it wrong twice over
    # (ADR-051).
    #
    # The recorded figure is `best_val_perplexity` - the MINIMUM of 40 noisy
    # 100-batch estimates, saved because it was the lowest. That makes it an ORDER
    # STATISTIC, not an unbiased estimate of anything, so testing whether the
    # full-set value sits in an interval *around it* is the wrong test: the
    # reference point is itself selected for being low.
    #
    # The right test is assumption-free. Re-run the recorded ESTIMATOR, take its
    # sampling error from the spread of its own batch losses rather than from an
    # iid formula, and ask whether the full-set value is consistent with the
    # distribution the record was drawn from. The iid formula understated that
    # spread 2.4x, because a batch is contiguous chunks and tokens inside a chunk
    # correlate - ADR-033's error, and G5 twenty lines below was already doing it
    # correctly.
    est = training_estimator(models['base'], tokens, cfg, dev.device, dtype,
                             blocks['base']['nll_sd'])
    dev_nats = abs(math.log(got) - est["mean_nll"])
    g4 = dev_nats <= 2 * est["se_measured"]
    z_rec = (math.log(rec_ppl) - est["mean_nll"]) / est["se_measured"]
    print(f"  recorded  {rec_ppl:.4f}  = min of {rec['steps'] // cfg.eval_interval if 'steps' in rec else 40} "
          f"evaluations, i.e. an order statistic")
    print(f"  estimator re-run: mean ppl {math.exp(est['mean_nll']):.4f}, "
          f"measured SE {est['se_measured']:.6f} nats "
          f"({est['inflation_vs_iid']}x the iid formula)")
    print(f"  full-set  {got:.4f} over {blocks['base']['tokens_scored']:,} tokens "
          f"-> {dev_nats / est['se_measured']:+.2f} SE from the estimator mean  "
          f"{'CONSISTENT' if g4 else 'DISAGREES - surface, do not overwrite'}")
    print(f"  recorded value sits {z_rec:+.2f} SE from that mean "
          f"-> optimistically biased by selection, see ADR-051")

    print("\nG5  precision — on the MEASURED design-effect-adjusted SE")
    b = blocks["sft"]
    c = b["clustering"]
    print(f"  per-token NLL sd {b['nll_sd']:.4f} over {b['tokens_scored']:,} tokens")
    print(f"  clustering: {c['k']:,} documents, mean {c['mean_group_size']} tokens, "
          f"ICC {c['icc']:.4f}, deff {c['deff']}, n_eff {c['n_eff']:,.0f}")
    print(f"  relative SE  iid {b['relative_se_iid'] * 100:.4f}%   "
          f"MEASURED {b['relative_se_measured'] * 100:.4f}%   "
          f"({b['inflation_vs_iid']}x the iid lower bound)")
    g5 = b["relative_se_measured"] <= cfg.eval_max_relative_se
    print(f"  bar <= {cfg.eval_max_relative_se * 100:.2f}%  -> {'PASS' if g5 else 'FAIL'}")
    if not g5:
        print("  PRE-COMMITTED RESPONSE (ADR-049): the precision claim does not hold.")
        print("  Report the true interval. Do NOT relax the bar or republish a")
        print("  friendlier number - the threshold doing its job is the point.")

    gates = {"G1_eval_set_integrity": True, "G2_environment": True,
             "G3_determinism": bool(g3), "G4_recompute_vs_recorded": bool(g4),
             "G5_precision_measured": bool(g5)}
    verdict = "GREEN" if all(gates.values()) else "RED"
    summary = "   ".join(f"{name.split('_')[0]} {'ok' if ok else 'FAIL'}"
                         for name, ok in gates.items())
    print(f"\n  {summary}")
    print(f"\n  PHASE 7: {verdict}")

    out = {
        "verdict": verdict, "gates": gates,
        "environment": env,
        "eval_sets": {k: v for k, v in hashes.items()},
        "checkpoints": ckpt_meta,
        "perplexity": blocks,
        "determinism": {"repeat_digest": second["nll_digest"],
                        "bit_identical": bool(g3)},
        "recompute_vs_recorded": {
            "metric": "val_perplexity",
            "recorded": rec_ppl,
            "recorded_is": "minimum over 40 evaluations - an order statistic, "
                           "optimistically biased (ADR-051)",
            "recomputed_full_set": got,
            "estimator_rerun": {k: v for k, v in est.items()},
            "estimator_mean_perplexity": round(math.exp(est["mean_nll"]), 4),
            "full_set_deviation_se": round(dev_nats / est["se_measured"], 2),
            "recorded_deviation_se": round(z_rec, 2),
            "consistent": bool(g4)},
        "precision_bar": {"bar": cfg.eval_max_relative_se,
                          "measured": b["relative_se_measured"],
                          "iid_lower_bound": b["relative_se_iid"],
                          "passes_on": "measured deff-adjusted SE"},
        "eval_stage_hash": cfg.stage_hash("eval"),
        "model_stage_hash": cfg.stage_hash("model"),
        "elapsed_s": round(time.time() - t0, 1),
    }
    p = write_json(REPO_ROOT / cfg.results_dir / "phase7_scorecard.json", out)
    print(f"wrote {p}")
    if verdict != "GREEN":
        raise SystemExit(1)


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase 7 eval harness.")
    ap.add_argument("command", choices=("scorecard",))
    ap.add_argument("--config", default=None)
    ap.add_argument("--require-device", default="cuda")
    ap.add_argument("--allow-cpu", action="store_true")
    args = ap.parse_args()
    cmd_scorecard(Config.load(args.config), args.require_device, args.allow_cpu)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
