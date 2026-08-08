"""Phase 5 — supervised fine-tuning. base.pt -> sft.pt.

The base model speaks English. It does not take instructions: prompt it with
"Write a story about a needle." and it continues the sentence rather than
answering it. This stage teaches the answer-shaped behaviour, and only that -
`sft_lr` is 6x below pretrain's because the job is to bend behaviour without
overwriting what the model knows.

**Loss is masked to the response.** Prompt positions carry `IGNORE`, so the model
is never rewarded for predicting the instruction. That is not a small detail: a
masking bug that is wrong still trains to a plausible-looking loss curve, because
predicting the instruction - which the model can already see in its own context -
is easy and pulls the average down. A broken mask therefore looks like a *healthy*
run. So the mask is asserted directly on the tensor the optimizer sees, before
step 1, and this trainer refuses to start until that assertion has run
(`python -m src.sft_data mask`). Test the property, never the proxy - the same
rule as the causal-mask check in Phase 3 (ADR-014).

**Checkpoint selection never touches the gate.** Validation here is a slice of the
*train-subject* pairs. The 200 held-out-subject prompts are the gate, and using
them to pick a checkpoint would turn the thing being measured into a training
signal.

**What could go wrong, named in advance.** Three epochs of narrow instruction data
can cost general fluency - catastrophic forgetting, bought as adherence. That is
exactly what the `is_story` and `not_degenerate` floors are for. If they break,
the remedy is mixing raw corpus batches back in, not lowering the floors.

    python -m src.sft train
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from config import REPO_ROOT, Config
from src.model import GPT
from src.sft_data import IGNORE
from src.train import build_optimizer
from utils.device import probe
from utils.io import write_json, write_text
from utils.seed import set_seed


class PairLoader:
    """Batches of masked instruction pairs.

    Unlike `ShardLoader`, which samples random windows out of a token stream, an
    SFT epoch is a *permutation*: every pair is seen exactly once per epoch. That
    matters for a 37k-example set - random sampling with replacement would show
    some pairs five times and others never, and "three epochs" would stop meaning
    anything.
    """

    def __init__(self, x: np.ndarray, y: np.ndarray, batch: int,
                 device: torch.device, seed: int):
        self.x, self.y = x, y
        self.batch = batch
        self.device = device
        self.rng = np.random.default_rng(seed)
        self.order = self.rng.permutation(len(x))
        self.pos = 0
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.x) // self.batch

    def next_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.pos + self.batch > len(self.order):
            self.order = self.rng.permutation(len(self.x))
            self.pos = 0
            self.epoch += 1
        idx = self.order[self.pos:self.pos + self.batch]
        self.pos += self.batch
        xb = torch.from_numpy(self.x[idx].astype(np.int64)).to(self.device)
        yb = torch.from_numpy(self.y[idx].astype(np.int64)).to(self.device)
        return xb, yb


def masked_loss(model: GPT, xb: torch.Tensor, yb: torch.Tensor) -> torch.Tensor:
    """Cross-entropy over response positions only.

    The model's own forward pass computes a loss when handed targets, but it does
    not know about IGNORE, so the loss is taken here from the logits. `yb` already
    carries IGNORE on prompt and padding positions (see `src.sft_data.pack`), and
    `ignore_index` drops exactly those terms from both the average and the
    gradient.
    """
    logits, _ = model(xb)
    return F.cross_entropy(logits.view(-1, logits.size(-1)), yb.view(-1),
                           ignore_index=IGNORE)


def sft_lr_at(step: int, total: int, cfg: Config) -> float:
    """Warmup then cosine, over the fine-tune's own (much shorter) horizon."""
    if step < cfg.sft_warmup_steps:
        return cfg.sft_lr * (step + 1) / cfg.sft_warmup_steps
    prog = (step - cfg.sft_warmup_steps) / max(total - cfg.sft_warmup_steps, 1)
    prog = min(max(prog, 0.0), 1.0)
    floor = cfg.sft_lr * cfg.sft_min_lr_ratio
    return floor + 0.5 * (cfg.sft_lr - floor) * (1 + math.cos(math.pi * prog))


@torch.no_grad()
def evaluate(model: GPT, xv: np.ndarray, yv: np.ndarray, batch: int,
             device: torch.device) -> float:
    """Full pass over the validation slice - it is small enough to not sample."""
    model.eval()
    total, n_batches = 0.0, 0
    for i in range(0, len(xv) - batch + 1, batch):
        xb = torch.from_numpy(xv[i:i + batch].astype(np.int64)).to(device)
        yb = torch.from_numpy(yv[i:i + batch].astype(np.int64)).to(device)
        total += float(masked_loss(model, xb, yb).item())
        n_batches += 1
    model.train()
    return total / max(n_batches, 1)


def require_mask_assertion(cfg: Config) -> None:
    """Re-run the mask assertion here, every time, before step 1.

    The first version of this read a JSON receipt written by
    `python -m src.sft_data mask`. That is a gate reading a summary, which this
    project has already been burned by twice - `verify_docs` skipping real claims
    while printing success (ADR-021), and a threshold trusted from a file instead
    of recomputed (ADR-029). Edit `pack()` after running the CLI and a receipt
    still says "passed".

    So the assertion is recomputed on live tensors instead. It costs one pair.
    """
    from src.sft_data import assert_mask

    rec = assert_mask(cfg, verbose=False)
    print(f"mask assertion recomputed: {rec['supervised_positions']} supervised "
          f"positions == {rec['response_tokens']} response tokens, "
          f"{rec['ignored_prompt_positions']} prompt positions ignored")


def load_tensors(cfg: Config) -> tuple[np.ndarray, ...]:
    d = REPO_ROOT / cfg.data_dir
    if not (d / "sft_x.npy").exists():
        raise SystemExit("run `python -m src.sft_data build` first.")
    x = np.load(d / "sft_x.npy")
    y = np.load(d / "sft_y.npy")
    # Deterministic split, and by INDEX rather than by shuffling the arrays, so
    # the same config always carves the same validation slice.
    n_val = max(int(cfg.sft_val_frac * len(x)), cfg.sft_micro_batch)
    rng = np.random.default_rng(cfg.seed)
    perm = rng.permutation(len(x))
    vi, ti = perm[:n_val], perm[n_val:]
    return x[ti], y[ti], x[vi], y[vi]


def train(cfg: Config) -> dict:
    require_mask_assertion(cfg)
    set_seed(cfg.seed, strict=cfg.strict_determinism)
    dev = probe()

    xt, yt, xv, yv = load_tensors(cfg)
    print(f"SFT data: {len(xt):,} train pairs, {len(xv):,} val pairs "
          f"(val is train-subject; the gate's held-out subjects are untouched)")

    base_path = REPO_ROOT / cfg.ckpt_dir / "base.pt"
    if not base_path.exists():
        raise SystemExit(f"missing {base_path} - Phase 4 must be green first.")
    ckpt = torch.load(base_path, map_location="cpu", weights_only=False)
    model = GPT(cfg).to(dev.device)
    model.load_state_dict(ckpt["model"])
    print(f"loaded base.pt at step {ckpt['step']}, "
          f"val ppl {ckpt.get('val_perplexity')}")
    print(f"  base train-stage hash: {ckpt.get('train_stage_hash')}")

    # Fresh optimizer state: base.pt's Adam moments belong to a cosine schedule
    # that finished. Carrying them in would apply pretrain's momentum to a
    # different objective.
    opt = build_optimizer(model, cfg)
    for g in opt.param_groups:
        g["lr"] = cfg.sft_lr

    per_epoch = len(xt) // (cfg.sft_micro_batch * cfg.sft_grad_accum_steps)
    total_steps = per_epoch * cfg.sft_epochs
    loader = PairLoader(xt, yt, cfg.sft_micro_batch, dev.device, cfg.seed)
    print(f"  {per_epoch:,} optimizer steps/epoch x {cfg.sft_epochs} epochs "
          f"= {total_steps:,} steps")
    print(f"  effective batch {cfg.sft_micro_batch * cfg.sft_grad_accum_steps} "
          f"sequences, lr {cfg.sft_lr:g} -> "
          f"{cfg.sft_lr * cfg.sft_min_lr_ratio:g}")

    # The loss at step 0 is the base model's loss on instruction data. It is the
    # honest starting point for the curve, so it is recorded before any update.
    start_val = evaluate(model, xv, yv, cfg.sft_micro_batch, dev.device)
    print(f"\n  base.pt on SFT val (before any update): {start_val:.4f}")

    ckpt_path = REPO_ROOT / cfg.ckpt_dir / "sft.pt"
    rows = ["step,split,loss,lr,epoch,elapsed_s"]
    rows.append(f"0,val,{start_val:.6f},0,0,0.0")
    best_val, best_step = start_val, 0
    started = time.time()
    amp = torch.autocast(device_type=dev.device.type, dtype=torch.bfloat16) \
        if dev.device.type == "cuda" else torch.autocast("cpu", enabled=False)

    model.train()
    for step in range(total_steps):
        lr = sft_lr_at(step, total_steps, cfg)
        for g in opt.param_groups:
            g["lr"] = lr

        opt.zero_grad(set_to_none=True)
        acc = 0.0
        for _ in range(cfg.sft_grad_accum_steps):
            xb, yb = loader.next_batch()
            with amp:
                loss = masked_loss(model, xb, yb)
            (loss / cfg.sft_grad_accum_steps).backward()
            acc += float(loss.item())
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
        tl = acc / cfg.sft_grad_accum_steps

        if (step + 1) % cfg.log_interval == 0:
            rows.append(f"{step + 1},train,{tl:.6f},{lr:.6e},{loader.epoch},"
                        f"{time.time() - started:.1f}")
            print(f"  step {step + 1:>5}/{total_steps}  loss {tl:.4f}  "
                  f"lr {lr:.2e}  epoch {loader.epoch}")

        if (step + 1) % cfg.sft_eval_interval == 0 or step + 1 == total_steps:
            vl = evaluate(model, xv, yv, cfg.sft_micro_batch, dev.device)
            rows.append(f"{step + 1},val,{vl:.6f},{lr:.6e},{loader.epoch},"
                        f"{time.time() - started:.1f}")
            better = vl < best_val
            print(f"  eval  step {step + 1:>5}  val loss {vl:.4f}"
                  f"{'   <- best' if better else ''}")
            if better:
                best_val, best_step = vl, step + 1
                torch.save({
                    "model": model.state_dict(),
                    "optimizer": opt.state_dict(),
                    "step": step + 1,
                    "best_val_loss": best_val,
                    "base_step": ckpt["step"],
                    "base_train_stage_hash": ckpt.get("train_stage_hash"),
                    "config": cfg.as_dict(),
                    "config_hash": cfg.hash(),
                    "sft_stage_hash": cfg.stage_hash("sft"),
                }, ckpt_path)
            _flush(cfg, rows)

    _flush(cfg, rows)
    elapsed = time.time() - started
    summary = {
        "steps": total_steps,
        "epochs": cfg.sft_epochs,
        "steps_per_epoch": per_epoch,
        "train_pairs": len(xt),
        "val_pairs": len(xv),
        "base_val_loss_before_sft": round(start_val, 6),
        "best_val_loss": round(best_val, 6),
        "best_step": best_step,
        "val_loss_improvement": round(start_val - best_val, 6),
        "lr": cfg.sft_lr,
        "elapsed_s": round(elapsed, 1),
        "device": dev.name,
        "sft_stage_hash": cfg.stage_hash("sft"),
        "config_hash": cfg.hash(),
    }
    write_json(REPO_ROOT / cfg.results_dir / "sft_train_summary.json", summary)
    print(f"\nsft.pt at step {best_step}, val loss {best_val:.4f} "
          f"(base started at {start_val:.4f}, "
          f"improved {start_val - best_val:.4f})")
    print(f"  {elapsed / 60:.1f} min on {dev.name}")
    return summary


def _flush(cfg: Config, rows: list[str]) -> None:
    write_text(REPO_ROOT / cfg.results_dir / "sft_curve.csv", "\n".join(rows) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="Supervised fine-tuning (Phase 5).")
    ap.add_argument("command", choices=("train",))
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    train(Config.load(args.config))


if __name__ == "__main__":
    main()
