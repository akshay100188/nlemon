"""Phase 4 — pretraining. The model learns to speak.

Next-token cross-entropy over the token shards, warmup + cosine learning rate,
gradient accumulation, bf16 autocast, periodic validation, and a checkpoint at
``checkpoints/base.pt``. Every logged step lands in ``results/loss_curve.csv``.

    python -m src.train                  # the full run
    python -m src.train --max-steps 200  # a pilot, for checking the plumbing

The loss curve is an artifact, not a debug print: Phase 8 publishes it.
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch

from config import REPO_ROOT, Config
from src.model import GPT
from utils.device import probe
from utils.io import write_json, write_text
from utils.seed import set_seed


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
class ShardLoader:
    """Samples fixed-length windows out of a flat uint16 token shard.

    Uses its own ``torch.Generator`` rather than global RNG, so the batch
    sequence depends only on the seed — not on how much RNG anything else in the
    process happened to consume. Phase 3 (ADR-015) is the reason that matters.
    """

    def __init__(self, path: Path, cfg: Config, device: torch.device, seed: int):
        if not path.exists():
            raise SystemExit(f"{path} not found - run `python -m src.tokenizer encode` first.")
        self.data = np.memmap(path, dtype=np.uint16, mode="r")
        self.ctx = cfg.context_len
        self.batch = cfg.micro_batch
        self.device = device
        self.gen = torch.Generator().manual_seed(seed)
        self.high = self.data.size - self.ctx - 1
        if self.high <= 0:
            raise SystemExit(f"{path.name} is too small for context_len {self.ctx}")

    def __len__(self) -> int:
        return int(self.data.size)

    def batch_of(self) -> tuple[torch.Tensor, torch.Tensor]:
        starts = torch.randint(self.high, (self.batch,), generator=self.gen)
        x = np.stack([self.data[s: s + self.ctx] for s in starts.tolist()])
        y = np.stack([self.data[s + 1: s + 1 + self.ctx] for s in starts.tolist()])
        xt = torch.from_numpy(x.astype(np.int64))
        yt = torch.from_numpy(y.astype(np.int64))
        if self.device.type == "cuda":
            return (xt.pin_memory().to(self.device, non_blocking=True),
                    yt.pin_memory().to(self.device, non_blocking=True))
        return xt.to(self.device), yt.to(self.device)


# --------------------------------------------------------------------------- #
# schedule + optimizer
# --------------------------------------------------------------------------- #
def lr_at(step: int, cfg: Config) -> float:
    """Linear warmup, then cosine decay to ``min_lr_ratio * lr``.

    Warmup exists because Adam's second-moment estimate is garbage for the first
    few dozen steps; stepping at full rate then is how a run diverges in its
    first minute. Cosine decay lets the model take fine steps at the end, when
    it is polishing rather than exploring.
    """
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / max(cfg.warmup_steps, 1)
    progress = (step - cfg.warmup_steps) / max(cfg.max_steps - cfg.warmup_steps, 1)
    progress = min(max(progress, 0.0), 1.0)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    min_lr = cfg.lr * cfg.min_lr_ratio
    return min_lr + coeff * (cfg.lr - min_lr)


def build_optimizer(model: GPT, cfg: Config) -> torch.optim.AdamW:
    """Weight-decay the matrices, not the biases and LayerNorms.

    Decaying a LayerNorm gain pulls it toward zero, which quietly scales the
    activations down; decaying a bias just fights the model. Only tensors with
    2+ dimensions — the things that actually multiply — get decay.
    """
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (decay if p.dim() >= 2 else no_decay).append(p)
    groups = [
        {"params": decay, "weight_decay": cfg.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=cfg.lr, betas=(cfg.beta1, cfg.beta2))


@torch.no_grad()
def evaluate(model: GPT, loaders: dict[str, ShardLoader], cfg: Config,
             dtype: torch.dtype, device: torch.device) -> dict[str, float]:
    """Mean loss over a fixed number of batches per split, in eval mode."""
    model.eval()
    out: dict[str, float] = {}
    for split, loader in loaders.items():
        losses = torch.zeros(cfg.eval_iters)
        for i in range(cfg.eval_iters):
            x, y = loader.batch_of()
            with torch.autocast(device_type=device.type, dtype=dtype,
                                enabled=device.type == "cuda"):
                _, loss = model(x, y)
            losses[i] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


# --------------------------------------------------------------------------- #
def train(cfg: Config, resume: bool = False) -> dict:
    set_seed(cfg.seed, strict=cfg.strict_determinism)
    dev = probe()
    device, dtype = dev.device, dev.dtype

    data_dir = REPO_ROOT / cfg.data_dir
    train_loader = ShardLoader(data_dir / "train.bin", cfg, device, cfg.seed)
    # A different seed for validation so the two streams are independent; the
    # eval loader is re-created each pass so every validation sees the *same*
    # batches and the curve is comparable point to point.
    val_seed = cfg.seed + 1

    model = GPT(cfg).to(device)
    opt = build_optimizer(model, cfg)

    ckpt_dir = REPO_ROOT / cfg.ckpt_dir
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "base.pt"

    start_step = 0
    best_val = float("inf")
    rows: list[str] = ["step,split,loss,perplexity,lr,tokens,elapsed_s"]
    if resume and ckpt_path.exists():
        state = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["optimizer"])
        start_step = state["step"]
        best_val = state.get("best_val_loss", float("inf"))
        print(f"resumed from {ckpt_path} at step {start_step}")

    tokens_per_step = cfg.micro_batch * cfg.grad_accum_steps * cfg.context_len
    print(f"{cfg.project_name}  ·  phase 4 · pretraining")
    print(f"train stage hash : {cfg.stage_hash('train')}")
    print(f"device           : {dev.name} ({str(dtype).replace('torch.', '')})")
    print(f"params           : {model.count_params():,}")
    print(f"tokens/step      : {tokens_per_step:,} "
          f"({cfg.micro_batch} x {cfg.grad_accum_steps} x {cfg.context_len})")
    print(f"total tokens     : {tokens_per_step * cfg.max_steps:,} over "
          f"{cfg.max_steps:,} steps")
    print(f"shard            : {len(train_loader):,} train tokens "
          f"({tokens_per_step * cfg.max_steps / len(train_loader):.2f} epochs)\n")

    model.train()
    started = time.time()
    last_log = started
    for step in range(start_step, cfg.max_steps):
        lr = lr_at(step, cfg)
        for group in opt.param_groups:
            group["lr"] = lr

        # gradient accumulation: grad_accum_steps micro-batches make one
        # optimizer step, so the effective batch is bigger than VRAM allows
        opt.zero_grad(set_to_none=True)
        total = 0.0
        for _ in range(cfg.grad_accum_steps):
            x, y = train_loader.batch_of()
            with torch.autocast(device_type=device.type, dtype=dtype,
                                enabled=device.type == "cuda"):
                _, loss = model(x, y)
            # scale so the accumulated gradient is the mean, not the sum
            (loss / cfg.grad_accum_steps).backward()
            total += loss.item() / cfg.grad_accum_steps

        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()

        done = step + 1
        if done % cfg.log_interval == 0 or step == start_step:
            now = time.time()
            tok_s = cfg.log_interval * tokens_per_step / max(now - last_log, 1e-9)
            last_log = now
            rows.append(f"{done},train,{total:.6f},{math.exp(min(total, 20)):.4f},"
                        f"{lr:.6e},{done * tokens_per_step},{now - started:.1f}")
            print(f"step {done:>6}/{cfg.max_steps}  loss {total:.4f}  "
                  f"ppl {math.exp(min(total, 20)):>8.2f}  lr {lr:.2e}  "
                  f"{tok_s:>7,.0f} tok/s")

        if done % cfg.eval_interval == 0 or done == cfg.max_steps:
            loaders = {"val": ShardLoader(data_dir / "val.bin", cfg, device, val_seed)}
            metrics = evaluate(model, loaders, cfg, dtype, device)
            vloss = metrics["val"]
            vppl = math.exp(min(vloss, 20))
            elapsed = time.time() - started
            rows.append(f"{done},val,{vloss:.6f},{vppl:.4f},{lr:.6e},"
                        f"{done * tokens_per_step},{elapsed:.1f}")
            better = vloss < best_val
            print(f"  eval  step {done:>6}  val loss {vloss:.4f}  "
                  f"val ppl {vppl:.3f}{'   <- best' if better else ''}")
            if better:
                best_val = vloss
                torch.save({
                    "model": model.state_dict(),
                    "optimizer": opt.state_dict(),
                    "step": done,
                    "best_val_loss": best_val,
                    "val_perplexity": vppl,
                    "config": cfg.as_dict(),
                    "config_hash": cfg.hash(),
                    "train_stage_hash": cfg.stage_hash("train"),
                }, ckpt_path)
            _flush_curve(cfg, rows)

    _flush_curve(cfg, rows)
    elapsed = time.time() - started
    summary = {
        "project_name": cfg.project_name,
        "config_hash": cfg.hash(),
        "train_stage_hash": cfg.stage_hash("train"),
        "steps": cfg.max_steps,
        "tokens_seen": cfg.max_steps * tokens_per_step,
        "best_val_loss": round(best_val, 6),
        "best_val_perplexity": round(math.exp(min(best_val, 20)), 4),
        "params": model.count_params(),
        "device": dev.name,
        "elapsed_seconds": round(elapsed, 1),
        "checkpoint": str(ckpt_path.relative_to(REPO_ROOT)).replace("\\", "/"),
    }
    write_json(REPO_ROOT / cfg.results_dir / "train_summary.json", summary)
    print(f"\nbest val loss {best_val:.4f}  "
          f"(perplexity {summary['best_val_perplexity']})")
    print(f"checkpoint: {ckpt_path}")
    print(f"elapsed: {elapsed / 60:.1f} min")
    return summary


def _flush_curve(cfg: Config, rows: list[str]) -> None:
    write_text(REPO_ROOT / cfg.results_dir / "loss_curve.csv", "\n".join(rows) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="nLemon-14 pretraining (Phase 4).")
    ap.add_argument("--config", default=None)
    ap.add_argument("--max-steps", type=int, default=None,
                    help="override max_steps for a pilot run")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    import dataclasses
    cfg = Config.load(args.config)
    if args.max_steps is not None:
        cfg = dataclasses.replace(cfg, max_steps=args.max_steps,
                                  warmup_steps=min(cfg.warmup_steps, args.max_steps // 10))
    train(cfg, resume=args.resume)


if __name__ == "__main__":
    main()
