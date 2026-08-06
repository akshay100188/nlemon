"""Perplexity floors, so the Phase 4 gate threshold means something.

"Validation perplexity below N" is only a claim if you know what N would be for
a model that learned nothing. These are the trivial predictors, measured on the
same held-out shard the real model is scored on:

  uniform  — every token equally likely. Perplexity = vocab_size, by definition.
  unigram  — token frequencies from train, ignoring all context.
  bigram   — the previous token only, with backoff to unigram.

A model that beats unigram has learned that some words are common. A model that
beats bigram has learned something about context. Neither is impressive; they
are the floor the real number has to clear before "it learned to speak" is
anything more than a hopeful adjective.

    python -m src.baselines
"""

from __future__ import annotations

import argparse
import math

import numpy as np

from config import REPO_ROOT, Config
from utils.io import write_json, write_text
from utils.seed import set_seed


def load_shard(cfg: Config, split: str) -> np.ndarray:
    path = REPO_ROOT / cfg.data_dir / f"{split}.bin"
    if not path.exists():
        raise SystemExit(f"{path} not found - run `python -m src.tokenizer encode` first.")
    return np.memmap(path, dtype=np.uint16, mode="r")


def unigram_nll(train: np.ndarray, val: np.ndarray, vocab: int) -> float:
    """Mean negative log-likelihood of val under train's token frequencies."""
    counts = np.bincount(np.asarray(train, dtype=np.int64), minlength=vocab).astype(np.float64)
    probs = (counts + 1.0) / (counts.sum() + vocab)          # Laplace smoothing
    logp = np.log(probs)
    return float(-logp[np.asarray(val, dtype=np.int64)].mean())


def bigram_nll(train: np.ndarray, val: np.ndarray, vocab: int,
               backoff: float = 0.1) -> float:
    """Mean NLL under P(next | previous), backed off to the unigram.

    A full vocab x vocab table is 64M float64 entries — 512 MB — so this uses a
    sparse count matrix built from the training shard.
    """
    from scipy import sparse  # noqa: F401  (imported lazily; optional dep)

    tr = np.asarray(train, dtype=np.int64)
    counts = sparse.coo_matrix(
        (np.ones(tr.size - 1, dtype=np.float32), (tr[:-1], tr[1:])),
        shape=(vocab, vocab),
    ).tocsr()
    row_sums = np.asarray(counts.sum(axis=1)).ravel()

    uni_counts = np.bincount(tr, minlength=vocab).astype(np.float64)
    uni = (uni_counts + 1.0) / (uni_counts.sum() + vocab)

    va = np.asarray(val, dtype=np.int64)
    prev, nxt = va[:-1], va[1:]
    # P = (1-b) * bigram + b * unigram, so unseen pairs still get mass
    pair = np.asarray(counts[prev, nxt]).ravel()
    denom = row_sums[prev]
    bigram_p = np.divide(pair, denom, out=np.zeros_like(pair, dtype=np.float64),
                         where=denom > 0)
    p = (1.0 - backoff) * bigram_p + backoff * uni[nxt]
    return float(-np.log(np.maximum(p, 1e-12)).mean())


def model_nll(cfg: Config, val: np.ndarray, checkpoint: str) -> dict:
    """Score the checkpoint on the *same* tokens the baselines just used.

    The gate's val perplexity comes from random 256-token windows; the baselines
    read a contiguous prefix. Both are per-token BPE negative log-likelihood on
    val.bin, but they are different subsets, so a ratio between them is only
    approximately apples-to-apples. This scores the model on exactly the array
    passed in, so "N times better than bigram" has one denominator and one
    numerator drawn from identical data.

    Non-overlapping windows: the first token of each window is predicted with no
    context, which slightly handicaps the model (1 token in context_len). The
    ratio it produces is therefore conservative.
    """
    import torch

    from src.sample import load_checkpoint
    from utils.device import probe

    dev = probe()
    model, _ = load_checkpoint(REPO_ROOT / checkpoint, dev.device)
    ctx = cfg.context_len
    usable = (val.size // ctx) * ctx
    windows = np.asarray(val[:usable], dtype=np.int64).reshape(-1, ctx)

    total_nll = 0.0
    total_tokens = 0
    batch = cfg.micro_batch
    with torch.no_grad():
        for start in range(0, windows.shape[0], batch):
            chunk = windows[start:start + batch]
            x = torch.from_numpy(chunk[:, :-1]).to(dev.device)
            y = torch.from_numpy(chunk[:, 1:]).to(dev.device)
            with torch.autocast(device_type=dev.device.type, dtype=dev.dtype,
                                enabled=dev.device.type == "cuda"):
                logits, _ = model(x)
            # sum, not mean, so windows of differing size cannot skew the average
            nll = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)).float(),
                y.reshape(-1), reduction="sum")
            total_nll += float(nll.item())
            total_tokens += int(y.numel())

    mean_nll = total_nll / total_tokens
    return {
        "checkpoint": checkpoint,
        "nll": round(mean_nll, 6),
        "perplexity": round(math.exp(mean_nll), 4),
        "tokens_scored": total_tokens,
        "windows": int(windows.shape[0]),
        "note": "non-overlapping windows; first token of each has no context",
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Perplexity floors for the Phase 4 gate.")
    ap.add_argument("--config", default=None)
    ap.add_argument("--val-tokens", type=int, default=2_000_000,
                    help="cap on validation tokens scored (0 = all)")
    ap.add_argument("--skip-bigram", action="store_true")
    ap.add_argument("--with-model", metavar="CKPT", default=None,
                    help="also score a checkpoint on the identical tokens")
    args = ap.parse_args()

    cfg = Config.load(args.config)
    set_seed(cfg.seed, strict=cfg.strict_determinism)

    train = load_shard(cfg, "train")
    val = load_shard(cfg, "val")
    if args.val_tokens and val.size > args.val_tokens:
        val = val[: args.val_tokens]

    print(f"{cfg.project_name}  ·  perplexity floors")
    print(f"train tokens : {train.size:,}")
    print(f"val tokens   : {val.size:,}\n")

    results = {
        "vocab_size": cfg.vocab_size,
        "train_tokens": int(train.size),
        "val_tokens": int(val.size),
        "baselines": {},
    }

    uniform_ppl = float(cfg.vocab_size)
    results["baselines"]["uniform"] = {
        "nll": round(math.log(cfg.vocab_size), 6),
        "perplexity": round(uniform_ppl, 4),
    }

    u = unigram_nll(train, val, cfg.vocab_size)
    results["baselines"]["unigram"] = {"nll": round(u, 6),
                                       "perplexity": round(math.exp(u), 4)}

    if not args.skip_bigram:
        try:
            b = bigram_nll(train, val, cfg.vocab_size)
            results["baselines"]["bigram"] = {"nll": round(b, 6),
                                              "perplexity": round(math.exp(b), 4)}
        except ImportError:
            print("scipy not installed - skipping the bigram floor\n")

    if args.with_model:
        results["model"] = model_nll(cfg, val, args.with_model)

    model_ppl = results.get("model", {}).get("perplexity")
    width = 54 if model_ppl else 36
    print(f"{'baseline':<12}{'NLL':>10}{'perplexity':>14}{'model is':>18}")
    print("-" * width)
    for name, v in results["baselines"].items():
        ratio = f"{v['perplexity'] / model_ppl:,.2f}x better" if model_ppl else ""
        print(f"{name:<12}{v['nll']:>10.4f}{v['perplexity']:>14,.2f}{ratio:>18}")
    if model_ppl:
        m = results["model"]
        scored = f"{m['tokens_scored']:,} tok"
        print("-" * width)
        print(f"{cfg.project_name:<12}{m['nll']:>10.4f}{m['perplexity']:>14,.4f}"
              f"{scored:>18}")
        # Ratios are computed and recorded here so nobody has to derive them by
        # hand in prose. That is exactly how a stale denominator got shipped.
        results["ratios_vs_model"] = {
            name: round(v["perplexity"] / model_ppl, 4)
            for name, v in results["baselines"].items()
        }
    print("-" * width)
    if model_ppl:
        print("\nModel and baselines are scored on the identical token array, so "
              "each ratio\nhas one numerator and one denominator drawn from the "
              "same data.")

    lines = [
        "# Perplexity floors",
        "",
        f"Measured on {val.size:,} held-out validation tokens with a "
        f"{cfg.vocab_size:,}-token vocabulary. These are what trivial predictors "
        "score; the trained model has to beat them before *learned to speak* "
        "means anything.",
        "",
        "| baseline | what it knows | NLL | perplexity |"
        + (" nLemon-14 is |" if model_ppl else ""),
        "|---|---|---|---|" + ("---|" if model_ppl else ""),
    ]
    knows = {
        "uniform": "nothing at all",
        "unigram": "which tokens are common",
        "bigram": "the previous token only",
    }
    for name, v in results["baselines"].items():
        row = (f"| {name} | {knows.get(name, '')} | {v['nll']:.4f} | "
               f"{v['perplexity']:,.2f} |")
        if model_ppl:
            row += f" **{v['perplexity'] / model_ppl:,.2f}x better** |"
        lines.append(row)
    if model_ppl:
        m = results["model"]
        lines += [
            "",
            f"**{cfg.project_name}: NLL {m['nll']:.4f}, perplexity "
            f"{m['perplexity']:,.4f}** over {m['tokens_scored']:,} of these same "
            f"tokens.",
            "",
            "Model and baselines are scored on the **identical token array** here, "
            "so the ratios have one numerator and one denominator drawn from the "
            "same data. All figures are per-token negative log-likelihood under "
            "the same 8,000-token BPE — perplexity is not comparable across "
            "tokenizers, so a number from a different vocabulary would not belong "
            "in this table.",
            "",
            f"The model is scored on non-overlapping {cfg.context_len}-token "
            "windows, so the first token of each window is predicted with no "
            "context. That handicaps the model slightly, which makes these ratios "
            "conservative.",
        ]
    lines += ["", "Generated by `python -m src.baselines`."]
    out = write_text(REPO_ROOT / cfg.results_dir / "perplexity_floors.md",
                     "\n".join(lines))
    write_json(REPO_ROOT / cfg.results_dir / "perplexity_floors.json", results)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
