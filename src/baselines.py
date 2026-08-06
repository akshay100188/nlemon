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


def main() -> None:
    ap = argparse.ArgumentParser(description="Perplexity floors for the Phase 4 gate.")
    ap.add_argument("--config", default=None)
    ap.add_argument("--val-tokens", type=int, default=2_000_000,
                    help="cap on validation tokens scored (0 = all)")
    ap.add_argument("--skip-bigram", action="store_true")
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

    print(f"{'baseline':<12}{'NLL':>10}{'perplexity':>14}")
    print("-" * 36)
    for name, v in results["baselines"].items():
        print(f"{name:<12}{v['nll']:>10.4f}{v['perplexity']:>14,.2f}")
    print("-" * 36)

    lines = [
        "# Perplexity floors",
        "",
        f"Measured on {val.size:,} held-out validation tokens with a "
        f"{cfg.vocab_size:,}-token vocabulary. These are what trivial predictors "
        "score; the trained model has to beat them before *learned to speak* "
        "means anything.",
        "",
        "| baseline | what it knows | NLL | perplexity |",
        "|---|---|---|---|",
    ]
    knows = {
        "uniform": "nothing at all",
        "unigram": "which tokens are common",
        "bigram": "the previous token only",
    }
    for name, v in results["baselines"].items():
        lines.append(f"| {name} | {knows.get(name, '')} | {v['nll']:.4f} | "
                     f"{v['perplexity']:,.2f} |")
    lines += ["", "Generated by `python -m src.baselines`."]
    out = write_text(REPO_ROOT / cfg.results_dir / "perplexity_floors.md",
                     "\n".join(lines))
    write_json(REPO_ROOT / cfg.results_dir / "perplexity_floors.json", results)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
