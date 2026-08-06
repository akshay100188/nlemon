"""Deterministic coherence proxy — is this a little story, or word salad?

The Phase 4 gate asks whether samples are "coherent little stories". That is a
human judgement, and it stays one: Akshay reads the gallery and calls it. This
module is the *automatic* half — it catches the failure modes a human would be
bored by (degenerate loops, invented words, one endless sentence) and gives
Phase 5/6 the machinery they need anyway (ADR-005).

The thresholds are not invented. Every statistic here is measured on **real
corpus documents** first, and generated text has to land inside the percentile
band that genuine TinyStories text occupies. That way the proxy answers a
falsifiable question — "does this look statistically like the corpus?" — instead
of encoding one person's guess about what a good story looks like.

A band is a two-sided test on purpose. Too much repetition is degenerate; too
*little* is also suspicious, because real children's stories repeat names and
phrases on purpose.

    python -m src.coherence reference   # build bands from the corpus
    python -m src.coherence check       # score a checkpoint's samples
"""

from __future__ import annotations

import argparse
import re
from collections import Counter
from typing import Iterable

import numpy as np

from config import REPO_ROOT, Config
from src.tokenizer import iter_docs, text_path
from utils.io import write_json, write_text
from utils.seed import set_seed

WORD = re.compile(r"[a-z']+")
SENTENCE_END = re.compile(r"[.!?]+")

# Metrics whose band is checked. Kept explicit so the gate report and the
# reference file cannot drift apart.
METRICS = (
    "repeated_4gram_rate",
    "max_immediate_repeat_run",
    "mean_sentence_words",
    "type_token_ratio",
    "known_word_rate",
)


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def words_of(text: str) -> list[str]:
    return WORD.findall(text.lower())


def repeated_ngram_rate(tokens: list[str], n: int = 4) -> float:
    """Share of n-grams that are not seen for the first time.

    The classic degeneracy signature: a model that falls into a loop emits the
    same 4-gram over and over, driving this toward 1.0.
    """
    if len(tokens) < n + 1:
        return 0.0
    grams = [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
    counts = Counter(grams)
    repeats = sum(c - 1 for c in counts.values())
    return repeats / len(grams)


def max_immediate_repeat_run(tokens: list[str]) -> int:
    """Longest run of the same word repeated back to back ("the the the")."""
    best = run = 1
    for a, b in zip(tokens, tokens[1:]):
        run = run + 1 if a == b else 1
        best = max(best, run)
    return best if tokens else 0


def mean_sentence_words(text: str) -> float:
    parts = [p for p in SENTENCE_END.split(text) if p.strip()]
    if not parts:
        return float(len(words_of(text)))
    return float(np.mean([len(words_of(p)) for p in parts]))


def metrics_for(text: str, known: set[str]) -> dict[str, float]:
    toks = words_of(text)
    if not toks:
        return {m: 0.0 for m in METRICS}
    return {
        "repeated_4gram_rate": round(repeated_ngram_rate(toks), 6),
        "max_immediate_repeat_run": float(max_immediate_repeat_run(toks)),
        "mean_sentence_words": round(mean_sentence_words(text), 4),
        "type_token_ratio": round(len(set(toks)) / len(toks), 6),
        "known_word_rate": round(
            sum(1 for t in toks if t in known) / len(toks), 6) if known else 0.0,
    }


# --------------------------------------------------------------------------- #
# reference bands from the corpus
# --------------------------------------------------------------------------- #
def build_known_words(cfg: Config) -> set[str]:
    """Vocabulary the model actually saw, from the training text."""
    known: set[str] = set()
    for doc in iter_docs(text_path(cfg, "train"), cfg.doc_separator,
                         limit=cfg.coherence_vocab_docs):
        known.update(words_of(doc))
    return known


def build_reference(cfg: Config, known: set[str]) -> dict:
    """Percentile bands for each metric over real validation documents."""
    per_doc: list[dict[str, float]] = []
    for doc in iter_docs(text_path(cfg, "val"), cfg.doc_separator,
                         limit=cfg.coherence_ref_docs):
        per_doc.append(metrics_for(doc, known))

    bands = {}
    for m in METRICS:
        vals = np.array([d[m] for d in per_doc], dtype=np.float64)
        bands[m] = {
            "low": round(float(np.percentile(vals, cfg.coherence_band_low_pct)), 6),
            "high": round(float(np.percentile(vals, cfg.coherence_band_high_pct)), 6),
            "median": round(float(np.median(vals)), 6),
        }
    return {
        "docs": len(per_doc),
        "known_words": len(known),
        "band_percentiles": [cfg.coherence_band_low_pct, cfg.coherence_band_high_pct],
        "bands": bands,
    }


def score_samples(texts: Iterable[str], known: set[str], reference: dict) -> dict:
    """Median metric across samples, checked against the corpus bands.

    The median, not the mean: one degenerate sample should not be averaged away
    by fifteen good ones, and it should not condemn them either. Per-sample
    detail is kept so a red result can be read rather than guessed at.
    """
    per_sample = [metrics_for(t, known) for t in texts]
    bands = reference["bands"]
    checks = {}
    for m in METRICS:
        vals = [s[m] for s in per_sample]
        median = round(float(np.median(vals)), 6)
        low, high = bands[m]["low"], bands[m]["high"]
        checks[m] = {
            "sample_median": median,
            "corpus_band": [low, high],
            "corpus_median": bands[m]["median"],
            "in_band": bool(low <= median <= high),
        }
    return {
        "n_samples": len(per_sample),
        "checks": checks,
        "passed": all(c["in_band"] for c in checks.values()),
        "per_sample": per_sample,
    }


# --------------------------------------------------------------------------- #
def cmd_reference(cfg: Config) -> dict:
    print("building known-word set from training text...")
    known = build_known_words(cfg)
    print(f"  {len(known):,} distinct words from {cfg.coherence_vocab_docs:,} docs")
    print(f"building reference bands from {cfg.coherence_ref_docs:,} val docs...")
    ref = build_reference(cfg, known)

    print(f"\n{'metric':<28}{'p' + str(cfg.coherence_band_low_pct):>10}"
          f"{'median':>10}{'p' + str(cfg.coherence_band_high_pct):>10}")
    print("-" * 58)
    for m in METRICS:
        b = ref["bands"][m]
        print(f"{m:<28}{b['low']:>10.4f}{b['median']:>10.4f}{b['high']:>10.4f}")

    write_json(REPO_ROOT / cfg.results_dir / "coherence_reference.json", ref)
    lines = [
        "# What real TinyStories text looks like",
        "",
        f"Measured on {ref['docs']:,} held-out validation documents, with a "
        f"known-word set of {ref['known_words']:,} words taken from "
        f"{cfg.coherence_vocab_docs:,} training documents.",
        "",
        "These bands are the reference the generated samples are judged against, "
        "so the coherence check tests *resemblance to the corpus* rather than "
        "anyone's opinion about good prose. Both edges matter: too much "
        "repetition is degenerate, too little is suspicious, because real "
        "children's stories repeat names and phrases deliberately.",
        "",
        f"| metric | p{cfg.coherence_band_low_pct} | median | "
        f"p{cfg.coherence_band_high_pct} |",
        "|---|---|---|---|",
    ]
    for m in METRICS:
        b = ref["bands"][m]
        lines.append(f"| `{m}` | {b['low']:.4f} | {b['median']:.4f} | {b['high']:.4f} |")
    lines += ["", "Generated by `python -m src.coherence reference`."]
    out = write_text(REPO_ROOT / cfg.results_dir / "coherence_reference.md",
                     "\n".join(lines))
    print(f"\nwrote {out}")
    return ref


def cmd_gate(cfg: Config, checkpoint: str) -> None:
    """The Phase 4 gate: validation perplexity + the coherence proxy.

    Perplexity is recomputed from the checkpoint rather than read out of
    train_summary.json. The summary is what the training run *said*; the gate
    should verify it independently, the same reason the parameter count is
    derived twice (ADR-013).
    """
    import json
    import math

    import torch

    from src.model import GPT  # noqa: F401  (loaded via sample.load_checkpoint)
    from src.sample import GALLERY_PROMPTS, load_checkpoint, generate
    from src.tokenizer import load as load_tokenizer
    from src.train import ShardLoader, evaluate
    from utils.device import probe

    ref_path = REPO_ROOT / cfg.results_dir / "coherence_reference.json"
    if not ref_path.exists():
        raise SystemExit("run `python -m src.coherence reference` first.")
    reference = json.loads(ref_path.read_text(encoding="utf-8"))

    dev = probe()
    model, state = load_checkpoint(REPO_ROOT / checkpoint, dev.device)
    tok = load_tokenizer(cfg)

    print(f"{cfg.project_name}  ·  phase 4 gate")
    print(f"train stage hash : {state.get('train_stage_hash', '?')}")
    print(f"checkpoint       : {checkpoint} @ step {state.get('step')}\n")

    # (a) perplexity, recomputed
    loader = ShardLoader(REPO_ROOT / cfg.data_dir / "val.bin", cfg,
                         dev.device, cfg.seed + 1)
    metrics = evaluate(model, {"val": loader}, cfg, dev.dtype, dev.device)
    val_loss = metrics["val"]
    val_ppl = math.exp(val_loss)
    ppl_ok = val_ppl <= cfg.val_ppl_threshold
    print(f"[{'PASS' if ppl_ok else 'FAIL'}] val perplexity {val_ppl:.4f} "
          f"<= {cfg.val_ppl_threshold} (loss {val_loss:.4f})")
    print(f"        floors: bigram 41.81, unigram 389.91, uniform {cfg.vocab_size:,}")

    # (b) coherence proxy on generated continuations
    known = build_known_words(cfg)
    eot = tok.token_to_id(cfg.doc_separator)
    texts = []
    for i in range(cfg.coherence_samples):
        prompt = GALLERY_PROMPTS[i % len(GALLERY_PROMPTS)]
        gen = torch.Generator(device="cpu").manual_seed(cfg.seed + i)
        ids = generate(model, tok.encode(prompt).ids, cfg.coherence_new_tokens,
                       cfg.coherence_temperature, cfg.coherence_top_k,
                       dev.device, eot_id=eot, generator=gen)
        texts.append(tok.decode(ids))

    scored = score_samples(texts, known, reference)
    print(f"\n[{'PASS' if scored['passed'] else 'FAIL'}] coherence proxy on "
          f"{scored['n_samples']} generated samples")
    print(f"        {'metric':<28}{'sample':>10}{'corpus band':>22}{'':>4}")
    for m, c in scored["checks"].items():
        band = f"[{c['corpus_band'][0]:.4f}, {c['corpus_band'][1]:.4f}]"
        print(f"        {m:<28}{c['sample_median']:>10.4f}{band:>22}"
              f"  {'ok' if c['in_band'] else 'OUT'}")

    passed = bool(ppl_ok and scored["passed"])
    result = {
        "checkpoint": checkpoint,
        "step": state.get("step"),
        "train_stage_hash": state.get("train_stage_hash"),
        "perplexity": {
            "val_loss": round(val_loss, 6),
            "val_perplexity": round(val_ppl, 4),
            "threshold": cfg.val_ppl_threshold,
            "passed": ppl_ok,
            "recomputed_from_checkpoint": True,
        },
        "coherence": {k: v for k, v in scored.items() if k != "per_sample"},
        "coherence_per_sample": scored["per_sample"],
        "passed": passed,
    }
    write_json(REPO_ROOT / cfg.results_dir / "phase4_gate.json", result)
    print(f"\nPHASE 4 GATE: {'PASS' if passed else 'FAIL'}")
    print("(the human half - 'are these little stories?' - is the gallery in "
          "results/samples/base_samples.md)")
    if not passed:
        raise SystemExit(1)


def main() -> None:
    ap = argparse.ArgumentParser(description="Deterministic coherence proxy.")
    ap.add_argument("command", choices=("reference", "gate"))
    ap.add_argument("--config", default=None)
    ap.add_argument("--checkpoint", default="checkpoints/base.pt")
    args = ap.parse_args()

    cfg = Config.load(args.config)
    set_seed(cfg.seed, strict=cfg.strict_determinism)
    if args.command == "reference":
        cmd_reference(cfg)
    else:
        cmd_gate(cfg, args.checkpoint)


if __name__ == "__main__":
    main()
