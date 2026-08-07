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
)

# Rare-event metrics, pooled across the whole sample set rather than taken as a
# per-document median (ADR-022).
#
# `known_word_rate` used to be a banded metric and was degenerate: over 95% of
# real validation documents contain zero out-of-vocabulary words, so p5 and p95
# both landed on 1.0 and the "band" was a point. It passed, but it could only
# ever answer "did the median document invent a word", which is not the question
# Phase 5 needs. SFT has to tell a plausible new name from garbage, and that is a
# character-level question, not a vocabulary-membership one.
#
# Two metrics replace it:
#   oov_rate         - pooled over all words in all samples, so a rare event has
#                      somewhere to vary. The band comes from bootstrapping real
#                      documents in groups the same size as the sample set.
#   oov_plausibility - mean character-trigram log-probability of the words that
#                      *are* out of vocabulary. "Timmothy" scores like English;
#                      "xqzvt" does not. This is the metric that survives SFT.
POOLED_METRICS = ("oov_rate", "oov_plausibility")
ALL_METRICS = METRICS + POOLED_METRICS


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
    }


# --------------------------------------------------------------------------- #
# character-level plausibility of out-of-vocabulary words
# --------------------------------------------------------------------------- #
BOUNDARY = "\x02"       # word-start padding for the trigram context
ADD_K = 0.5


class CharModel:
    """Character trigram model over corpus words.

    Answers the question vocabulary membership cannot: is this unknown word
    shaped like English? A name the model invented and a random byte sequence are
    both out of vocabulary, and only one of them is a failure.
    """

    def __init__(self, words: Iterable[str]):
        self.counts: dict[str, Counter] = {}
        self.alphabet: set[str] = set()
        for w in words:
            padded = BOUNDARY * 2 + w
            self.alphabet.update(w)
            for i in range(2, len(padded)):
                ctx = padded[i - 2:i]
                self.counts.setdefault(ctx, Counter())[padded[i]] += 1
        self.vocab_size = max(len(self.alphabet), 1)

    def logprob(self, word: str) -> float:
        """Mean log-probability per character. Higher is more English-shaped."""
        if not word:
            return 0.0
        padded = BOUNDARY * 2 + word
        total = 0.0
        for i in range(2, len(padded)):
            ctx, ch = padded[i - 2:i], padded[i]
            row = self.counts.get(ctx)
            if row is None:
                total += np.log(1.0 / self.vocab_size)
            else:
                total += np.log((row.get(ch, 0) + ADD_K)
                                / (sum(row.values()) + ADD_K * self.vocab_size))
        return total / (len(padded) - 2)


def pooled_metrics(texts: Iterable[str], known: set[str],
                   char_model: CharModel) -> dict[str, float]:
    """Rare-event metrics pooled over every word in every sample."""
    total = 0
    oov: list[str] = []
    for text in texts:
        for w in words_of(text):
            total += 1
            if w not in known:
                oov.append(w)
    if not total:
        return {"oov_rate": 0.0, "oov_plausibility": float("nan")}
    # No OOV words means there is nothing to score. NaN, not 0.0 - zero would
    # read as "maximally implausible" and quietly fail a band it never entered.
    plausibility = (float(np.mean([char_model.logprob(w) for w in oov]))
                    if oov else float("nan"))
    return {
        "oov_rate": round(len(oov) / total, 8),
        "oov_plausibility": (round(plausibility, 6)
                             if plausibility == plausibility else float("nan")),
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


def build_reference(cfg: Config, known: set[str],
                    char_model: CharModel) -> dict:
    """Percentile bands over real validation documents.

    Per-document metrics band on their per-document distribution. The pooled
    metrics cannot: out-of-vocabulary words are rare enough that most single
    documents contain none, which is what made `known_word_rate` degenerate. They
    are bootstrapped instead — real documents are drawn in groups the same size
    as the sample set we will judge, and the band is the spread across groups.
    Same statistic, same sample size, so the comparison is like for like.
    """
    docs = list(iter_docs(text_path(cfg, "val"), cfg.doc_separator,
                          limit=cfg.coherence_ref_docs))
    per_doc = [metrics_for(d, known) for d in docs]

    bands = {}
    for m in METRICS:
        vals = np.array([d[m] for d in per_doc], dtype=np.float64)
        bands[m] = {
            "low": round(float(np.percentile(vals, cfg.coherence_band_low_pct)), 6),
            "high": round(float(np.percentile(vals, cfg.coherence_band_high_pct)), 6),
            "median": round(float(np.median(vals)), 6),
        }

    rng = np.random.default_rng(cfg.seed)
    group = min(cfg.coherence_samples, len(docs))
    pooled_draws: dict[str, list[float]] = {m: [] for m in POOLED_METRICS}
    for _ in range(cfg.coherence_bootstrap):
        idx = rng.choice(len(docs), size=group, replace=False)
        stats = pooled_metrics([docs[i] for i in idx], known, char_model)
        for m in POOLED_METRICS:
            v = stats[m]
            if v == v:  # skip NaN: a group with no OOV words says nothing here
                pooled_draws[m].append(v)
    for m in POOLED_METRICS:
        vals = np.array(pooled_draws[m], dtype=np.float64)
        if vals.size == 0:
            bands[m] = {"low": float("nan"), "high": float("nan"),
                        "median": float("nan"), "draws": 0}
            continue
        bands[m] = {
            "low": round(float(np.percentile(vals, cfg.coherence_band_low_pct)), 6),
            "high": round(float(np.percentile(vals, cfg.coherence_band_high_pct)), 6),
            "median": round(float(np.median(vals)), 6),
            "draws": int(vals.size),
        }

    return {
        "docs": len(per_doc),
        "known_words": len(known),
        "band_percentiles": [cfg.coherence_band_low_pct, cfg.coherence_band_high_pct],
        "bootstrap_groups": cfg.coherence_bootstrap,
        "bootstrap_group_size": group,
        "bands": bands,
    }


def score_samples(texts: Iterable[str], known: set[str], reference: dict,
                  char_model: CharModel | None = None) -> dict:
    """Score samples against the corpus bands.

    Per-document metrics use the median across samples, not the mean: one
    degenerate sample should not be averaged away by fifteen good ones, nor
    condemn them. Pooled metrics use every word in every sample, because that is
    how their band was built.
    """
    texts = list(texts)
    per_sample = [metrics_for(t, known) for t in texts]
    bands = reference["bands"]
    checks = {}

    for m in METRICS:
        vals = [s[m] for s in per_sample]
        value = round(float(np.median(vals)), 6)
        low, high = bands[m]["low"], bands[m]["high"]
        checks[m] = {
            "sample_value": value,
            "statistic": "median across samples",
            "corpus_band": [low, high],
            "corpus_median": bands[m]["median"],
            "in_band": bool(low <= value <= high),
        }

    if char_model is not None:
        pooled = pooled_metrics(texts, known, char_model)
        for m in POOLED_METRICS:
            band = bands.get(m)
            value = pooled[m]
            if band is None or band["low"] != band["low"]:
                checks[m] = {"sample_value": value, "statistic": "pooled",
                             "corpus_band": None, "in_band": True,
                             "note": "no corpus band available"}
                continue
            if value != value:  # NaN: no OOV words at all
                checks[m] = {
                    "sample_value": None, "statistic": "pooled",
                    "corpus_band": [band["low"], band["high"]],
                    "corpus_median": band["median"], "in_band": True,
                    "note": "no out-of-vocabulary words to score",
                }
                continue
            checks[m] = {
                "sample_value": round(value, 6),
                "statistic": "pooled over all words in all samples",
                "corpus_band": [band["low"], band["high"]],
                "corpus_median": band["median"],
                "in_band": bool(band["low"] <= value <= band["high"]),
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
    print("building character trigram model over those words...")
    char_model = CharModel(known)
    print(f"  {len(char_model.counts):,} contexts, alphabet {char_model.vocab_size}")
    print(f"building reference bands from {cfg.coherence_ref_docs:,} val docs "
          f"({cfg.coherence_bootstrap} bootstrap groups for pooled metrics)...")
    ref = build_reference(cfg, known, char_model)

    print(f"\n{'metric':<28}{'p' + str(cfg.coherence_band_low_pct):>10}"
          f"{'median':>10}{'p' + str(cfg.coherence_band_high_pct):>10}   width")
    print("-" * 68)
    for m in ALL_METRICS:
        b = ref["bands"][m]
        width = b["high"] - b["low"]
        flag = "  DEGENERATE" if width == 0 else ""
        print(f"{m:<28}{b['low']:>10.4f}{b['median']:>10.4f}{b['high']:>10.4f}"
              f"{width:>9.4f}{flag}")

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
        f"| metric | statistic | p{cfg.coherence_band_low_pct} | median | "
        f"p{cfg.coherence_band_high_pct} |",
        "|---|---|---|---|---|",
    ]
    for m in ALL_METRICS:
        b = ref["bands"][m]
        stat = "pooled, bootstrapped" if m in POOLED_METRICS else "per document"
        lines.append(f"| `{m}` | {stat} | {b['low']:.4f} | {b['median']:.4f} | "
                     f"{b['high']:.4f} |")
    lines += [
        "",
        "## Why two kinds of statistic",
        "",
        "The first four metrics vary document to document, so their band is the "
        f"p{cfg.coherence_band_low_pct}-p{cfg.coherence_band_high_pct} spread across "
        f"{ref['docs']:,} real documents.",
        "",
        "The last two cannot be measured that way. Out-of-vocabulary words are rare "
        "enough that most single documents contain none, which is exactly what made the "
        "old `known_word_rate` metric degenerate: p5 and p95 both landed on 1.0 and the "
        "band was a point (ADR-022). They are **pooled** over every word in a group of "
        f"{ref['bootstrap_group_size']} documents instead, and the band is the spread "
        f"across {ref['bootstrap_groups']} bootstrap resamples — the same statistic at "
        "the same sample size as the generated set it will judge.",
        "",
        "`oov_plausibility` is the mean character-trigram log-probability of the words "
        "that are out of vocabulary. Vocabulary membership cannot tell an invented name "
        "from garbage, because both are unknown; character shape can. This is the metric "
        "Phase 5 needs when SFT starts producing names the corpus never used.",
        "",
        "Generated by `python -m src.coherence reference`.",
    ]
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
    char_model = CharModel(known)
    eot = tok.token_to_id(cfg.doc_separator)
    texts = []
    for i in range(cfg.coherence_samples):
        prompt = GALLERY_PROMPTS[i % len(GALLERY_PROMPTS)]
        gen = torch.Generator(device="cpu").manual_seed(cfg.seed + i)
        ids = generate(model, tok.encode(prompt).ids, cfg.coherence_new_tokens,
                       cfg.coherence_temperature, cfg.coherence_top_k,
                       dev.device, eot_id=eot, generator=gen)
        texts.append(tok.decode(ids))

    scored = score_samples(texts, known, reference, char_model)
    print(f"\n[{'PASS' if scored['passed'] else 'FAIL'}] coherence proxy on "
          f"{scored['n_samples']} generated samples")
    print(f"        {'metric':<28}{'sample':>10}{'corpus band':>24}{'':>4}")
    for m, c in scored["checks"].items():
        if c["corpus_band"] is None or c["sample_value"] is None:
            print(f"        {m:<28}{'n/a':>10}{'':>24}  {c.get('note', '')}")
            continue
        band = f"[{c['corpus_band'][0]:.4f}, {c['corpus_band'][1]:.4f}]"
        print(f"        {m:<28}{c['sample_value']:>10.4f}{band:>24}"
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
