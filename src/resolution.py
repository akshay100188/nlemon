"""Phase 6 — does a story *end*, or does it just stop?

The axis DPO is asked to teach. It is chosen because **token-level cross-entropy
cannot express it**: a story that resolves and a story that trails off are both
locally fluent and both are high-probability continuations, so SFT's loss is
blind to the difference by construction. DPO learns from contrast, which is the
one thing that can see it. That is what makes this DPO rather than SFT with a
different optimizer (ADR-038).

**The markers are mined, not written.** A hand-listed set of "ending words" would
be taste wearing a lab coat - the thing ADR-018 rejected when it built the
coherence bands from corpus percentiles instead of from opinion. So: split every
training document into sentences, compare the word distribution of **final**
sentences against **non-final** sentences, and keep the words the corpus itself
puts at the end. TinyStories is formulaic enough that this works, and the
formulaic-ness is a property of the corpus rather than an assumption about it.

Scoring uses a smoothed log-odds ratio with an uninformative prior, not raw
frequency, because raw frequency just returns `the` and `and`.

**What this must not become.** `is_story` already checks that a response ends on
terminal punctuation, and sft.pt scores 98.1% on it. If `resolution` were just
that again the axis would be saturated and the phase would be measuring nothing.
The two are deliberately different: `is_story` asks whether the text stopped
cleanly, `resolution` asks whether the *narrative* closed. A response can end on
a perfect full stop mid-arc, and that is the case this metric exists to catch.

    python -m src.resolution markers    # mine the corpus-derived ending markers
    python -m src.resolution measure    # corpus ceiling + sft.pt floor
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter

from config import REPO_ROOT, Config
from src.coherence import words_of
from src.tokenizer import iter_docs, text_path
from utils.io import write_json, write_text

SENT = re.compile(r"[^.!?]+[.!?]+")

# Words too common to be markers of anything. Kept tiny on purpose: the log-odds
# scoring already suppresses ubiquitous words, and a long hand-written stoplist
# would be the taste this module exists to avoid.
BORING = frozenset("the a an and or but of to in on at it is was were".split())


def sentences(text: str) -> list[str]:
    return [s.strip() for s in SENT.findall(text) if s.strip()]


def mine_markers(cfg: Config) -> dict:
    """Words the corpus puts in final sentences and not elsewhere.

    Smoothed log-odds: log( (f_end + a) / (N_end + aV) ) - log( (f_mid + a) /
    (N_mid + aV) ). The prior `a` keeps rare words from scoring infinitely, so a
    word seen twice cannot outrank one seen two thousand times.
    """
    end_c: Counter = Counter()
    mid_c: Counter = Counter()
    docs = 0
    for doc in iter_docs(text_path(cfg, "train"), cfg.doc_separator,
                         limit=cfg.resolution_scan_docs):
        sents = sentences(doc)
        if len(sents) < cfg.checker_min_sentences:
            continue
        docs += 1
        for w in words_of(sents[-1]):
            end_c[w] += 1
        for s in sents[:-1]:
            for w in words_of(s):
                mid_c[w] += 1

    vocab = set(end_c) | set(mid_c)
    n_end, n_mid = sum(end_c.values()), sum(mid_c.values())
    a, V = 1.0, len(vocab)
    scored = {}
    for w in vocab:
        if w in BORING or len(w) < 3:
            continue
        if end_c[w] < cfg.resolution_min_marker_count:
            continue
        lo = (math.log((end_c[w] + a) / (n_end + a * V))
              - math.log((mid_c[w] + a) / (n_mid + a * V)))
        scored[w] = lo
    top = sorted(scored.items(), key=lambda kv: -kv[1])[:cfg.resolution_markers]
    return {
        "docs": docs, "final_sentence_words": n_end, "other_words": n_mid,
        "vocab": V,
        "markers": {w: round(v, 4) for w, v in top},
        "marker_min_count": cfg.resolution_min_marker_count,
    }


def load_markers(cfg: Config) -> dict[str, float]:
    p = REPO_ROOT / cfg.results_dir / "resolution_markers.json"
    if not p.exists():
        raise SystemExit("run `python -m src.resolution markers` first.")
    return json.loads(p.read_text(encoding="utf-8"))["markers"]


def score_one(cfg: Config, text: str, markers: dict[str, float]) -> dict:
    """Did the narrative close?

    Three signals, reported apart so a later gate cannot blend them:

    * `terminal`   - ends on sentence-final punctuation. This is `is_story`'s
                     clause, included only so the overlap is visible rather than
                     hidden; it is near-saturated and carries no information.
    * `marker`     - the final sentence contains at least one word the corpus
                     puts at endings.
    * `resolved`   - both. The metric the phase is about.
    """
    sents = sentences(text)
    stripped = text.strip()
    terminal = bool(stripped) and stripped.endswith((".", "!", "?", '"'))
    last = sents[-1] if sents else ""
    hits = [w for w in words_of(last) if w in markers]
    strength = max((markers[w] for w in hits), default=0.0)
    return {
        "terminal": terminal,
        "marker": bool(hits),
        "resolved": bool(terminal and hits),
        "marker_strength": round(strength, 4),
        "markers_hit": hits[:5],
        "sentences": len(sents),
    }


def rate(rows: list[dict], key: str) -> float:
    return round(sum(1 for r in rows if r[key]) / max(len(rows), 1), 6)


def cmd_markers(cfg: Config) -> None:
    print(f"mining ending markers from {cfg.resolution_scan_docs:,} docs...")
    out = mine_markers(cfg)
    print(f"  {out['docs']:,} docs with >= {cfg.checker_min_sentences} sentences")
    print(f"  {out['final_sentence_words']:,} words in final sentences, "
          f"{out['other_words']:,} elsewhere")
    items = list(out["markers"].items())
    print(f"\n  top {min(30, len(items))} corpus-derived ending markers "
          f"(log-odds end vs mid):")
    for i in range(0, min(30, len(items)), 3):
        print("    " + "  ".join(f"{w:<14}{v:>6.2f}" for w, v in items[i:i + 3]))
    write_json(REPO_ROOT / cfg.results_dir / "resolution_markers.json", out)
    print(f"\nwrote {REPO_ROOT / cfg.results_dir / 'resolution_markers.json'}")


def cmd_measure(cfg: Config) -> None:
    """The pre-flight check: does this axis have headroom, or is it saturated?

    Same discipline as the length census and the aboutness check - measure the
    ceiling and the floor before a single preference pair is built, because an
    axis the model already saturates is an axis DPO cannot demonstrate anything
    on, and that has to be found now rather than at the gate.
    """
    markers = load_markers(cfg)
    d = REPO_ROOT / cfg.data_dir
    r = REPO_ROOT / cfg.results_dir

    held = json.loads((d / "sft_heldout.json").read_text(encoding="utf-8"))
    corpus_rows = [score_one(cfg, p["response"], markers) for p in held]

    out = {"n": len(held), "markers": len(markers), "stages": {}}
    print(f"resolution on {len(held)} held-out prompts\n")
    print(f"  {'stage':<22} {'terminal':>9} {'marker':>8} {'RESOLVED':>9}")
    print(f"  {'corpus (ceiling)':<22} {rate(corpus_rows,'terminal'):>8.1%} "
          f"{rate(corpus_rows,'marker'):>7.1%} {rate(corpus_rows,'resolved'):>8.1%}")
    out["stages"]["corpus"] = {k: rate(corpus_rows, k)
                               for k in ("terminal", "marker", "resolved")}

    for label in ("base", "sft"):
        p = r / f"sft_scores_{label}.json"
        if not p.exists():
            continue
        resp = json.loads(p.read_text(encoding="utf-8"))["responses"]
        rows = [score_one(cfg, t, markers) for t in resp]
        out["stages"][label] = {k: rate(rows, k)
                                for k in ("terminal", "marker", "resolved")}
        out[f"{label}_per_prompt"] = rows
        print(f"  {label + '.pt':<22} {rate(rows,'terminal'):>8.1%} "
              f"{rate(rows,'marker'):>7.1%} {rate(rows,'resolved'):>8.1%}")

    if "sft" in out["stages"]:
        c = out["stages"]["corpus"]["resolved"]
        s = out["stages"]["sft"]["resolved"]
        print(f"\n  headroom for DPO: {(c - s) * 100:.1f} points "
              f"(sft {s:.1%} -> corpus ceiling {c:.1%})")
        print(f"  compare is_story, which sft already scores 98.1% on:")
        print(f"    resolution is NOT is_story - terminal punctuation is "
              f"{out['stages']['sft']['terminal']:.1%},")
        print(f"    but only {s:.1%} of those endings actually close the story.")
    write_json(r / "resolution_headroom.json", out)
    print(f"\nwrote {r / 'resolution_headroom.json'}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Story resolution (Phase 6).")
    ap.add_argument("command", choices=("markers", "measure"))
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = Config.load(args.config)
    (cmd_markers if args.command == "markers" else cmd_measure)(cfg)


if __name__ == "__main__":
    main()
