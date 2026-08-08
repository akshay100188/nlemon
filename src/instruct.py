"""Phase 5 — instruction pairs, with subjects held out rather than phrasings.

Builds `"Write a story about a kitten." -> <story>` pairs by mining a subject
from each TinyStories document and templating a prompt around it.

**The held-out split is over subjects, not phrasings** (ADR-023). Holding out
phrasings would test whether the model generalises across four sentence
templates; holding out subjects tests whether it learned to follow an
instruction about a thing it was never instructed about. The second is the
harder claim and the one that survives "did it just memorise your templates".

**The leak this design can spring.** The prompt contains the subject word, so a
model that merely echoes its prompt scores well on subject-mention without
following anything. That is measurable rather than hypothetical: `base.pt` never
saw these instructions, so if *base* scores high on subject-mention, the checker
is reading its own answer back off the prompt. The base floor is therefore two
things at once - the anchor for the threshold, and the leak detector - and a high
base score means a broken checker, not a weak floor.

    python -m src.instruct subjects   # mine the subject pool, show the split
    python -m src.instruct pairs      # build train pairs + held-out prompts
"""

from __future__ import annotations

import argparse
import random
import re
from collections import Counter

from config import REPO_ROOT, Config
from src.tokenizer import iter_docs, text_path
from utils.io import write_json, write_text
from utils.seed import set_seed

WORD = re.compile(r"[a-z]+")
# A determiner immediately before a word is a cheap, corpus-derived noun test.
# No POS tagger, no new dependency, and the evidence is countable.
DETERMINER = re.compile(r"\b(?:a|an|the)\s+([a-z]{3,})\b")

# Prompt phrasings. All of them appear in both train and held-out sets on
# purpose: the split is over subjects, so phrasing is held constant.
TEMPLATES = (
    "Write a story about {subject}.",
    "Tell me a story about {subject}.",
    "Write a short story about {subject}.",
    "Can you write a story about {subject}?",
)

# Words that follow a determiner constantly but are not story subjects.
STOPWORDS = frozenset("""
    little big small good bad old new other next last first second few many
    same other whole other another every each some most more less very much
    only just other next time day way thing things lot bit end top side
    front back part kind sort couple pair group bunch number rest
""".split())

# Words that can follow a noun but not sit between a determiner and its head.
# "a bird was" -> bird is the head; "a brave bird" -> brave is a modifier.
HEAD_FOLLOWERS = frozenset("""
    was were is are had has said says asked told saw went came looked wanted
    liked loved felt found made took gave ran jumped played smiled laughed
    that who which and but or so then because in on at to with for from of
    near under over into onto around beside behind before after while when
    named called full happy sad again there here too very
""".split())

NEXT_WORD = re.compile(r"\s+([a-z]+)")


def mine_subjects(cfg: Config) -> tuple[Counter, Counter, Counter, Counter]:
    """Count determiner-followed words: occurrences, heads, doc hits, and aboutness.

    Three filters stack here, and they catch different things.

    **Headedness** (`heads / counts`) separates a noun from an adjective. The
    determiner test alone cannot: "a brave bird" makes `brave` look like a
    subject, and "Write a story about a brave." is not an instruction. A head is
    followed by punctuation or a word that cannot be a noun ("a bird was...");
    a modifier is followed by the actual noun.

    **Aboutness** (`doc_about / doc_hits`) separates a subject from a passing
    mention, and it is the filter attempt #1 lacked (ADR-030). Headedness cannot
    do this job and never could: in "a floor", `floor` **is** the head. A story
    genuinely about a rabbit says "rabbit" repeatedly; a story that merely
    happens near a floor says "floor" once. So a document counts toward a
    subject's aboutness only if the word occurs at least `sft_min_aboutness`
    times in it.

    **Absolute count** filters rarity. This one is worth keeping separate in the
    reporting even though it is the same threshold, because it fails a *different
    class* of word: `moral` fails aboutness (common, never a protagonist), while
    `lizard` fails rarity (a fine protagonist, just scarce in TinyStories). A
    pool that shrinks needs to say which cause did it - the two have different
    implications for whether the remaining pool is any good.
    """
    counts: Counter = Counter()
    heads: Counter = Counter()
    doc_hits: Counter = Counter()
    doc_about: Counter = Counter()
    for doc in iter_docs(text_path(cfg, "train"), cfg.doc_separator,
                         limit=cfg.sft_subject_scan_docs):
        low = doc.lower()
        seen: set[str] = set()
        for m in DETERMINER.finditer(low):
            w = m.group(1)
            if w in STOPWORDS:
                continue
            counts[w] += 1
            nxt = NEXT_WORD.match(low, m.end())
            if nxt is None or nxt.group(1) in HEAD_FOLLOWERS:
                heads[w] += 1
            seen.add(w)
        if not seen:
            continue
        words = WORD.findall(low)
        freq = Counter(words)
        for w in seen:
            doc_hits[w] += 1
            if freq[w] >= cfg.sft_min_aboutness:
                doc_about[w] += 1
    return counts, heads, doc_hits, doc_about


def head_ratios(counts: Counter, heads: Counter) -> dict[str, float]:
    return {w: heads[w] / n for w, n in counts.items() if n}


def split_subjects(cfg: Config, counts: Counter) -> dict:
    """Disjoint train / held-out subject pools, chosen deterministically.

    Split by rank-stratified sampling rather than by frequency, so the held-out
    pool is not systematically rarer than the train pool - which would make the
    held-out set harder for a reason that has nothing to do with instruction
    following.
    """
    pool = [w for w, n in counts.most_common(cfg.sft_subject_pool)
            if n >= cfg.sft_min_subject_count]
    rng = random.Random(cfg.seed)
    held: list[str] = []
    train: list[str] = []
    # walk the frequency-ordered list in blocks, taking a fixed share of each
    block = max(int(round(1 / cfg.sft_heldout_subject_frac)), 2)
    for i in range(0, len(pool), block):
        chunk = pool[i:i + block]
        rng.shuffle(chunk)
        held.append(chunk[0])
        train.extend(chunk[1:])
    return {
        "train_subjects": sorted(train),
        "heldout_subjects": sorted(held),
        "counts": {w: counts[w] for w in pool},
    }


def subject_phrase(word: str) -> str:
    """'kitten' -> 'a kitten'; keeps prompts grammatical without a lexicon."""
    return ("an " if word[0] in "aeiou" else "a ") + word


def build_pairs(cfg: Config, subjects: set[str], limit: int,
                split_name: str) -> list[dict]:
    """Pair each document with one subject it is actually about.

    The subject must appear in the document, so the response genuinely answers
    the prompt. Where several candidates appear, the earliest wins - the thing a
    story opens on is the thing it is about, far more often than the most
    frequent word.
    """
    rng = random.Random(cfg.seed + (0 if split_name == "train" else 1))
    pairs: list[dict] = []
    dropped_unabout = 0
    for doc in iter_docs(text_path(cfg, "train"), cfg.doc_separator,
                         limit=cfg.sft_pair_scan_docs):
        low = doc.lower()
        freq = Counter(WORD.findall(low))
        # Attempt #2: pick the MOST-REPEATED candidate, not the earliest one
        # (ADR-030). "Earliest determiner" picks whatever the story walked past
        # first, which is how `Write a story about a floor.` got paired with a
        # story that says "floor" once. What a story keeps returning to is what
        # it is about; ties break on position, so the choice stays deterministic.
        best = None
        best_key = None
        for m in DETERMINER.finditer(low):
            w = m.group(1)
            if w not in subjects:
                continue
            key = (freq[w], -m.start())
            if best_key is None or key > best_key:
                best, best_key = w, key
        if best is None:
            continue
        # ...and the winner still has to clear the aboutness bar outright. A pool
        # subject is one that CAN be a protagonist; this pair has to show that it
        # IS one here.
        if freq[best] < cfg.sft_min_aboutness:
            dropped_unabout += 1
            continue
        words = doc.split()
        if not (cfg.sft_min_response_words <= len(words) <= cfg.sft_max_response_words):
            continue
        template = TEMPLATES[rng.randrange(len(TEMPLATES))]
        pairs.append({
            "subject": best,
            "prompt": template.format(subject=subject_phrase(best)),
            "response": doc,
            "response_words": len(words),
            "subject_occurrences": freq[best],
        })
        if len(pairs) >= limit:
            break
    if dropped_unabout:
        print(f"    dropped {dropped_unabout:,} docs whose best candidate appeared "
              f"< {cfg.sft_min_aboutness}x (passing mention, not a subject)")
    return pairs


def format_example(cfg: Config, prompt: str, response: str) -> str:
    """The on-the-wire format. Prompt, separator line, response, EOT."""
    return f"{prompt}\n{response}{cfg.doc_separator}"


def cmd_subjects(cfg: Config) -> None:
    print(f"scanning {cfg.sft_subject_scan_docs:,} training docs for subjects...")
    counts, heads, doc_hits, doc_about = mine_subjects(cfg)
    ratios = head_ratios(counts, heads)
    about_ratios = {w: doc_about[w] / n for w, n in doc_hits.items() if n}

    considered = [w for w, n in counts.most_common(cfg.sft_subject_pool * 3)
                  if n >= cfg.sft_min_subject_count]
    drop_head = [w for w in considered
                 if ratios.get(w, 0.0) < cfg.sft_min_head_ratio]
    survives_head = [w for w in considered if w not in set(drop_head)]
    # Two causes, reported apart (ADR-030). A word can fail because it is never
    # what a story is about, or because it is simply scarce - and `lizard` versus
    # `moral` is exactly that distinction.
    drop_about = [w for w in survives_head
                  if about_ratios.get(w, 0.0) < cfg.sft_min_aboutness_ratio]
    drop_thin = [w for w in survives_head
                 if w not in set(drop_about)
                 and doc_about[w] < cfg.sft_min_subject_count]
    kept = Counter({w: doc_about[w] for w in survives_head
                    if w not in set(drop_about) and w not in set(drop_thin)})

    print(f"  {len(counts):,} candidate words, {len(considered)} above the "
          f"{cfg.sft_min_subject_count} occurrence floor")
    print(f"\n  filter 1  headedness >= {cfg.sft_min_head_ratio:.0%}"
          f"           dropped {len(drop_head):>4}  (modifiers)")
    print(f"            e.g. {', '.join(sorted(drop_head, key=lambda w: -counts[w])[:10])}")
    print(f"  filter 2  aboutness ratio >= {cfg.sft_min_aboutness_ratio:.0%}"
          f"    dropped {len(drop_about):>4}  (never what a story is ABOUT)")
    print(f"            e.g. {', '.join(sorted(drop_about, key=lambda w: -counts[w])[:12])}")
    print(f"  filter 3  >= {cfg.sft_min_subject_count} aboutness docs"
          f"        dropped {len(drop_thin):>4}  (fine subjects, just scarce)")
    print(f"            e.g. {', '.join(sorted(drop_thin, key=lambda w: -doc_about[w])[:12])}")
    print(f"\n  pool after all three filters: {len(kept)} subjects "
          f"(attempt #1 had {cfg.sft_subject_pool} considered, no aboutness filter)")

    split = split_subjects(cfg, kept)
    train, held = split["train_subjects"], split["heldout_subjects"]

    print(f"  pool capped at {cfg.sft_subject_pool}")
    print(f"  train subjects   : {len(train)}")
    print(f"  held-out subjects: {len(held)}")
    print(f"  overlap          : {len(set(train) & set(held))}  (must be 0)")
    freq_t = [counts[w] for w in train]
    freq_h = [counts[w] for w in held]
    print(f"  median corpus frequency: train {sorted(freq_t)[len(freq_t)//2]:,}  "
          f"held-out {sorted(freq_h)[len(freq_h)//2]:,}")
    print(f"\n  held-out sample: {', '.join(held[:14])}")

    write_json(REPO_ROOT / cfg.results_dir / "sft_subjects.json", {
        "attempt": 2,
        "scan_docs": cfg.sft_subject_scan_docs,
        "pool": cfg.sft_subject_pool,
        "min_count": cfg.sft_min_subject_count,
        "heldout_fraction": cfg.sft_heldout_subject_frac,
        "min_head_ratio": cfg.sft_min_head_ratio,
        "min_aboutness": cfg.sft_min_aboutness,
        "min_aboutness_ratio": cfg.sft_min_aboutness_ratio,
        # The two shrinkage causes stay apart: a word can fail because no story
        # is ever about it, or because it is scarce. Different implications.
        "considered": len(considered),
        "dropped_as_modifiers": sorted(drop_head, key=lambda w: -counts[w])[:60],
        "dropped_as_not_about": sorted(drop_about, key=lambda w: -counts[w])[:80],
        "dropped_as_too_scarce": sorted(drop_thin, key=lambda w: -doc_about[w])[:80],
        "n_dropped_modifiers": len(drop_head),
        "n_dropped_not_about": len(drop_about),
        "n_dropped_too_scarce": len(drop_thin),
        "pool_size": len(kept),
        "n_train": len(train), "n_heldout": len(held),
        "overlap": len(set(train) & set(held)),
        "train_subjects": train, "heldout_subjects": held,
        "frequency": split["counts"],
        "aboutness_docs": {w: doc_about[w] for w in train + held},
        "aboutness_ratio": {w: round(about_ratios.get(w, 0.0), 4)
                            for w in train + held},
        "head_ratio": {w: round(ratios.get(w, 0.0), 4)
                       for w in train + held},
    })
    print(f"\nwrote {REPO_ROOT / cfg.results_dir / 'sft_subjects.json'}")


def cmd_pairs(cfg: Config) -> None:
    import json
    path = REPO_ROOT / cfg.results_dir / "sft_subjects.json"
    if not path.exists():
        raise SystemExit("run `python -m src.instruct subjects` first.")
    split = json.loads(path.read_text(encoding="utf-8"))
    train_subjects = set(split["train_subjects"])
    held_subjects = set(split["heldout_subjects"])

    print(f"building train pairs (subjects: {len(train_subjects)})...")
    train_pairs = build_pairs(cfg, train_subjects, cfg.sft_max_pairs, "train")
    print(f"  {len(train_pairs):,} pairs")

    print(f"building held-out prompts (subjects: {len(held_subjects)}, disjoint)...")
    held_pairs = build_pairs(cfg, held_subjects, cfg.sft_eval_prompts, "heldout")
    print(f"  {len(held_pairs):,} prompts")

    used_train = {p["subject"] for p in train_pairs}
    used_held = {p["subject"] for p in held_pairs}
    leak = used_train & used_held
    print(f"\nsubject overlap between splits: {len(leak)}  (must be 0)")
    if leak:
        raise SystemExit(f"held-out subjects appear in training pairs: {sorted(leak)[:10]}")

    lengths = sorted(p["response_words"] for p in train_pairs)
    p5 = lengths[int(0.05 * len(lengths))]
    p95 = lengths[int(0.95 * len(lengths))]
    print(f"response length band from training pairs: p5 {p5}, p95 {p95} words")

    out_dir = REPO_ROOT / cfg.data_dir
    write_json(out_dir / "sft_train.json", train_pairs)
    write_json(out_dir / "sft_heldout.json", held_pairs)
    write_json(REPO_ROOT / cfg.results_dir / "sft_pairs_summary.json", {
        "train_pairs": len(train_pairs),
        "heldout_prompts": len(held_pairs),
        "distinct_train_subjects": len(used_train),
        "distinct_heldout_subjects": len(used_held),
        "subject_overlap": len(leak),
        "response_words_p5": p5,
        "response_words_p95": p95,
        "templates": list(TEMPLATES),
    })

    lines = [
        "# Instruction pairs",
        "",
        f"{len(train_pairs):,} training pairs over {len(used_train)} subjects; "
        f"{len(held_pairs)} held-out prompts over {len(used_held)} **disjoint** "
        "subjects.",
        "",
        "The split is over **subjects, not phrasings** (ADR-023). All four templates "
        "appear in both sets, so phrasing is held constant and the held-out set asks "
        "whether the model follows an instruction about a thing it was never instructed "
        "about. Holding out phrasings instead would have measured generalisation across "
        "four sentence forms, which is a different and easier question.",
        "",
        f"Response length band from the training pairs: p5 {p5}, p95 {p95} words.",
        "",
        "## Templates",
        "",
    ] + [f"- `{t}`" for t in TEMPLATES] + [
        "",
        "## Held-out subjects",
        "",
        ", ".join(f"`{s}`" for s in sorted(used_held)),
        "",
        "Generated by `python -m src.instruct pairs`.",
    ]
    write_text(REPO_ROOT / cfg.results_dir / "sft_pairs.md", "\n".join(lines))
    print(f"wrote {out_dir / 'sft_train.json'} and {out_dir / 'sft_heldout.json'}")


def cmd_aboutness(cfg: Config) -> None:
    """Is each pair's response actually ABOUT its subject? (ADR-030)

    `build_pairs` requires the subject to *appear* in the document. Appearing is
    not aboutness. "Once upon a time a girl dropped her toy on the floor" contains
    "a floor" as the head of a determiner phrase, so the miner accepts it, and the
    pair then teaches `Write a story about a floor.` -> a story that says "floor"
    once in passing. The model learns that faithfully, and then a checker asking
    "did you mention the subject" marks it wrong.

    The headedness filter cannot catch this: in "a floor", `floor` *is* the head.
    Nor can any stopword list I would have to keep complete - the residue includes
    `while` ("after a while") and `sudden` ("all of a sudden"), where the idiom
    makes an abstract noun the head of its phrase.

    The proxy is repetition. A story genuinely about a rabbit says "rabbit" over
    and over; a story that merely happens near a floor says "floor" once. This is
    the check that should have run before the subject_mention bar was registered -
    the answerability twin of the length census in `src/sft_data.py`, and the one
    that was missed.
    """
    import json

    from src.coherence import words_of

    train = json.loads((REPO_ROOT / cfg.data_dir / "sft_train.json")
                       .read_text(encoding="utf-8"))
    counts: Counter = Counter()
    per_subject: dict[str, list[int]] = {}
    for p in train:
        n = words_of(p["response"]).count(p["subject"])
        counts[min(n, 5)] += 1
        per_subject.setdefault(p["subject"], []).append(n)

    tot = sum(counts.values())
    print(f"subject occurrences per response, over {tot:,} training pairs:")
    for k in sorted(counts):
        print(f"  {'5+' if k >= 5 else k:>3}x : {counts[k]:>6,}  {counts[k]/tot:>6.1%}")
    once_or_less = (counts[0] + counts[1]) / tot
    print(f"\n  {once_or_less:.1%} of pairs mention the subject at most ONCE.")
    print(f"  Those teach 'write about X' -> a story that says X in passing.")

    means = {s: sum(v) / len(v) for s, v in per_subject.items() if len(v) >= 20}
    ranked = sorted(means.items(), key=lambda kv: kv[1])
    print(f"\n  weakest aboutness (mean occurrences per response):")
    print("    " + ", ".join(f"{s} {m:.2f}" for s, m in ranked[:15]))
    print(f"  strongest:")
    print("    " + ", ".join(f"{s} {m:.2f}" for s, m in ranked[-10:][::-1]))

    write_json(REPO_ROOT / cfg.results_dir / "sft_aboutness.json", {
        "pairs": tot,
        "occurrence_histogram": {("5+" if k >= 5 else str(k)): counts[k]
                                 for k in sorted(counts)},
        "frac_at_most_once": round(once_or_less, 6),
        "subject_mean_occurrences": {s: round(m, 4) for s, m in ranked},
        "weakest": [s for s, _ in ranked[:20]],
        "strongest": [s for s, _ in ranked[-20:][::-1]],
    })
    print(f"\nwrote {REPO_ROOT / cfg.results_dir / 'sft_aboutness.json'}")


def main() -> None:
    ap = argparse.ArgumentParser(description="SFT instruction pairs (Phase 5).")
    ap.add_argument("command", choices=("subjects", "pairs", "aboutness"))
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = Config.load(args.config)
    set_seed(cfg.seed, strict=cfg.strict_determinism)
    {"subjects": cmd_subjects, "pairs": cmd_pairs,
     "aboutness": cmd_aboutness}[args.command](cfg)


if __name__ == "__main__":
    main()
