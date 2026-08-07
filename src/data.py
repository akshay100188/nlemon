"""Phase 1 — corpus on disk.

Pulls TinyStories (ADR-001), repairs the encoding damage it ships with
(ADR-009), writes plain-text train/val shards, prints corpus stats, dumps a
deterministic peek at three stories, and records everything it did in
``data/manifest.json`` — resolved dataset commit, license, file hashes,
repair counts, config hash. That manifest is what makes the corpus step
auditable rather than "I downloaded something once".

    python -m src.data              # build the corpus
    python -m src.data --dry-run    # just probe the hub repo and report
"""

from __future__ import annotations

import argparse
import hashlib
import random
import time
from collections import Counter
from pathlib import Path

import ftfy
import numpy as np
from tqdm import tqdm

from config import REPO_ROOT, Config
from utils.io import write_json, write_text
from utils.seed import set_seed

SPLITS = {"train": "max_train_docs", "validation": "max_val_docs"}
TEXT_COLUMN = "text"

# --------------------------------------------------------------------------- #
# Encoding repair (ADR-009)
#
# The upstream text is double-encoded in places: UTF-8 bytes that were once
# decoded as cp1252 and re-saved. ftfy repairs most of it. Two things it cannot:
#
#   * some curly quotes lost their third UTF-8 byte upstream, leaving the corpse
#     chr(0xE2)+chr(0x20AC) with nothing to reconstruct from;
#   * for one variant of that corpse ftfy re-encodes into U+2020 DAGGER — a
#     legitimate-looking character the tokenizer would learn cleanly, which is
#     worse than obvious garbage.
#
# Verified against the raw dataset: TinyStories contains zero daggers of its
# own, so every dagger here is a dead quote. ftfy's default config uncurls
# quotes to ASCII, so ASCII '"' is the consistent target for open and close
# alike. Source below is pure ASCII on purpose — a file about mojibake should
# not itself depend on being read in the right encoding.
# --------------------------------------------------------------------------- #
RESIDUE: tuple[tuple[str, str], ...] = (
    (chr(0x00E2) + chr(0x20AC), '"'),  # unrepairable curly-quote corpse
    (chr(0x2020), '"'),                # DAGGER, mis-resurrected by ftfy
    (chr(0x00A0), " "),                # no-break space
    (chr(0x2009), " "),                # thin space
    (chr(0x200A), " "),                # hair space
    (chr(0x200B), ""),                 # zero-width space
    (chr(0x00AD), ""),                 # soft hyphen
    (chr(0xFFFD), ""),                 # replacement char: the byte is already gone
    # Orphaned mojibake prefix, left behind when 'A-circumflex' is followed by
    # ASCII: 'wasn<C2>'t', 'loved.<C2>'. ftfy cannot read those as mojibake (C2
    # plus an ASCII byte is not a valid pair) so it leaves them, and every one of
    # the 9 occurrences audited in the 1.84 GiB shard was of this kind - none was
    # a real capital A-circumflex.
    #
    # Position in this tuple does not matter. An earlier comment here claimed the
    # rule had to run last because the no-break-space and soft-hyphen rules would
    # strand the prefix; that was wrong. ftfy repairs those two pairs before
    # RESIDUE runs, so they never reach these rules at all. Verified in
    # tests/test_residue.py: 400 permutations, identical output, no rule pattern
    # contains another.
    (chr(0x00C2), ""),
)

# If any of these survive cleaning, the repair has a hole worth knowing about.
# Deliberately broader than RESIDUE: it flags characters that are *usually*
# damage but can be legitimate (a lowercase a-circumflex is correct in
# "papier-mache"), so the audit records context and a human adjudicates rather
# than the cleaner guessing. See RESIDUE_FOOTNOTE in the manifest.
SUSPICIOUS = (chr(0x00E2), chr(0x20AC), chr(0x2020), chr(0xFFFD),
              chr(0x00C2), chr(0x00C3))

# How many context snippets to keep per suspicious codepoint in the manifest.
RESIDUE_SAMPLES = 5
RESIDUE_CONTEXT = 55


def clean_text(raw: str) -> tuple[str, bool]:
    """Repair upstream encoding damage. Returns ``(text, changed)``."""
    text = ftfy.fix_text(raw)
    for bad, good in RESIDUE:
        if bad in text:
            text = text.replace(bad, good)
    return text, text != raw


# --------------------------------------------------------------------------- #
# hub
# --------------------------------------------------------------------------- #
def resolve_dataset(cfg: Config) -> dict:
    """Pin the dataset to an exact commit and capture its license from the Hub."""
    from huggingface_hub import HfApi

    info = HfApi().dataset_info(cfg.dataset_name, revision=cfg.dataset_revision)
    card = info.cardData or {}
    license_ = card.get("license", "UNKNOWN")
    if isinstance(license_, list):
        license_ = ", ".join(license_)
    return {
        "name": cfg.dataset_name,
        "requested_revision": cfg.dataset_revision,
        "resolved_revision": info.sha,
        "license": license_,
    }


def load_splits(cfg: Config, revision: str):
    from datasets import load_dataset

    ds = load_dataset(cfg.dataset_name, revision=revision)
    missing = set(SPLITS) - set(ds.keys())
    if missing:
        raise SystemExit(
            f"{cfg.dataset_name}@{revision[:7]} is missing split(s) {sorted(missing)}; "
            f"found {sorted(ds.keys())}. We use the dataset's own train/validation "
            f"boundary rather than inventing one (ADR-006) — if upstream drops it, "
            f"that is a decision to record, not to paper over."
        )
    for split in SPLITS:
        cols = ds[split].column_names
        if TEXT_COLUMN not in cols:
            raise SystemExit(
                f"split '{split}' has columns {cols}, expected a '{TEXT_COLUMN}' column"
            )
    return ds


# --------------------------------------------------------------------------- #
# writing + stats
# --------------------------------------------------------------------------- #
def write_split(split_ds, out_path: Path, separator: str, limit: int) -> dict:
    """Stream one split to a .txt shard, cleaning, hashing and measuring as we go.

    Documents are delimited by the configured separator on its own line — the
    same boundary token the tokenizer will learn in Phase 2.

    Two repairs are applied, both counted rather than silent: encoding damage
    (ADR-009) and documents that are empty after stripping, which are dropped
    (ADR-008). Neither is a tunable threshold, so neither adds a hyperparameter.
    """
    n_total = len(split_ds)
    n_read = n_total if limit <= 0 else min(limit, n_total)
    delimiter = f"\n{separator}\n"

    sha = hashlib.sha256()
    counts: list[int] = []
    n_chars = 0
    n_bytes = 0
    n_empty = 0
    n_repaired = 0
    residual: Counter[str] = Counter()
    residual_ctx: dict[str, list[str]] = {}

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        for i in tqdm(range(n_read), desc=f"writing {out_path.name}", unit="doc"):
            text, changed = clean_text(split_ds[i][TEXT_COLUMN])
            n_repaired += changed
            text = text.strip()
            if not text:
                n_empty += 1
                continue
            for ch in SUSPICIOUS:
                if ch not in text:
                    continue
                key = f"U+{ord(ch):04X}"
                residual[key] += text.count(ch)
                # Keep a few snippets so the remainder is a recorded footnote
                # rather than an unexplained number someone finds later.
                seen = residual_ctx.setdefault(key, [])
                if len(seen) < RESIDUE_SAMPLES:
                    at = text.index(ch)
                    seen.append(
                        text[max(0, at - RESIDUE_CONTEXT): at + RESIDUE_CONTEXT]
                        .replace("\n", " ")
                    )
            chunk = text + delimiter
            f.write(chunk)
            encoded = chunk.encode("utf-8")
            sha.update(encoded)
            n_bytes += len(encoded)
            n_chars += len(text)
            counts.append(text.count(" ") + 1)

    word_counts = np.asarray(counts, dtype=np.int32)
    n_docs = int(word_counts.size)

    return {
        "file": out_path.name,
        "docs": n_docs,
        "docs_read": n_read,
        "docs_available": n_total,
        "empty_docs_dropped": n_empty,
        "docs_encoding_repaired": n_repaired,
        "residual_suspicious_chars": dict(residual),
        "residual_context": residual_ctx,
        "truncated": n_read < n_total,
        "chars": int(n_chars),
        "words": int(word_counts.sum()),
        "bytes": int(n_bytes),
        "words_per_doc": {
            "min": int(word_counts.min()) if n_docs else 0,
            "mean": round(float(word_counts.mean()), 1) if n_docs else 0.0,
            "p50": int(np.percentile(word_counts, 50)) if n_docs else 0,
            "p95": int(np.percentile(word_counts, 95)) if n_docs else 0,
            "max": int(word_counts.max()) if n_docs else 0,
        },
        "sha256": sha.hexdigest(),
    }


def print_stats(stats: dict[str, dict]) -> None:
    print("\ncorpus stats")
    print("-" * 72)
    print(f"{'split':<12}{'docs':>10}{'words':>14}{'chars':>14}{'MiB':>9}{'p50 w':>8}")
    for split, s in stats.items():
        print(
            f"{split:<12}{s['docs']:>10,}{s['words']:>14,}{s['chars']:>14,}"
            f"{s['bytes'] / 1024 ** 2:>9.1f}{s['words_per_doc']['p50']:>8}"
        )
    print("-" * 72)
    total_words = sum(s["words"] for s in stats.values())
    # ~1.3 tokens per word is the usual English BPE ballpark; a sanity check
    # against the Phase 4 token budget, not a measurement.
    print(f"total words  : {total_words:,}  (~{int(total_words * 1.3):,} BPE tokens, rough)")
    for split, s in stats.items():
        wp = s["words_per_doc"]
        print(f"{split:<12} words/doc  min {wp['min']}  mean {wp['mean']}  "
              f"p50 {wp['p50']}  p95 {wp['p95']}  max {wp['max']}")
        if s["docs_encoding_repaired"]:
            pct = 100 * s["docs_encoding_repaired"] / max(s["docs_read"], 1)
            print(f"{split:<12} repaired encoding in {s['docs_encoding_repaired']:,} docs "
                  f"({pct:.1f}% of read) (ADR-009)")
        if s["empty_docs_dropped"]:
            print(f"{split:<12} dropped {s['empty_docs_dropped']:,} empty docs "
                  f"of {s['docs_read']:,} read (ADR-008)")
        residual = s["residual_suspicious_chars"]
        if not residual:
            print(f"{split:<12} residual suspicious chars: none")
        else:
            n = sum(residual.values())
            print(f"{split:<12} residual suspicious chars: {n} "
                  f"({', '.join(f'{k} x{v}' for k, v in sorted(residual.items()))}) "
                  f"- context in manifest, adjudicated in ADR-009")
        if s["truncated"]:
            print(f"{split:<12} TRUNCATED to {s['docs_read']:,} of "
                  f"{s['docs_available']:,} docs by config")


def write_peek(split_ds, out_path: Path, n: int, seed: int, dataset: dict) -> list[int]:
    """Dump n stories chosen deterministically from the seed."""
    idx = sorted(random.Random(seed).sample(range(len(split_ds)), n))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Corpus peek — TinyStories",
        "",
        f"Dataset `{dataset['name']}` @ `{dataset['resolved_revision'][:12]}` "
        f"(license: {dataset['license']}).",
        f"{n} stories drawn from the **train** split with seed `{seed}` — "
        f"rerunning picks the same ones. Shown after the ADR-009 encoding "
        f"repair, i.e. exactly as the tokenizer will see them.",
        "",
    ]
    for rank, i in enumerate(idx, 1):
        text, _ = clean_text(split_ds[i][TEXT_COLUMN])
        lines += [f"## {rank}. train[{i}]", "", "> " + text.strip().replace("\n", "\n> "), ""]
    write_text(out_path, "\n".join(lines))
    return idx


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description="Build the nLemon-14 corpus.")
    ap.add_argument("--config", default=None, help="path to the run config yaml")
    ap.add_argument("--dry-run", action="store_true",
                    help="resolve and report the dataset without writing shards")
    args = ap.parse_args()

    cfg = Config.load(args.config)
    set_seed(cfg.seed, strict=cfg.strict_determinism)

    print(f"{cfg.project_name}  ·  phase 1 · data")
    print(f"config hash : {cfg.hash()}")
    print(f"seed        : {cfg.seed}\n")

    dataset = resolve_dataset(cfg)
    print(f"dataset     : {dataset['name']}")
    print(f"revision    : {dataset['resolved_revision']}  "
          f"(requested '{dataset['requested_revision']}')")
    print(f"license     : {dataset['license']}")

    ds = load_splits(cfg, dataset["resolved_revision"])
    for split in SPLITS:
        print(f"  {split:<11}: {len(ds[split]):,} docs, columns {ds[split].column_names}")

    if args.dry_run:
        print("\ndry run — nothing written.")
        return

    data_dir = REPO_ROOT / cfg.data_dir
    started = time.time()
    stats = {
        split: write_split(
            ds[split],
            data_dir / f"{'val' if split == 'validation' else split}.txt",
            cfg.doc_separator,
            getattr(cfg, limit_attr),
        )
        for split, limit_attr in SPLITS.items()
    }
    print_stats(stats)

    peek_path = REPO_ROOT / cfg.results_dir / "samples" / "corpus_peek.md"
    peek_idx = write_peek(ds["train"], peek_path, cfg.peek_samples, cfg.seed, dataset)
    try:  # configured dirs may legitimately point outside the repo
        peek_ref = peek_path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        peek_ref = peek_path.as_posix()

    manifest = {
        "project_name": cfg.project_name,
        "config_hash": cfg.hash(),
        "data_stage_hash": cfg.stage_hash("data"),
        "seed": cfg.seed,
        "dataset": dataset,
        "splits": stats,
        "peek": {"file": peek_ref, "train_indices": peek_idx},
        "built_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "build_seconds": round(time.time() - started, 1),
    }
    manifest_path = write_json(data_dir / "manifest.json", manifest)

    print(f"\nwrote {data_dir / 'train.txt'}")
    print(f"wrote {data_dir / 'val.txt'}")
    print(f"wrote {peek_path}")
    print(f"wrote {manifest_path}")
    print(f"\ndone in {manifest['build_seconds']}s")


if __name__ == "__main__":
    main()
