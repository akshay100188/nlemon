"""Phase 2 — our own subword vocabulary.

Trains a byte-level BPE on the corpus, encodes both splits to ``uint16`` token
shards, and exposes ``encode`` / ``decode``.

Byte-level is the load-bearing choice (ADR-011): every input byte is in the
alphabet, so there is no OOV, no unknown token, and the encode/decode roundtrip
is lossless by construction rather than by luck. That is what the Phase 2 gate
checks.

    python -m src.tokenizer train      # corpus -> data/tokenizer.json
    python -m src.tokenizer encode     # corpus -> data/{train,val}.bin
    python -m src.tokenizer check      # the gate: roundtrip + vocab size
    python -m src.tokenizer sweep      # evidence for tokenizer_train_docs
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Iterator

import numpy as np
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors, trainers
from tqdm import tqdm

from config import REPO_ROOT, Config
from utils.io import write_json, write_text
from utils.seed import set_seed

# vocab_size is 8_000, so a token id always fits in uint16. Asserted at encode
# time rather than assumed — a later vocab bump must fail loudly, not silently
# wrap around and corrupt every shard.
TOKEN_DTYPE = np.uint16
ENCODE_BATCH = 2_000


def tokenizer_path(cfg: Config) -> Path:
    return REPO_ROOT / cfg.data_dir / "tokenizer.json"


def shard_path(cfg: Config, split: str) -> Path:
    return REPO_ROOT / cfg.data_dir / f"{split}.bin"


def text_path(cfg: Config, split: str) -> Path:
    return REPO_ROOT / cfg.data_dir / f"{split}.txt"


# --------------------------------------------------------------------------- #
# corpus iteration
# --------------------------------------------------------------------------- #
def iter_docs(path: Path, separator: str, limit: int = 0) -> Iterator[str]:
    """Yield documents from a .txt shard, splitting on the separator line.

    Streams: the train shard is 1.8 GiB and this machine does not have that to
    spare.
    """
    buf: list[str] = []
    n = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.rstrip("\n") == separator:
                doc = "".join(buf).strip()
                buf.clear()
                if doc:
                    yield doc
                    n += 1
                    if limit and n >= limit:
                        return
            else:
                buf.append(line)
    tail = "".join(buf).strip()
    if tail:
        yield tail


# --------------------------------------------------------------------------- #
# train
# --------------------------------------------------------------------------- #
def build_tokenizer(cfg: Config) -> Tokenizer:
    """A GPT-2-lineage byte-level BPE, configured for a lossless roundtrip."""
    tok = Tokenizer(models.BPE())
    # add_prefix_space=False: we must not invent a leading space that the
    # decoder would then have to strip, which is a classic roundtrip break.
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
    tok.decoder = decoders.ByteLevel()
    tok.post_processor = processors.ByteLevel(trim_offsets=False)
    return tok


def train(cfg: Config, train_docs: int | None = None, quiet: bool = False) -> Tokenizer:
    docs = cfg.tokenizer_train_docs if train_docs is None else train_docs
    src = text_path(cfg, "train")
    if not src.exists():
        raise SystemExit(f"{src} not found - run `python -m src.data` first.")

    tok = build_tokenizer(cfg)
    trainer = trainers.BpeTrainer(
        vocab_size=cfg.vocab_size,
        min_frequency=cfg.tokenizer_min_frequency,
        special_tokens=[cfg.doc_separator],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=not quiet,
    )
    tok.train_from_iterator(
        iter_docs(src, cfg.doc_separator, limit=docs),
        trainer=trainer,
        length=docs or None,
    )
    return tok


# --------------------------------------------------------------------------- #
# encode
# --------------------------------------------------------------------------- #
def load(cfg: Config) -> Tokenizer:
    path = tokenizer_path(cfg)
    if not path.exists():
        raise SystemExit(f"{path} not found - run `python -m src.tokenizer train` first.")
    return Tokenizer.from_file(str(path))


def encode(tok: Tokenizer, text: str) -> list[int]:
    return tok.encode(text).ids


def decode(tok: Tokenizer, ids: list[int]) -> str:
    return tok.decode(ids)


def encode_split(cfg: Config, tok: Tokenizer, split: str) -> dict:
    """Stream a text shard into a flat uint16 .bin of token ids.

    Every document is terminated by the separator token, so the boundary the
    model learns is a real token rather than a formatting accident.
    """
    if tok.get_vocab_size() > np.iinfo(TOKEN_DTYPE).max + 1:
        raise SystemExit(
            f"vocab {tok.get_vocab_size()} exceeds {TOKEN_DTYPE.__name__} range - "
            f"widen TOKEN_DTYPE before encoding, or ids will wrap silently."
        )
    eot = tok.token_to_id(cfg.doc_separator)
    if eot is None:
        raise SystemExit(f"separator {cfg.doc_separator!r} is not in the vocabulary")

    src = text_path(cfg, split)
    dst = shard_path(cfg, split)
    n_tokens = 0
    n_docs = 0
    batch: list[str] = []
    started = time.time()

    with open(dst, "wb") as out:
        def flush() -> None:
            nonlocal n_tokens, n_docs
            if not batch:
                return
            ids: list[int] = []
            for enc in tok.encode_batch(batch):
                ids.extend(enc.ids)
                ids.append(eot)
            np.asarray(ids, dtype=TOKEN_DTYPE).tofile(out)
            n_tokens += len(ids)
            n_docs += len(batch)
            batch.clear()

        for doc in tqdm(iter_docs(src, cfg.doc_separator), desc=f"encoding {split}", unit="doc"):
            batch.append(doc)
            if len(batch) >= ENCODE_BATCH:
                flush()
        flush()

    return {
        "file": dst.name,
        "docs": n_docs,
        "tokens": n_tokens,
        "bytes": dst.stat().st_size,
        "dtype": TOKEN_DTYPE.__name__,
        "seconds": round(time.time() - started, 1),
    }


# --------------------------------------------------------------------------- #
# gate
# --------------------------------------------------------------------------- #
def roundtrip_check(cfg: Config, tok: Tokenizer, n_docs: int) -> dict:
    """The Phase 2 gate: exact string match on held-out documents.

    Held out means the **validation** split, which the tokenizer never saw
    during training. A roundtrip that only worked on training text would prove
    nothing about the vocabulary's coverage.
    """
    failures: list[dict] = []
    checked = 0
    checked_chars = 0
    for doc in iter_docs(text_path(cfg, "val"), cfg.doc_separator, limit=n_docs):
        back = decode(tok, encode(tok, doc))
        checked += 1
        checked_chars += len(doc)
        if back != doc:
            at = next((i for i, (a, b) in enumerate(zip(doc, back)) if a != b), min(len(doc), len(back)))
            if len(failures) < 5:
                failures.append({
                    "doc_index": checked - 1,
                    "first_diff_at": at,
                    "expected": doc[max(0, at - 30): at + 30],
                    "got": back[max(0, at - 30): at + 30],
                })
    return {
        "docs_checked": checked,
        "chars_checked": checked_chars,
        "failures": len(failures),
        "examples": failures,
        "passed": not failures,
    }


def measure_compression(cfg: Config, tok: Tokenizer, n_docs: int = 2000) -> dict:
    """Tokens per word on held-out text — how much the vocabulary actually buys."""
    tokens = 0
    words = 0
    chars = 0
    for doc in iter_docs(text_path(cfg, "val"), cfg.doc_separator, limit=n_docs):
        tokens += len(encode(tok, doc))
        words += len(doc.split())
        chars += len(doc)
    return {
        "docs": n_docs,
        "tokens": tokens,
        "words": words,
        "tokens_per_word": round(tokens / max(words, 1), 4),
        "chars_per_token": round(chars / max(tokens, 1), 4),
    }


# --------------------------------------------------------------------------- #
# commands
# --------------------------------------------------------------------------- #
def cmd_train(cfg: Config) -> None:
    print(f"training byte-level BPE: vocab={cfg.vocab_size:,} "
          f"min_freq={cfg.tokenizer_min_frequency} "
          f"docs={cfg.tokenizer_train_docs:,}")
    started = time.time()
    tok = train(cfg)
    out = tokenizer_path(cfg)
    tok.save(str(out))
    elapsed = round(time.time() - started, 1)

    meta = {
        "project_name": cfg.project_name,
        "config_hash": cfg.hash(),
        "tokenizer_stage_hash": cfg.stage_hash("tokenizer"),
        "seed": cfg.seed,
        "vocab_size": tok.get_vocab_size(),
        "vocab_size_config": cfg.vocab_size,
        "min_frequency": cfg.tokenizer_min_frequency,
        "trained_on_docs": cfg.tokenizer_train_docs,
        "separator": cfg.doc_separator,
        "separator_id": tok.token_to_id(cfg.doc_separator),
        "train_seconds": elapsed,
        "built_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    write_json(out.parent / "tokenizer_meta.json", meta)

    print(f"\nvocab size   : {tok.get_vocab_size():,} (config {cfg.vocab_size:,})")
    print(f"separator id : {meta['separator_id']}")
    print(f"trained in   : {elapsed}s")
    print(f"wrote {out}")


def cmd_encode(cfg: Config) -> None:
    tok = load(cfg)
    stats = {split: encode_split(cfg, tok, split) for split in ("train", "val")}

    print("\ntoken shards")
    print("-" * 64)
    print(f"{'split':<8}{'docs':>12}{'tokens':>16}{'MiB':>10}")
    for split, s in stats.items():
        print(f"{split:<8}{s['docs']:>12,}{s['tokens']:>16,}{s['bytes'] / 1024 ** 2:>10.1f}")
    print("-" * 64)

    meta_path = REPO_ROOT / cfg.data_dir / "tokenizer_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["shards"] = stats
    write_json(meta_path, meta)
    print(f"updated {meta_path}")


def cmd_check(cfg: Config, n_docs: int) -> None:
    tok = load(cfg)
    print(f"{cfg.project_name}  ·  phase 2 gate")
    print(f"tokenizer stage hash : {cfg.stage_hash('tokenizer')}\n")

    size_ok = tok.get_vocab_size() == cfg.vocab_size
    print(f"[{'PASS' if size_ok else 'FAIL'}] vocab size = {tok.get_vocab_size():,} "
          f"(config {cfg.vocab_size:,})")

    rt = roundtrip_check(cfg, tok, n_docs)
    print(f"[{'PASS' if rt['passed'] else 'FAIL'}] lossless roundtrip on "
          f"{rt['docs_checked']:,} held-out val docs "
          f"({rt['chars_checked']:,} chars), {rt['failures']} mismatch(es)")
    for ex in rt["examples"]:
        print(f"        doc {ex['doc_index']} at char {ex['first_diff_at']}")
        print(f"          expected {ex['expected']!r}")
        print(f"          got      {ex['got']!r}")

    comp = measure_compression(cfg, tok)
    print(f"\ncompression on held-out val: {comp['tokens_per_word']} tokens/word, "
          f"{comp['chars_per_token']} chars/token")

    result = {
        "vocab_size_ok": size_ok,
        "roundtrip": rt,
        "compression": comp,
        "tokenizer_stage_hash": cfg.stage_hash("tokenizer"),
        "passed": bool(size_ok and rt["passed"]),
    }
    out = write_json(REPO_ROOT / cfg.results_dir / "tokenizer_gate.json", result)
    print(f"wrote {out}")

    print(f"\nPHASE 2 GATE: {'PASS' if result['passed'] else 'FAIL'}")
    if not result["passed"]:
        raise SystemExit(1)


def cmd_sweep(cfg: Config, sizes: list[int]) -> None:
    """Evidence for tokenizer_train_docs: does more training text still help?"""
    rows = []
    for n in sizes:
        started = time.time()
        tok = train(cfg, train_docs=n, quiet=True)
        comp = measure_compression(cfg, tok, n_docs=2000)
        vocab = set(tok.get_vocab())
        rows.append({
            "train_docs": n,
            "vocab_size": tok.get_vocab_size(),
            "tokens_per_word": comp["tokens_per_word"],
            "chars_per_token": comp["chars_per_token"],
            "seconds": round(time.time() - started, 1),
            "_vocab": vocab,
        })
        print(f"  {n:>8,} docs -> {comp['tokens_per_word']:.4f} tokens/word "
              f"({rows[-1]['seconds']}s)")

    base = rows[-1]["_vocab"]
    for r in rows:
        r["vocab_overlap_vs_largest"] = round(len(r["_vocab"] & base) / len(base), 4)
        del r["_vocab"]

    write_sweep_report(cfg, rows)


def write_sweep_report(cfg: Config, rows: list[dict]) -> None:
    """Render the sweep evidence. Split out so the report can be regenerated
    from the saved rows without retraining six tokenizers."""
    # Conclusion is computed, not narrated: the smallest rung whose compression is
    # within 0.5% of the best one seen.
    best = min(r["tokens_per_word"] for r in rows)
    plateau = min(r["train_docs"] for r in rows
                  if r["tokens_per_word"] <= best * 1.005)
    chosen = next((r for r in rows if r["train_docs"] == cfg.tokenizer_train_docs), None)
    spread = max(r["tokens_per_word"] for r in rows) - best
    lo, hi = rows[0], rows[-1]
    doc_ratio = hi["train_docs"] / max(lo["train_docs"], 1)
    time_ratio = hi["seconds"] / max(lo["seconds"], 0.1)

    out = REPO_ROOT / cfg.results_dir / "tokenizer_subset_sweep.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# How much text does an 8k BPE actually need?",
        "",
        f"Held-out measurement on 2,000 **validation** documents. Seed `{cfg.seed}`, "
        f"vocab `{cfg.vocab_size:,}`, min_frequency `{cfg.tokenizer_min_frequency}`.",
        "",
        "`tokenizer_train_docs` bounds how much text the BPE *trains* on; the whole "
        "corpus is encoded regardless. The question is where more training text stops "
        "buying compression.",
        "",
        "| train docs | vocab | tokens/word | chars/token | vocab overlap vs largest | train time |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        mark = "  **<- config**" if chosen and r is chosen else ""
        lines.append(
            f"| {r['train_docs']:,} | {r['vocab_size']:,} | {r['tokens_per_word']} | "
            f"{r['chars_per_token']} | {r['vocab_overlap_vs_largest']:.1%} | "
            f"{r['seconds']}s{mark} |"
        )
    lines += [
        "",
        "## What this shows",
        "",
        f"Compression plateaus at **{plateau:,} documents** — the smallest rung within "
        f"0.5% of the best result measured. Across a {doc_ratio:.0f}x increase in "
        f"training text the spread is only {spread:.4f} tokens/word, while training "
        f"time grows {time_ratio:.0f}x.",
        "",
        "Training times are wall-clock on one laptop and move with system load — treat "
        "them as orders of magnitude, not measurements. The compression and overlap "
        "columns are deterministic.",
        "",
    ]
    if chosen:
        lines += [
            f"`tokenizer_train_docs` is set to **{chosen['train_docs']:,}**. That is past "
            f"the compression plateau, and its vocabulary is "
            f"{chosen['vocab_overlap_vs_largest']:.1%} identical to one trained on "
            f"{max(r['train_docs'] for r in rows):,} documents — for "
            f"{rows[-1]['seconds'] / max(chosen['seconds'], 0.1):.0f}x less training time.",
            "",
            "Compression is not the only thing at stake, which is why vocabulary overlap "
            "is measured too: two tokenizers can compress identically while disagreeing "
            "on which rare tokens earned a slot. Overlap is what justifies not simply "
            "taking the cheapest rung.",
            "",
        ]
    lines += ["Generated by `python -m src.tokenizer sweep`."]
    write_text(out, "\n".join(lines))

    write_json(out.with_suffix(".json"), {
        "seed": cfg.seed,
        "vocab_size": cfg.vocab_size,
        "min_frequency": cfg.tokenizer_min_frequency,
        "tokenizer_stage_hash": cfg.stage_hash("tokenizer"),
        "plateau_docs": plateau,
        "configured_docs": cfg.tokenizer_train_docs,
        "rows": rows,
    })
    print(f"\nwrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description="nLemon-14 tokenizer (Phase 2).")
    ap.add_argument("command",
                    choices=("train", "encode", "check", "sweep", "sweep-report"))
    ap.add_argument("--config", default=None)
    ap.add_argument("--check-docs", type=int, default=2000,
                    help="held-out docs for the roundtrip gate")
    ap.add_argument("--sweep-sizes", type=int, nargs="+",
                    default=[25_000, 50_000, 100_000, 200_000, 400_000])
    args = ap.parse_args()

    cfg = Config.load(args.config)
    set_seed(cfg.seed, strict=cfg.strict_determinism)

    if args.command == "train":
        cmd_train(cfg)
    elif args.command == "encode":
        cmd_encode(cfg)
    elif args.command == "check":
        cmd_check(cfg, args.check_docs)
    elif args.command == "sweep":
        cmd_sweep(cfg, args.sweep_sizes)
    else:
        src = REPO_ROOT / cfg.results_dir / "tokenizer_subset_sweep.json"
        if not src.exists():
            raise SystemExit(f"{src} not found - run `sweep` first.")
        write_sweep_report(cfg, json.loads(src.read_text(encoding="utf-8"))["rows"])


if __name__ == "__main__":
    main()
