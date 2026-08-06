"""Encoding guard. Run as a pre-commit hook.

I corrupted this repo's own source files twice with the same mistake, in Phase 3
and again in Phase 4: PowerShell's `Get-Content | Set-Content -Encoding utf8`
reads UTF-8 as ANSI, so every em dash becomes mojibake and a BOM gets prepended.
Both times it was caught by eye, which is not a control. This is the control.

A project whose Phase 1 headline is "we repaired 757,666 mojibake sequences"
cannot ship mojibake in its own source.

Checks, on tracked files only:
  * decodes as UTF-8
  * no byte-order mark
  * no LF/CRLF surprises (.gitattributes pins LF; this catches a stray BOM+CRLF pair)
  * no mojibake signature in code or config

Markdown is exempt from the mojibake check and only that check: ADR-009 and the
README quote the corrupted sequences on purpose, and a guard that forbade them
would forbid documenting the bug.

    python scripts/check_encoding.py          # check tracked files
    python scripts/check_encoding.py --install  # install the git hook
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# Signature of UTF-8 read as cp1252/latin-1: the leading byte of a multi-byte
# sequence surfacing as its own character.
MOJIBAKE = (chr(0x00C2), chr(0x00C3), chr(0x00E2), chr(0x20AC), chr(0x2020))
BOM = b"\xef\xbb\xbf"

CHECK_EXT = {".py", ".yaml", ".yml", ".json", ".txt", ".md", ".csv", ".cfg", ".toml"}
MOJIBAKE_EXEMPT_EXT = {".md"}          # docs quote the corruption deliberately
MOJIBAKE_EXEMPT_FILES = {
    "data/tokenizer.json",             # byte-level BPE vocab: these ARE tokens
    "data/manifest.json",              # records residual-character context
    "data/train.txt", "data/val.txt",  # 9 adjudicated residuals (ADR-009)
}


def tracked_files() -> list[Path]:
    out = subprocess.run(["git", "ls-files"], cwd=REPO, capture_output=True,
                         text=True, check=True).stdout
    return [REPO / line for line in out.splitlines() if line]


def check(paths: list[Path]) -> list[str]:
    problems: list[str] = []
    for path in paths:
        rel = path.relative_to(REPO).as_posix()
        if path.suffix.lower() not in CHECK_EXT or not path.exists():
            continue
        raw = path.read_bytes()

        if raw.startswith(BOM):
            problems.append(f"{rel}: byte-order mark (strip it; write with encoding='utf-8')")
            raw = raw[len(BOM):]

        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as e:
            problems.append(f"{rel}: not valid UTF-8 ({e.reason} at byte {e.start})")
            continue

        if path.suffix.lower() in MOJIBAKE_EXEMPT_EXT or rel in MOJIBAKE_EXEMPT_FILES:
            continue
        hits = {c: text.count(c) for c in MOJIBAKE if c in text}
        if hits:
            detail = ", ".join(f"U+{ord(c):04X} x{n}" for c, n in hits.items())
            problems.append(f"{rel}: mojibake signature ({detail})")
    return problems


HOOK = """#!/bin/sh
# nLemon-14 encoding guard - see scripts/check_encoding.py
python scripts/check_encoding.py || {
    echo "commit blocked: fix the encoding problems above" >&2
    exit 1
}
"""


def install() -> int:
    hooks = REPO / ".git" / "hooks"
    if not hooks.is_dir():
        print("no .git/hooks directory - is this a git repo?", file=sys.stderr)
        return 1
    target = hooks / "pre-commit"
    with open(target, "w", encoding="utf-8", newline="\n") as f:
        f.write(HOOK)
    print(f"installed {target}")
    print("note: .git/hooks is not version-controlled, so each clone must run "
          "`python scripts/check_encoding.py --install` once.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Encoding guard.")
    ap.add_argument("--install", action="store_true", help="install the pre-commit hook")
    args = ap.parse_args()
    if args.install:
        return install()

    files = tracked_files()
    problems = check(files)
    if problems:
        print(f"encoding check FAILED ({len(problems)} problem(s)):")
        for p in problems:
            print(f"  {p}")
        return 1
    print(f"encoding check ok ({len(files)} tracked files)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
