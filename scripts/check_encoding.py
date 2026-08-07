"""Encoding guard. Run as a pre-commit hook.

I corrupted this repo's own source twice with the same mistake, in Phase 3 and
again in Phase 4: PowerShell's `Get-Content | Set-Content -Encoding utf8` reads
UTF-8 as ANSI, so every em dash becomes mojibake and a BOM gets prepended. Both
times a human eye caught it, which is not a control. This is the control.

A project whose Phase 1 headline is "we repaired 757,666 mojibake sequences"
cannot ship mojibake in its own source.

Two different checks, because there are two different failures:

1. MALFORMED - bytes that are not valid UTF-8, a BOM, or a mojibake signature in
   code or config. This is the PowerShell failure.

2. CLEAN BUT WRONG - the harder one. Running ftfy over the docs "repaired"
   ADR-009's example quotes into well-formed UTF-8 that says the wrong thing.
   Check 1 passes it: the output is legal. That is ADR-009's own lesson one level
   up, the dangerous corruption is the one that looks correct. So the examples
   are pinned by exact codepoint sequence in GOLDEN, and flattening them fails
   the guard even though nothing is malformed.

Markdown is exempt from check 1's mojibake scan and only that: the ADRs quote
corrupted bytes on purpose, and a guard forbidding them would forbid documenting
the bug. GOLDEN is what keeps that exemption from being a hole.

This file is pure ASCII on purpose, verified by its own test. A file about
mojibake must not depend on being read in the right encoding.

    python scripts/check_encoding.py            # check tracked files
    python scripts/check_encoding.py --install  # install the git hook
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# Signature of UTF-8 read as cp1252/latin-1: the leading byte of a multi-byte
# sequence surfacing as a character in its own right.
MOJIBAKE = (chr(0x00C2), chr(0x00C3), chr(0x00E2), chr(0x20AC), chr(0x2020))
BOM = b"\xef\xbb\xbf"

CHECK_EXT = {".py", ".yaml", ".yml", ".json", ".txt", ".md", ".csv", ".cfg", ".toml"}
MOJIBAKE_EXEMPT_EXT = {".md"}          # docs quote the corruption deliberately
MOJIBAKE_EXEMPT_FILES = {
    "data/tokenizer.json",             # byte-level BPE vocab: these ARE tokens
    "data/manifest.json",              # records residual-character context
    "data/train.txt", "data/val.txt",  # 9 adjudicated residuals (ADR-009)
    "scripts/check_encoding.py",       # GOLDEN below names the sequences
}

# Golden examples: the corruption that is supposed to be there. Built from chr()
# so the source stays ASCII.
_MOJI_RSQUO = chr(0x00E2) + chr(0x20AC) + chr(0x2122)   # mojibake of U+2019
_MOJI_LDQUO = chr(0x00E2) + chr(0x20AC) + chr(0x0153)   # mojibake of U+201C
_MOJI_EACUTE = chr(0x00C3) + chr(0x00A9)                # mojibake of U+00E9
_MOJI_NTILDE = chr(0x00C3) + chr(0x00B1)                # mojibake of U+00F1
_MOJI_SHY = chr(0x00C2) + chr(0x00AD)                   # mojibake of U+00AD
_MOJI_EMOJI = chr(0x00E2) + chr(0x00A4) + chr(0x00EF)   # mojibake'd heart emoji
_DAGGER = chr(0x2020)                                   # what ftfy invented
_PAPIER = "papier-m" + chr(0x00E2) + "ch" + chr(0x00E9)  # legitimate French

GOLDEN: dict[str, list[tuple[str, str]]] = {
    "ADR.md": [
        ("mojibake'd right single quote (ADR-009 example)", _MOJI_RSQUO),
        ("mojibake'd left double quote (ADR-009 example)", _MOJI_LDQUO),
        ("mojibake'd e-acute (ADR-009 tail)", _MOJI_EACUTE),
        ("mojibake'd n-tilde (ADR-009 tail)", _MOJI_NTILDE),
        ("mojibake'd soft hyphen (ADR-009 tail)", _MOJI_SHY),
        ("mojibake'd emoji fragment (ADR-009 footnote)", _MOJI_EMOJI),
        # No U+2020 entry here: ADR-009 names the dagger in words rather than
        # printing one. Pinning it would assert something the file never
        # contained - a golden list has to record what is there, not what I
        # assumed was. README.md does print one, and pins it.
        ("legitimate French a-circumflex (the ADR-009 false positive)", _PAPIER),
    ],
    "README.md": [
        ("mojibake'd right single quote (Cleaning section)", _MOJI_RSQUO),
        ("U+2020 DAGGER, the character ftfy invented", _DAGGER),
        ("legitimate French a-circumflex (the ADR-009 false positive)", _PAPIER),
    ],
}


def tracked_files() -> list[Path]:
    out = subprocess.run(["git", "ls-files"], cwd=REPO, capture_output=True,
                         text=True, check=True).stdout
    return [REPO / line for line in out.splitlines() if line]


def check_malformed(paths: list[Path]) -> list[str]:
    problems: list[str] = []
    for path in paths:
        rel = path.relative_to(REPO).as_posix()
        if path.suffix.lower() not in CHECK_EXT or not path.exists():
            continue
        raw = path.read_bytes()

        if raw.startswith(BOM):
            problems.append(f"{rel}: byte-order mark (write with encoding='utf-8')")
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


def check_golden() -> list[str]:
    """The documented corruption must still be corrupt."""
    problems = []
    for rel, entries in GOLDEN.items():
        path = REPO / rel
        if not path.exists():
            problems.append(f"{rel}: missing, cannot verify golden examples")
            continue
        text = path.read_text(encoding="utf-8")
        for label, needle in entries:
            if needle not in text:
                cps = " ".join(f"U+{ord(c):04X}" for c in needle)
                problems.append(
                    f"{rel}: golden example gone - {label} [{cps}]. Something "
                    f"'fixed' the evidence; the text is now clean and wrong."
                )
    return problems


def check_self_ascii() -> list[str]:
    """This file claims to be pure ASCII. Verify the claim."""
    raw = Path(__file__).read_bytes()
    bad = [i for i, b in enumerate(raw) if b > 127]
    if bad:
        return [f"scripts/check_encoding.py: {len(bad)} non-ASCII byte(s), first "
                f"at offset {bad[0]}. A guard against mojibake must not contain any."]
    return []


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
    print("note: .git/hooks is not version-controlled, so each clone runs "
          "`python scripts/check_encoding.py --install` once.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Encoding guard.")
    ap.add_argument("--install", action="store_true", help="install the pre-commit hook")
    args = ap.parse_args()
    if args.install:
        return install()

    files = tracked_files()
    problems = check_malformed(files) + check_golden() + check_self_ascii()
    if problems:
        print(f"encoding check FAILED ({len(problems)} problem(s)):")
        for p in problems:
            print(f"  {p}")
        return 1
    golden_count = sum(len(v) for v in GOLDEN.values())
    print(f"encoding check ok ({len(files)} tracked files, "
          f"{golden_count} golden examples intact)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
