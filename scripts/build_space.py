"""Assemble the HuggingFace Space from the repo, and prove the copies are the repo.

    python -m scripts.build_space            # build into build/space/
    python -m scripts.build_space --verify   # re-check an existing build

The Space needs `model.py`, `config.py`, `utils/` and `tokenizer.json`. The
tempting move is to copy them by hand once. That is how a Space starts serving an
architecture that has quietly drifted from the one the scorecard measured - the
same shape as every other instance in ADR-046's list: two things that are supposed
to be identical, validated separately, with nothing comparing them.

So the build is a script, and it **asserts every vendored file is byte-identical
to its source** before writing a manifest. `--verify` re-checks a built folder
against the repo at any later time, which is what makes "the Space runs the
measured model" a checkable claim rather than a promise.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "build" / "space"

# (source in repo, destination inside the Space). Flattened on purpose: the Space
# has no package structure, so `from model import GPT` and `from config import
# Config` must resolve at the root.
VENDORED: tuple[tuple[str, str], ...] = (
    ("src/model.py", "model.py"),
    ("config.py", "config.py"),
    # No utils/__init__.py: `utils` is a namespace package in this repo and
    # resolves without one on Python 3.3+. Listing a file that does not exist
    # would fail the build, which is the correct behaviour and how this was found.
    ("utils/device.py", "utils/device.py"),
    ("utils/io.py", "utils/io.py"),
    ("utils/seed.py", "utils/seed.py"),
    ("data/tokenizer.json", "tokenizer.json"),
)

# Space-specific files, authored for the Space and living in space/.
OWN: tuple[str, ...] = ("app.py", "requirements.txt", "README.md")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check(verify_only: bool) -> int:
    problems: list[str] = []
    rows: list[tuple[str, str, str]] = []

    if not verify_only:
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "utils").mkdir(exist_ok=True)

    for src_rel, dst_rel in VENDORED:
        src, dst = REPO / src_rel, OUT / dst_rel
        if not src.exists():
            problems.append(f"source missing: {src_rel}")
            continue
        if not verify_only:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
        if not dst.exists():
            problems.append(f"not built: {dst_rel}")
            continue
        a, b = sha256(src), sha256(dst)
        rows.append((f"{src_rel} -> {dst_rel}", b[:16], "ok" if a == b else "DRIFT"))
        if a != b:
            problems.append(f"{dst_rel} is NOT byte-identical to {src_rel}\n"
                            f"      repo  {a}\n      space {b}")

    for name in OWN:
        src, dst = REPO / "space" / name, OUT / name
        if not src.exists():
            problems.append(f"space/{name} missing from the repo")
            continue
        if not verify_only:
            shutil.copy2(src, dst)
        if not dst.exists():
            problems.append(f"not built: {name}")
            continue
        a, b = sha256(src), sha256(dst)
        rows.append((f"space/{name} -> {name}", b[:16], "ok" if a == b else "DRIFT"))
        if a != b:
            problems.append(f"{name} differs from space/{name}")

    print(f"{'file':<44} {'sha256':<18} state")
    for what, digest, state in rows:
        print(f"  {what:<42} {digest}...  {state}")

    if problems:
        print("\nBUILD REFUSED:")
        for p in problems:
            print(f"  - {p}")
        return 1

    print(f"\n  {len(rows)} files, all byte-identical to their sources")
    print(f"  built at {OUT}")
    print("\n  Upload the CONTENTS of that folder to the Space (not the folder).")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Build the HF Space from the repo.")
    ap.add_argument("--verify", action="store_true",
                    help="re-check an existing build without copying")
    args = ap.parse_args()
    return check(args.verify)


if __name__ == "__main__":
    sys.exit(main())
