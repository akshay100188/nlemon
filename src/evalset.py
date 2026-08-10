"""Phase 7 — the frozen evaluation sets, and the guard that keeps them frozen.

    python -m src.evalset verify     # assert both sets match their registered hashes
    python -m src.evalset hashes     # print what is on disk, for registration

This module exists **before** the harness, deliberately. An eval set is the one
thing a saturated or inconvenient metric could be quietly reselected against:
nothing in a scorecard reveals that the set it was computed over is not the set
that was registered. Bars are frozen in config and cross-checked on every gate
run; the eval set gets the same treatment for the same reason (ADR-048).

Two sets, because Phase 7 reports two families of metric:

  data/val.bin           4,682,459 tokens - the perplexity set. The ENTIRE
                         validation split, not a sample, which is what makes it
                         a fixed set with nothing to reselect and no seed to
                         choose (ADR-020's "large fixed evaluation set").
  data/sft_heldout.json  312 prompts over 78 held-out subjects - the feature set,
                         disjoint by subject from everything training touched.

Both are regenerable from recorded inputs, so committing the hash rather than the
artifact is enough: `data/*.bin` and `data/*.json` are gitignored precisely
because they are derived, and a hash is the part that cannot be silently wrong.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

from config import REPO_ROOT, Config

# The frozen sets, keyed by the config field holding the registered hash.
FROZEN: tuple[tuple[str, str, str], ...] = (
    ("data/val.bin", "eval_val_sha256", "perplexity set (full validation split)"),
    ("data/sft_heldout.json", "eval_heldout_sha256", "feature set (78 held-out subjects)"),
)


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def assert_frozen(cfg: Config, verbose: bool = True) -> dict[str, str]:
    """Refuse to proceed unless every eval set matches its registered hash.

    Reported as a mismatch, never repaired. A harness that silently re-hashes
    whatever it finds has no tamper-evidence at all - it just always agrees with
    itself, which is the failure mode ADR-047 found in the device field's absence.
    """
    seen: dict[str, str] = {}
    drift: list[str] = []
    missing: list[str] = []

    for rel, field, what in FROZEN:
        path = REPO_ROOT / rel
        registered = getattr(cfg, field)
        if not path.exists():
            missing.append(f"{rel} is absent - regenerate it before evaluating ({what})")
            continue
        actual = sha256_of(path)
        seen[rel] = actual
        ok = actual == registered
        if verbose:
            print(f"  {'ok  ' if ok else 'DRIFT'}  {rel:<24} {actual[:16]}...  "
                  f"({what})")
        if not ok:
            drift.append(f"{rel}\n      registered {registered}\n      on disk    {actual}")

    if missing:
        raise SystemExit("eval set missing:\n  " + "\n  ".join(missing))
    if drift:
        raise SystemExit(
            "EVAL SET HASH MISMATCH - refusing to evaluate:\n  "
            + "\n  ".join(drift)
            + "\n\nThe set on disk is not the set Phase 7 registered. Either it was "
              "rebuilt from different inputs, or a different set is being scored. "
              "Regenerate from the recorded inputs, or - if the change is intended - "
              "re-register the hash in its own commit BEFORE any number is read from "
              "the new set (ADR-048).")
    if verbose:
        print(f"  eval sets verified against registered hashes ({len(seen)} sets)")
    return seen


def cmd_hashes(cfg: Config) -> None:
    for rel, field, what in FROZEN:
        path = REPO_ROOT / rel
        if not path.exists():
            print(f"  {rel:<24} ABSENT")
            continue
        print(f"  {rel:<24} {sha256_of(path)}")
        print(f"  {'':<24} registered as {field}  ({what})")


def main() -> int:
    ap = argparse.ArgumentParser(description="Frozen Phase 7 evaluation sets.")
    ap.add_argument("command", choices=("verify", "hashes"))
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = Config.load(args.config)
    if args.command == "hashes":
        cmd_hashes(cfg)
        return 0
    assert_frozen(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
