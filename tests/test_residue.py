"""Regression test for the ADR-009 encoding repair.

Phase 5 formats instruction/response pairs through `clean_text`, so this is the
last moment to pin its behaviour before another phase depends on it.

**This test corrected the thing it was written to protect.** It was written
believing `RESIDUE` ordering was load-bearing — that the no-break-space and
soft-hyphen rules strand an `A-circumflex` prefix which the orphan rule must
clear afterwards, so the orphan rule had to run last. `src/data.py` said so in a
comment and ADR-009 implied it.

It is false. `ftfy` repairs `A-circumflex`+no-break-space and
`A-circumflex`+soft-hyphen *before* `RESIDUE` ever runs, so those pairs never
reach the rules. The orphans that actually survived in the corpus were
`A-circumflex` followed by **ASCII** (`wasn<C2>'t`, `loved.<C2>`), which ftfy
cannot interpret as mojibake and leaves alone. The rules never stranded
anything.

Measured: 401 random permutations of `RESIDUE` produce identical output on every
fixture, and no rule pattern contains another. The rule set is **order
independent**, and that is what this test now protects — along with the
structural invariant that keeps it so. A documented hazard that does not exist is
its own kind of misinformation: it makes people preserve an ordering for a reason
that was never true, and fear touching it.

    python -m tests.test_residue
"""

from __future__ import annotations

import sys

from src.data import RESIDUE, clean_text

# Named so the test reads as text, not as codepoints.
A = chr(0x00C2)       # orphaned mojibake prefix (capital A with circumflex)
NBSP = chr(0x00A0)
SHY = chr(0x00AD)
E2 = chr(0x00E2)
EURO = chr(0x20AC)
TM = chr(0x2122)
DAGGER = chr(0x2020)
ACIRC = chr(0x00E2)   # lowercase, legitimate in French
EACUTE = chr(0x00E9)
FFFD = chr(0xFFFD)

CASES: list[tuple[str, str, str]] = [
    # label, input, expected output
    ("ftfy repairs a right single quote",
     "wasn" + E2 + EURO + TM + "t", "wasn't"),
    ("unrepairable corpse becomes an ASCII quote",
     "said " + E2 + EURO + ".", 'said ".'),
    ("the dagger ftfy invented becomes a quote",
     '"hi' + DAGGER + " and", '"hi" and'),

    # --- interacting rules: these are the ones that detect a reordering ---
    ("INTERACTING: prefix + no-break space",
     'here?"' + A + NBSP + "it asked", 'here?" it asked'),
    ("INTERACTING: prefix + soft hyphen",
     "wasn" + A + SHY + "'t sure", "wasn't sure"),
    ("INTERACTING: bare stranded prefix",
     "loved." + A, "loved."),

    # --- things that must survive untouched ---
    ("legitimate French is left alone",
     "papier-m" + ACIRC + "ch" + EACUTE, "papier-m" + ACIRC + "ch" + EACUTE),
    ("bare no-break space becomes a space",
     "two" + NBSP + "words", "two words"),
    ("replacement char is dropped",
     "lost" + FFFD + "byte", "lostbyte"),
    ("clean text is unchanged",
     "Once upon a time, Tom said \"hello\".", "Once upon a time, Tom said \"hello\"."),
]


def check(label: str, got, want) -> bool:
    ok = got == want
    print(f"  [{'ok  ' if ok else 'FAIL'}] {label}")
    if not ok:
        print(f"           got  {got!r}")
        print(f"           want {want!r}")
    return ok


def test_cases() -> int:
    print(f"clean_text fixtures ({len(CASES)}):")
    return sum(0 if check(label, clean_text(raw)[0], want) else 1
               for label, raw, want in CASES)


PERMUTATIONS = 400


def test_order_independence() -> int:
    """Every permutation of RESIDUE must give the same answer.

    This is the property that actually holds, replacing the ordering dependency
    the test originally asserted and disproved. If it ever fails, someone has
    added a rule that interacts with another, and the ordering has to become
    explicit and documented rather than incidental.
    """
    import random

    import src.data as data

    original = data.RESIDUE
    random.seed(0)
    perms = [tuple(random.sample(original, len(original)))
             for _ in range(PERMUTATIONS)]

    failures = 0
    try:
        for label, raw, want in CASES:
            outputs = set()
            for perm in perms:
                data.RESIDUE = perm
                outputs.add(data.clean_text(raw)[0])
            if len(outputs) != 1 or outputs.pop() != want:
                print(f"  [FAIL] {label}: ordering changes the result")
                failures += 1
    finally:
        data.RESIDUE = original

    print(f"\norder independence ({PERMUTATIONS} permutations):")
    if not failures:
        print(f"  [ok  ] all {len(CASES)} fixtures identical under every permutation")
    return failures


def test_no_overlapping_rules() -> int:
    """The structural reason order does not matter.

    If one rule's pattern contains another's, the order they run in decides the
    output. Today none does. This fails the moment that changes, which is the
    moment ordering stops being incidental and has to be designed.
    """
    overlaps = [(repr(a), repr(b)) for a, _ in RESIDUE for b, _ in RESIDUE
                if a != b and a in b]
    print("\nstructural invariant:")
    if overlaps:
        print(f"  [FAIL] rule patterns overlap: {overlaps}. Order now decides the "
              f"result - make it explicit and document why.")
        return 1
    print(f"  [ok  ] no rule pattern contains another ({len(RESIDUE)} rules)")
    return 0


def main() -> int:
    failures = (test_cases() + test_order_independence()
                + test_no_overlapping_rules())
    print()
    if failures:
        print(f"RESIDUE REGRESSION: FAIL ({failures} problem(s))")
        return 1
    print(f"RESIDUE REGRESSION: PASS ({len(CASES)} fixtures, "
          f"{PERMUTATIONS} permutations, {len(RESIDUE)} rules non-overlapping)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
