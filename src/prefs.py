"""Phase 6 — on-policy preference pairs, matched so that subject fidelity is the
only systematic difference between chosen and rejected.

Sample `k` responses from `sft.pt` for each training prompt, then pair a
subject-faithful one against a subject-unfaithful one **from the same prompt**.
Both sides are the model's own output, so DPO re-ranks a distribution it already
samples rather than being pulled off it.

**The matched control is the load-bearing decision in this phase.** Chosen and
rejected must be comparable on length and `is_story`, so the only thing that
systematically separates them is whether the response is about the requested
subject. Skip it and the preference signal quietly encodes "longer is better" or
"ends with a period is better" - inside the pairs, where no gate can see it, with
the floors sitting green while DPO optimises the exact trade they exist to catch.
It is also the answer to the sharpest question this phase invites: *how do you
know DPO taught subject-fidelity and not length?* Because the pairs were matched
on length, so length could not be the signal.

**And the match has to hold jointly, not marginally** (ADR-039). Matching on
length and separately on `is_story` is not enough, because the aboutness filter
layered on top can correlate with length - a story genuinely about its subject
plausibly runs longer - and if it does, the pairs **un-match themselves at the
point of selection**. So the matching happens *after* the aboutness filter, on
the candidates that actually survive it, and `stats` reports the parity that
results rather than the parity that was intended.

Chosen must clear aboutness >= 2, not bare mention: selecting on bare mention is
literally an instruction to say the word more often.

Pairs come from the **310 train subjects only**. The 78 held-out subjects are the
eval and never appear here.

    python -m src.prefs sample    # generate k candidates per prompt from sft.pt
    python -m src.prefs build     # filter, match, write pairs
    python -m src.prefs stats     # the pre-flight: did the match survive?
"""

from __future__ import annotations

import argparse
import json
import random

from config import REPO_ROOT, Config
from src.checker import _subject_forms, check_one, load_band
from src.coherence import words_of
from utils.io import write_json, write_text


def occurrences(text: str, subject: str) -> int:
    forms = set(_subject_forms(subject))
    return sum(1 for w in words_of(text) if w in forms)


def cmd_sample(cfg: Config) -> None:
    """k samples per prompt from sft.pt, on-policy.

    Prompts are drawn from the training pairs, so every subject here is a train
    subject. Seeds are per (prompt, draw) so the whole set regenerates exactly.
    """
    import torch

    from src.sample import generate, load_checkpoint
    from src.tokenizer import load as load_tokenizer
    from utils.device import probe
    from utils.seed import set_seed

    set_seed(cfg.seed, strict=cfg.strict_determinism)
    train = json.loads((REPO_ROOT / cfg.data_dir / "sft_train.json")
                       .read_text(encoding="utf-8"))
    held = json.loads((REPO_ROOT / cfg.data_dir / "sft_heldout.json")
                      .read_text(encoding="utf-8"))
    held_subjects = {p["subject"] for p in held}

    rng = random.Random(cfg.seed)
    # One prompt per subject-instance, sampled without replacement, capped.
    idx = list(range(len(train)))
    rng.shuffle(idx)
    picked, seen = [], set()
    for i in idx:
        p = train[i]
        if p["subject"] in held_subjects:
            raise SystemExit(f"held-out subject {p['subject']!r} in train pairs")
        key = (p["subject"], p["prompt"])
        if key in seen:
            continue
        seen.add(key)
        picked.append(p)
        if len(picked) >= cfg.pref_prompts:
            break
    print(f"{len(picked):,} prompts over {len({p['subject'] for p in picked})} "
          f"train subjects; {cfg.pref_samples} samples each "
          f"= {len(picked) * cfg.pref_samples:,} generations")

    dev = probe()
    model, _ = load_checkpoint(REPO_ROOT / cfg.ckpt_dir / "sft.pt", dev.device)
    tok = load_tokenizer(cfg)
    eot = tok.token_to_id(cfg.doc_separator)

    out = []
    for i, p in enumerate(picked):
        wire = p["prompt"] + "\n"
        ids = tok.encode(wire).ids
        cands = []
        for j in range(cfg.pref_samples):
            g = torch.Generator(device="cpu").manual_seed(cfg.seed + i * 97 + j)
            gen = generate(model, ids, cfg.sft_gate_new_tokens,
                           cfg.sft_gate_temperature, cfg.sft_gate_top_k or None,
                           dev.device, eot_id=eot, generator=g)
            full = tok.decode(gen)
            cands.append(full[len(wire):] if full.startswith(wire)
                         else full[len(p["prompt"]):])
        out.append({"subject": p["subject"], "prompt": p["prompt"],
                    "candidates": cands})
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(picked)}")

    write_json(REPO_ROOT / cfg.data_dir / "pref_samples.json", out)
    print(f"wrote {REPO_ROOT / cfg.data_dir / 'pref_samples.json'}")


def cmd_build(cfg: Config) -> None:
    """Filter to faithful/unfaithful candidates, then match on length.

    Order matters and it is the ADR-039 point: aboutness filtering happens
    FIRST, matching happens on what survives. Matching a pool and then filtering
    it would leave the pairs matched on paper and drifted in fact.
    """
    band = load_band(cfg)
    src = REPO_ROOT / cfg.data_dir / "pref_samples.json"
    if not src.exists():
        raise SystemExit("run `python -m src.prefs sample` first.")
    groups = json.loads(src.read_text(encoding="utf-8"))

    rng = random.Random(cfg.seed + 5)
    pairs, drops = [], {"no_faithful": 0, "no_unfaithful": 0, "no_match": 0}
    for g in groups:
        subj = g["subject"]
        scored = []
        for c in g["candidates"]:
            row = check_one(cfg, subj, c, band)
            row["text"] = c
            row["occ"] = occurrences(c, subj)
            scored.append(row)
        # aboutness FIRST
        faithful = [r for r in scored if r["occ"] >= cfg.sft_min_aboutness]
        unfaithful = [r for r in scored if not r["subject_mention"]]
        if not faithful:
            drops["no_faithful"] += 1
            continue
        if not unfaithful:
            drops["no_unfaithful"] += 1
            continue
        # ...then match, on the survivors only. Pick the (chosen, rejected) pair
        # whose word counts are closest, requiring both to agree on is_story so
        # story-ness cannot be the signal either.
        best, best_gap = None, None
        for c in faithful:
            for r in unfaithful:
                if c["is_story"] != r["is_story"]:
                    continue
                gap = abs(c["words"] - r["words"])
                if gap > cfg.pref_max_length_gap:
                    continue
                if best_gap is None or gap < best_gap:
                    best, best_gap = (c, r), gap
        if best is None:
            drops["no_match"] += 1
            continue
        c, r = best
        pairs.append({
            "subject": subj, "prompt": g["prompt"],
            "chosen": c["text"], "rejected": r["text"],
            "chosen_words": c["words"], "rejected_words": r["words"],
            "chosen_occ": c["occ"], "rejected_occ": r["occ"],
            "chosen_is_story": c["is_story"], "rejected_is_story": r["is_story"],
            "length_gap": best_gap,
        })

    print(f"built {len(pairs):,} pairs from {len(groups):,} prompts")
    for k, v in drops.items():
        print(f"  dropped {v:>5,}  {k}")
    write_json(REPO_ROOT / cfg.data_dir / "pref_pairs.json", pairs)
    print(f"wrote {REPO_ROOT / cfg.data_dir / 'pref_pairs.json'}")


def _welch(a: list[float], b: list[float]) -> tuple[float, float]:
    """Welch t and a normal-approx two-sided p. No SciPy in this project."""
    import math
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return 0.0, 1.0
    ma, mb = sum(a) / na, sum(b) / nb
    va = sum((x - ma) ** 2 for x in a) / (na - 1)
    vb = sum((x - mb) ** 2 for x in b) / (nb - 1)
    se = math.sqrt(va / na + vb / nb)
    if se == 0:
        return 0.0, 1.0
    t = (ma - mb) / se
    return t, math.erfc(abs(t) / math.sqrt(2))


def cmd_stats(cfg: Config) -> None:
    """The pre-flight: did the match survive the aboutness filter?

    This is Phase 6's equivalent of the surviving-length-distribution census in
    Phase 5 - a validity check on the *training signal* rather than on the model,
    run before anything trains. If chosen skews longer than rejected here, the
    preference signal contains a length gradient and the phase's central claim
    ("length could not be the signal") is false before DPO starts.
    """
    p = REPO_ROOT / cfg.data_dir / "pref_pairs.json"
    if not p.exists():
        raise SystemExit("run `python -m src.prefs build` first.")
    pairs = json.loads(p.read_text(encoding="utf-8"))
    cw = [x["chosen_words"] for x in pairs]
    rw = [x["rejected_words"] for x in pairs]
    co = [x["chosen_occ"] for x in pairs]
    ro = [x["rejected_occ"] for x in pairs]
    n = len(pairs)

    t, pv = _welch([float(x) for x in cw], [float(x) for x in rw])
    mean_gap = sum(x["length_gap"] for x in pairs) / max(n, 1)
    story_c = sum(1 for x in pairs if x["chosen_is_story"]) / max(n, 1)
    story_r = sum(1 for x in pairs if x["rejected_is_story"]) / max(n, 1)
    subs = len({x["subject"] for x in pairs})

    print(f"pair set: {n:,} pairs over {subs} train subjects\n")
    print(f"  LENGTH PARITY, measured AFTER the aboutness filter (ADR-039):")
    print(f"    chosen   mean {sum(cw)/n:>7.2f} words   median "
          f"{sorted(cw)[n//2]:>4}")
    print(f"    rejected mean {sum(rw)/n:>7.2f} words   median "
          f"{sorted(rw)[n//2]:>4}")
    print(f"    difference    {(sum(cw)-sum(rw))/n:>+7.2f} words   "
          f"mean |gap| within pair {mean_gap:.2f}")
    print(f"    Welch t {t:+.3f}, p {pv:.4f}  -> "
          f"{'PARITY HOLDS' if pv > 0.05 else 'DRIFTED - re-match required'}")
    print(f"\n  IS_STORY PARITY (matched exactly within each pair):")
    print(f"    chosen {story_c:.1%}   rejected {story_r:.1%}   "
          f"{'ok' if abs(story_c - story_r) < 1e-9 else 'MISMATCH'}")
    print(f"\n  ABOUTNESS - the signal that IS supposed to differ:")
    print(f"    chosen   mean {sum(co)/n:.2f} occurrences, min "
          f"{min(co)}  (filter requires >= {cfg.sft_min_aboutness})")
    print(f"    rejected mean {sum(ro)/n:.2f} occurrences, max {max(ro)}  "
          f"(must be 0 - no mention at all)")

    ok = pv > 0.05 and abs(story_c - story_r) < 1e-9 and max(ro) == 0
    print(f"\n  PRE-FLIGHT: {'PASS' if ok else 'FAIL'}")
    write_json(REPO_ROOT / cfg.results_dir / "pref_pairs_stats.json", {
        "pairs": n, "subjects": subs,
        "chosen_words_mean": round(sum(cw) / n, 3),
        "rejected_words_mean": round(sum(rw) / n, 3),
        "length_diff": round((sum(cw) - sum(rw)) / n, 3),
        "mean_abs_gap": round(mean_gap, 3),
        "welch_t": round(t, 4), "welch_p": round(pv, 6),
        "length_parity_holds": bool(pv > 0.05),
        "is_story_chosen": round(story_c, 6), "is_story_rejected": round(story_r, 6),
        "chosen_occ_mean": round(sum(co) / n, 3), "chosen_occ_min": min(co),
        "rejected_occ_mean": round(sum(ro) / n, 3), "rejected_occ_max": max(ro),
        "preflight_pass": bool(ok),
    })
    if not ok:
        raise SystemExit(1)


def main() -> None:
    ap = argparse.ArgumentParser(description="DPO preference pairs (Phase 6).")
    ap.add_argument("command", choices=("sample", "build", "stats"))
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = Config.load(args.config)
    {"sample": cmd_sample, "build": cmd_build, "stats": cmd_stats}[args.command](cfg)


if __name__ == "__main__":
    main()
