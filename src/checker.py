"""The deterministic feature-checker (ADR-005), reporting sub-scores separately.

Four checks, and they are **not** blended into one adherence scalar, because
they move for different reasons and a single number hides which one a fine-tune
actually bought (ADR-024):

* `subject_mention` - does the response talk about the thing that was asked for?
  This is the sub-score that carries the instruction-following claim, **and** it
  is the leak detector. `base.pt` never saw these instructions, so it has no
  reason to name the subject unless the prompt is handing the checker its own
  answer. A high base score means a broken checker, not a weak floor - the
  opposite response to a low one.

* `length_band` - is the response the length of a story from this corpus? Bands
  come from the training pairs, not from taste.

* `is_story` - **a degeneracy floor, deliberately excluded from the delta.**
  `base.pt` emits stories unconditionally, so this is near-saturated before SFT
  touches anything: `known_word_rate`'s degenerate band wearing a different hat.
  It has almost no headroom, so a green here is not evidence that SFT worked. It
  earns its place only as a floor: a fine-tune that stops producing stories must
  fail.

* `not_degenerate` - reuses the Phase 4 corpus bands. Also a floor.

The headline is therefore two numbers, not one: subject-mention and length-band.

    python -m src.checker demo   # run the checker over the held-out responses
"""

from __future__ import annotations

import argparse
import json
import re

from config import REPO_ROOT, Config
from src.coherence import max_immediate_repeat_run, repeated_ngram_rate, words_of

SENTENCE_END = re.compile(r"[.!?]")

# The sub-scores that carry the instruction-following claim, versus the ones that
# only ever act as floors. Kept as data so the gate cannot quietly blend them.
DELTA_SCORES = ("subject_mention", "length_band")
FLOOR_SCORES = ("is_story", "not_degenerate")
ALL_SCORES = DELTA_SCORES + FLOOR_SCORES


def _subject_forms(subject: str) -> tuple[str, ...]:
    """Crude morphology: the word, its plural, and a trimmed stem.

    Deliberately generous. Under-counting a genuine mention would understate the
    fine-tune, and the number this feeds is a *delta over base* - both sides get
    the same generosity, so a loose matcher cancels rather than flatters.
    """
    forms = {subject, subject + "s", subject + "es"}
    if subject.endswith("y"):
        forms.add(subject[:-1] + "ies")
    if len(subject) > 4:
        forms.add(subject[:-1])
    return tuple(forms)


def check_one(cfg: Config, subject: str, response: str, band: tuple[int, int]) -> dict:
    """Score one response. `response` must exclude the prompt."""
    toks = words_of(response)
    n_words = len(toks)
    token_set = set(toks)

    forms = _subject_forms(subject)
    mentioned = any(f in token_set for f in forms)

    low, high = band
    in_band = low <= n_words <= high

    sentences = [s for s in SENTENCE_END.split(response) if s.strip()]
    # "Is it a story": several complete sentences and some narrative motion.
    is_story = (len(sentences) >= cfg.checker_min_sentences
                and n_words >= cfg.checker_min_story_words
                and response.strip().endswith((".", "!", "?", '"')))

    rep4 = repeated_ngram_rate(toks, 4) if toks else 0.0
    run = max_immediate_repeat_run(toks) if toks else 0
    not_degenerate = (rep4 <= cfg.checker_max_repeat_rate
                      and run <= cfg.checker_max_repeat_run)

    return {
        "subject": subject,
        "words": n_words,
        "subject_mention": bool(mentioned),
        "length_band": bool(in_band),
        "is_story": bool(is_story),
        "not_degenerate": bool(not_degenerate),
        "repeated_4gram_rate": round(rep4, 6),
        "max_repeat_run": int(run),
        "sentences": len(sentences),
    }


def shuffled_control(cfg: Config, rows: list[dict], responses: list[str],
                     band: tuple[int, int]) -> float:
    """Subject-mention when each response is scored against the WRONG subject.

    This is what separates three different things that a raw subject-mention
    rate blends together:

      chance      - the shuffled rate. How often a story mentions some unrelated
                    common noun anyway.
      echo        - matched rate minus shuffled, for a model that was never
                    instructed. A language model conditioned on a prompt
                    containing "bunny" will say "bunny"; that is ordinary
                    conditioning, not instruction-following.
      instruction - what SFT has to buy on top of both.

    Without this control, base's matched rate is uninterpretable: high could mean
    a broken checker, or could mean an LM behaving exactly as an LM does.
    """
    import random

    rng = random.Random(cfg.seed)
    subjects = [r["subject"] for r in rows]
    shuffled = subjects[:]
    # derangement: nobody keeps their own subject
    for _ in range(64):
        rng.shuffle(shuffled)
        if all(a != b for a, b in zip(subjects, shuffled)):
            break
    hits = sum(1 for subj, resp in zip(shuffled, responses)
               if check_one(cfg, subj, resp, band)["subject_mention"])
    return round(hits / max(len(rows), 1), 6)


def aggregate(rows: list[dict]) -> dict:
    """Rates per sub-score, never blended."""
    n = max(len(rows), 1)
    out = {s: round(sum(1 for r in rows if r[s]) / n, 6) for s in ALL_SCORES}
    out["n"] = len(rows)
    out["mean_words"] = round(sum(r["words"] for r in rows) / n, 2)
    return out


def load_band(cfg: Config) -> tuple[int, int]:
    """The band comes from the pairs that SURVIVE the token budget.

    Attempt #1 took p5/p95 of the pairs as built, before the 256-token filter
    removed the long tail. That over-states the top of the band: the filter
    censors above ~190 words (ADR-028), so the raw p95 describes lengths the
    training set no longer contains and the model was never shown. On attempt
    #2's pairs the raw band is 103-202 and the surviving band is 103-190 - and
    202 words cannot even be emitted inside the 245-token generation cap, so the
    raw upper edge would have been inert by construction, which is exactly the
    defect ADR-028 was written about.

    So: the census's surviving band wins when it exists. It is the band the model
    was taught and the band it can physically produce.
    """
    census = REPO_ROOT / cfg.results_dir / "sft_budget_census.json"
    if census.exists():
        c = json.loads(census.read_text(encoding="utf-8"))
        sw = c.get("surviving_words")
        if sw:
            return int(sw["p5"]), int(sw["p95"])
    path = REPO_ROOT / cfg.results_dir / "sft_pairs_summary.json"
    if not path.exists():
        raise SystemExit("run `python -m src.instruct pairs` first.")
    s = json.loads(path.read_text(encoding="utf-8"))
    return int(s["response_words_p5"]), int(s["response_words_p95"])


def cmd_demo(cfg: Config) -> None:
    """Sanity: the checker must score real corpus responses near-perfectly.

    If a genuine TinyStories response paired with its own prompt does not score
    high, the checker is broken before any model is judged by it. This is the
    upper reference point the model floors are read against.
    """
    band = load_band(cfg)
    pairs = json.loads((REPO_ROOT / cfg.data_dir / "sft_heldout.json")
                       .read_text(encoding="utf-8"))
    rows = [check_one(cfg, p["subject"], p["response"], band) for p in pairs]
    agg = aggregate(rows)
    control = shuffled_control(cfg, rows, [p["response"] for p in pairs], band)
    print(f"checker on {agg['n']} real corpus responses (the ceiling):")
    print(f"  length band                : {band[0]}-{band[1]} words")
    for s in ALL_SCORES:
        tag = "delta" if s in DELTA_SCORES else "floor"
        print(f"  {s:<26} {agg[s]:>8.1%}   ({tag})")
    print(f"  {'subject_mention (shuffled)':<26} {control:>8.1%}   (chance control)")
    print(f"  {'mean words':<26} {agg['mean_words']:>8.2f}")


def score_checkpoint(cfg: Config, checkpoint: str, label: str) -> dict:
    """Generate held-out responses from a checkpoint and score them.

    Decoding is read from `sft_gate_*`, which is pinned separately from the
    global decode defaults on purpose (ADR-025): a later phase changing the
    global default must not silently re-run this gate at settings it was never
    pre-registered against. The values used are recorded in the result.
    """
    import torch

    from src.sample import generate, load_checkpoint
    from src.tokenizer import load as load_tokenizer
    from utils.device import probe
    from utils.seed import set_seed

    set_seed(cfg.seed, strict=cfg.strict_determinism)
    band = load_band(cfg)
    prompts = json.loads((REPO_ROOT / cfg.data_dir / "sft_heldout.json")
                         .read_text(encoding="utf-8"))

    dev = probe()
    model, state = load_checkpoint(REPO_ROOT / checkpoint, dev.device)
    tok = load_tokenizer(cfg)
    eot = tok.token_to_id(cfg.doc_separator)

    rows, samples, responses = [], [], []
    for i, p in enumerate(prompts):
        # The model sees the instruction; the checker sees only what came after.
        wire = p["prompt"] + "\n"
        ids = tok.encode(wire).ids
        gen = torch.Generator(device="cpu").manual_seed(cfg.seed + i)
        out = generate(model, ids, cfg.sft_gate_new_tokens, cfg.sft_gate_temperature,
                       cfg.sft_gate_top_k or None, dev.device, eot_id=eot,
                       generator=gen)
        full = tok.decode(out)
        response = full[len(wire):] if full.startswith(wire) else full[len(p["prompt"]):]
        rows.append(check_one(cfg, p["subject"], response, band))
        responses.append(response)
        if len(samples) < 8:
            samples.append({"prompt": p["prompt"], "subject": p["subject"],
                            "response": response.strip()[:400]})
        if (i + 1) % 50 == 0:
            print(f"    {i + 1}/{len(prompts)}")

    agg = aggregate(rows)
    agg["subject_mention_shuffled"] = shuffled_control(cfg, rows, responses, band)
    agg["subject_mention_above_chance"] = round(
        agg["subject_mention"] - agg["subject_mention_shuffled"], 6)
    return {
        "label": label,
        "checkpoint": checkpoint,
        "step": state.get("step"),
        "decoding": {"temperature": cfg.sft_gate_temperature,
                     "top_k": cfg.sft_gate_top_k,
                     "new_tokens": cfg.sft_gate_new_tokens},
        "length_band": list(band),
        "scores": agg,
        "per_prompt": rows,
        "responses": responses,   # kept so controls can be recomputed without regenerating
        "samples": samples,
    }


def cmd_score(cfg: Config, checkpoint: str, label: str) -> None:
    print(f"scoring {label} ({checkpoint}) on held-out prompts at "
          f"T={cfg.sft_gate_temperature} k={cfg.sft_gate_top_k}...")
    result = score_checkpoint(cfg, checkpoint, label)
    agg = result["scores"]
    print(f"\n{label}: {agg['n']} held-out prompts, disjoint subjects")
    for s in ALL_SCORES:
        tag = "delta" if s in DELTA_SCORES else "floor"
        print(f"  {s:<26} {agg[s]:>8.1%}   ({tag})")
    print(f"  {'subject_mention (shuffled)':<26} {agg['subject_mention_shuffled']:>8.1%}"
          f"   (chance control)")
    print(f"  {'  -> above chance':<26} {agg['subject_mention_above_chance']:>8.1%}")
    print(f"  {'mean words':<26} {agg['mean_words']:>8.2f}")

    from utils.io import write_json
    out = write_json(REPO_ROOT / cfg.results_dir / f"sft_scores_{label}.json", result)
    print(f"wrote {out}")


# --------------------------------------------------------------------------- #
# the gate
# --------------------------------------------------------------------------- #

def cmd_oov(cfg: Config) -> None:
    """The oov_plausibility watch, on the responses rather than the gallery.

    Phase 4 scored `oov_rate` and `oov_plausibility` on 16 continuations of the
    Phase 4 gallery prompts and found zero out-of-vocabulary words, so the
    plausibility band never fired - it stayed dormant through the whole phase,
    validated only against fixtures.

    The 200 held-out instruction responses are the condition the band was built
    for: ~28k words of instruction-conditioned output, from a stage where pushing
    off-distribution is a real risk. And **which class of word fires matters more
    than whether the cell lights up**. Word-shaped inventions mean the band is
    doing its job. Garbage means the fine-tune bought adherence by breaking
    coherence, which is the same failure the is_story floor guards against,
    arriving through a different metric.
    """
    from src.coherence import CharModel, build_known_words, pooled_metrics, words_of

    ref_p = REPO_ROOT / cfg.results_dir / "coherence_reference.json"
    if not ref_p.exists():
        raise SystemExit("run `python -m src.coherence reference` first.")
    ref = json.loads(ref_p.read_text(encoding="utf-8"))
    known = build_known_words(cfg)
    char = CharModel(known)
    print(f"known-word set: {len(known):,} words\n")

    out = {"known_words": len(known), "bands": {
        m: ref["bands"].get(m) for m in ("oov_rate", "oov_plausibility")}}
    for label in ("base", "sft"):
        p = REPO_ROOT / cfg.results_dir / f"sft_scores_{label}.json"
        if not p.exists():
            continue
        texts = json.loads(p.read_text(encoding="utf-8"))["responses"]
        nw = sum(len(words_of(t)) for t in texts)
        pooled = pooled_metrics(texts, known, char)
        oov: dict[str, int] = {}
        for t in texts:
            for w in words_of(t):
                if w not in known:
                    oov[w] = oov.get(w, 0) + 1
        ranked = sorted(oov.items(), key=lambda kv: -kv[1])
        rate_band = ref["bands"]["oov_rate"]
        plaus_band = ref["bands"]["oov_plausibility"]
        rate_ok = rate_band["low"] <= pooled["oov_rate"] <= rate_band["high"]
        v = pooled["oov_plausibility"]

        print(f"{label}: {len(texts)} responses, {nw:,} words")
        print(f"  oov_rate         {pooled['oov_rate']:.6f}  "
              f"band [{rate_band['low']:.6f}, {rate_band['high']:.6f}]  "
              f"{'ok' if rate_ok else 'OUT OF BAND'}")
        if v != v:
            print(f"  oov_plausibility DORMANT - zero OOV words to score")
        else:
            n_oov = sum(oov.values())
            # A pooled mean over a handful of words cannot be compared to a band
            # bootstrapped from corpus groups holding far more (ADR-022). Saying
            # "out of band" on n=3 would be reading noise as a finding.
            thin = n_oov < 30
            ok = plaus_band["low"] <= v <= plaus_band["high"]
            print(f"  oov_plausibility {v:.4f}  "
                  f"band [{plaus_band['low']:.4f}, {plaus_band['high']:.4f}]  "
                  f"{'ok' if ok else 'OUT OF BAND'}"
                  f"{f'  (but n={n_oov} OOV words - too thin to read)' if thin else ''}")
        print(f"  {len(oov)} distinct OOV words"
              + (":" if oov else " - nothing invented"))
        for w, c in ranked[:20]:
            print(f"    {w:<20} x{c:<4} char-trigram logprob {char.logprob(w):>7.3f}")
        print()
        out[label] = {
            "responses": len(texts), "words": nw,
            "oov_rate": pooled["oov_rate"], "oov_rate_in_band": bool(rate_ok),
            "oov_plausibility": None if v != v else round(v, 6),
            "n_oov_tokens": sum(oov.values()), "n_oov_distinct": len(oov),
            "oov_words": [{"word": w, "count": c,
                           "char_logprob": round(char.logprob(w), 4)}
                          for w, c in ranked],
        }

    from utils.io import write_json
    write_json(REPO_ROOT / cfg.results_dir / "sft_oov.json", out)
    print(f"wrote {REPO_ROOT / cfg.results_dir / 'sft_oov.json'}")


def _se_diff(p: float, n: int) -> float:
    """SE of a paired difference, at the unpaired upper bound.

    The true SE depends on the discordant-pair count and is smaller, so using the
    unpaired bound makes every bar derived from it slightly conservative. Being
    conservative in the direction that makes the gate harder to pass is the only
    safe way to round.
    """
    import math
    return math.sqrt(2 * p * (1 - p) / max(n, 1))


def derive_bars(cfg: Config, base: dict) -> dict:
    """Recompute every bar from base's recorded scores and the fixed rules.

    Two rules, both pre-registered, neither fitted to sft.pt:

      delta scores : bar = base + the agreed delta
      floor scores : bar = base - z * SE, so a breach is a regression that can be
                     told apart from noise (ADR-029)

    Then assert the recomputed values match the frozen numbers in the config. A
    gate that reads a threshold off a summary can be handed a wrong summary; a
    gate that recomputes it and cross-checks cannot.
    """
    n = base["scores"]["n"]
    deltas = {"subject_mention": cfg.sft_gate_subject_mention_delta,
              "length_band": cfg.sft_gate_length_band_delta}
    frozen = {"subject_mention": cfg.sft_gate_subject_mention_min,
              "length_band": cfg.sft_gate_length_band_min,
              "is_story": cfg.sft_gate_is_story_min,
              "not_degenerate": cfg.sft_gate_not_degenerate_min}

    bars, drift = {}, []
    for s in ALL_SCORES:
        p = base["scores"][s]
        if s in DELTA_SCORES:
            bar, rule = p + deltas[s], f"base {p:.1%} + delta {deltas[s]:.1%}"
        else:
            allow = cfg.sft_gate_floor_z * _se_diff(p, n)
            bar = p - allow
            rule = (f"base {p:.1%} - {cfg.sft_gate_floor_z} x SE "
                    f"{_se_diff(p, n):.3%} = -{allow:.1%}")
        bars[s] = {"bar": round(bar, 6), "frozen": frozen[s], "base": p,
                   "rule": rule}
        if abs(bar - frozen[s]) > 0.001:
            drift.append(f"{s}: rule gives {bar:.4f}, config says {frozen[s]:.4f}")
    if drift:
        raise SystemExit("pre-registered bars do not match the rule that set "
                         "them:\n  " + "\n  ".join(drift))
    return bars


def cmd_gate(cfg: Config) -> None:
    base_p = REPO_ROOT / cfg.results_dir / "sft_scores_base.json"
    sft_p = REPO_ROOT / cfg.results_dir / "sft_scores_sft.json"
    for p in (base_p, sft_p):
        if not p.exists():
            raise SystemExit(f"missing {p.name} - run `python -m src.checker "
                             f"score --checkpoint ... --label ...` first.")
    base = json.loads(base_p.read_text(encoding="utf-8"))
    sft = json.loads(sft_p.read_text(encoding="utf-8"))

    # Comparing two stages under different decoding would measure the decoder,
    # not the fine-tune. Cheap to check, fatal if wrong, so it is checked.
    if base["decoding"] != sft["decoding"]:
        raise SystemExit(f"decoding differs:\n  base {base['decoding']}\n  "
                         f"sft  {sft['decoding']}")
    if base["length_band"] != sft["length_band"]:
        raise SystemExit("length band differs between the two scorings")

    bars = derive_bars(cfg, base)
    d = base["decoding"]
    print(f"Phase 5 gate  -  T={d['temperature']} k={d['top_k']} "
          f"new_tokens={d['new_tokens']}, {base['scores']['n']} held-out prompts, "
          f"band {base['length_band'][0]}-{base['length_band'][1]} words\n")

    print(f"  {'sub-score':<17} {'base':>7} {'sft':>7} {'delta':>8} "
          f"{'bar':>7}  verdict   rule")
    results = {}
    for s in ALL_SCORES:
        b, v = base["scores"][s], sft["scores"][s]
        bar = bars[s]["bar"]
        ok = v >= bar
        results[s] = {"base": b, "sft": v, "delta": round(v - b, 6),
                      "bar": bar, "pass": bool(ok),
                      "kind": "delta" if s in DELTA_SCORES else "floor",
                      "rule": bars[s]["rule"]}
        print(f"  {s:<17} {b:>6.1%} {v:>6.1%} {(v - b) * 100:>+6.1f}pt {bar:>6.1%}  "
              f"{'PASS' if ok else 'FAIL':<8}  {bars[s]['rule']}")

    # The validity check. This can invalidate a PASS, which is the point: if
    # subject-mention climbed because the model names more nouns rather than the
    # right one, the headline stops meaning adherence (ADR-026).
    shuf_b = base["scores"]["subject_mention_shuffled"]
    shuf_s = sft["scores"]["subject_mention_shuffled"]
    shuf_ok = shuf_s <= cfg.sft_gate_shuffled_max
    above_b = base["scores"]["subject_mention_above_chance"]
    above_s = sft["scores"]["subject_mention_above_chance"]
    print(f"\n  validity: shuffled-subject control")
    print(f"    shuffled rate      base {shuf_b:>6.1%}  sft {shuf_s:>6.1%}  "
          f"max {cfg.sft_gate_shuffled_max:.1%}  "
          f"{'OK' if shuf_ok else 'INVALIDATES THE PASS'}")
    print(f"    above chance       base {above_b:>6.1%}  sft {above_s:>6.1%}  "
          f"({(above_s - above_b) * 100:+.1f}pt)")

    deltas_pass = all(results[s]["pass"] for s in DELTA_SCORES)
    floors_pass = all(results[s]["pass"] for s in FLOOR_SCORES)
    green = deltas_pass and floors_pass and shuf_ok
    print(f"\n  deltas  {'PASS' if deltas_pass else 'FAIL'}    "
          f"floors  {'PASS' if floors_pass else 'FAIL'}    "
          f"validity  {'OK' if shuf_ok else 'BROKEN'}")
    print(f"\n  PHASE 5: {'GREEN' if green else 'RED'}")

    from utils.io import write_json
    write_json(REPO_ROOT / cfg.results_dir / "sft_gate.json", {
        "green": bool(green), "deltas_pass": bool(deltas_pass),
        "floors_pass": bool(floors_pass), "validity_ok": bool(shuf_ok),
        "decoding": d, "n": base["scores"]["n"],
        "length_band": base["length_band"],
        "scores": results,
        "shuffled": {"base": shuf_b, "sft": shuf_s,
                     "max": cfg.sft_gate_shuffled_max,
                     "above_chance_base": above_b, "above_chance_sft": above_s},
        "mean_words": {"base": base["scores"]["mean_words"],
                       "sft": sft["scores"]["mean_words"]},
        "config_hash": cfg.hash(), "sft_stage_hash": cfg.stage_hash("sft"),
    })
    print(f"wrote {REPO_ROOT / cfg.results_dir / 'sft_gate.json'}")
    _write_gate_report(cfg, base, sft, results, bars, green, deltas_pass,
                       floors_pass, shuf_ok)
    if not green:
        raise SystemExit(1)


def _write_gate_report(cfg: Config, base: dict, sft: dict, results: dict,
                       bars: dict, green: bool, deltas_pass: bool,
                       floors_pass: bool, shuf_ok: bool) -> None:
    from utils.io import write_text

    d = base["decoding"]
    lo, hi = base["length_band"]
    lines = [
        f"# Phase 5 gate: {'GREEN' if green else 'RED'}",
        "",
        f"`base.pt` versus `sft.pt` on {base['scores']['n']} held-out prompts over "
        "**disjoint subjects**, at identical pinned decoding "
        f"(T={d['temperature']}, top-k={d['top_k']}, {d['new_tokens']} new tokens), "
        "scored by the same deterministic checker.",
        "",
        "Every bar below is recomputed by `src/checker.py gate` from base's "
        "recorded scores and cross-checked against the frozen config value. "
        "Deltas were agreed before `src/sft.py` existed (ADR-027); the floors were "
        "set by rule before `sft.pt` was scored (ADR-029).",
        "",
        "| sub-score | kind | base | sft | delta | bar | verdict | how the bar was set |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for s in ALL_SCORES:
        r = results[s]
        lines.append(
            f"| `{s}` | {r['kind']} | {r['base']:.1%} | {r['sft']:.1%} | "
            f"{r['delta'] * 100:+.1f} pt | {r['bar']:.1%} | "
            f"**{'PASS' if r['pass'] else 'FAIL'}** | {r['rule']} |")

    sb = base["scores"]["subject_mention_shuffled"]
    ss = sft["scores"]["subject_mention_shuffled"]
    lines += [
        "",
        f"Mean response length: base {base['scores']['mean_words']:.2f} words, "
        f"sft {sft['scores']['mean_words']:.2f}. Band {lo}-{hi} words.",
        "",
        "## Validity: the shuffled-subject control",
        "",
        "This can **invalidate a pass**, not only fail one. If subject-mention "
        "climbs while the shuffled rate climbs with it, the model has learned to "
        "name more nouns rather than the right one, and the headline stops meaning "
        "adherence (ADR-026).",
        "",
        f"| | base | sft | limit |",
        "|---|---|---|---|",
        f"| shuffled subject-mention | {sb:.1%} | {ss:.1%} | "
        f"{cfg.sft_gate_shuffled_max:.1%} |",
        f"| matched, above chance | "
        f"{base['scores']['subject_mention_above_chance']:.1%} | "
        f"{sft['scores']['subject_mention_above_chance']:.1%} | - |",
        "",
        f"Verdict: **{'valid' if shuf_ok else 'INVALID'}**. Subject-mention rose "
        f"{results['subject_mention']['delta'] * 100:+.1f} points while the "
        f"shuffled rate moved {(ss - sb) * 100:+.1f}, so the gain is adherence "
        "rather than noun-spraying.",
        "",
        "## Outcome",
        "",
        f"- deltas: **{'PASS' if deltas_pass else 'FAIL'}**",
        f"- floors: **{'PASS' if floors_pass else 'FAIL'}**",
        f"- validity: **{'OK' if shuf_ok else 'BROKEN'}**",
        "",
        f"**PHASE 5: {'GREEN' if green else 'RED'}**",
        "",
        "Generated by `python -m src.checker gate`.",
    ]
    write_text(REPO_ROOT / cfg.results_dir / "sft_gate.md", "\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser(description="Deterministic feature-checker (Phase 5).")
    ap.add_argument("command", choices=("demo", "score", "gate", "oov"))
    ap.add_argument("--config", default=None)
    ap.add_argument("--checkpoint", default="checkpoints/base.pt")
    ap.add_argument("--label", default="base")
    args = ap.parse_args()
    cfg = Config.load(args.config)
    if args.command == "demo":
        cmd_demo(cfg)
    elif args.command == "gate":
        cmd_gate(cfg)
    elif args.command == "oov":
        cmd_oov(cfg)
    else:
        cmd_score(cfg, args.checkpoint, args.label)


if __name__ == "__main__":
    main()
