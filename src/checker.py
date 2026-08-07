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


def main() -> None:
    ap = argparse.ArgumentParser(description="Deterministic feature-checker (Phase 5).")
    ap.add_argument("command", choices=("demo", "score"))
    ap.add_argument("--config", default=None)
    ap.add_argument("--checkpoint", default="checkpoints/base.pt")
    ap.add_argument("--label", default="base")
    args = ap.parse_args()
    cfg = Config.load(args.config)
    if args.command == "demo":
        cmd_demo(cfg)
    else:
        cmd_score(cfg, args.checkpoint, args.label)


if __name__ == "__main__":
    main()
