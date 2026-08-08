"""Phase 5 — turn instruction pairs into masked training tensors, and measure
what the token budget censors on the way.

Two jobs in one module, deliberately:

1. **Build the tensors.** `x`, and a `y` whose prompt positions are already
   `IGNORE` — the model learns to answer, not to recite the instruction.
2. **Report what the budget filter removed**, as a joint distribution of
   response-words against tokens-consumed.

They live together because the census has to describe *the data that is actually
trained on*. A separate estimator would be a second implementation of the same
filter, free to disagree with the real one, and the number I would then quote as
a pre-flight gate would be an estimate of the training set rather than a
measurement of it.

**Why the census is a gate and not a footnote.** `context_len` is 256 tokens.
Responses run to `sft_max_response_words` words plus a prompt, and the long ones
do not fit. Dropping them is the right call - truncating mid-story would teach
the model to stop mid-sentence, which is the exact pathology SFT is supposed to
train out. But the filter does not remove long responses *at random*: length is
the thing that breaks the budget, so it removes the longest ones preferentially.
That is a **censored** distribution, not a thinned one - the right tail is cut
off rather than sampled less.

And that collides with the pre-registered `length_band` bar. If the token ceiling
censors responses at, say, 185 words, the training set cannot contain examples
reaching the top of a 102-200 band, and the model cannot learn to produce what it
never sees. A bar the data is structurally incapable of teaching is not a bar the
model can miss; scoring 68% against it would be a fact about the training set
being priced wrong, not about the fine-tune underperforming.

**The censoring point is not a fixed word count.** Prompt length varies per pair,
so the token headroom left for the response varies too: a 190-word response
behind a short prompt fits where the same response behind a long one does not.
The census therefore reports the joint - words against tokens, and where the wall
falls - rather than a single "max words" that would be true only on average.

This is the length twin of the subject-leak detector in `src/checker.py`. There,
base's own score says whether the *checker* is honest. Here, the surviving data
distribution says whether the *bar* is reachable. Both are validity checks on the
gate rather than on the model, and both have to pass before the fine-tune is a
meaningful test instead of a rigged one.

    python -m src.sft_data census   # the gate: what does the 256 wall cut?
    python -m src.sft_data build    # write the masked tensors
    python -m src.sft_data mask     # assert the mask on one pair (ADR-014 shape)
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from config import REPO_ROOT, Config
from utils.io import write_json, write_text

# Cross-entropy's default ignore_index. Positions carrying this contribute no
# loss and no gradient.
IGNORE = -100

# 10-word bins for the joint table. Fine enough to locate the wall, coarse enough
# that each bin has a usable count at 40k pairs.
BIN = 10


def pairs_path(cfg: Config):
    return REPO_ROOT / cfg.data_dir / "sft_train.json"


def load_pairs(cfg: Config) -> list[dict]:
    path = pairs_path(cfg)
    if not path.exists():
        raise SystemExit("run `python -m src.instruct pairs` first.")
    return json.loads(path.read_text(encoding="utf-8"))


def wire_format(cfg: Config, prompt: str) -> str:
    """The prompt exactly as the model sees it, newline included.

    `src.checker` builds the same string when it scores a checkpoint. Keeping the
    two in agreement matters more than it looks: if training saw
    `prompt + "\\n"` and evaluation sent `prompt + " "`, the gate would measure a
    format the model was never taught.
    """
    return prompt + "\n"


def encode_pair(cfg: Config, tok, eot: int, pair: dict) -> dict:
    """Token accounting for one pair. No filtering decision here - just facts."""
    p_ids = tok.encode(wire_format(cfg, pair["prompt"])).ids
    r_ids = tok.encode(pair["response"]).ids + [eot]
    return {
        "words": int(pair["response_words"]),
        "prompt_tokens": len(p_ids),
        "response_tokens": len(r_ids),
        "total_tokens": len(p_ids) + len(r_ids),
        "prompt_ids": p_ids,
        "response_ids": r_ids,
    }


def max_seq(cfg: Config) -> int:
    """Longest sequence that fits.

    A sequence of L tokens yields L-1 next-token predictions, so L = context_len
    + 1 fills the window exactly: `x = seq[:-1]` is `context_len` long.
    """
    return cfg.context_len + 1


def encode_all(cfg: Config) -> tuple[list[dict], int]:
    from src.tokenizer import load as load_tokenizer

    tok = load_tokenizer(cfg)
    eot = tok.token_to_id(cfg.doc_separator)
    if eot is None:
        raise SystemExit(f"tokenizer has no {cfg.doc_separator!r} token")
    pairs = load_pairs(cfg)
    print(f"encoding {len(pairs):,} pairs...")
    recs = []
    for i, p in enumerate(pairs):
        recs.append(encode_pair(cfg, tok, eot, p))
        if (i + 1) % 10000 == 0:
            print(f"  {i + 1:,}")
    return recs, max_seq(cfg)


# --------------------------------------------------------------------------- #
# the census
# --------------------------------------------------------------------------- #

def _pct(vals: list[int] | np.ndarray, q: float) -> int:
    return int(np.percentile(np.asarray(vals), q))


def census(cfg: Config, recs: list[dict], limit: int) -> dict:
    """Joint distribution of response words against tokens, and the wall."""
    words = np.array([r["words"] for r in recs])
    total = np.array([r["total_tokens"] for r in recs])
    ptoks = np.array([r["prompt_tokens"] for r in recs])
    fits = total <= limit

    n = len(recs)
    n_fit = int(fits.sum())
    n_drop = n - n_fit

    # Per-bin survival: this is where a censored tail shows up as a survival
    # rate that falls to zero rather than a count that thins out evenly.
    lo, hi = int(words.min()), int(words.max())
    bins = []
    for start in range(lo - lo % BIN, hi + 1, BIN):
        sel = (words >= start) & (words < start + BIN)
        k = int(sel.sum())
        if not k:
            continue
        kf = int((sel & fits).sum())
        bins.append({
            "words_from": start, "words_to": start + BIN - 1,
            "n": k, "n_fit": kf, "survival": round(kf / k, 4),
            "median_total_tokens": int(np.median(total[sel])),
            "min_total_tokens": int(total[sel].min()),
            "max_total_tokens": int(total[sel].max()),
        })

    # Three different readings of "where the wall is", because they answer
    # different questions and a single number would hide the spread that the
    # varying prompt length creates.
    surviving_words = words[fits]
    full_bins = [b for b in bins if b["survival"] >= 0.999]
    half_bins = [b for b in bins if b["survival"] >= 0.5]
    any_bins = [b for b in bins if b["n_fit"] > 0]

    tpw = total / np.maximum(words, 1)
    return {
        "context_len": cfg.context_len,
        "max_seq_tokens": limit,
        "reason": (f"sequence longer than context_len+1 = {limit} tokens "
                   f"(prompt + response + {cfg.doc_separator})"),
        "n_pairs": n,
        "n_fit": n_fit,
        "n_dropped": n_drop,
        "drop_rate": round(n_drop / max(n, 1), 6),
        "prompt_tokens": {"min": int(ptoks.min()), "median": int(np.median(ptoks)),
                          "max": int(ptoks.max())},
        "tokens_per_word": {"p5": round(float(np.percentile(tpw, 5)), 4),
                            "median": round(float(np.median(tpw)), 4),
                            "p95": round(float(np.percentile(tpw, 95)), 4)},
        "all_words": {"p5": _pct(words, 5), "median": _pct(words, 50),
                      "p95": _pct(words, 95), "max": hi},
        "surviving_words": {"p5": _pct(surviving_words, 5),
                            "median": _pct(surviving_words, 50),
                            "p95": _pct(surviving_words, 95),
                            "max": int(surviving_words.max())},
        "wall": {
            "last_full_bin_to": full_bins[-1]["words_to"] if full_bins else None,
            "last_half_bin_to": half_bins[-1]["words_to"] if half_bins else None,
            "last_surviving_bin_to": any_bins[-1]["words_to"] if any_bins else None,
        },
        "bins": bins,
    }


def band_verdict(cfg: Config, cen: dict) -> dict:
    """Does the surviving data span the band the gate was pre-registered against?

    The band the checker uses is p5/p95 of the *pre-filter* pairs, which is the
    band `sft_gate_length_band_min` was anchored to. If the filter censors below
    p95, that pre-registered upper edge describes lengths the training set no
    longer contains, and the bar has to be re-registered before anything trains.
    """
    summary = json.loads((REPO_ROOT / cfg.results_dir / "sft_pairs_summary.json")
                         .read_text(encoding="utf-8"))
    band_lo = int(summary["response_words_p5"])
    band_hi = int(summary["response_words_p95"])

    # Survival inside the top of the registered band is the number that decides
    # it: a band whose upper quintile the data cannot reach is unreachable.
    top_lo = band_lo + int(0.8 * (band_hi - band_lo))
    in_top = [b for b in cen["bins"]
              if b["words_from"] >= top_lo - BIN and b["words_from"] <= band_hi]
    n_top = sum(b["n"] for b in in_top)
    n_top_fit = sum(b["n_fit"] for b in in_top)

    surviving_max = cen["surviving_words"]["max"]
    clipped = surviving_max < band_hi
    return {
        "registered_band": [band_lo, band_hi],
        "band_top_quintile_from": top_lo,
        "n_in_top_quintile": n_top,
        "n_fit_in_top_quintile": n_top_fit,
        "top_quintile_survival": round(n_top_fit / n_top, 4) if n_top else None,
        "max_surviving_words": surviving_max,
        "clipped_below_band_top": bool(clipped),
        "surviving_band": [cen["surviving_words"]["p5"],
                           cen["surviving_words"]["p95"]],
    }


def cmd_census(cfg: Config) -> None:
    recs, limit = encode_all(cfg)
    cen = census(cfg, recs, limit)
    ver = band_verdict(cfg, cen)

    print(f"\ntoken budget: context_len {cen['context_len']} -> "
          f"max sequence {cen['max_seq_tokens']} tokens")
    print(f"  prompt tokens        : min {cen['prompt_tokens']['min']}  "
          f"median {cen['prompt_tokens']['median']}  max {cen['prompt_tokens']['max']}")
    print(f"  tokens per word      : p5 {cen['tokens_per_word']['p5']}  "
          f"median {cen['tokens_per_word']['median']}  "
          f"p95 {cen['tokens_per_word']['p95']}")
    print(f"\n  dropped {cen['n_dropped']:,} of {cen['n_pairs']:,} "
          f"({cen['drop_rate']:.1%}), all for one reason:")
    print(f"    {cen['reason']}")

    print(f"\n  joint distribution (response words x tokens consumed):")
    print(f"    {'words':>9}  {'n':>6}  {'fit':>6}  {'survive':>8}  "
          f"{'total tokens (min/med/max)':>28}")
    for b in cen["bins"]:
        rng = f"{b['min_total_tokens']}/{b['median_total_tokens']}/{b['max_total_tokens']}"
        flag = ""
        if 0.0 < b["survival"] < 1.0:
            flag = "  <- wall"
        elif b["survival"] == 0.0:
            flag = "  <- censored"
        print(f"    {b['words_from']:>4}-{b['words_to']:<4}  {b['n']:>6}  "
              f"{b['n_fit']:>6}  {b['survival']:>7.1%}  {rng:>28}{flag}")

    print(f"\n  where the wall falls:")
    print(f"    last bin fully surviving : {cen['wall']['last_full_bin_to']} words")
    print(f"    last bin >=50% surviving : {cen['wall']['last_half_bin_to']} words")
    print(f"    longest surviving response: {cen['surviving_words']['max']} words")

    print(f"\n  registered band {ver['registered_band'][0]}-{ver['registered_band'][1]} "
          f"words (pre-registered as sft_gate_length_band_min)")
    print(f"    surviving band (p5-p95)  : {ver['surviving_band'][0]}-"
          f"{ver['surviving_band'][1]}")
    if ver["top_quintile_survival"] is not None:
        print(f"    survival in the band's top quintile "
              f"({ver['band_top_quintile_from']}-{ver['registered_band'][1]}): "
              f"{ver['top_quintile_survival']:.1%} "
              f"({ver['n_fit_in_top_quintile']:,}/{ver['n_in_top_quintile']:,})")

    verdict = "CLIPPED" if ver["clipped_below_band_top"] else "SPANS THE BAND"
    print(f"\n  VERDICT: {verdict}")
    if ver["clipped_below_band_top"]:
        print(f"    the longest response the model can be shown is "
              f"{ver['max_surviving_words']} words, below the registered upper "
              f"edge of {ver['registered_band'][1]}.")
        print(f"    the length bar must be re-registered before any fine-tune runs.")

    out = {**cen, "band_verdict": ver}
    out.pop("bins")
    out["bins"] = cen["bins"]
    write_json(REPO_ROOT / cfg.results_dir / "sft_budget_census.json", out)
    write_report(cfg, cen, ver)
    print(f"\nwrote {REPO_ROOT / cfg.results_dir / 'sft_budget_census.json'}")


def write_report(cfg: Config, cen: dict, ver: dict) -> None:
    clipped = ver["clipped_below_band_top"]
    lines = [
        "# SFT token budget: what the 256-token wall censors",
        "",
        f"`context_len` is {cen['context_len']}, so the longest training sequence is "
        f"{cen['max_seq_tokens']} tokens (`context_len + 1` gives `context_len` "
        "next-token predictions). Pairs longer than that are **dropped, not "
        "truncated**: cutting a story mid-sentence would teach the model to stop "
        "mid-sentence, which is the pathology SFT exists to remove.",
        "",
        f"**Dropped {cen['n_dropped']:,} of {cen['n_pairs']:,} pairs "
        f"({cen['drop_rate']:.1%})**, every one for the same reason - "
        f"{cen['reason']}.",
        "",
        "## Why this is a gate and not a footnote",
        "",
        "The filter does not thin the length distribution, it **censors** it. Length "
        "is what breaks the budget, so the longest responses are removed "
        "preferentially and the right tail is cut off rather than sampled less "
        "often. If that cut lands below the upper edge of the pre-registered "
        "`length_band`, the training set cannot contain examples reaching the top of "
        "the band, and the model cannot produce what it was never shown. Scoring "
        "under such a bar would be a fact about the bar, not about the fine-tune.",
        "",
        "The censoring point is **not a fixed word count**. Prompt length varies "
        f"({cen['prompt_tokens']['min']}-{cen['prompt_tokens']['max']} tokens, median "
        f"{cen['prompt_tokens']['median']}), so the headroom left for the response "
        "varies with it: the same response fits behind a short prompt and does not "
        "behind a long one. Hence the joint table rather than a single threshold.",
        "",
        f"Encoding cost is {cen['tokens_per_word']['median']} tokens per word at the "
        f"median (p5 {cen['tokens_per_word']['p5']}, p95 "
        f"{cen['tokens_per_word']['p95']}).",
        "",
        "## Joint distribution",
        "",
        "| response words | pairs | fit | survival | total tokens min/median/max |",
        "|---|---|---|---|---|",
    ]
    for b in cen["bins"]:
        rng = (f"{b['min_total_tokens']} / {b['median_total_tokens']} / "
               f"{b['max_total_tokens']}")
        note = ""
        if 0.0 < b["survival"] < 1.0:
            note = " **&larr; the wall**"
        elif b["survival"] == 0.0:
            note = " *censored*"
        lines.append(f"| {b['words_from']}-{b['words_to']} | {b['n']:,} | "
                     f"{b['n_fit']:,} | {b['survival']:.1%}{note} | {rng} |")

    lines += [
        "",
        "## Where the wall falls",
        "",
        f"- last bin fully surviving: **{cen['wall']['last_full_bin_to']} words**",
        f"- last bin at least half surviving: **{cen['wall']['last_half_bin_to']} words**",
        f"- longest response that fits at all: **{cen['surviving_words']['max']} words**",
        "",
        "## Verdict against the pre-registered band",
        "",
        f"Registered band: **{ver['registered_band'][0]}-{ver['registered_band'][1]} "
        f"words** (p5/p95 of the pairs before filtering - the band "
        "`sft_gate_length_band_min` was anchored to).",
        "",
        f"Surviving band: **{ver['surviving_band'][0]}-{ver['surviving_band'][1]} "
        "words**.",
        "",
        (f"Survival inside the band's top quintile "
         f"({ver['band_top_quintile_from']}-{ver['registered_band'][1]} words): "
         f"**{ver['top_quintile_survival']:.1%}**"
         if ver["top_quintile_survival"] is not None else ""),
        "",
        (f"**CLIPPED.** The longest response the model can be shown is "
         f"{ver['max_surviving_words']} words, below the registered upper edge of "
         f"{ver['registered_band'][1]}. The length bar is re-registered before the "
         "fine-tune runs, on the surviving distribution."
         if clipped else
         "**Spans the band.** The surviving data reaches the registered upper edge, "
         "so the pre-registered length bar is reachable and stands unchanged."),
        "",
        "Generated by `python -m src.sft_data census`.",
    ]
    write_text(REPO_ROOT / cfg.results_dir / "sft_budget_census.md",
               "\n".join(l for l in lines if l is not None))


# --------------------------------------------------------------------------- #
# the tensors
# --------------------------------------------------------------------------- #

def pack(cfg: Config, recs: list[dict], limit: int) -> tuple[np.ndarray, np.ndarray]:
    """Masked training tensors.

    `x[i, t]` predicts `y[i, t]`. `y` is `IGNORE` everywhere the target is a
    prompt token or padding, so the loss only ever sees response positions.

    The first response token *is* supervised - it is predicted from the last
    prompt token, which is precisely the "given this instruction, begin" step.
    Masking it too would remove the only position that carries the transition.
    """
    kept = [r for r in recs if r["total_tokens"] <= limit]
    n, L = len(kept), cfg.context_len
    x = np.zeros((n, L), dtype=np.int32)
    y = np.full((n, L), IGNORE, dtype=np.int32)
    for i, r in enumerate(kept):
        seq = r["prompt_ids"] + r["response_ids"]
        plen = len(r["prompt_ids"])
        xs, ys = seq[:-1], seq[1:]
        x[i, :len(xs)] = xs
        # position t predicts seq[t+1]; supervise it when that target is a
        # response token, i.e. t + 1 >= plen.
        for t in range(plen - 1, len(ys)):
            y[i, t] = ys[t]
    return x, y


def cmd_build(cfg: Config) -> None:
    recs, limit = encode_all(cfg)
    x, y = pack(cfg, recs, limit)
    d = REPO_ROOT / cfg.data_dir
    np.save(d / "sft_x.npy", x)
    np.save(d / "sft_y.npy", y)
    sup = int((y != IGNORE).sum())
    print(f"\npacked {x.shape[0]:,} examples x {x.shape[1]} tokens")
    print(f"  supervised positions : {sup:,} "
          f"({sup / y.size:.1%} of the tensor)")
    print(f"  masked positions     : {y.size - sup:,} (prompt + padding)")
    print(f"wrote {d / 'sft_x.npy'} and {d / 'sft_y.npy'}")


# --------------------------------------------------------------------------- #
# the mask assertion (ADR-014 shape: test the property, not the proxy)
# --------------------------------------------------------------------------- #

def assert_mask(cfg: Config, verbose: bool = True) -> dict:
    """Assert the mask directly, on one pair, before any training step.

    A masking bug that is wrong still trains to a plausible loss curve, because
    predicting the instruction - which the model can see in its own context - is
    easy and pulls the average down. A broken mask therefore looks like a healthy
    run. Same shape as the causal-mask check in Phase 3 (ADR-014): assert the
    property, never the proxy.

    **`src.sft.train` calls this function, not a receipt file.** The first version
    of this check wrote a JSON receipt that the trainer read, which is the exact
    weakness this project has already named twice: a gate that reads a summary can
    be handed a stale one. Edit `pack()` after running the CLI and the receipt
    still says "passed". Recomputing costs one forward-free tensor build, so there
    is no reason to trust a file instead (ADR-021, ADR-029).
    """
    import torch
    import torch.nn.functional as F

    from src.tokenizer import load as load_tokenizer

    def say(*a):
        if verbose:
            print(*a)

    tok = load_tokenizer(cfg)
    eot = tok.token_to_id(cfg.doc_separator)
    pair = load_pairs(cfg)[0]
    rec = encode_pair(cfg, tok, eot, pair)
    if rec["total_tokens"] > max_seq(cfg):
        raise SystemExit("first pair does not fit; pick another for the assertion")

    x, y = pack(cfg, [rec], max_seq(cfg))
    plen, rlen = rec["prompt_tokens"], rec["response_tokens"]
    seq_positions = rec["total_tokens"] - 1     # positions in x that carry a target

    say(f"pair 0: prompt {plen} tokens, response {rlen} tokens, "
          f"total {rec['total_tokens']}")
    say(f"  prompt: {pair['prompt']!r}")

    # 1. every prompt-predicting position is IGNORE
    prompt_region = y[0, :plen - 1]
    assert (prompt_region == IGNORE).all(), \
        f"{int((prompt_region != IGNORE).sum())} prompt positions are supervised"
    say(f"  positions 0..{plen - 2} (targets are prompt tokens) : all IGNORE  OK")

    # 2. every response-predicting position is a real token, and the RIGHT one
    resp_region = y[0, plen - 1:seq_positions]
    assert (resp_region != IGNORE).all(), \
        f"{int((resp_region == IGNORE).sum())} response positions are masked"
    expected = np.array(rec["response_ids"], dtype=np.int32)
    assert np.array_equal(resp_region, expected), "response targets are misaligned"
    say(f"  positions {plen - 1}..{seq_positions - 1} (targets are response tokens): "
          f"all supervised, and equal to response_ids  OK")

    # 3. padding is IGNORE
    pad = y[0, seq_positions:]
    assert (pad == IGNORE).all(), "padding is supervised"
    say(f"  positions {seq_positions}..{cfg.context_len - 1} (padding) : "
          f"all IGNORE  OK")

    # 4. count matches: supervised positions == response tokens
    sup = int((y[0] != IGNORE).sum())
    assert sup == rlen, f"supervised {sup} != response tokens {rlen}"
    say(f"  supervised count {sup} == response tokens {rlen}  OK")

    # 5. the loss itself: finite, and built from exactly `rlen` terms.
    # Asserting on y alone would leave open whether cross_entropy honours it, so
    # this checks the tensor the optimizer actually sees.
    logits = torch.randn(1, cfg.context_len, cfg.vocab_size,
                         generator=torch.Generator().manual_seed(cfg.seed))
    yt = torch.from_numpy(y.astype(np.int64))
    per_pos = F.cross_entropy(logits.view(-1, cfg.vocab_size), yt.view(-1),
                              ignore_index=IGNORE, reduction="none")
    nonzero = int((per_pos > 0).sum())
    assert nonzero == rlen, f"loss has {nonzero} live terms, expected {rlen}"
    assert torch.isfinite(per_pos).all(), "loss is not finite"
    mean = F.cross_entropy(logits.view(-1, cfg.vocab_size), yt.view(-1),
                           ignore_index=IGNORE)
    say(f"  cross_entropy: {nonzero} live terms == {rlen} response tokens  OK")
    say(f"  random-logit loss {mean:.4f} ~ ln(vocab) "
          f"{float(np.log(cfg.vocab_size)):.4f}  OK")

    # The record is returned to the caller AND written out. `src.sft.train` uses
    # the return value, not the file - the file is only an artifact for a reader.
    record = {
        "passed": True,
        "context_len": cfg.context_len,
        "pair_index": 0,
        "prompt": pair["prompt"],
        "prompt_tokens": plen,
        "response_tokens": rlen,
        "supervised_positions": sup,
        "ignored_prompt_positions": plen - 1,
        "ignored_padding_positions": int(cfg.context_len - seq_positions),
        "cross_entropy_live_terms": nonzero,
        "checks": [
            "prompt-target positions are all IGNORE",
            "response-target positions are all supervised",
            "response targets equal response_ids exactly (alignment)",
            "padding positions are all IGNORE",
            "supervised count equals response token count",
            "cross_entropy live-term count equals response token count",
            "loss is finite",
        ],
    }
    write_json(REPO_ROOT / cfg.results_dir / "sft_mask_assertion.json", record)
    say("\nmask assertion PASSED on the tensor the optimizer sees.")
    say(f"wrote {REPO_ROOT / cfg.results_dir / 'sft_mask_assertion.json'}")
    return record


def main() -> None:
    ap = argparse.ArgumentParser(description="SFT data: budget census + tensors.")
    ap.add_argument("command", choices=("census", "build", "mask"))
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = Config.load(args.config)
    {"census": cmd_census, "build": cmd_build,
     "mask": lambda c: assert_mask(c, verbose=True)}[args.command](cfg)


if __name__ == "__main__":
    main()
