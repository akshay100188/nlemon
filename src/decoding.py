"""What temperature and top-k actually do, measured before SFT needs the answer.

Sampling knobs change what the model *says* without changing anything it knows.
That becomes dangerous at Phase 5: if `sft.pt` is compared to `base.pt` under
different decoding, the measurement is of the decoder, not the fine-tune — and
"the model got worse" and "my decoding is wrong" produce the same-looking output.

So this sweeps the grid once against the corpus bands from
`results/coherence_reference.json` and pins one pair. Every cross-stage
comparison from Phase 5 on uses that pair unchanged.

Read the two failure directions in the table:

* **too cold** — the distribution collapses onto its mode. Repetition climbs, the
  type/token ratio falls, and the model loops. Fluent, and saying nothing.
* **too hot** — the tail gets sampled. Known-word rate falls as the model invents
  character sequences, and sentences stop resolving.

The corpus band brackets both, which is why the reference is two-sided.

    python -m src.decoding sweep
"""

from __future__ import annotations

import argparse
import json
import time

import torch

from config import REPO_ROOT, Config
from src.coherence import METRICS, build_known_words, score_samples
from src.sample import GALLERY_PROMPTS, generate, load_checkpoint
from src.tokenizer import load as load_tokenizer
from utils.device import probe
from utils.io import write_json, write_text
from utils.seed import set_seed


def generate_cell(cfg: Config, model, tok, device, temperature: float,
                  top_k: int) -> list[str]:
    """Fixed prompts and fixed seeds, so cells differ only by the knobs."""
    eot = tok.token_to_id(cfg.doc_separator)
    texts = []
    for i in range(cfg.decode_sweep_samples):
        prompt = GALLERY_PROMPTS[i % len(GALLERY_PROMPTS)]
        gen = torch.Generator(device="cpu").manual_seed(cfg.seed + i)
        ids = generate(model, tok.encode(prompt).ids, cfg.decode_sweep_new_tokens,
                       temperature, top_k or None, device, eot_id=eot, generator=gen)
        texts.append(tok.decode(ids))
    return texts


def cmd_sweep(cfg: Config, checkpoint: str) -> None:
    ref_path = REPO_ROOT / cfg.results_dir / "coherence_reference.json"
    if not ref_path.exists():
        raise SystemExit("run `python -m src.coherence reference` first.")
    reference = json.loads(ref_path.read_text(encoding="utf-8"))

    # A sweep that never measures the value it is justifying justifies nothing.
    # The first run of this reported the pinned pair as "OUT OF BAND" purely
    # because 0.8 was not in the grid and a None row read as falsy — a report
    # stating something false, which is the ADR-021 failure exactly.
    if cfg.coherence_temperature not in cfg.decode_sweep_temperatures:
        raise SystemExit(
            f"pinned temperature {cfg.coherence_temperature} is not in "
            f"decode_sweep_temperatures {list(cfg.decode_sweep_temperatures)} - "
            f"add it, or the sweep cannot speak about the pinned pair."
        )
    if cfg.coherence_top_k not in cfg.decode_sweep_top_k:
        raise SystemExit(
            f"pinned top_k {cfg.coherence_top_k} is not in "
            f"decode_sweep_top_k {list(cfg.decode_sweep_top_k)}."
        )

    dev = probe()
    model, state = load_checkpoint(REPO_ROOT / checkpoint, dev.device)
    tok = load_tokenizer(cfg)
    known = build_known_words(cfg)

    rows = []
    total = len(cfg.decode_sweep_temperatures) * len(cfg.decode_sweep_top_k)
    print(f"sweeping {total} cells x {cfg.decode_sweep_samples} samples "
          f"({checkpoint} @ step {state.get('step')})\n")
    started = time.time()
    for temperature in cfg.decode_sweep_temperatures:
        for top_k in cfg.decode_sweep_top_k:
            texts = generate_cell(cfg, model, tok, dev.device, temperature, top_k)
            scored = score_samples(texts, known, reference)
            row = {
                "temperature": temperature,
                "top_k": top_k,
                "in_band": scored["passed"],
                "metrics": {m: scored["checks"][m]["sample_median"] for m in METRICS},
                "out_of_band": [m for m in METRICS if not scored["checks"][m]["in_band"]],
            }
            rows.append(row)
            flag = "in band" if row["in_band"] else "OUT: " + ",".join(row["out_of_band"])
            print(f"  T={temperature:<4} k={top_k or 'off':<4} "
                  f"rep4={row['metrics']['repeated_4gram_rate']:.4f} "
                  f"ttr={row['metrics']['type_token_ratio']:.4f} "
                  f"known={row['metrics']['known_word_rate']:.4f}  {flag}")

    pinned = {"temperature": cfg.coherence_temperature, "top_k": cfg.coherence_top_k}
    pinned_row = next((r for r in rows
                       if r["temperature"] == pinned["temperature"]
                       and r["top_k"] == pinned["top_k"]), None)
    in_band = [r for r in rows if r["in_band"]]

    bands = reference["bands"]
    lines = [
        "# What temperature and top-k actually do",
        "",
        f"`{checkpoint}` at step {state.get('step')}, "
        f"{cfg.decode_sweep_samples} samples per cell, "
        f"{cfg.decode_sweep_new_tokens} new tokens, fixed prompts and seeds so cells "
        "differ only by the knobs. Each cell is scored against the corpus bands from "
        "[coherence_reference.md](coherence_reference.md).",
        "",
        "Sampling changes what the model says, not what it knows. The point of measuring "
        "it here is Phase 5: comparing `sft.pt` to `base.pt` under different decoding "
        "would measure the decoder, not the fine-tune.",
        "",
        "| T | top-k | repeated 4-gram | type/token | known-word | mean sent. | verdict |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        m = r["metrics"]
        verdict = "in band" if r["in_band"] else "**out**: " + ", ".join(r["out_of_band"])
        star = " **<- pinned**" if r is pinned_row else ""
        lines.append(
            f"| {r['temperature']} | {r['top_k'] or 'off'} | "
            f"{m['repeated_4gram_rate']:.4f} | {m['type_token_ratio']:.4f} | "
            f"{m['known_word_rate']:.4f} | {m['mean_sentence_words']:.2f} | "
            f"{verdict}{star} |"
        )
    lines += [
        "",
        f"Corpus band for reference: repeated 4-gram "
        f"[{bands['repeated_4gram_rate']['low']:.4f}, "
        f"{bands['repeated_4gram_rate']['high']:.4f}], type/token "
        f"[{bands['type_token_ratio']['low']:.4f}, "
        f"{bands['type_token_ratio']['high']:.4f}], known-word "
        f"[{bands['known_word_rate']['low']:.4f}, "
        f"{bands['known_word_rate']['high']:.4f}].",
        "",
        "## The two failure directions",
        "",
        "**Too cold** collapses the distribution onto its mode: repetition climbs, the "
        "type/token ratio falls, and the model loops on a phrase. It stays perfectly "
        "grammatical while saying nothing, which is why fluency is not the thing to "
        "measure.",
        "",
        "**Too hot** samples the tail: known-word rate falls as the model invents "
        "character sequences that a byte-level BPE is perfectly happy to emit, and "
        "sentences stop resolving.",
        "",
        "A one-sided check would miss one of these. That is the argument for the "
        "two-sided corpus band, restated from ADR-018 with the failure it prevents now "
        "visible in the table.",
        "",
        "## What is pinned, and why it matters from here",
        "",
    ]
    lines.append(
        f"**temperature {pinned['temperature']}, top-k {pinned['top_k']}** — "
        + ("inside every corpus band"
           if pinned_row["in_band"]
           else "**outside the band on " + ", ".join(pinned_row["out_of_band"]) + "**")
        + f". {len(in_band)} of {len(rows)} swept cells land in band, so the choice is "
          "not unique; it is pinned so that it is *fixed*, not because it is the only "
          "defensible pair."
    )
    lines += [
        "",
        "Phase 5 and Phase 6 compare checkpoints, and a comparison is only about the "
        "checkpoints if everything else is held still. Decoding is the easiest thing to "
        "vary by accident and the hardest to spot afterwards: a fine-tune that looks "
        "degenerate at one temperature can look fine at another, and neither reading is "
        "about the fine-tune. So the pair above is config, not a CLI default, and the "
        "before/after galleries use it unchanged.",
        "",
        "Generated by `python -m src.decoding sweep`.",
    ]
    out = write_text(REPO_ROOT / cfg.results_dir / "decoding_sweep.md", "\n".join(lines))
    write_json(out.with_suffix(".json"), {
        "checkpoint": checkpoint,
        "step": state.get("step"),
        "samples_per_cell": cfg.decode_sweep_samples,
        "new_tokens": cfg.decode_sweep_new_tokens,
        "pinned": pinned,
        "pinned_in_band": bool(pinned_row and pinned_row["in_band"]),
        "cells_in_band": len(in_band),
        "cells_total": len(rows),
        "rows": rows,
    })
    # pinned_row cannot be None here - the grid membership is asserted above -
    # but say which of the three states it is rather than collapsing to a bool.
    verdict = "in band" if pinned_row["in_band"] else \
        "OUT OF BAND on " + ", ".join(pinned_row["out_of_band"])
    print(f"\n{len(in_band)}/{len(rows)} cells in band; pinned "
          f"T={pinned['temperature']} k={pinned['top_k']} {verdict}")
    print(f"swept in {time.time() - started:.0f}s")
    print(f"wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Decoding sweep (Phase 4).")
    ap.add_argument("command", choices=("sweep",))
    ap.add_argument("--config", default=None)
    ap.add_argument("--checkpoint", default="checkpoints/base.pt")
    args = ap.parse_args()

    cfg = Config.load(args.config)
    set_seed(cfg.seed, strict=cfg.strict_determinism)
    cmd_sweep(cfg, args.checkpoint)


if __name__ == "__main__":
    main()
