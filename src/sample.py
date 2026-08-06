"""Phase 4/7 — inference. Ask the model to say something.

    python -m src.sample --prompt "Once upon a time"
    python -m src.sample --checkpoint checkpoints/base.pt --temperature 0.8 --top-k 40
    python -m src.sample --gallery      # deterministic sample set -> results/samples/

Sampling knobs are arguments, not config: they are how you *inspect* a
checkpoint, not part of what produced it. The gallery is the exception — it is
an artifact Phase 8 publishes, so it is seeded and reproducible.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from config import REPO_ROOT, Config
from src.model import GPT
from src.tokenizer import load as load_tokenizer
from utils.device import probe
from utils.io import write_text
from utils.seed import set_seed

# Fixed prompts for the gallery. Chosen to span the corpus: a classic opening, a
# mid-sentence continuation, a dialogue, and a bare noun phrase.
GALLERY_PROMPTS = (
    "Once upon a time",
    "Lily and Tom went to the park",
    'The little dog said, "',
    "One day, a boy found a shiny",
)


def load_checkpoint(path: Path, device: torch.device) -> tuple[GPT, dict]:
    if not path.exists():
        raise SystemExit(f"{path} not found - run `python -m src.train` first.")
    state = torch.load(path, map_location=device, weights_only=False)
    cfg = Config(**state["config"])
    model = GPT(cfg).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    return model, state


@torch.no_grad()
def generate(model: GPT, ids: list[int], max_new_tokens: int, temperature: float,
             top_k: int | None, device: torch.device, eot_id: int | None = None,
             generator: torch.Generator | None = None) -> list[int]:
    """Autoregressive sampling.

    temperature < 1 sharpens the distribution (safer, more repetitive), > 1
    flattens it (more surprising, more likely to wander). top_k truncates to the
    k most likely tokens before sampling, which removes the long tail of
    nonsense that a 14M model still assigns non-zero mass to.
    """
    out = list(ids)
    for _ in range(max_new_tokens):
        # crop to the context window - the model has no positions beyond it
        window = out[-model.cfg.context_len:]
        x = torch.tensor([window], dtype=torch.long, device=device)
        logits, _ = model(x)
        logits = logits[0, -1, :]

        if temperature <= 0:                      # greedy
            nxt = int(torch.argmax(logits).item())
        else:
            logits = logits / temperature
            if top_k:
                k = min(top_k, logits.size(-1))
                kth = torch.topk(logits, k).values[-1]
                logits = logits.masked_fill(logits < kth, float("-inf"))
            # Sample on the CPU with a CPU generator: the draw then depends only
            # on the seed, not on which device ran the forward pass, so a gallery
            # generated on this laptop reproduces on a CPU-only machine.
            probs = F.softmax(logits, dim=-1).float().cpu()
            nxt = int(torch.multinomial(probs, 1, generator=generator).item())

        if eot_id is not None and nxt == eot_id:
            break
        out.append(nxt)
    return out


def sample_text(model: GPT, tok, prompt: str, args, device: torch.device,
                generator: torch.Generator | None = None) -> str:
    ids = tok.encode(prompt).ids
    eot = tok.token_to_id(model.cfg.doc_separator)
    full = generate(model, ids, args.max_new_tokens, args.temperature,
                    args.top_k, device, eot_id=eot, generator=generator)
    return tok.decode(full)


def cmd_gallery(model: GPT, tok, state: dict, args, device: torch.device,
                cfg: Config) -> None:
    """A reproducible sample set: same seed, same prompts, same output."""
    lines = [
        f"# Samples — {cfg.project_name}",
        "",
        f"Checkpoint `{args.checkpoint}` at step {state.get('step', '?')} "
        f"(val perplexity {state.get('val_perplexity', float('nan')):.3f}).",
        f"temperature `{args.temperature}`, top_k `{args.top_k}`, "
        f"seed `{cfg.seed}`, max {args.max_new_tokens} new tokens.",
        "",
        "Prompts are fixed; rerunning reproduces these exact continuations.",
        "",
    ]
    for i, prompt in enumerate(GALLERY_PROMPTS, 1):
        gen = torch.Generator(device="cpu").manual_seed(cfg.seed + i)
        text = sample_text(model, tok, prompt, args, device, generator=gen)
        continuation = text[len(prompt):]
        lines += [
            f"## {i}. {prompt!r}",
            "",
            f"> **{prompt}**{continuation}",
            "",
        ]
    out = write_text(REPO_ROOT / cfg.results_dir / "samples" / "base_samples.md",
                     "\n".join(lines))
    print(f"wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description="nLemon-14 sampling.")
    ap.add_argument("--config", default=None)
    ap.add_argument("--checkpoint", default="checkpoints/base.pt")
    ap.add_argument("--prompt", default="Once upon a time")
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--num-samples", type=int, default=1)
    ap.add_argument("--gallery", action="store_true",
                    help="write the reproducible sample gallery instead")
    args = ap.parse_args()

    cfg = Config.load(args.config)
    set_seed(cfg.seed, strict=cfg.strict_determinism)
    dev = probe()
    model, state = load_checkpoint(REPO_ROOT / args.checkpoint, dev.device)
    tok = load_tokenizer(cfg)

    if args.gallery:
        cmd_gallery(model, tok, state, args, dev.device, cfg)
        return

    for i in range(args.num_samples):
        gen = torch.Generator(device="cpu").manual_seed(cfg.seed + i)
        started = time.time()
        text = sample_text(model, tok, args.prompt, args, dev.device, generator=gen)
        print(f"--- sample {i + 1}/{args.num_samples} "
              f"({time.time() - started:.1f}s) ---")
        print(text)
        print()


if __name__ == "__main__":
    main()
