"""Phase 3 — the GPT, from scratch.

A decoder-only transformer in the nanoGPT lineage: learned token and positional
embeddings, pre-norm blocks, causal multi-head self-attention, an MLP with GELU,
and an output head whose weights are tied to the token embedding.

    python -m src.model summary    # shape + parameter accounting
    python -m src.model check      # the gate: param budget + single-batch overfit

Nothing here reads a hyperparameter from anywhere but the config.
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

from config import REPO_ROOT, Config
from utils.device import probe
from utils.io import write_json, write_text
from utils.seed import set_seed


# --------------------------------------------------------------------------- #
# modules
# --------------------------------------------------------------------------- #
class CausalSelfAttention(nn.Module):
    """Multi-head self-attention with a causal mask.

    The mask is what makes this *language modelling*: position t may attend to
    positions <= t, never ahead. Without it the model could read the answer off
    the next token and the loss would collapse for the wrong reason.
    """

    def __init__(self, cfg: Config):
        super().__init__()
        if cfg.d_model % cfg.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.dropout = cfg.dropout

        # one projection producing q, k and v together, then split
        self.c_attn = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=cfg.bias)
        self.c_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=cfg.bias)
        self.resid_dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(C, dim=2)
        # (B, T, C) -> (B, n_heads, T, head_dim)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # is_causal builds the triangular mask internally and uses a fused
        # kernel where one is available.
        y = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))


class MLP(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        hidden = cfg.mlp_ratio * cfg.d_model
        self.c_fc = nn.Linear(cfg.d_model, hidden, bias=cfg.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(hidden, cfg.d_model, bias=cfg.bias)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.c_proj(self.gelu(self.c_fc(x))))


class Block(nn.Module):
    """Pre-norm: normalise going *into* each sublayer, add the result back.

    The residual stream itself is never normalised, so gradients reach the
    embeddings by an unobstructed path. Post-norm (the original 2017 layout)
    needs a warmup schedule to train stably at depth; pre-norm does not.
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.ln_2 = nn.LayerNorm(cfg.d_model)
        self.mlp = MLP(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.context_len, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        # Weight tying: the output head *is* the token embedding, transposed.
        # Saves 3.07M parameters here - 22% of the model - and ties "predicting
        # a token" to "representing a token", which is the same knowledge.
        self.lm_head.weight = self.tok_emb.weight

        self.apply(self._init_weights)
        # Scale the projections that write into the residual stream, so its
        # variance does not grow with depth (GPT-2's trick).
        for name, p in self.named_parameters():
            if name.endswith("c_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layers))

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        B, T = idx.shape
        if T > self.cfg.context_len:
            raise ValueError(f"sequence of {T} exceeds context_len {self.cfg.context_len}")

        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.reshape(-1)
            )
        return logits, loss

    # -- parameter accounting ------------------------------------------------
    def count_params(self, non_embedding: bool = False) -> int:
        """Total trainable parameters.

        Tied weights are shared objects, so deduplicate by identity — counting
        ``lm_head`` and ``tok_emb`` separately would inflate the total by 3.07M
        and hide the fact that tying happened at all.
        """
        seen: dict[int, torch.nn.Parameter] = {}
        for p in self.parameters():
            if p.requires_grad:
                seen[id(p)] = p
        total = sum(p.numel() for p in seen.values())
        if non_embedding:
            total -= self.pos_emb.weight.numel()
        return total

    def param_breakdown(self) -> dict[str, int]:
        """Where the parameters actually live, for the scorecard and the lesson."""
        emb = self.tok_emb.weight.numel()
        pos = self.pos_emb.weight.numel()
        per_block = sum(p.numel() for p in self.blocks[0].parameters())
        return {
            "token_embedding": emb,
            "positional_embedding": pos,
            "blocks_total": per_block * self.cfg.n_layers,
            "per_block": per_block,
            "final_layernorm": sum(p.numel() for p in self.ln_f.parameters()),
            "output_head_untied_cost": emb,  # what tying saves
            "total": self.count_params(),
        }


# --------------------------------------------------------------------------- #
# an independent check on the count
# --------------------------------------------------------------------------- #
def expected_params(cfg: Config) -> int:
    """Parameter count derived from the config by hand, not from the modules.

    This exists to disagree with ``count_params()``. Counting the model's own
    tensors proves only that the model is self-consistent — if the head were
    accidentally untied, or a block silently dropped, the count would still be
    "correct" for whatever got built. Two independent derivations that agree are
    evidence; one is a tautology.
    """
    d, r = cfg.d_model, cfg.mlp_ratio
    b = 1 if cfg.bias else 0

    embeddings = cfg.vocab_size * d + cfg.context_len * d      # head is tied
    layernorm = 2 * d                                          # weight + bias
    attn = (d * 3 * d + b * 3 * d) + (d * d + b * d)           # qkv + out proj
    mlp = (d * r * d + b * r * d) + (r * d * d + b * d)        # fc + proj
    per_block = 2 * layernorm + attn + mlp
    return embeddings + cfg.n_layers * per_block + layernorm


# --------------------------------------------------------------------------- #
# gate
# --------------------------------------------------------------------------- #
@dataclass
class OverfitResult:
    steps: int
    first_loss: float
    final_loss: float
    min_loss: float
    target: float
    random_baseline: float
    passed: bool
    seconds: float
    history: list[tuple[int, float]]


def overfit_one_batch(cfg: Config, device: torch.device, dtype: torch.dtype) -> OverfitResult:
    """Drive the loss to ~0 on a single batch.

    A correctly wired model can memorise one batch. If it cannot, something is
    disconnected — and this catches it in seconds instead of after a night of
    pretraining that quietly goes nowhere. Dropout is disabled: we want raw
    capacity here, not regularisation fighting the test.

    Reseeds first. Without it the verdict depends on how much RNG the checks
    *before* it happened to consume: adding the causality check moved the final
    loss from 0.079 to 0.593 and flipped the gate, while the model was
    unchanged. A gate that depends on test ordering is not measuring the model.
    """
    set_seed(cfg.seed, strict=cfg.strict_determinism)
    shard = REPO_ROOT / cfg.data_dir / "val.bin"
    if not shard.exists():
        raise SystemExit(f"{shard} not found - run `python -m src.tokenizer encode` first.")

    data = np.memmap(shard, dtype=np.uint16, mode="r")
    span = cfg.micro_batch * (cfg.context_len + 1)
    if data.size < span:
        raise SystemExit("val.bin is too small for one batch")
    chunk = np.asarray(data[:span], dtype=np.int64).reshape(cfg.micro_batch, -1)
    x = torch.from_numpy(chunk[:, :-1]).to(device)
    y = torch.from_numpy(chunk[:, 1:]).to(device)

    model = GPT(cfg).to(device)
    model.train()
    for m in model.modules():           # capacity test, not a regularisation test
        if isinstance(m, nn.Dropout):
            m.p = 0.0

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.overfit_lr, betas=(0.9, 0.95))
    autocast = torch.autocast(device_type=device.type, dtype=dtype) \
        if device.type == "cuda" else torch.autocast(device_type="cpu", enabled=False)

    history: list[tuple[int, float]] = []
    first = min_loss = float("nan")
    started = time.time()
    for step in range(1, cfg.overfit_steps + 1):
        with autocast:
            _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        value = loss.item()
        if step == 1:
            first = value
            min_loss = value
        min_loss = min(min_loss, value)
        if step == 1 or step % 25 == 0 or step == cfg.overfit_steps:
            history.append((step, round(value, 5)))

    return OverfitResult(
        steps=cfg.overfit_steps,
        first_loss=round(first, 5),
        final_loss=round(value, 5),
        min_loss=round(min_loss, 5),
        target=cfg.overfit_target_loss,
        # A fresh model should sit near ln(vocab_size); far off means the init
        # or the loss reduction is wrong before training even starts.
        random_baseline=round(math.log(cfg.vocab_size), 4),
        passed=bool(min_loss <= cfg.overfit_target_loss),
        seconds=round(time.time() - started, 1),
        history=history,
    )


def causality_check(cfg: Config, device: torch.device) -> dict:
    """Prove the mask actually masks.

    The overfit test cannot do this. A model with *no* causal mask memorises a
    batch faster, not slower — it can read the answer off the following token.
    So a green overfit is fully compatible with broken masking, and the bug
    would only show up as a model that generates beautifully during training and
    gibberish at inference.

    The property: changing the token at position t must not alter the logits at
    any position < t. Run in eval mode so dropout does not add noise.
    """
    model = GPT(cfg).to(device).eval()
    torch.manual_seed(cfg.seed)
    T = min(32, cfg.context_len)
    idx = torch.randint(0, cfg.vocab_size, (1, T), device=device)

    with torch.no_grad():
        base, _ = model(idx)
        t = T // 2
        poked = idx.clone()
        # force a genuinely different token at position t
        poked[0, t] = (idx[0, t] + 1) % cfg.vocab_size
        after, _ = model(poked)

        before_max = (base[0, :t] - after[0, :t]).abs().max().item()
        at_and_after_max = (base[0, t:] - after[0, t:]).abs().max().item()

    # Positions before t must be bit-identical; positions from t on must change,
    # otherwise the input is being ignored and the first check is vacuous.
    passed = before_max == 0.0 and at_and_after_max > 0.0
    return {
        "seq_len": T,
        "poked_position": t,
        "max_delta_before": before_max,
        "max_delta_from_poke": at_and_after_max,
        "passed": bool(passed),
    }


def cmd_summary(cfg: Config) -> None:
    model = GPT(cfg)
    bd = model.param_breakdown()
    print(f"{cfg.project_name}  ·  phase 3 · architecture")
    print(f"model stage hash : {cfg.stage_hash('model')}\n")
    print(f"d_model {cfg.d_model}  layers {cfg.n_layers}  heads {cfg.n_heads}  "
          f"head_dim {cfg.head_dim}  ctx {cfg.context_len}  mlp x{cfg.mlp_ratio}")
    print()
    print(f"{'component':<28}{'params':>14}{'share':>9}")
    print("-" * 51)
    total = bd["total"]
    for key in ("token_embedding", "positional_embedding", "blocks_total",
                "final_layernorm"):
        print(f"{key:<28}{bd[key]:>14,}{bd[key] / total:>8.1%}")
    print("-" * 51)
    print(f"{'total (tied head)':<28}{total:>14,}")
    print(f"{'per block':<28}{bd['per_block']:>14,}")
    print(f"{'saved by weight tying':<28}{bd['output_head_untied_cost']:>14,}")
    print(f"{'non-embedding':<28}{model.count_params(non_embedding=True):>14,}")


def cmd_check(cfg: Config) -> None:
    print(f"{cfg.project_name}  ·  phase 3 gate")
    print(f"model stage hash : {cfg.stage_hash('model')}\n")

    model = GPT(cfg)
    actual = model.count_params()
    predicted = expected_params(cfg)

    # (a1) two independent derivations must agree exactly
    formula_ok = actual == predicted
    print(f"[{'PASS' if formula_ok else 'FAIL'}] module count == analytic formula: "
          f"{actual:,} vs {predicted:,}")

    # (a2) and the result must sit inside the claimed budget
    low = cfg.param_budget * (1 - cfg.param_budget_tol)
    high = cfg.param_budget * (1 + cfg.param_budget_tol)
    budget_ok = low <= actual <= high
    print(f"[{'PASS' if budget_ok else 'FAIL'}] within budget {cfg.param_budget:,} "
          f"+/-{cfg.param_budget_tol:.0%}: {actual:,} "
          f"({actual / cfg.param_budget:.2%} of budget)")

    # tying is a claim the README makes; check it rather than trust it
    tied = model.lm_head.weight is model.tok_emb.weight
    print(f"[{'PASS' if tied else 'FAIL'}] output head tied to token embedding")

    dev = probe()
    print(f"\ndevice: {dev.name} ({str(dev.dtype).replace('torch.', '')})")

    cau = causality_check(cfg, dev.device)
    print(f"[{'PASS' if cau['passed'] else 'FAIL'}] causal mask: poking position "
          f"{cau['poked_position']} of {cau['seq_len']} moves logits by "
          f"{cau['max_delta_before']} before it, "
          f"{cau['max_delta_from_poke']:.4f} from it onward")

    ov = overfit_one_batch(cfg, dev.device, dev.dtype)
    print(f"[{'PASS' if ov.passed else 'FAIL'}] overfit one batch: "
          f"loss {ov.first_loss} -> {ov.final_loss} in {ov.steps} steps "
          f"(min {ov.min_loss}, target <= {ov.target}, {ov.seconds}s)")
    print(f"        random-init baseline ln(vocab) = {ov.random_baseline}, "
          f"first step {ov.first_loss}")
    print("        " + "  ".join(f"{s}:{v}" for s, v in ov.history[:8]))

    passed = bool(formula_ok and budget_ok and tied and cau["passed"] and ov.passed)
    result = {
        "model_stage_hash": cfg.stage_hash("model"),
        "params": {
            "counted": actual,
            "analytic": predicted,
            "agree": formula_ok,
            "budget": cfg.param_budget,
            "within_budget": budget_ok,
            "breakdown": model.param_breakdown(),
            "non_embedding": model.count_params(non_embedding=True),
        },
        "weight_tying": tied,
        "causal_mask": cau,
        "overfit": {
            "steps": ov.steps, "first_loss": ov.first_loss,
            "final_loss": ov.final_loss, "min_loss": ov.min_loss,
            "target": ov.target, "random_baseline": ov.random_baseline,
            "passed": ov.passed, "seconds": ov.seconds, "history": ov.history,
            "device": dev.name,
        },
        "passed": passed,
    }
    out = write_json(REPO_ROOT / cfg.results_dir / "model_gate.json", result)
    print(f"\nwrote {out}")
    print(f"\nPHASE 3 GATE: {'PASS' if passed else 'FAIL'}")
    if not passed:
        raise SystemExit(1)


def cmd_margin(cfg: Config, step_grid: list[int], lr_grid: list[float]) -> None:
    """Evidence for overfit_steps / overfit_lr.

    A gate needs margin, not a lucky pass. This reports the *worst* seed for
    each setting, because that is what the gate has to survive.
    """
    import dataclasses

    dev = probe()
    print(f"device: {dev.name}\n")
    rows = []
    for lr in lr_grid:
        for steps in step_grid:
            per_seed = []
            for seed in cfg.overfit_margin_seeds:
                trial = dataclasses.replace(cfg, seed=seed, overfit_steps=steps,
                                            overfit_lr=lr)
                r = overfit_one_batch(trial, dev.device, dev.dtype)
                per_seed.append(r.min_loss)
                print(f"  lr={lr:.0e} steps={steps:<5} seed={seed:<5} "
                      f"min_loss={r.min_loss:.4f} ({r.seconds}s)")
            worst = max(per_seed)
            rows.append({
                "lr": lr, "steps": steps,
                "per_seed_min_loss": per_seed,
                "worst_min_loss": round(worst, 5),
                "passes_target": bool(worst <= cfg.overfit_target_loss),
                "margin_x": round(cfg.overfit_target_loss / worst, 2) if worst else None,
            })
            print(f"  -> worst-of-{len(per_seed)} = {worst:.4f}\n")

    chosen = next((r for r in rows if r["lr"] == cfg.overfit_lr
                   and r["steps"] == cfg.overfit_steps), None)
    lines = [
        "# How much margin does the single-batch overfit gate have?",
        "",
        f"Target: min loss <= `{cfg.overfit_target_loss}`. Each setting is run "
        f"across seeds `{list(cfg.overfit_margin_seeds)}` and reported by its "
        f"**worst** seed — that is what the gate must survive.",
        "",
        "| lr | steps | worst min loss | margin vs target | passes |",
        "|---|---|---|---|---|",
    ]
    for r in rows:
        mark = "  **<- config**" if chosen and r is chosen else ""
        margin = f"{r['margin_x']}x" if r["margin_x"] else "-"
        lines.append(
            f"| {r['lr']:.0e} | {r['steps']:,} | {r['worst_min_loss']} | {margin} | "
            f"{'yes' if r['passes_target'] else 'NO'}{mark} |"
        )
    lines += [
        "",
        "## What this shows",
        "",
        "A single-batch overfit is supposed to be a wiring check, but at too few "
        "steps it becomes a test of luck: the spread across seeds is wider than "
        "the distance to the target, so the same correct model passes or fails "
        "depending on initialisation. The chosen setting is the cheapest one "
        "whose *worst* seed clears the target with room to spare.",
        "",
        "Raising the learning rate does not substitute for steps — the higher rate "
        "oscillates rather than converging, and its worst seed is further from the "
        "target, not closer.",
        "",
        "Generated by `python -m src.model margin`.",
    ]
    out = write_text(REPO_ROOT / cfg.results_dir / "overfit_margin.md",
                     "\n".join(lines))
    write_json(out.with_suffix(".json"), {
        "target": cfg.overfit_target_loss,
        "seeds": list(cfg.overfit_margin_seeds),
        "configured": {"lr": cfg.overfit_lr, "steps": cfg.overfit_steps},
        "rows": rows,
    })
    print(f"wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description="nLemon-14 model (Phase 3).")
    ap.add_argument("command", choices=("summary", "check", "margin"))
    ap.add_argument("--config", default=None)
    ap.add_argument("--margin-steps", type=int, nargs="+", default=[300, 600, 1000])
    ap.add_argument("--margin-lrs", type=float, nargs="+", default=[1e-3, 3e-3])
    args = ap.parse_args()

    cfg = Config.load(args.config)
    set_seed(cfg.seed, strict=cfg.strict_determinism)
    if args.command == "summary":
        cmd_summary(cfg)
    elif args.command == "check":
        cmd_check(cfg)
    else:
        cmd_margin(cfg, args.margin_steps, args.margin_lrs)


if __name__ == "__main__":
    main()
