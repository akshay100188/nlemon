"""nLemon-14 config — the single source of truth (spec §4.1).

Every hyperparameter and the global seed live in ``configs/nlemon_14m.yaml``.
This module loads that file into a frozen dataclass. Nothing else in the repo
may hardcode a hyperparameter, and every artifact records ``Config.hash()`` so
a scorecard can always be traced back to the exact settings that produced it.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "nlemon_14m.yaml"

# Which fields actually determine each stage's output (ADR-010).
#
# `Config.hash()` fingerprints the whole config, so adding a field for a later
# phase changes it — which would make a Phase 1 artifact look stale even though
# the corpus on disk is byte-for-byte identical. A stage hash covers only the
# inputs that stage really depends on, so "did this artifact change?" and "did
# the config change somewhere else?" stop being the same question.
STAGE_FIELDS: dict[str, tuple[str, ...]] = {
    "data": (
        "seed", "dataset_name", "dataset_revision", "doc_separator",
        "max_train_docs", "max_val_docs", "peek_samples",
    ),
    "tokenizer": (
        "seed", "vocab_size", "doc_separator",
        "tokenizer_train_docs", "tokenizer_min_frequency",
    ),
    # Model shape only. dropout is included because it changes the forward pass,
    # but the gate thresholds are not: moving a threshold must not make the
    # architecture look like it changed.
    "model": (
        "vocab_size", "d_model", "n_layers", "n_heads", "context_len",
        "dropout", "mlp_ratio", "bias",
    ),
    # Everything that changes the weights in base.pt. Includes the model fields
    # (a different architecture is a different training run) and
    # strict_determinism, which does not change the model's shape but does
    # change its trained values — ADR-016 measured the drift. Excludes pure
    # bookkeeping (log_interval, ckpt_interval): how often we write a line to a
    # csv must not make the checkpoint look different.
    "train": (
        "seed", "strict_determinism",
        "vocab_size", "d_model", "n_layers", "n_heads", "context_len",
        "dropout", "mlp_ratio", "bias",
        "micro_batch", "grad_accum_steps", "lr", "max_steps", "warmup_steps",
        "weight_decay", "beta1", "beta2", "grad_clip", "min_lr_ratio",
        # eval_iters is here but should not be (ADR-020): it changes the reported
        # metric, not the weights. Moving it now would shift base.pt's recorded
        # hash for no real change; it migrates to an eval stage in Phase 7.
        "eval_iters",
    ),
}

# Fields deliberately outside every stage hash, with the reason.
#
# STAGE_FIELDS is a thing that can be wrong by omission: forget to add a field a
# stage depends on and its stage hash under-reports a real change, silently. This
# turns that silence into an error — a new field must be classified as either
# affecting some stage's output or explicitly not.
STAGE_EXEMPT: dict[str, str] = {
    "project_name": "display label; flows into artifacts but changes no output",
    "data_dir": "path",
    "ckpt_dir": "path",
    "results_dir": "path",
    "log_interval": "bookkeeping: how often a csv row is written",
    "ckpt_interval": "bookkeeping",
    "eval_interval": "bookkeeping: how often validation runs, not what it measures",
    "val_ppl_threshold": "gate threshold; moving a bar must not make an artifact look changed",
    "param_budget": "gate threshold",
    "param_budget_tol": "gate threshold",
    "overfit_steps": "gate threshold (Phase 3 wiring check, produces no artifact)",
    "overfit_target_loss": "gate threshold",
    "overfit_lr": "gate threshold",
    "overfit_margin_seeds": "gate evidence sweep",
    "coherence_ref_docs": "gate threshold",
    "coherence_vocab_docs": "gate threshold",
    "coherence_band_low_pct": "gate threshold",
    "coherence_band_high_pct": "gate threshold",
    "coherence_samples": "gate threshold",
    "coherence_bootstrap": "gate threshold",
    "coherence_new_tokens": "gate threshold",
    "coherence_temperature": "decoding knob; changes what the model says, not what it knows",
    "coherence_top_k": "decoding knob; changes what the model says, not what it knows",
    "decode_sweep_temperatures": "evidence sweep for the pinned decoding pair",
    "decode_sweep_top_k": "evidence sweep",
    "decode_sweep_samples": "evidence sweep",
    "decode_sweep_new_tokens": "evidence sweep",
    # SFT dataset construction. These belong to a "sft" stage hash, added when
    # sft.py lands; until a checkpoint records one, they change no artifact.
    "sft_subject_scan_docs": "sft dataset construction (stage added with sft.py)",
    "sft_pair_scan_docs": "sft dataset construction",
    "sft_subject_pool": "sft dataset construction",
    "sft_min_subject_count": "sft dataset construction",
    "sft_min_head_ratio": "sft dataset construction",
    "sft_heldout_subject_frac": "sft dataset construction",
    "sft_max_pairs": "sft dataset construction",
    "sft_eval_prompts": "sft evaluation set size",
    "sft_min_response_words": "sft dataset construction",
    "sft_max_response_words": "sft dataset construction",
    "checker_min_sentences": "gate threshold",
    "checker_min_story_words": "gate threshold",
    "checker_max_repeat_rate": "gate threshold",
    "checker_max_repeat_run": "gate threshold",
    "sft_gate_temperature": "gate decoding, pinned separately from the global default",
    "sft_gate_top_k": "gate decoding, pinned separately",
    "sft_gate_new_tokens": "gate decoding, pinned separately",
    "sft_gate_subject_mention_min": "pre-registered gate threshold",
    "sft_gate_length_band_min": "pre-registered gate threshold",
    "sft_gate_is_story_min": "pre-registered gate threshold",
    "sft_gate_not_degenerate_min": "pre-registered gate threshold",
    "sft_gate_shuffled_max": "pre-registered gate threshold",
}


def assert_stage_coverage() -> None:
    """Every config field is in a stage hash or explicitly exempt.

    Also catches the reverse mistake: a stage or exemption naming a field that no
    longer exists, which would silently narrow a hash.
    """
    known = {f.name for f in fields(Config)}
    covered: set[str] = set().union(*STAGE_FIELDS.values())

    unknown_in_stages = {
        f"{stage}:{name}"
        for stage, names in STAGE_FIELDS.items()
        for name in names if name not in known
    }
    unknown_exempt = set(STAGE_EXEMPT) - known
    unclassified = known - covered - set(STAGE_EXEMPT)

    problems = []
    if unclassified:
        problems.append(
            f"config field(s) {sorted(unclassified)} are in no stage hash and not "
            f"listed in STAGE_EXEMPT. Add them to the stage(s) whose output they "
            f"change, or to STAGE_EXEMPT with a reason."
        )
    if unknown_in_stages:
        problems.append(f"STAGE_FIELDS names nonexistent field(s): {sorted(unknown_in_stages)}")
    if unknown_exempt:
        problems.append(f"STAGE_EXEMPT names nonexistent field(s): {sorted(unknown_exempt)}")
    if problems:
        raise ValueError("stage-hash coverage: " + " | ".join(problems))


@dataclass(frozen=True)
class Config:
    """Immutable run configuration. Field order is the hash order."""

    project_name: str = "nLemon-14"  # flows into every artifact + the scorecard
    seed: int = 1337
    strict_determinism: bool = True

    # model
    vocab_size: int = 8000
    d_model: int = 384
    n_layers: int = 6
    n_heads: int = 6
    context_len: int = 256
    dropout: float = 0.1
    mlp_ratio: int = 4
    bias: bool = True

    # training
    micro_batch: int = 16
    grad_accum_steps: int = 4
    lr: float = 3e-4
    max_steps: int = 20000
    warmup_steps: int = 1000
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    min_lr_ratio: float = 0.1
    eval_interval: int = 500
    eval_iters: int = 100
    log_interval: int = 20
    ckpt_interval: int = 2000

    # Phase 4 gate
    val_ppl_threshold: float = 8.0
    coherence_ref_docs: int = 2000
    coherence_vocab_docs: int = 100_000
    coherence_band_low_pct: int = 5
    coherence_band_high_pct: int = 95
    coherence_samples: int = 16
    coherence_bootstrap: int = 400
    coherence_new_tokens: int = 200
    coherence_temperature: float = 0.8
    coherence_top_k: int = 40

    # decoding sweep (Phase 4 -> pins decoding for Phase 5/6 comparisons)
    decode_sweep_temperatures: tuple[float, ...] = (0.2, 0.5, 0.7, 0.8, 0.9, 1.1, 1.4)
    decode_sweep_top_k: tuple[int, ...] = (0, 20, 40)
    decode_sweep_samples: int = 6
    decode_sweep_new_tokens: int = 160

    # SFT instruction pairs (Phase 5)
    sft_subject_scan_docs: int = 200_000
    sft_pair_scan_docs: int = 400_000
    sft_subject_pool: int = 400
    sft_min_subject_count: int = 200
    sft_min_head_ratio: float = 0.5
    sft_heldout_subject_frac: float = 0.2
    sft_max_pairs: int = 40_000
    sft_eval_prompts: int = 200
    sft_min_response_words: int = 60
    sft_max_response_words: int = 220
    checker_min_sentences: int = 3
    checker_min_story_words: int = 40
    checker_max_repeat_rate: float = 0.0333
    checker_max_repeat_run: int = 2
    # Pinned in the gate's own config, deliberately not aliased to the global
    # decode defaults - see ADR-025.
    sft_gate_temperature: float = 0.8
    sft_gate_top_k: int = 40
    sft_gate_new_tokens: int = 200
    # Pre-registered before src/sft.py existed - see ADR-027 and git history.
    sft_gate_subject_mention_min: float = 0.60
    sft_gate_length_band_min: float = 0.70
    sft_gate_is_story_min: float = 0.69
    sft_gate_not_degenerate_min: float = 0.745
    sft_gate_shuffled_max: float = 0.10

    # data
    dataset_name: str = "roneneldan/TinyStories"
    dataset_revision: str = "main"
    doc_separator: str = "<|endoftext|>"
    max_train_docs: int = 0  # 0 = entire split
    max_val_docs: int = 0
    peek_samples: int = 3

    # tokenizer
    tokenizer_train_docs: int = 200_000  # bounds BPE *training*, not encoding
    tokenizer_min_frequency: int = 2

    # architecture gate (Phase 3)
    param_budget: int = 14_000_000
    param_budget_tol: float = 0.10
    overfit_steps: int = 600
    overfit_target_loss: float = 0.1
    overfit_lr: float = 1e-3
    overfit_margin_seeds: tuple[int, ...] = (1337, 7, 99)

    # paths (relative to the repo root)
    data_dir: str = "data"
    ckpt_dir: str = "checkpoints"
    results_dir: str = "results"

    # -- derived ------------------------------------------------------------
    @property
    def head_dim(self) -> int:
        if self.d_model % self.n_heads:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
            )
        return self.d_model // self.n_heads

    @property
    def effective_batch(self) -> int:
        """Tokens-per-optimizer-step is what training actually sees (ADR-004)."""
        return self.micro_batch * self.grad_accum_steps

    # -- io -----------------------------------------------------------------
    @staticmethod
    def load(path: str | Path | None = None) -> "Config":
        path = Path(path) if path else DEFAULT_CONFIG_PATH
        with open(path, "r", encoding="utf-8") as f:
            raw: dict[str, Any] = yaml.safe_load(f) or {}

        known = {f.name for f in fields(Config)}
        unknown = set(raw) - known
        if unknown:
            raise ValueError(
                f"{path}: unknown config key(s) {sorted(unknown)}. "
                f"Add the field to Config or remove it from the YAML — silent "
                f"drops would break the reproducibility claim."
            )

        cfg = Config(**raw)
        cfg.validate()
        return cfg

    def validate(self) -> None:
        assert_stage_coverage()  # fail at load, not at the phase that needs it
        _ = self.head_dim  # raises if d_model is not divisible by n_heads
        if self.warmup_steps > self.max_steps:
            raise ValueError("warmup_steps must not exceed max_steps")
        for name in ("vocab_size", "d_model", "n_layers", "n_heads", "context_len",
                     "micro_batch", "grad_accum_steps", "max_steps"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def hash(self) -> str:
        """Short, stable fingerprint of every field. Recorded in every artifact."""
        blob = json.dumps(self.as_dict(), sort_keys=True).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()[:12]

    def stage_hash(self, stage: str) -> str:
        """Fingerprint of only the fields that determine ``stage``'s output.

        Two builds with the same stage hash must produce identical artifacts for
        that stage, regardless of what changed elsewhere in the config.
        """
        try:
            fields_ = STAGE_FIELDS[stage]
        except KeyError:
            raise ValueError(
                f"unknown stage {stage!r}; known stages: {sorted(STAGE_FIELDS)}"
            ) from None
        subset = {name: getattr(self, name) for name in fields_}
        blob = json.dumps(subset, sort_keys=True).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()[:12]


def resolve(attr_value: str, *parts: str) -> Path:
    """Join a configured relative directory onto the repo root."""
    return REPO_ROOT.joinpath(attr_value, *parts)


if __name__ == "__main__":
    cfg = Config.load()
    print(json.dumps(cfg.as_dict(), indent=2, sort_keys=True))
    print(f"\nconfig_hash = {cfg.hash()}")
    for stage in STAGE_FIELDS:
        print(f"  stage_hash[{stage:<9}] = {cfg.stage_hash(stage)}")
    print(f"head_dim = {cfg.head_dim}   effective_batch = {cfg.effective_batch}")
