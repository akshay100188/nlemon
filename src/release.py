"""Phase 8 — export the shippable checkpoint, and prove it is the measured one.

    python -m src.release export     # write the release file + record its hash
    python -m src.release verify     # re-check a written release against the record

`sft.pt` is 165.9 MB and cannot ship: GitHub's hard limit is 100 MB. The optimizer
state is 2/3 of that file and is useless to anyone running the model, so the
release is **weights only**.

**fp32, not bf16, and that is the load-bearing decision.** bf16 would be 33.8 MB
instead of 55.3 MB, and it would be a *different model* from the one Phase 7
certified - the scorecard's 5.2662 was measured on fp32 weights. Shipping bf16
weights under an fp32 scorecard is ADR-047's device confound wearing a new hat:
the artifact would not be the thing that was measured. So the export asserts
**tensor-for-tensor bit-identity** with the checkpoint the scorecard read, and
refuses to write if that fails. The identity IS the certification.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from config import REPO_ROOT, Config
from utils.io import write_json

SOURCE = "checkpoints/sft.pt"
RELEASE = "checkpoints/nlemon-14-sft-weights.pt"
RECORD = "release_manifest.json"

# Keys carried into the release: provenance and shape, never optimizer state.
KEEP = ("model", "config", "config_hash", "sft_stage_hash", "step",
        "base_step", "base_train_stage_hash", "best_val_loss")


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def param_counts(state: dict) -> dict:
    """Both counts, because a reader recomputing from the file will get the larger.

    The output head is tied to the token embedding, so `state_dict` stores the same
    tensor under two names and summing its values double-counts the embedding. The
    published figure comes from `model.parameters()`, which de-duplicates. Shipping
    only one number would make the card contradict the file to anyone who checks -
    and the readers who check are the audience.
    """
    sd_sum = sum(v.numel() for v in state["model"].values())
    cfg = Config(**state["config"])
    emb = cfg.vocab_size * cfg.d_model
    return {"state_dict_sum": sd_sum, "tied_embedding": emb,
            "unique_parameters": sd_sum - emb,
            "arithmetic": f"{sd_sum:,} - {cfg.vocab_size}x{cfg.d_model} = {sd_sum - emb:,}"}


def cmd_export(cfg: Config) -> None:
    src = REPO_ROOT / SOURCE
    dst = REPO_ROOT / RELEASE
    full = torch.load(src, map_location="cpu", weights_only=False)

    slim = {k: v for k, v in full.items() if k in KEEP}
    missing = [k for k in KEEP if k not in full]
    if missing:
        print(f"  note: absent from source, omitted: {', '.join(missing)}")
    torch.save(slim, dst)

    # The assertion the whole export exists for: the shipped weights must be the
    # weights Phase 7 measured, tensor for tensor.
    back = torch.load(dst, map_location="cpu", weights_only=False)
    keys_ok = set(back["model"]) == set(full["model"])
    diff = [k for k in full["model"]
            if not torch.equal(full["model"][k], back["model"][k])]
    dtypes = {str(v.dtype) for v in back["model"].values()}
    if not keys_ok or diff:
        dst.unlink(missing_ok=True)
        raise SystemExit(
            f"REFUSING TO SHIP: release is not bit-identical to {SOURCE}. "
            f"keys match={keys_ok}, differing tensors={len(diff)}. The scorecard "
            f"measured the source; an artifact that is not the source is not "
            f"certified by it.")
    if dtypes != {"torch.float32"}:
        dst.unlink(missing_ok=True)
        raise SystemExit(
            f"REFUSING TO SHIP: release dtypes {dtypes}, expected float32 only. "
            f"Phase 7 measured fp32 weights; a cast would ship a different model "
            f"than was certified.")

    counts = param_counts(full)
    digest = sha256_of(dst)
    mb = dst.stat().st_size / 1e6
    print(f"  wrote {RELEASE}  {mb:.1f} MB")
    print(f"  bit-identical to {SOURCE}: all {len(full['model'])} tensors, fp32")
    print(f"  sha256 {digest}")
    print(f"  params: state_dict sums to {counts['state_dict_sum']:,}, "
          f"unique {counts['unique_parameters']:,} (tied head)")

    rec = write_json(REPO_ROOT / RECORD, {
        "artifact": RELEASE,
        "sha256": digest,
        "bytes": dst.stat().st_size,
        "source_checkpoint": SOURCE,
        "source_sha256": sha256_of(src),
        "bit_identical_to_source": True,
        "tensors": len(full["model"]),
        "dtype": "float32",
        "why_not_bf16": "bf16 would be 33.8 MB but a different model than the "
                        "Phase 7 scorecard measured; the identity is the "
                        "certification (ADR-047's confound in a new hat).",
        "parameters": counts,
        "step": full.get("step"),
        "sft_stage_hash": full.get("sft_stage_hash"),
        "config_hash": full.get("config_hash"),
        "model_stage_hash": cfg.stage_hash("model"),
        "excluded": ["optimizer"],
    })
    print(f"  recorded in {rec.name if hasattr(rec, 'name') else rec}")


def cmd_verify(cfg: Config) -> None:
    dst = REPO_ROOT / RELEASE
    recp = REPO_ROOT / RECORD
    if not recp.exists():
        raise SystemExit(f"{RECORD} absent - run `export` first.")
    rec = json.loads(recp.read_text(encoding="utf-8"))
    if not dst.exists():
        raise SystemExit(
            f"{RELEASE} absent. It is gitignored by design - the hash in {RECORD} "
            f"is what ships in-repo, and the file ships as a Release asset. "
            f"Rebuild with `python -m src.release export`.")
    actual = sha256_of(dst)
    ok = actual == rec["sha256"]
    print(f"  recorded {rec['sha256']}")
    print(f"  on disk  {actual}")
    print(f"  -> {'MATCH' if ok else 'MISMATCH - do not publish this file'}")
    if not ok:
        raise SystemExit(1)
    counts = param_counts(torch.load(dst, map_location="cpu", weights_only=False))
    same = counts == rec["parameters"]
    print(f"  model-card arithmetic reconciles to the shipped file: {same}")
    print(f"    {counts['arithmetic']}")
    if not same:
        raise SystemExit(1)


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase 8 release export.")
    ap.add_argument("command", choices=("export", "verify"))
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = Config.load(args.config)
    (cmd_export if args.command == "export" else cmd_verify)(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
