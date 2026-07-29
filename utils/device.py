"""Device / precision probe (spec §5, Phase 1).

Run it as ``python -m utils.device`` before any training stage: it tells you
whether this machine can honour the bf16-on-CUDA assumption in the config, and
how much VRAM you have to fit the batch into.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

BYTES_PER_MIB = 1024 ** 2


@dataclass(frozen=True)
class DeviceReport:
    device: torch.device
    name: str
    cuda: bool
    bf16: bool
    total_mib: int
    free_mib: int
    capability: str

    @property
    def dtype(self) -> torch.dtype:
        """bf16 where supported, else fp32. Never fp16 — no loss-scaling here."""
        return torch.bfloat16 if self.bf16 else torch.float32


def probe() -> DeviceReport:
    if torch.cuda.is_available():
        idx = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(idx)
        free, total = torch.cuda.mem_get_info(idx)
        return DeviceReport(
            device=torch.device("cuda", idx),
            name=props.name,
            cuda=True,
            bf16=torch.cuda.is_bf16_supported(),
            total_mib=total // BYTES_PER_MIB,
            free_mib=free // BYTES_PER_MIB,
            capability=f"sm_{props.major}{props.minor}",
        )
    return DeviceReport(
        device=torch.device("cpu"),
        name="cpu",
        cuda=False,
        bf16=hasattr(torch, "bfloat16"),
        total_mib=0,
        free_mib=0,
        capability="n/a",
    )


def get_device() -> torch.device:
    return probe().device


def report(prefix: str = "") -> DeviceReport:
    r = probe()
    print(f"{prefix}torch        : {torch.__version__}")
    print(f"{prefix}cuda build   : {torch.version.cuda or 'cpu-only'}")
    print(f"{prefix}cuda avail   : {r.cuda}")
    print(f"{prefix}device       : {r.name} ({r.capability})")
    if r.cuda:
        print(f"{prefix}vram         : {r.free_mib} MiB free / {r.total_mib} MiB total")
    print(f"{prefix}bf16         : {r.bf16}")
    print(f"{prefix}train dtype  : {str(r.dtype).replace('torch.', '')}")
    if not r.cuda:
        print(f"{prefix}NOTE         : CPU only — data and tokenizer stages are fine, "
              f"pretraining will be impractically slow.")
    return r


if __name__ == "__main__":
    report()
