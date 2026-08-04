"""Global determinism (spec §4.2).

``set_seed`` is called at the top of every entry point. One seed, everywhere —
that is what makes "clone → one command per stage → identical scorecard"
a claim we can defend rather than a hope.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch

# cuBLAS reads this once, when the CUDA context is created. Setting it inside
# set_seed() would be too late if anything has already touched the GPU, so it is
# set at import: every entry point imports this module before it does any work.
# Without it, torch.use_deterministic_algorithms(True) raises on the first
# matmul it cannot make reproducible.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def set_seed(seed: int, *, strict: bool = True) -> int:
    """Seed python / numpy / torch and pin cuDNN to deterministic kernels.

    ``strict=True`` also forces ``torch.use_deterministic_algorithms``, which is
    what actually buys bit-reproducible *training* (ADR-016). Seeding alone is
    not enough: the seed fixes initialisation, but the backward pass accumulates
    with atomics whose order varies run to run. Measured on this model, seeding
    alone gave final losses of 0.01515 / 0.01034 / 0.01264 across three runs of
    an identical setup; with strict on, runs are bit-identical for a ~7%
    throughput cost.

    Set ``strict=False`` only to measure that cost, or if a later phase needs an
    op with no deterministic implementation — in which case the reproducibility
    claim needs re-stating, not quietly dropping.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    torch.use_deterministic_algorithms(strict)
    return seed
