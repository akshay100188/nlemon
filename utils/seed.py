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


def set_seed(seed: int, *, strict: bool = False) -> int:
    """Seed python / numpy / torch and pin cuDNN to deterministic kernels.

    ``strict=True`` additionally forces ``torch.use_deterministic_algorithms``,
    which turns any remaining nondeterministic CUDA kernel into a hard error
    instead of a silent divergence. It is off by default because it can reject
    otherwise-fine ops; turn it on when auditing a reproducibility claim.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if strict:
        # cuBLAS needs this set before the first CUDA context for reduction determinism.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)

    return seed
