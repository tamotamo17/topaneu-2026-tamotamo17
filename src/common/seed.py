"""Global seeding utilities for reproducibility.

This module centralizes random seed handling so that fixing a single seed in
the YAML config reproduces the *same* model weight initialization (and, when
``deterministic`` is enabled, the same training trajectory) across all tasks:
classification, segmentation, and detection.

Call :func:`set_seed` once at the very start of a training script, *before*
building the model, so that weight initialization draws from a deterministic
RNG state.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int = 42, deterministic: bool = False) -> int:
    """Seed all RNGs used during training.

    Seeds Python's ``random``, NumPy, and PyTorch (CPU + all CUDA devices),
    and sets ``PYTHONHASHSEED``. Because model weight initialization draws from
    PyTorch's global RNG, calling this before building the model guarantees the
    same initial weights for a given seed.

    Args:
        seed: The random seed to apply.
        deterministic: If True, also force deterministic algorithms and disable
            cuDNN benchmarking so that the full training run (not just weight
            init) is reproducible. This can be slower and a few ops may raise if
            no deterministic implementation exists.

    Returns:
        The seed that was applied (for logging convenience).
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        # Make GPU/cuDNN ops reproducible at the cost of some speed.
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        # Required by some deterministic CUDA kernels (e.g. certain matmuls).
        # Set unconditionally: a pre-existing incompatible value (e.g. ":16:8")
        # would make use_deterministic_algorithms raise.
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        # warn_only=True: fall back to non-deterministic kernels (with a warning)
        # for ops lacking a deterministic implementation, instead of crashing
        # training. Weight-init reproducibility does not depend on this flag; it
        # only tightens the per-step training trajectory.
        torch.use_deterministic_algorithms(True, warn_only=True)

    return seed


def seed_worker(worker_id: int) -> None:
    """DataLoader ``worker_init_fn`` for reproducible multi-worker data loading.

    Without this, every worker process inherits the same base RNG state, so
    augmentation/random transforms repeat across workers. Deriving each worker's
    seed from ``torch.initial_seed()`` (which PyTorch sets per worker from the
    base seed) gives distinct-but-deterministic seeds.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
