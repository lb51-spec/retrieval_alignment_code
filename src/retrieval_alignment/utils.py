"""General utilities for the DW-MQL implementation.

Utility scope is described in ``docs/code_architecture.md``. Keep this module
free of algorithm-specific loss logic.
"""

from __future__ import annotations

import random

import numpy as np

try:  # pragma: no cover - exercised when the project dependencies are installed.
    import torch
except ModuleNotFoundError:  # pragma: no cover - keeps imports clean pre-install.
    torch = None  # type: ignore[assignment]


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch random generators."""

    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
