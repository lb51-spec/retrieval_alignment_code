"""Density-ratio weight utilities for DW-MQL+ANN.

The density head in DW-MQL+ANN estimates discounted occupancy-ratio weights
used to reweight Bellman residuals toward the ANN-truncated target policy's
stationary distribution. Following ``docs/algorithm_notes.md``, the basic
stabilized parameterization is:

``w_i = min(softplus(u_i) + eps, w_max)``
``wbar_i = w_i / mean_j(w_j)``

This module only implements tensor utilities for positive weights, clipping,
self-normalisation, ESS, and entropy. It is independent of FAISS, replay
buffers, networks, and training loops.
"""

from __future__ import annotations

from typing import Any

try:  # pragma: no cover - exercised when the project dependencies are installed.
    import torch
    import torch.nn.functional as F
except ModuleNotFoundError:  # pragma: no cover - keeps imports clean pre-install.
    torch = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]


def _require_torch() -> None:
    if torch is None or F is None:
        raise ModuleNotFoundError("PyTorch is required to compute density weights.")


def _as_vector(values: Any, *, name: str) -> "torch.Tensor":
    _require_torch()
    tensor = torch.as_tensor(values)
    if not torch.is_floating_point(tensor):
        tensor = tensor.float()
    if tensor.ndim == 2 and tensor.shape[1] == 1:
        tensor = tensor.squeeze(1)
    if tensor.ndim != 1:
        raise ValueError(f"{name} must have shape [batch] or [batch, 1].")
    if tensor.numel() == 0:
        raise ValueError(f"{name} must be non-empty.")
    if torch.any(~torch.isfinite(tensor)):
        raise ValueError(f"{name} must contain only finite values.")
    return tensor


def _validate_raw_weights(weights: Any, *, name: str = "weights") -> "torch.Tensor":
    tensor = _as_vector(weights, name=name)
    if torch.any(tensor < 0):
        raise ValueError(f"{name} must be nonnegative.")
    return tensor


def _validate_eps(eps: float) -> float:
    if not isinstance(eps, int | float) or float(eps) <= 0:
        raise ValueError("eps must be a positive finite scalar.")
    eps_float = float(eps)
    if not torch.isfinite(torch.tensor(eps_float)):
        raise ValueError("eps must be a positive finite scalar.")
    return eps_float


def positive_density_weights(logits: Any, eps: float = 1e-8) -> "torch.Tensor":
    """Map raw density logits to strictly positive weights.

    Args:
        logits: Raw density-head outputs shaped ``[batch]`` or ``[batch, 1]``.
        eps: Positive offset added after softplus.

    Returns:
        Positive weights shaped ``[batch]``.
    """

    eps_float = _validate_eps(eps)
    logit_vec = _as_vector(logits, name="logits")
    return F.softplus(logit_vec) + eps_float


def clip_density_weights(weights: Any, max_weight: float) -> "torch.Tensor":
    """Clip finite nonnegative density weights to ``max_weight``."""

    weight_vec = _validate_raw_weights(weights)
    if not isinstance(max_weight, int | float) or float(max_weight) <= 0:
        raise ValueError("max_weight must be positive.")
    max_weight_float = float(max_weight)
    if not torch.isfinite(torch.tensor(max_weight_float, dtype=weight_vec.dtype)):
        raise ValueError("max_weight must be positive and finite.")
    return torch.clamp(weight_vec, max=max_weight_float)


def self_normalize_weights(weights: Any, eps: float = 1e-8) -> "torch.Tensor":
    """Normalize nonnegative weights by their batch mean.

    The result has mean approximately one. All-zero weights are rejected because
    they cannot represent a usable density ratio.
    """

    eps_float = _validate_eps(eps)
    weight_vec = _validate_raw_weights(weights)
    mean_weight = weight_vec.mean()
    if mean_weight <= eps_float:
        raise ValueError("weights must not be all zero for self-normalisation.")
    return weight_vec / mean_weight


def make_density_weights(
    logits: Any,
    max_weight: float | None = None,
    normalize: bool = True,
    eps: float = 1e-8,
) -> "torch.Tensor":
    """Create stabilized density weights from raw logits.

    This applies softplus positivity, optional clipping, and optional batch
    self-normalisation in the order specified by the project notes.
    """

    weights = positive_density_weights(logits, eps=eps)
    if max_weight is not None:
        weights = clip_density_weights(weights, max_weight=max_weight)
    if normalize:
        weights = self_normalize_weights(weights, eps=eps)
    return weights


def effective_sample_size(weights: Any, eps: float = 1e-8) -> "torch.Tensor":
    """Compute ESS ``(sum_i w_i)^2 / (sum_i w_i^2 + eps)``."""

    eps_float = _validate_eps(eps)
    weight_vec = _validate_raw_weights(weights)
    numerator = weight_vec.sum().square()
    denominator = weight_vec.square().sum() + eps_float
    return numerator / denominator


def weight_entropy(weights: Any, eps: float = 1e-8) -> "torch.Tensor":
    """Compute entropy of weights after normalizing them to probabilities."""

    eps_float = _validate_eps(eps)
    weight_vec = _validate_raw_weights(weights)
    total = weight_vec.sum()
    if total <= eps_float:
        raise ValueError("weights must have positive sum for entropy.")
    probs = weight_vec / total
    return -(probs * torch.log(probs.clamp_min(eps_float))).sum()
