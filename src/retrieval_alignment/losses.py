"""ANN-truncated Bellman target and residual utilities.

The utilities here implement the Bellman residual from ``docs/equations.md``:

``delta_q^ann(s, a, r, s') = r + gamma E_{a'~pi_k(.|s')} q(s', a') - q(s, a)``.

They are deliberately small tensor functions. They do not know about FAISS,
candidate construction, neural networks, replay buffers, density-ratio heads,
or training loops. Candidate probabilities are expected to already represent
the masked ANN-truncated policy from ``src/retrieval_alignment/policies.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

try:  # pragma: no cover - exercised when the project dependencies are installed.
    import torch
except ModuleNotFoundError:  # pragma: no cover - keeps imports clean pre-install.
    torch = None  # type: ignore[assignment]


@dataclass(slots=True)
class LossOutput:
    """Container for future loss values and diagnostics."""

    loss: "torch.Tensor | Any"
    components: dict[str, "torch.Tensor | Any"] = field(default_factory=dict)


def _require_torch() -> None:
    if torch is None:
        raise ModuleNotFoundError("PyTorch is required to compute DW-MQL losses.")


def _as_float_tensor(values: Any, *, name: str) -> "torch.Tensor":
    _require_torch()
    tensor = torch.as_tensor(values)
    if not torch.is_floating_point(tensor):
        tensor = tensor.float()
    if torch.any(~torch.isfinite(tensor)):
        raise ValueError(f"{name} must contain only finite values.")
    return tensor


def _as_vector(values: Any, *, name: str) -> "torch.Tensor":
    tensor = _as_float_tensor(values, name=name)
    if tensor.ndim == 2 and tensor.shape[1] == 1:
        tensor = tensor.squeeze(1)
    if tensor.ndim != 1:
        raise ValueError(f"{name} must have shape [batch] or [batch, 1].")
    return tensor


def _validate_gamma(gamma: float) -> float:
    if not isinstance(gamma, int | float):
        raise ValueError("gamma must be a scalar in [0, 1].")
    gamma_float = float(gamma)
    if not 0.0 <= gamma_float <= 1.0:
        raise ValueError("gamma must be in [0, 1].")
    return gamma_float


def _validate_candidate_tensors(
    candidate_q_values: Any,
    candidate_probs: Any,
    mask: Any,
) -> tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
    q_values = _as_float_tensor(candidate_q_values, name="candidate_q_values")
    probs = _as_float_tensor(candidate_probs, name="candidate_probs").to(device=q_values.device)
    mask_tensor = torch.as_tensor(mask, dtype=torch.bool, device=q_values.device)

    if q_values.ndim != 2:
        raise ValueError("candidate_q_values must have shape [batch, num_candidates].")
    if probs.shape != q_values.shape:
        raise ValueError(
            "candidate_probs shape must match candidate_q_values shape: "
            f"got {tuple(probs.shape)} and {tuple(q_values.shape)}."
        )
    if mask_tensor.shape != q_values.shape:
        raise ValueError(
            "mask shape must match candidate_q_values shape: "
            f"got {tuple(mask_tensor.shape)} and {tuple(q_values.shape)}."
        )
    return q_values, probs, mask_tensor


def masked_expected_q_value(
    candidate_q_values: Any,
    candidate_probs: Any,
    mask: Any,
) -> "torch.Tensor":
    """Compute ``E_{a'~pi_k(.|s')} Q(s', a')`` over valid candidates.

    Args:
        candidate_q_values: Candidate Q-values with shape
            ``[batch, num_candidates]``.
        candidate_probs: Candidate probabilities with the same shape. These
            should already be the ANN-truncated policy probabilities.
        mask: Boolean candidate-validity mask with the same shape.

    Returns:
        Tensor with shape ``[batch]``. Invalid candidates contribute exactly
        zero, even if their input probability is nonzero.
    """

    q_values, probs, mask_tensor = _validate_candidate_tensors(
        candidate_q_values,
        candidate_probs,
        mask,
    )
    valid = mask_tensor.to(dtype=q_values.dtype)
    return (q_values * probs * valid).sum(dim=1)


def bellman_target(
    rewards: Any,
    dones: Any,
    expected_next_q: Any,
    gamma: float,
) -> "torch.Tensor":
    """Compute the terminal-safe ANN-truncated Bellman target.

    The target is ``reward + gamma * (1 - done) * expected_next_q`` and always
    returns shape ``[batch]``.
    """

    gamma_float = _validate_gamma(gamma)
    reward_vec = _as_vector(rewards, name="rewards")
    done_vec = _as_vector(dones, name="dones").to(device=reward_vec.device)
    next_q_vec = _as_vector(expected_next_q, name="expected_next_q").to(device=reward_vec.device)

    if done_vec.shape != reward_vec.shape:
        raise ValueError(f"dones shape {tuple(done_vec.shape)} must match rewards shape {tuple(reward_vec.shape)}.")
    if next_q_vec.shape != reward_vec.shape:
        raise ValueError(
            "expected_next_q shape must match rewards shape: "
            f"got {tuple(next_q_vec.shape)} and {tuple(reward_vec.shape)}."
        )
    if torch.any((done_vec < 0) | (done_vec > 1)):
        raise ValueError("dones must be boolean-like or numeric values in [0, 1].")

    return reward_vec + gamma_float * (1.0 - done_vec) * next_q_vec


def bellman_residual(target: Any, current_q: Any) -> "torch.Tensor":
    """Compute ``target - Q(s, a)`` with shape ``[batch]``."""

    target_vec = _as_vector(target, name="target")
    current_q_vec = _as_vector(current_q, name="current_q").to(device=target_vec.device)
    if current_q_vec.shape != target_vec.shape:
        raise ValueError(
            f"current_q shape {tuple(current_q_vec.shape)} must match target shape {tuple(target_vec.shape)}."
        )
    return target_vec - current_q_vec


def ann_bellman_residual(
    current_q: Any,
    rewards: Any,
    dones: Any,
    candidate_q_values: Any,
    candidate_probs: Any,
    mask: Any,
    gamma: float,
) -> "torch.Tensor":
    """Compute the full ANN-truncated Bellman residual in one call."""

    expected_next_q = masked_expected_q_value(candidate_q_values, candidate_probs, mask)
    target = bellman_target(rewards, dones, expected_next_q, gamma)
    return bellman_residual(target, current_q)


def weighted_squared_bellman_loss(residual: Any, weights: Any | None = None) -> "torch.Tensor":
    """Return ``mean(weights * residual**2)``.

    If ``weights`` is ``None``, this is the unweighted mean squared Bellman
    residual. Weights must be finite, nonnegative, and shaped ``[batch]`` or
    ``[batch, 1]``.
    """

    residual_vec = _as_vector(residual, name="residual")
    squared = residual_vec.square()
    if weights is None:
        return squared.mean()

    weight_vec = _as_vector(weights, name="weights").to(device=residual_vec.device)
    if weight_vec.shape != residual_vec.shape:
        raise ValueError(
            f"weights shape {tuple(weight_vec.shape)} must match residual shape {tuple(residual_vec.shape)}."
        )
    if torch.any(weight_vec < 0):
        raise ValueError("weights must be nonnegative.")
    return (weight_vec * squared).mean()
