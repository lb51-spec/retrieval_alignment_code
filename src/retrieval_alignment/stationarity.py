"""Discounted stationarity residual utilities for DW-MQL+ANN.

The density-correction side of DW-MQL+ANN uses the DICE/MWL-style identity
from ``docs/equations.md``:

``E_mu[w(s,a) (gamma f(s', pi_k) - f(s,a))]
  + (1 - gamma) E_d0[f(s0, pi_k)] = 0``.

For ANN-truncated policies, ``f(s, pi_k)`` is a masked candidate expectation
over the retrieved candidate set. This module provides only pure tensor
utilities for that stationarity moment. It is independent of FAISS, networks,
replay buffers, and training loops.
"""

from __future__ import annotations

from typing import Any

try:  # pragma: no cover - exercised when project dependencies are installed.
    import torch
except ModuleNotFoundError:  # pragma: no cover - keeps imports clean pre-install.
    torch = None  # type: ignore[assignment]


def _require_torch() -> None:
    if torch is None:
        raise ModuleNotFoundError("PyTorch is required to compute stationarity residuals.")


def _as_float_tensor(values: Any, *, name: str) -> "torch.Tensor":
    _require_torch()
    tensor = torch.as_tensor(values)
    if not torch.is_floating_point(tensor):
        tensor = tensor.float()
    if tensor.numel() == 0:
        raise ValueError(f"{name} must be non-empty.")
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
    if not torch.isfinite(torch.tensor(gamma_float)):
        raise ValueError("gamma must be finite.")
    return gamma_float


def _validate_candidate_tensors(
    candidate_f_values: Any,
    candidate_probs: Any,
    mask: Any,
) -> tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
    f_values = _as_float_tensor(candidate_f_values, name="candidate_f_values")
    probs = _as_float_tensor(candidate_probs, name="candidate_probs").to(device=f_values.device)
    mask_tensor = torch.as_tensor(mask, dtype=torch.bool, device=f_values.device)

    if f_values.ndim != 2:
        raise ValueError("candidate_f_values must have shape [batch, num_candidates].")
    if probs.shape != f_values.shape:
        raise ValueError(
            "candidate_probs shape must match candidate_f_values shape: "
            f"got {tuple(probs.shape)} and {tuple(f_values.shape)}."
        )
    if mask_tensor.shape != f_values.shape:
        raise ValueError(
            f"mask shape {tuple(mask_tensor.shape)} must match candidate_f_values shape {tuple(f_values.shape)}."
        )
    if torch.any(~mask_tensor.any(dim=1)):
        raise ValueError("Each batch row must contain at least one valid candidate.")
    return f_values, probs, mask_tensor


def masked_expected_test_value(
    candidate_f_values: Any,
    candidate_probs: Any,
    mask: Any,
) -> "torch.Tensor":
    """Compute ``E_{a~pi_k(.|s)} f(s,a)`` over valid candidates.

    Invalid candidates contribute exactly zero, even if their probability input
    is nonzero. Candidate probabilities are expected to already be masked and
    normalized by ``src/retrieval_alignment/policies.py``.
    """

    f_values, probs, mask_tensor = _validate_candidate_tensors(
        candidate_f_values,
        candidate_probs,
        mask,
    )
    valid = mask_tensor.to(dtype=f_values.dtype)
    return (f_values * probs * valid).sum(dim=1)


def transition_stationarity_terms(
    current_f: Any,
    next_expected_f: Any,
    gamma: float,
) -> "torch.Tensor":
    """Return per-transition terms ``gamma * f(s', pi_k) - f(s,a)``."""

    gamma_float = _validate_gamma(gamma)
    current_vec = _as_vector(current_f, name="current_f")
    next_vec = _as_vector(next_expected_f, name="next_expected_f").to(device=current_vec.device)
    if next_vec.shape != current_vec.shape:
        raise ValueError(
            "next_expected_f shape must match current_f shape: "
            f"got {tuple(next_vec.shape)} and {tuple(current_vec.shape)}."
        )
    return gamma_float * next_vec - current_vec


def weighted_stationarity_mean(weights: Any, transition_terms: Any) -> "torch.Tensor":
    """Compute ``mean(weights * transition_terms)`` for stationarity."""

    weight_vec = _as_vector(weights, name="weights")
    term_vec = _as_vector(transition_terms, name="transition_terms").to(device=weight_vec.device)
    if term_vec.shape != weight_vec.shape:
        raise ValueError(
            f"transition_terms shape {tuple(term_vec.shape)} must match weights shape {tuple(weight_vec.shape)}."
        )
    if torch.any(weight_vec < 0):
        raise ValueError("weights must be nonnegative.")
    return (weight_vec * term_vec).mean()


def initial_state_term(initial_expected_f: Any, gamma: float) -> "torch.Tensor":
    """Compute ``(1 - gamma) * mean(f(s0, pi_k))``."""

    gamma_float = _validate_gamma(gamma)
    initial_vec = _as_vector(initial_expected_f, name="initial_expected_f")
    return (1.0 - gamma_float) * initial_vec.mean()


def stationarity_moment(
    weights: Any,
    current_f: Any,
    next_candidate_f_values: Any,
    next_candidate_probs: Any,
    next_mask: Any,
    gamma: float,
    initial_expected_f: Any | None = None,
) -> "torch.Tensor":
    """Compute the DICE/MWL stationarity moment under an ANN-truncated policy.

    The transition part is
    ``mean(weights * (gamma * E_pi_k[f(s', a')] - f(s,a)))``.
    If ``initial_expected_f`` is provided, the initial-state term
    ``(1 - gamma) * mean(initial_expected_f)`` is added.
    """

    next_expected_f = masked_expected_test_value(
        next_candidate_f_values,
        next_candidate_probs,
        next_mask,
    )
    terms = transition_stationarity_terms(current_f, next_expected_f, gamma)
    moment = weighted_stationarity_mean(weights, terms)
    if initial_expected_f is not None:
        moment = moment + initial_state_term(initial_expected_f, gamma)
    return moment


def squared_stationarity_loss(moment: Any) -> "torch.Tensor":
    """Return the scalar squared stationarity moment."""

    moment_tensor = _as_float_tensor(moment, name="moment")
    if moment_tensor.ndim != 0:
        raise ValueError("moment must be a scalar tensor.")
    return moment_tensor.square()
