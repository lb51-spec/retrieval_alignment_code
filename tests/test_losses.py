"""Tests for ANN-truncated Bellman utility functions."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from retrieval_alignment.losses import (
    ann_bellman_residual,
    bellman_residual,
    bellman_target,
    masked_expected_q_value,
    weighted_squared_bellman_loss,
)


def test_masked_expected_q_value_shape() -> None:
    q_values = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    probs = torch.tensor([[0.25, 0.75], [0.5, 0.5]])
    mask = torch.tensor([[True, True], [True, True]])

    expected = masked_expected_q_value(q_values, probs, mask)

    assert expected.shape == (2,)


def test_invalid_candidates_contribute_zero() -> None:
    q_values = torch.tensor([[1.0, 1000.0]])
    probs = torch.tensor([[1.0, 1.0]])
    mask = torch.tensor([[True, False]])

    expected = masked_expected_q_value(q_values, probs, mask)

    assert expected.item() == pytest.approx(1.0)


def test_invalid_probabilities_do_not_affect_expected_value() -> None:
    q_values = torch.tensor([[2.0, 100.0, -50.0]])
    probs = torch.tensor([[0.5, 999.0, 999.0]])
    mask = torch.tensor([[True, False, False]])

    expected = masked_expected_q_value(q_values, probs, mask)

    assert expected.item() == pytest.approx(1.0)


def test_terminal_done_removes_bootstrap_term() -> None:
    rewards = torch.tensor([1.0, 2.0])
    dones = torch.tensor([False, True])
    expected_next_q = torch.tensor([10.0, 10.0])

    target = bellman_target(rewards, dones, expected_next_q, gamma=0.5)

    assert target.tolist() == pytest.approx([6.0, 2.0])


def test_gamma_validation() -> None:
    with pytest.raises(ValueError, match="gamma"):
        bellman_target(torch.tensor([1.0]), torch.tensor([0.0]), torch.tensor([2.0]), gamma=-0.1)

    with pytest.raises(ValueError, match="gamma"):
        bellman_target(torch.tensor([1.0]), torch.tensor([0.0]), torch.tensor([2.0]), gamma=1.1)


def test_reward_and_current_q_shape_handling() -> None:
    rewards = torch.tensor([[1.0], [2.0]])
    dones = torch.tensor([[0.0], [0.0]])
    expected_next_q = torch.tensor([[3.0], [4.0]])
    current_q = torch.tensor([[0.5], [1.0]])

    target = bellman_target(rewards, dones, expected_next_q, gamma=1.0)
    residual = bellman_residual(target, current_q)

    assert target.shape == (2,)
    assert residual.tolist() == pytest.approx([3.5, 5.0])


def test_ann_bellman_residual_shape() -> None:
    residual = ann_bellman_residual(
        current_q=torch.tensor([0.5, 1.0]),
        rewards=torch.tensor([1.0, 2.0]),
        dones=torch.tensor([0.0, 1.0]),
        candidate_q_values=torch.tensor([[2.0, 4.0], [10.0, 20.0]]),
        candidate_probs=torch.tensor([[0.5, 0.5], [1.0, 0.0]]),
        mask=torch.tensor([[True, True], [True, False]]),
        gamma=0.5,
    )

    assert residual.shape == (2,)


def test_weighted_and_unweighted_loss() -> None:
    residual = torch.tensor([1.0, 2.0])
    weights = torch.tensor([0.5, 2.0])

    unweighted = weighted_squared_bellman_loss(residual)
    weighted = weighted_squared_bellman_loss(residual, weights)

    assert unweighted.item() == pytest.approx(2.5)
    assert weighted.item() == pytest.approx(4.25)


def test_negative_weights_raise_value_error() -> None:
    with pytest.raises(ValueError, match="nonnegative"):
        weighted_squared_bellman_loss(torch.tensor([1.0, 2.0]), torch.tensor([1.0, -1.0]))


def test_non_finite_inputs_raise_value_error() -> None:
    with pytest.raises(ValueError, match="finite"):
        masked_expected_q_value(
            torch.tensor([[1.0, float("nan")]]),
            torch.tensor([[0.5, 0.5]]),
            torch.tensor([[True, True]]),
        )

    with pytest.raises(ValueError, match="finite"):
        weighted_squared_bellman_loss(torch.tensor([1.0, float("inf")]))


def test_shape_mismatch_raises_clear_value_error() -> None:
    with pytest.raises(ValueError, match="shape"):
        masked_expected_q_value(
            torch.tensor([[1.0, 2.0]]),
            torch.tensor([[1.0]]),
            torch.tensor([[True, True]]),
        )

    with pytest.raises(ValueError, match="shape"):
        bellman_residual(torch.tensor([1.0, 2.0]), torch.tensor([1.0]))
