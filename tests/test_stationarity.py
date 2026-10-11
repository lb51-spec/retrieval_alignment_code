"""Tests for discounted stationarity residual utilities."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from retrieval_alignment.stationarity import (
    initial_state_term,
    masked_expected_test_value,
    squared_stationarity_loss,
    stationarity_moment,
    transition_stationarity_terms,
    weighted_stationarity_mean,
)


def test_masked_expected_test_value_shape() -> None:
    values = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    probs = torch.tensor([[0.5, 0.5], [0.25, 0.75]])
    mask = torch.tensor([[True, True], [True, True]])

    expected = masked_expected_test_value(values, probs, mask)

    assert expected.shape == (2,)


def test_invalid_candidates_contribute_zero() -> None:
    values = torch.tensor([[2.0, 100.0]])
    probs = torch.tensor([[0.5, 999.0]])
    mask = torch.tensor([[True, False]])

    expected = masked_expected_test_value(values, probs, mask)

    assert expected.item() == pytest.approx(1.0)


def test_all_invalid_rows_raise_value_error() -> None:
    with pytest.raises(ValueError, match="at least one valid candidate"):
        masked_expected_test_value(
            torch.tensor([[1.0, 2.0]]),
            torch.tensor([[0.5, 0.5]]),
            torch.tensor([[False, False]]),
        )


def test_transition_term_equals_gamma_next_minus_current() -> None:
    terms = transition_stationarity_terms(
        current_f=torch.tensor([1.0, 2.0]),
        next_expected_f=torch.tensor([3.0, 4.0]),
        gamma=0.5,
    )

    assert terms.tolist() == pytest.approx([0.5, 0.0])


def test_gamma_validation() -> None:
    with pytest.raises(ValueError, match="gamma"):
        transition_stationarity_terms(torch.tensor([1.0]), torch.tensor([2.0]), gamma=-0.1)

    with pytest.raises(ValueError, match="gamma"):
        initial_state_term(torch.tensor([1.0]), gamma=1.1)


def test_weighted_stationarity_mean_shape_and_value() -> None:
    mean = weighted_stationarity_mean(
        weights=torch.tensor([[1.0], [2.0]]),
        transition_terms=torch.tensor([3.0, 4.0]),
    )

    assert mean.ndim == 0
    assert mean.item() == pytest.approx(5.5)


def test_negative_weights_raise_value_error() -> None:
    with pytest.raises(ValueError, match="nonnegative"):
        weighted_stationarity_mean(torch.tensor([1.0, -1.0]), torch.tensor([1.0, 2.0]))


def test_initial_state_term_value() -> None:
    term = initial_state_term(torch.tensor([[2.0], [4.0]]), gamma=0.75)

    assert term.item() == pytest.approx(0.75)


def test_full_stationarity_moment_without_initial_term() -> None:
    moment = stationarity_moment(
        weights=torch.tensor([1.0, 2.0]),
        current_f=torch.tensor([1.0, 2.0]),
        next_candidate_f_values=torch.tensor([[3.0, 5.0], [4.0, 10.0]]),
        next_candidate_probs=torch.tensor([[1.0, 0.0], [0.5, 0.5]]),
        next_mask=torch.tensor([[True, False], [True, False]]),
        gamma=0.5,
    )

    assert moment.item() == pytest.approx(-0.75)


def test_full_stationarity_moment_with_initial_term() -> None:
    moment = stationarity_moment(
        weights=torch.tensor([1.0, 2.0]),
        current_f=torch.tensor([1.0, 2.0]),
        next_candidate_f_values=torch.tensor([[3.0, 5.0], [4.0, 10.0]]),
        next_candidate_probs=torch.tensor([[1.0, 0.0], [0.5, 0.5]]),
        next_mask=torch.tensor([[True, False], [True, False]]),
        gamma=0.5,
        initial_expected_f=torch.tensor([2.0, 4.0]),
    )

    assert moment.item() == pytest.approx(0.75)


def test_squared_loss_returns_scalar_moment_squared() -> None:
    loss = squared_stationarity_loss(torch.tensor(3.0))

    assert loss.ndim == 0
    assert loss.item() == pytest.approx(9.0)


def test_non_finite_inputs_raise_value_error() -> None:
    with pytest.raises(ValueError, match="finite"):
        masked_expected_test_value(
            torch.tensor([[1.0, float("nan")]]),
            torch.tensor([[1.0, 0.0]]),
            torch.tensor([[True, True]]),
        )

    with pytest.raises(ValueError, match="finite"):
        squared_stationarity_loss(torch.tensor(float("inf")))


def test_shape_mismatch_raises_value_error() -> None:
    with pytest.raises(ValueError, match="shape"):
        masked_expected_test_value(
            torch.tensor([[1.0, 2.0]]),
            torch.tensor([[1.0]]),
            torch.tensor([[True, True]]),
        )

    with pytest.raises(ValueError, match="shape"):
        transition_stationarity_terms(torch.tensor([1.0, 2.0]), torch.tensor([1.0]), gamma=0.9)
