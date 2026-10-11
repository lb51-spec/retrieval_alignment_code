"""Tests for before/after synthetic smoke evaluation helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from retrieval_alignment.config import ModelConfig, SyntheticDatasetConfig
from retrieval_alignment.evaluate import BeforeAfterSmokeResult, run_before_after_smoke_evaluation
from retrieval_alignment.networks import build_networks
from retrieval_alignment.pipeline import PipelineSmokeResult
from retrieval_alignment.synthetic import make_synthetic_offline_data
from retrieval_alignment.trainer import TrainLoopSmokeResult, default_smoke_alternating_schedule


def _data_model_optimizer(seed: int = 246):
    data = make_synthetic_offline_data(
        SyntheticDatasetConfig(
            num_transitions=40,
            state_dim=4,
            action_dim=3,
            num_actions=10,
            seed=seed,
            terminal_prob=0.2,
        )
    )
    torch.manual_seed(seed)
    model = build_networks(
        ModelConfig(
            state_dim=4,
            action_dim=3,
            embedding_dim=8,
            hidden_dims=(16,),
            pair_feature_mode="concat_product",
        )
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    return data, model, optimizer


def test_before_after_smoke_returns_result_types() -> None:
    data, model, optimizer = _data_model_optimizer()

    result = run_before_after_smoke_evaluation(
        data=data,
        model=model,
        optimizer=optimizer,
        num_steps=2,
        batch_size=6,
        k=4,
        gamma=0.95,
        generator=torch.Generator().manual_seed(31),
    )

    assert isinstance(result, BeforeAfterSmokeResult)
    assert isinstance(result.before, PipelineSmokeResult)
    assert isinstance(result.train_loop, TrainLoopSmokeResult)
    assert isinstance(result.after, PipelineSmokeResult)


def test_before_after_smoke_deltas_have_expected_finite_keys() -> None:
    data, model, optimizer = _data_model_optimizer()

    result = run_before_after_smoke_evaluation(
        data=data,
        model=model,
        optimizer=optimizer,
        num_steps=2,
        batch_size=6,
        k=4,
        gamma=0.95,
        generator=torch.Generator().manual_seed(31),
    )

    assert set(result.deltas) == {
        "bellman_loss_delta",
        "stationarity_loss_delta",
        "ess_delta",
        "mean_recall_delta",
        "exact_backend_recall_mean_delta",
        "exact_backend_overlap_fraction_mean_delta",
        "exact_backend_best_value_gap_mean_delta",
    }
    for value in result.deltas.values():
        assert isinstance(value, float)
        assert torch.isfinite(torch.tensor(value))


def test_before_after_smoke_retrieval_audit_deltas_are_finite() -> None:
    data, model, optimizer = _data_model_optimizer()

    result = run_before_after_smoke_evaluation(
        data=data,
        model=model,
        optimizer=optimizer,
        num_steps=2,
        batch_size=6,
        k=4,
        gamma=0.95,
        generator=torch.Generator().manual_seed(31),
    )

    for key in (
        "before_exact_backend_recall_mean",
        "after_exact_backend_recall_mean",
        "before_exact_backend_overlap_fraction_mean",
        "after_exact_backend_overlap_fraction_mean",
        "before_exact_backend_best_value_gap_mean",
        "after_exact_backend_best_value_gap_mean",
    ):
        assert key in result.diagnostics
        assert torch.isfinite(torch.tensor(result.diagnostics[key]))

    for key in (
        "exact_backend_recall_mean_delta",
        "exact_backend_overlap_fraction_mean_delta",
        "exact_backend_best_value_gap_mean_delta",
    ):
        assert key in result.deltas
        assert torch.isfinite(torch.tensor(result.deltas[key]))


def test_before_after_smoke_works_with_alternating_schedule() -> None:
    data, model, optimizer = _data_model_optimizer()

    result = run_before_after_smoke_evaluation(
        data=data,
        model=model,
        optimizer=optimizer,
        num_steps=3,
        batch_size=6,
        k=4,
        gamma=0.95,
        generator=torch.Generator().manual_seed(31),
        alternating_schedule=default_smoke_alternating_schedule(),
        assert_gradient_flow=True,
    )

    assert isinstance(result.train_loop, TrainLoopSmokeResult)
    assert result.diagnostics["alternating_schedule_used"] is True
    assert result.train_loop.diagnostics["assert_gradient_flow"] is True


def test_before_after_smoke_invalid_num_steps_raises_value_error() -> None:
    data, model, optimizer = _data_model_optimizer()

    with pytest.raises(ValueError, match="num_steps"):
        run_before_after_smoke_evaluation(
            data=data,
            model=model,
            optimizer=optimizer,
            num_steps=0,
            batch_size=6,
            k=4,
            gamma=0.95,
        )


def test_before_after_smoke_invalid_gamma_raises_value_error() -> None:
    data, model, optimizer = _data_model_optimizer()

    with pytest.raises(ValueError, match="gamma"):
        run_before_after_smoke_evaluation(
            data=data,
            model=model,
            optimizer=optimizer,
            num_steps=2,
            batch_size=6,
            k=4,
            gamma=1.5,
        )


def test_before_after_smoke_does_not_assert_loss_decrease() -> None:
    data, model, optimizer = _data_model_optimizer()

    result = run_before_after_smoke_evaluation(
        data=data,
        model=model,
        optimizer=optimizer,
        num_steps=1,
        batch_size=6,
        k=4,
        gamma=0.95,
        generator=torch.Generator().manual_seed(31),
    )

    assert "bellman_loss_delta" in result.deltas
    assert result.diagnostics["benchmark_evaluation"] is False
