"""Tests for synthetic evaluation diagnostics."""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from retrieval_alignment.config import ModelConfig, SyntheticDatasetConfig, default_smoke_profile  # noqa: E402
from retrieval_alignment.evaluation import (  # noqa: E402
    EvaluationConfig,
    EvaluationSummary,
    evaluate_synthetic_model,
    validate_evaluation_config,
)
from retrieval_alignment.networks import build_networks  # noqa: E402
from retrieval_alignment.synthetic import make_synthetic_offline_data  # noqa: E402


def _data_and_model():
    data = make_synthetic_offline_data(
        SyntheticDatasetConfig(num_transitions=24, num_actions=9, state_dim=4, action_dim=3)
    )
    torch.manual_seed(0)
    model = build_networks(ModelConfig(state_dim=4, action_dim=3, embedding_dim=8))
    return data, model


def test_evaluation_config_validation() -> None:
    assert validate_evaluation_config(EvaluationConfig(num_batches=1)).num_batches == 1

    with pytest.raises(ValueError, match="num_batches"):
        validate_evaluation_config(EvaluationConfig(num_batches=0))
    with pytest.raises(ValueError, match="batch_size"):
        validate_evaluation_config(EvaluationConfig(batch_size=0))
    with pytest.raises(ValueError, match="k"):
        validate_evaluation_config(EvaluationConfig(k=0))
    with pytest.raises(ValueError, match="temperature"):
        validate_evaluation_config(EvaluationConfig(temperature=0.0))
    with pytest.raises(ValueError, match="index_backend"):
        validate_evaluation_config(EvaluationConfig(index_backend="bad"))
    with pytest.raises(ValueError, match="index_metric"):
        validate_evaluation_config(EvaluationConfig(index_metric="bad"))


def test_evaluate_synthetic_model_returns_finite_summary() -> None:
    data, model = _data_and_model()
    profile = replace(default_smoke_profile(), batch_size=4, num_actions=9, k=3)

    summary = evaluate_synthetic_model(
        data,
        model,
        EvaluationConfig(num_batches=2, batch_size=4, k=3),
        profile,
        task_name="unit_eval",
    )

    assert isinstance(summary, EvaluationSummary)
    assert summary.task_name == "unit_eval"
    assert summary.num_batches == 2
    assert summary.synthetic_only is True
    assert summary.benchmark_claim is False
    for value in summary.metrics.values():
        assert torch.isfinite(torch.tensor(value))
    assert "mean_bellman_loss" in summary.metrics
    assert "mean_exact_backend_recall" in summary.metrics
