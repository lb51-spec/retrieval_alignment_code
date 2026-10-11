"""Tests for forward-only pipeline smoke wiring."""

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
from retrieval_alignment.networks import build_networks
from retrieval_alignment.pipeline import PipelineSmokeResult, run_pipeline_smoke_step
from retrieval_alignment.synthetic import make_synthetic_offline_data


def _data_and_model(seed: int = 123):
    config = SyntheticDatasetConfig(
        num_transitions=32,
        state_dim=4,
        action_dim=3,
        num_actions=10,
        seed=seed,
        terminal_prob=0.2,
    )
    data = make_synthetic_offline_data(config)
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
    return data, model


def _result(seed: int = 123) -> PipelineSmokeResult:
    data, model = _data_and_model(seed)
    return run_pipeline_smoke_step(
        data,
        model,
        batch_size=6,
        k=4,
        gamma=0.95,
        generator=torch.Generator().manual_seed(999),
        temperature=1.0,
        density_max_weight=10.0,
    )


def test_smoke_step_returns_result() -> None:
    assert isinstance(_result(), PipelineSmokeResult)


def test_smoke_step_works_with_exact_dot_backend() -> None:
    result = _result()

    assert result.diagnostics["index_backend"] == "exact"
    assert result.diagnostics["index_metric"] == "dot"
    assert result.diagnostics["selected_index_backend"] == "exact"
    assert result.diagnostics["selected_index_metric"] == "dot"
    assert result.diagnostics["exact_backend_recall_mean"] == pytest.approx(1.0)
    assert result.exact_backend_recall_at_k is not None
    assert torch.allclose(result.exact_backend_recall_at_k, torch.ones_like(result.exact_backend_recall_at_k))


def test_smoke_step_works_with_exact_cosine_backend() -> None:
    data, model = _data_and_model()

    result = run_pipeline_smoke_step(
        data,
        model,
        batch_size=6,
        k=4,
        gamma=0.95,
        generator=torch.Generator().manual_seed(999),
        index_backend="exact",
        index_metric="cosine",
    )

    assert isinstance(result, PipelineSmokeResult)
    assert result.diagnostics["index_backend"] == "exact"
    assert result.diagnostics["index_metric"] == "cosine"
    assert result.diagnostics["selected_index_backend"] == "exact"
    assert result.diagnostics["selected_index_metric"] == "cosine"
    assert result.diagnostics["exact_backend_recall_mean"] == pytest.approx(1.0)
    assert torch.isfinite(result.bellman_loss)


def test_exact_backend_overlap_fraction_mean_is_finite() -> None:
    result = _result()

    assert result.exact_backend_overlap_fraction is not None
    assert torch.isfinite(result.exact_backend_overlap_fraction).all()
    assert torch.isfinite(torch.tensor(result.diagnostics["exact_backend_overlap_fraction_mean"]))
    assert result.diagnostics["exact_backend_overlap_fraction_mean"] == pytest.approx(1.0)


def test_exact_backend_best_value_gap_mean_is_finite() -> None:
    result = _result()

    assert result.exact_backend_best_value_gap is not None
    assert torch.isfinite(result.exact_backend_best_value_gap).all()
    assert torch.isfinite(torch.tensor(result.diagnostics["exact_backend_best_value_gap_mean"]))
    assert result.diagnostics["exact_backend_best_value_gap_mean"] == pytest.approx(0.0)


def test_candidate_shapes() -> None:
    result = _result()

    assert result.candidate_ids.shape[0] == result.batch_size
    assert result.candidate_ids.shape == result.candidate_mask.shape
    assert result.candidate_probs.shape == result.candidate_ids.shape


def test_candidate_probs_sum_to_one_over_valid_candidates() -> None:
    result = _result()

    assert torch.allclose(result.candidate_probs.sum(dim=1), torch.ones(result.batch_size))
    assert torch.equal(result.candidate_probs[result.candidate_mask == 0], torch.zeros_like(result.candidate_probs[result.candidate_mask == 0]))


def test_q_and_bellman_shapes() -> None:
    result = _result()

    assert result.current_q.shape == (result.batch_size,)
    assert result.next_candidate_q.shape == result.candidate_ids.shape
    assert result.bellman_target.shape == (result.batch_size,)
    assert result.bellman_residual.shape == (result.batch_size,)


def test_scalar_losses_and_diagnostics_are_finite() -> None:
    result = _result()

    assert result.bellman_loss.ndim == 0
    assert torch.isfinite(result.bellman_loss)
    assert result.ess.ndim == 0
    assert torch.isfinite(result.ess)
    assert result.weight_entropy.ndim == 0
    assert torch.isfinite(result.weight_entropy)
    assert result.stationarity_moment.ndim == 0
    assert torch.isfinite(result.stationarity_moment)
    assert result.stationarity_loss.ndim == 0
    assert torch.isfinite(result.stationarity_loss)


def test_density_weights_shape() -> None:
    result = _result()

    assert result.density_weights.shape == (result.batch_size,)


def test_recall_at_k_is_finite_and_in_unit_interval() -> None:
    result = _result()

    assert result.recall_at_k.shape == (result.batch_size,)
    assert torch.isfinite(result.recall_at_k).all()
    assert torch.all((result.recall_at_k >= 0) & (result.recall_at_k <= 1))


def test_deterministic_behavior_with_fixed_seed_and_model_seed() -> None:
    first = _result(seed=321)
    second = _result(seed=321)

    assert torch.equal(first.candidate_ids, second.candidate_ids)
    assert torch.equal(first.candidate_mask, second.candidate_mask)
    assert torch.allclose(first.current_q, second.current_q)
    assert torch.allclose(first.bellman_residual, second.bellman_residual)
    assert torch.allclose(first.density_weights, second.density_weights)


def test_invalid_k_raises_value_error() -> None:
    data, model = _data_and_model()

    with pytest.raises(ValueError, match="k"):
        run_pipeline_smoke_step(data, model, batch_size=4, k=0, gamma=0.9)

    with pytest.raises(ValueError, match="number of actions"):
        run_pipeline_smoke_step(data, model, batch_size=4, k=99, gamma=0.9)


def test_invalid_gamma_raises_value_error() -> None:
    data, model = _data_and_model()

    with pytest.raises(ValueError, match="gamma"):
        run_pipeline_smoke_step(data, model, batch_size=4, k=3, gamma=1.1)
