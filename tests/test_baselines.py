"""Tests for trivial synthetic baseline scaffolds."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from retrieval_alignment.baselines import (  # noqa: E402
    BaselineResult,
    BehaviorCloningBaseline,
    CQLBaseline,
    IQLBaseline,
    NearestActionBaseline,
    PLACEHOLDER_BASELINE_NAMES,
    RandomCandidateBaseline,
    TD3BCBaseline,
    evaluate_synthetic_baseline,
    get_baseline,
)
from retrieval_alignment.benchmarks import SyntheticBenchmarkTask  # noqa: E402
from retrieval_alignment.config import SyntheticDatasetConfig  # noqa: E402
from retrieval_alignment.evaluation import EvaluationConfig  # noqa: E402


def _task() -> SyntheticBenchmarkTask:
    return SyntheticBenchmarkTask(
        SyntheticDatasetConfig(
            num_transitions=24,
            state_dim=4,
            action_dim=3,
            num_actions=7,
            seed=123,
        )
    )


def test_random_candidate_baseline_scores_and_selects_actions() -> None:
    baseline = RandomCandidateBaseline(seed=42)
    states = torch.zeros(5, 4)
    action_features = torch.zeros(7, 3)

    scores = baseline.predict_action_scores(states, action_features)
    selected = baseline.select_actions(states, action_features, k=3)

    assert scores.shape == (5, 7)
    assert selected.shape == (5, 3)
    assert baseline.synthetic_only is True
    assert baseline.metadata["benchmark_claim"] is False


def test_random_candidate_baseline_is_deterministic_with_seed() -> None:
    states = torch.zeros(4, 4)
    action_features = torch.zeros(6, 3)
    first = RandomCandidateBaseline(seed=9).predict_action_scores(states, action_features)
    second = RandomCandidateBaseline(seed=9).predict_action_scores(states, action_features)

    assert torch.allclose(first, second)


def test_nearest_action_baseline_scores_with_matching_dims() -> None:
    baseline = NearestActionBaseline()
    states = torch.randn(5, 3)
    action_features = torch.randn(7, 3)

    scores = baseline.predict_action_scores(states, action_features)
    selected = baseline.select_actions(states, action_features, k=2)

    assert scores.shape == (5, 7)
    assert selected.shape == (5, 2)
    assert torch.all(torch.isfinite(scores))


def test_nearest_action_baseline_handles_state_action_dim_mismatch() -> None:
    baseline = NearestActionBaseline()
    action_features = torch.randn(7, 3)

    short_scores = baseline.predict_action_scores(torch.randn(5, 2), action_features)
    long_scores = baseline.predict_action_scores(torch.randn(5, 5), action_features)

    assert short_scores.shape == (5, 7)
    assert long_scores.shape == (5, 7)


def test_get_baseline_registry() -> None:
    assert isinstance(get_baseline("random", seed=1), RandomCandidateBaseline)
    assert isinstance(get_baseline("nearest"), NearestActionBaseline)

    with pytest.raises(ValueError, match="Unknown baseline"):
        get_baseline("not_a_real_baseline_xyz")


def test_evaluate_synthetic_baseline_returns_result() -> None:
    task = _task()
    baseline = RandomCandidateBaseline(seed=7)

    result = evaluate_synthetic_baseline(
        baseline,
        task,
        EvaluationConfig(num_batches=1, batch_size=4, k=3),
    )

    assert isinstance(result, BaselineResult)
    assert result.baseline_name == "random"
    assert result.synthetic_only is True
    assert result.benchmark_claim is False
    assert result.diagnostics["score_shape"] == (4, 7)
    assert result.diagnostics["selected_shape"] == (4, 3)
    assert torch.isfinite(torch.tensor(result.metrics["mean_score"]))


def test_baseline_evaluate_method_delegates_to_helper() -> None:
    task = _task()
    baseline = NearestActionBaseline()

    result = baseline.evaluate(task, EvaluationConfig(num_batches=1, batch_size=4, k=2))

    assert isinstance(result, BaselineResult)
    assert result.baseline_name == "nearest"
    assert result.benchmark_claim is False


def test_baseline_result_implemented_field() -> None:
    result = BaselineResult(baseline_name="random")
    assert result.implemented is True


def test_placeholder_baselines_raise_not_implemented() -> None:
    for cls in (BehaviorCloningBaseline, CQLBaseline, IQLBaseline, TD3BCBaseline):
        bl = cls()
        with pytest.raises(NotImplementedError):
            bl.fit(None)
        with pytest.raises(NotImplementedError):
            bl.predict_action_scores(None, None)
        with pytest.raises(NotImplementedError):
            bl.select_actions(None, None)
        with pytest.raises(NotImplementedError):
            bl.evaluate(None)


def test_get_baseline_returns_placeholder_instances() -> None:
    assert isinstance(get_baseline("behavior_cloning"), BehaviorCloningBaseline)
    assert isinstance(get_baseline("cql"), CQLBaseline)
    assert isinstance(get_baseline("iql"), IQLBaseline)
    assert isinstance(get_baseline("td3_bc"), TD3BCBaseline)


def test_placeholder_baseline_names_frozenset() -> None:
    assert "behavior_cloning" in PLACEHOLDER_BASELINE_NAMES
    assert "cql" in PLACEHOLDER_BASELINE_NAMES
    assert "iql" in PLACEHOLDER_BASELINE_NAMES
    assert "td3_bc" in PLACEHOLDER_BASELINE_NAMES
    assert "random" not in PLACEHOLDER_BASELINE_NAMES
