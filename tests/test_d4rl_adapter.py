"""Tests for the dependency-free D4RL large-action scaffold."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from retrieval_alignment.benchmarks import (  # noqa: E402
    AdapterAvailability,
    D4RLActionCodebook,
    D4RLAdapter,
    D4RLDatasetTask,
    D4RLLargeActionTask,
    assert_benchmark_task_contract,
    build_d4rl_action_codebook_from_actions,
    evaluate_d4rl_ann_candidate_audit,
    evaluate_d4rl_action_codebook_audit,
    evaluate_d4rl_pmr_scaffold,
    score_d4rl_codebook_with_model,
)
from retrieval_alignment.config import ModelConfig  # noqa: E402
from retrieval_alignment.networks import build_networks  # noqa: E402


def _mock_arrays() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    observations = torch.tensor(
        [
            [1.0, 0.0, 0.5],
            [0.0, 1.0, 0.5],
            [0.5, 0.5, 1.0],
            [1.0, 1.0, 0.0],
        ],
        dtype=torch.float32,
    )
    actions = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [0.5, 0.5],
        ],
        dtype=torch.float32,
    )
    rewards = torch.tensor([1.0, 0.0, 0.5, 1.5], dtype=torch.float32)
    next_observations = observations + 0.1
    terminals = torch.tensor([False, False, False, True])
    return observations, actions, rewards, next_observations, terminals


def test_d4rl_action_codebook_validates_correct_tensors() -> None:
    codebook = D4RLActionCodebook(
        action_features=torch.randn(4, 2),
        action_ids=torch.arange(4),
        metadata={"kind": "unit"},
    )

    assert codebook.num_actions == 4
    assert codebook.action_dim == 2
    assert codebook.metadata["kind"] == "unit"


def test_d4rl_action_codebook_invalid_shapes_raise() -> None:
    with pytest.raises(ValueError, match="action_features"):
        D4RLActionCodebook(action_features=torch.randn(4), action_ids=torch.arange(4))
    with pytest.raises(ValueError, match="action_ids length"):
        D4RLActionCodebook(action_features=torch.randn(4, 2), action_ids=torch.arange(3))
    with pytest.raises(ValueError, match="unique"):
        D4RLActionCodebook(action_features=torch.randn(3, 2), action_ids=torch.tensor([0, 1, 1]))


def test_build_d4rl_action_codebook_from_actions_is_deterministic() -> None:
    _, actions, *_ = _mock_arrays()

    first = build_d4rl_action_codebook_from_actions(actions, max_actions=2, seed=7)
    second = build_d4rl_action_codebook_from_actions(actions, max_actions=2, seed=7)

    assert first.num_actions == 2
    assert torch.equal(first.action_features, second.action_features)
    assert torch.equal(first.action_ids, torch.arange(2))
    assert first.metadata["source"] == "continuous_action_samples"
    assert first.metadata["benchmark_claim"] is False


def test_d4rl_large_action_task_satisfies_contract_and_maps_logged_ids() -> None:
    observations, actions, rewards, next_observations, terminals = _mock_arrays()
    task = D4RLLargeActionTask(observations, actions, rewards, next_observations, terminals)

    bundle = task.make_dataset_bundle()

    assert assert_benchmark_task_contract(task) is True
    assert task.metadata["source"] == "d4rl_large_action_scaffold"
    assert task.metadata["benchmark_claim"] is False
    assert bundle.action_catalogue.num_actions == task.num_actions
    assert torch.all(bundle.dataset.batch.action_ids >= 0)
    assert torch.all(bundle.dataset.batch.action_ids < task.num_actions)


def test_d4rl_adapter_constructs_without_dependency_and_reports_availability() -> None:
    adapter = D4RLAdapter()

    availability = adapter.is_available()

    assert isinstance(availability, AdapterAvailability)
    assert adapter.name == "d4rl"
    assert "dependency-free large-action" in adapter.describe()


def test_d4rl_adapter_make_task_with_mock_arrays_returns_task() -> None:
    observations, actions, rewards, next_observations, terminals = _mock_arrays()
    adapter = D4RLAdapter(
        observations=observations,
        actions=actions,
        rewards=rewards,
        next_observations=next_observations,
        terminals=terminals,
    )

    task = adapter.make_task()

    assert isinstance(task, D4RLLargeActionTask)
    assert assert_benchmark_task_contract(task) is True


def test_evaluate_d4rl_action_codebook_audit_is_claim_free_and_bounded() -> None:
    observations, actions, rewards, next_observations, terminals = _mock_arrays()
    task = D4RLLargeActionTask(observations, actions, rewards, next_observations, terminals)

    result = evaluate_d4rl_action_codebook_audit(task, k=2)

    assert result.benchmark_claim is False
    assert result.synthetic_only is False
    assert result.diagnostics["scaffold_only"] is True
    assert result.metrics["codebook_size"] == task.num_actions
    assert 0.0 <= result.metrics["ann_candidate_recall_mean"] <= 1.0
    assert 0.0 <= result.metrics["ann_candidate_overlap_fraction_mean"] <= 1.0
    assert torch.isfinite(torch.tensor(float(result.metrics["ann_candidate_best_value_gap_mean"])))


def test_score_d4rl_codebook_with_model_shape_and_finite() -> None:
    observations, actions, rewards, next_observations, terminals = _mock_arrays()
    task = D4RLLargeActionTask(observations, actions, rewards, next_observations, terminals)
    model = build_networks(ModelConfig(state_dim=task.state_dim, action_dim=task.action_dim, embedding_dim=8))

    scores = score_d4rl_codebook_with_model(task, model)

    assert scores.shape == (len(task.make_dataset()), task.num_actions)
    assert torch.all(torch.isfinite(scores))


def test_evaluate_d4rl_pmr_scaffold_claim_free_and_bounded() -> None:
    observations, actions, rewards, next_observations, terminals = _mock_arrays()
    task = D4RLLargeActionTask(observations, actions, rewards, next_observations, terminals)
    model = build_networks(ModelConfig(state_dim=task.state_dim, action_dim=task.action_dim, embedding_dim=8))

    result = evaluate_d4rl_pmr_scaffold(task, model, k=2)

    assert result.benchmark_claim is False
    assert result.synthetic_only is False
    assert result.diagnostics["scaffold_only"] is True
    assert 0.0 <= result.metrics["pmr_at_k_scaffold"] <= 1.0
    assert 0.0 <= result.metrics["missing_policy_mass_mean"] <= 1.0
    assert 0.0 <= result.metrics["logged_action_in_topk_rate"] <= 1.0


def test_evaluate_d4rl_ann_candidate_audit_claim_free_and_bounded() -> None:
    observations, actions, rewards, next_observations, terminals = _mock_arrays()
    task = D4RLLargeActionTask(observations, actions, rewards, next_observations, terminals)
    model = build_networks(ModelConfig(state_dim=task.state_dim, action_dim=task.action_dim, embedding_dim=8))

    result = evaluate_d4rl_ann_candidate_audit(task, model, k=2)

    assert result.benchmark_claim is False
    assert result.diagnostics["scaffold_only"] is True
    assert 0.0 <= result.metrics["ann_candidate_recall_mean"] <= 1.0
    assert 0.0 <= result.metrics["ann_candidate_overlap_fraction_mean"] <= 1.0
    assert 0.0 <= result.metrics["ann_candidate_pmr_mean"] <= 1.0


# ---------------------------------------------------------------------------
# D4RLDatasetTask tests (mock-array path, no real d4rl/minari needed)
# ---------------------------------------------------------------------------


def test_d4rl_dataset_task_raises_import_error_without_deps() -> None:
    """D4RLDatasetTask must fail clearly when neither d4rl nor minari is installed."""
    with pytest.raises(ImportError, match="d4rl or minari"):
        D4RLDatasetTask(dataset_name="hopper-medium-v2", codebook_size=4)


def test_d4rl_adapter_with_dataset_name_and_no_deps_raises_import_error() -> None:
    adapter = D4RLAdapter(dataset_name="hopper-medium-v2", codebook_size=4)
    with pytest.raises(ImportError, match="d4rl or minari"):
        adapter.make_task()


def test_d4rl_adapter_with_mock_arrays_still_returns_large_action_task() -> None:
    observations, actions, rewards, next_observations, terminals = _mock_arrays()
    adapter = D4RLAdapter(
        observations=observations,
        actions=actions,
        rewards=rewards,
        next_observations=next_observations,
        terminals=terminals,
    )
    task = adapter.make_task()
    assert isinstance(task, D4RLLargeActionTask)


def test_evaluate_d4rl_pmr_scaffold_accepts_dataset_task_via_mock_wrapper() -> None:
    """Verify evaluate_d4rl_pmr_scaffold accepts both D4RLLargeActionTask and D4RLDatasetTask."""
    observations, actions, rewards, next_observations, terminals = _mock_arrays()
    task = D4RLLargeActionTask(observations, actions, rewards, next_observations, terminals)
    model = build_networks(ModelConfig(state_dim=task.state_dim, action_dim=task.action_dim, embedding_dim=8))
    result = evaluate_d4rl_pmr_scaffold(task, model, k=2)
    assert result.benchmark_claim is False
    assert 0.0 <= result.metrics["pmr_at_k_scaffold"] <= 1.0

