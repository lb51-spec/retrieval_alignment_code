"""Tests for dependency-free RecSim / RecSim NG scaffold contracts."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from retrieval_alignment.baselines import NearestActionBaseline  # noqa: E402
from retrieval_alignment.benchmarks import (  # noqa: E402
    ActionCatalogue,
    AdapterAvailability,
    RecSimAdapter,
    RecSimMockSlateTask,
    RecSimSlateMetadata,
    assert_benchmark_task_contract,
    evaluate_recsim_ann_candidate_audit,
    evaluate_recsim_pmr_scaffold,
    evaluate_recsim_slate_scaffold,
    score_recsim_catalogue_with_model,
)
from retrieval_alignment.config import ModelConfig  # noqa: E402
from retrieval_alignment.networks import build_networks  # noqa: E402


def test_recsim_slate_metadata_validation() -> None:
    metadata = RecSimSlateMetadata(
        slate_size=3,
        num_users=4,
        num_items=16,
        item_feature_dim=5,
        user_feature_dim=4,
        metadata={"kind": "unit"},
    )

    assert metadata.slate_size == 3
    assert metadata.metadata["kind"] == "unit"

    with pytest.raises(ValueError, match="slate_size"):
        RecSimSlateMetadata(slate_size=0, num_items=1, item_feature_dim=1, user_feature_dim=1)
    with pytest.raises(ValueError, match="num_items"):
        RecSimSlateMetadata(slate_size=1, num_items=0, item_feature_dim=1, user_feature_dim=1)
    with pytest.raises(ValueError, match="metadata"):
        RecSimSlateMetadata(slate_size=1, num_items=1, item_feature_dim=1, user_feature_dim=1, metadata=None)


def test_recsim_mock_slate_task_satisfies_contract() -> None:
    task = RecSimMockSlateTask(
        num_users=3,
        num_items=8,
        user_feature_dim=4,
        item_feature_dim=4,
        slate_size=2,
        num_interactions=12,
        seed=7,
    )

    assert assert_benchmark_task_contract(task) is True
    assert task.metadata["source"] == "recsim_mock_slate"
    assert task.metadata["benchmark_claim"] is False
    assert isinstance(task.make_action_catalogue_contract(), ActionCatalogue)
    assert len(task.make_dataset_bundle().dataset) == 12


def test_recsim_adapter_constructs_without_dependency() -> None:
    adapter = RecSimAdapter()

    availability = adapter.is_available()

    assert isinstance(availability, AdapterAvailability)
    assert adapter.name == "recsim"
    assert "mock slate" in adapter.describe()


def test_recsim_adapter_make_mock_slate_task() -> None:
    adapter = RecSimAdapter()

    task = adapter.make_task(task_name="mock_slate", num_users=2, num_items=6, slate_size=2, num_interactions=8)

    assert isinstance(task, RecSimMockSlateTask)
    assert assert_benchmark_task_contract(task) is True


def test_recsim_adapter_real_task_fails_clearly() -> None:
    adapter = RecSimAdapter()

    with pytest.raises((ImportError, NotImplementedError), match="RecSim|simulator|future adapter"):
        adapter.make_task(task_name="real_environment")


def test_evaluate_recsim_slate_scaffold_claim_free_and_finite() -> None:
    task = RecSimMockSlateTask(num_users=3, num_items=8, slate_size=2, num_interactions=12, seed=3)

    result = evaluate_recsim_slate_scaffold(task)

    assert result.benchmark_claim is False
    assert result.synthetic_only is False
    assert result.diagnostics["scaffold_only"] is True
    assert result.metrics["num_users"] == 3
    assert result.metrics["num_items"] == 8
    assert result.metrics["slate_size"] == 2
    assert 0.0 <= result.metrics["observed_item_coverage"] <= 1.0
    assert torch.isfinite(torch.tensor(float(result.metrics["avg_reward"])))


def test_evaluate_recsim_slate_scaffold_optional_score_shapes() -> None:
    task = RecSimMockSlateTask(
        num_users=3,
        num_items=8,
        user_feature_dim=4,
        item_feature_dim=4,
        slate_size=2,
        num_interactions=12,
        seed=5,
    )
    baseline = NearestActionBaseline(seed=0).fit(
        task.make_dataset(),
        action_features=task.make_action_catalogue()[0],
        action_ids=task.make_action_catalogue()[1],
    )

    result = evaluate_recsim_slate_scaffold(task, model_or_baseline=baseline, k=2)

    assert result.diagnostics["score_shape"] == [12, 8]
    assert result.diagnostics["topk_shape"] == [12, 2]


def test_score_recsim_catalogue_with_model_shape_and_finite() -> None:
    task = RecSimMockSlateTask(num_users=3, num_items=8, slate_size=2, num_interactions=12, seed=6)
    model = build_networks(ModelConfig(state_dim=task.state_dim, action_dim=task.action_dim, embedding_dim=8))

    scores = score_recsim_catalogue_with_model(task, model)

    assert scores.shape == (len(task.make_dataset()), task.num_actions)
    assert torch.all(torch.isfinite(scores))


def test_evaluate_recsim_pmr_scaffold_claim_free_and_bounded() -> None:
    task = RecSimMockSlateTask(num_users=3, num_items=8, slate_size=2, num_interactions=12, seed=6)
    model = build_networks(ModelConfig(state_dim=task.state_dim, action_dim=task.action_dim, embedding_dim=8))

    result = evaluate_recsim_pmr_scaffold(task, model, k=2)

    assert result.benchmark_claim is False
    assert result.synthetic_only is False
    assert result.diagnostics["scaffold_only"] is True
    assert 0.0 <= result.metrics["pmr_at_k_scaffold"] <= 1.0
    assert 0.0 <= result.metrics["missing_policy_mass_mean"] <= 1.0
    assert 0.0 <= result.metrics["logged_item_in_topk_rate"] <= 1.0


def test_evaluate_recsim_ann_candidate_audit_claim_free_and_bounded() -> None:
    task = RecSimMockSlateTask(num_users=3, num_items=8, slate_size=2, num_interactions=12, seed=6)
    model = build_networks(ModelConfig(state_dim=task.state_dim, action_dim=task.action_dim, embedding_dim=8))

    result = evaluate_recsim_ann_candidate_audit(task, model, k=2)

    assert result.benchmark_claim is False
    assert result.diagnostics["scaffold_only"] is True
    assert 0.0 <= result.metrics["ann_candidate_recall_mean"] <= 1.0
    assert 0.0 <= result.metrics["ann_candidate_overlap_fraction_mean"] <= 1.0
    assert 0.0 <= result.metrics["ann_candidate_pmr_mean"] <= 1.0
