"""Tests for benchmark scaffold task abstractions."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from retrieval_alignment.benchmarks import (
    ActionCatalogue,
    AdapterAvailability,
    BenchmarkEvaluationProtocol,
    D4RLAdapter,
    EvaluationResult,
    MockExternalBenchmarkTask,
    MovieLensAdapter,
    OfflineDatasetBundle,
    RecSimAdapter,
    SyntheticBenchmarkTask,
    available_external_adapters,
    assert_benchmark_task_contract,
    get_external_adapter,
    synthetic_data_to_dataset_bundle,
    validate_action_catalogue,
    validate_benchmark_evaluation_protocol,
    validate_offline_dataset_bundle,
)
from retrieval_alignment.config import ModelConfig, SyntheticDatasetConfig, default_smoke_profile
from retrieval_alignment.datasets import InMemoryReplayDataset
from retrieval_alignment.evaluation import EvaluationConfig
from retrieval_alignment.networks import build_networks
from retrieval_alignment.synthetic import make_synthetic_offline_data


def _valid_catalogue() -> ActionCatalogue:
    return ActionCatalogue(
        action_features=torch.randn(4, 3),
        action_ids=torch.arange(4),
        metadata={"kind": "unit"},
    )


def test_action_catalogue_valid_construction() -> None:
    catalogue = _valid_catalogue()

    assert catalogue.num_actions == 4
    assert catalogue.action_dim == 3
    assert catalogue.metadata["kind"] == "unit"
    assert validate_action_catalogue(catalogue) is catalogue


def test_action_catalogue_invalid_feature_shape_raises() -> None:
    with pytest.raises(ValueError, match="action_features"):
        ActionCatalogue(action_features=torch.randn(4), action_ids=torch.arange(4))


def test_action_catalogue_action_id_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="action_ids length"):
        ActionCatalogue(action_features=torch.randn(4, 3), action_ids=torch.arange(3))


def test_action_catalogue_empty_catalogue_raises() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        ActionCatalogue(action_features=torch.empty(0, 3), action_ids=torch.empty(0, dtype=torch.long))


def test_action_catalogue_duplicate_ids_raise() -> None:
    with pytest.raises(ValueError, match="unique"):
        ActionCatalogue(action_features=torch.randn(4, 3), action_ids=torch.tensor([0, 1, 1, 2]))


def test_offline_dataset_bundle_valid_construction() -> None:
    dataset = InMemoryReplayDataset(
        states=torch.randn(5, 4),
        actions=torch.randn(5, 3),
        rewards=torch.zeros(5),
        next_states=torch.randn(5, 4),
        dones=torch.zeros(5),
    )
    bundle = OfflineDatasetBundle(
        dataset=dataset,
        action_catalogue=_valid_catalogue(),
        metadata={"task": "unit"},
    )

    assert validate_offline_dataset_bundle(bundle) is bundle
    assert len(bundle.dataset) == 5
    assert bundle.metadata["task"] == "unit"


def test_offline_dataset_bundle_empty_dataset_raises() -> None:
    class EmptyDataset:
        def __len__(self) -> int:
            return 0

    with pytest.raises(ValueError, match="non-empty"):
        OfflineDatasetBundle(dataset=EmptyDataset(), action_catalogue=_valid_catalogue())


def test_synthetic_data_to_dataset_bundle_works() -> None:
    data = make_synthetic_offline_data(SyntheticDatasetConfig(num_transitions=8, num_actions=5))

    bundle = synthetic_data_to_dataset_bundle(data)

    assert isinstance(bundle, OfflineDatasetBundle)
    assert len(bundle.dataset) == 8
    assert bundle.action_catalogue.num_actions == 5
    assert bundle.metadata["synthetic_only"] is True
    assert bundle.metadata["benchmark_claim"] is False


def test_benchmark_evaluation_protocol_claim_free_validation() -> None:
    protocol = BenchmarkEvaluationProtocol(
        task_name="synthetic_debug",
        metrics=["mean_bellman_loss"],
        diagnostics=("mean_ess",),
        required_splits=[],
    )

    assert validate_benchmark_evaluation_protocol(protocol) is protocol
    assert protocol.synthetic_only is True
    assert protocol.benchmark_claim is False


def test_benchmark_evaluation_protocol_rejects_benchmark_claim_true() -> None:
    with pytest.raises(ValueError, match="benchmark_claim"):
        BenchmarkEvaluationProtocol(task_name="bad", benchmark_claim=True)


def test_mock_external_benchmark_task_creates_valid_contracts() -> None:
    task = MockExternalBenchmarkTask(seed=7)

    bundle = task.make_dataset_bundle()
    catalogue = task.make_action_catalogue_contract()
    result = task.evaluate_policy_or_model(model=None)

    assert assert_benchmark_task_contract(task) is True
    assert task.task_type == "mock_external"
    assert len(bundle.dataset) == 6
    assert catalogue.num_actions == 4
    assert result.benchmark_claim is False


def test_synthetic_benchmark_task_creates_valid_dataset_and_catalogue() -> None:
    task = SyntheticBenchmarkTask(
        SyntheticDatasetConfig(
            num_transitions=12,
            state_dim=4,
            action_dim=3,
            num_actions=7,
            seed=123,
        ),
        name="unit_synthetic",
    )

    dataset = task.make_dataset()
    action_features, action_ids = task.make_action_catalogue()

    assert len(dataset) == 12
    assert action_features.shape == (7, 3)
    assert action_ids.shape == (7,)
    assert task.state_dim == 4
    assert task.action_dim == 3
    assert task.num_actions == 7
    assert task.metadata["synthetic_only"] is True
    assert task.metadata["benchmark_claim"] is False


def test_synthetic_benchmark_task_make_dataset_bundle() -> None:
    task = SyntheticBenchmarkTask(SyntheticDatasetConfig(num_transitions=10, num_actions=6))

    bundle = task.make_dataset_bundle()
    catalogue = task.make_action_catalogue_contract()

    assert isinstance(bundle, OfflineDatasetBundle)
    assert isinstance(catalogue, ActionCatalogue)
    assert len(bundle.dataset) == 10
    assert bundle.action_catalogue.num_actions == 6


def test_synthetic_benchmark_task_evaluation_placeholder() -> None:
    task = SyntheticBenchmarkTask(SyntheticDatasetConfig(num_transitions=8, num_actions=5))

    result = task.evaluate_policy_or_model(model=None)

    assert isinstance(result, EvaluationResult)
    assert result.task_name == task.name
    assert result.synthetic_only is True
    assert result.benchmark_claim is False
    assert result.metrics["num_transitions"] == 8
    assert result.metrics["num_actions"] == 5


def test_synthetic_benchmark_task_evaluation_loop_with_model() -> None:
    task = SyntheticBenchmarkTask(SyntheticDatasetConfig(num_transitions=16, num_actions=6))
    torch.manual_seed(0)
    model = build_networks(ModelConfig(state_dim=4, action_dim=3, embedding_dim=8))

    result = task.evaluate_policy_or_model(
        model,
        evaluation_config=EvaluationConfig(num_batches=1, batch_size=4, k=3),
        base_config=default_smoke_profile(),
    )

    assert isinstance(result, EvaluationResult)
    assert result.synthetic_only is True
    assert result.benchmark_claim is False
    assert "mean_bellman_loss" in result.metrics
    assert "evaluation_summary" in result.diagnostics


def test_d4rl_adapter_constructs_without_dependency() -> None:
    adapter = D4RLAdapter()

    availability = adapter.is_available()

    assert isinstance(availability, AdapterAvailability)
    assert adapter.name == "d4rl"
    assert adapter.requires_dependency is True
    assert "Future adapter" in adapter.describe()


def test_d4rl_adapter_make_task_fails_clearly() -> None:
    adapter = D4RLAdapter()

    with pytest.raises((ImportError, NotImplementedError), match="D4RL|Minari|future adapter"):
        adapter.make_task()


def test_movielens_adapter_missing_path_reports_unavailable(tmp_path: Path) -> None:
    adapter = MovieLensAdapter(data_path=tmp_path / "missing")

    availability = adapter.is_available()

    assert availability.available is False
    assert "does not exist" in availability.message


def test_movielens_adapter_existing_path_reports_available_but_make_task_unimplemented(tmp_path: Path) -> None:
    adapter = MovieLensAdapter(data_path=tmp_path)

    availability = adapter.validate()

    assert availability.available is True
    assert "exists" in availability.message
    with pytest.raises(NotImplementedError, match="MovieLens"):
        adapter.make_task()


def test_recsim_adapter_constructs_without_dependency() -> None:
    adapter = RecSimAdapter()

    availability = adapter.is_available()

    assert isinstance(availability, AdapterAvailability)
    assert adapter.name == "recsim"
    assert adapter.requires_dependency is True


def test_recsim_adapter_make_task_fails_clearly() -> None:
    adapter = RecSimAdapter()

    with pytest.raises((ImportError, NotImplementedError), match="RecSim|future adapter"):
        adapter.make_task()


def test_external_adapter_registry() -> None:
    assert isinstance(get_external_adapter("d4rl"), D4RLAdapter)
    assert isinstance(get_external_adapter("movielens"), MovieLensAdapter)
    assert isinstance(get_external_adapter("recsim"), RecSimAdapter)

    with pytest.raises(ValueError, match="Unknown external adapter"):
        get_external_adapter("unknown")


def test_available_external_adapters_returns_expected_entries() -> None:
    adapters = available_external_adapters()

    assert set(adapters) == {"d4rl", "movielens", "recsim"}
    for value in adapters.values():
        assert isinstance(value["availability"], AdapterAvailability)
        assert value["metadata"]["implemented"] is False
