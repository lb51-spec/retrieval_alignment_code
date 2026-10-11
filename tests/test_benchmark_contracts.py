"""Compliance tests for benchmark task data contracts."""

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
    MockExternalBenchmarkTask,
    SyntheticBenchmarkTask,
    assert_benchmark_task_contract,
)
from retrieval_alignment.config import SyntheticDatasetConfig  # noqa: E402


def test_mock_external_benchmark_task_satisfies_contract() -> None:
    task = MockExternalBenchmarkTask(seed=123)

    assert assert_benchmark_task_contract(task) is True
    assert task.task_type == "mock_external"
    assert task.metadata["benchmark_claim"] is False


def test_synthetic_benchmark_task_satisfies_contract() -> None:
    task = SyntheticBenchmarkTask(SyntheticDatasetConfig(num_transitions=8, num_actions=5))

    assert assert_benchmark_task_contract(task) is True


def test_invalid_task_missing_required_methods_fails_clearly() -> None:
    class InvalidTask:
        name = "invalid"
        task_type = "invalid"
        state_dim = 1
        action_dim = 1
        num_actions = 1
        metadata = {}

        def evaluate_policy_or_model(self, model):
            del model
            return None

    with pytest.raises(AssertionError, match="make_dataset_bundle"):
        assert_benchmark_task_contract(InvalidTask())


def test_invalid_task_bad_metadata_fails_clearly() -> None:
    class BadMetadataTask:
        name = "bad"
        task_type = "mock_external"
        state_dim = 3
        action_dim = 2
        num_actions = 4
        metadata = None

    with pytest.raises(AssertionError, match="metadata"):
        assert_benchmark_task_contract(BadMetadataTask())


def test_mock_external_evaluation_placeholder_benchmark_claim_false() -> None:
    task = MockExternalBenchmarkTask(seed=0)

    result = task.evaluate_policy_or_model(model=None)

    assert result.benchmark_claim is False
    assert result.synthetic_only is True
    assert result.metrics["num_transitions"] == 6
    assert result.metrics["num_actions"] == 4
