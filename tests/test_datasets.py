"""Tests for replay batch and in-memory dataset containers."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from retrieval_alignment.datasets import InMemoryReplayDataset, ReplayBatch


def _batch_tensors(n: int = 5) -> dict[str, torch.Tensor]:
    return {
        "states": torch.arange(n * 3, dtype=torch.float32).reshape(n, 3),
        "actions": torch.arange(n * 2, dtype=torch.float32).reshape(n, 2),
        "rewards": torch.arange(n, dtype=torch.float32),
        "next_states": torch.arange(n * 3, dtype=torch.float32).reshape(n, 3) + 1.0,
        "dones": torch.zeros(n),
        "action_ids": torch.arange(n),
        "next_logged_action_ids": torch.arange(n) + 10,
    }


def test_replay_batch_valid_construction() -> None:
    batch = ReplayBatch(**_batch_tensors())

    assert batch.batch_size == 5


def test_rewards_and_dones_are_squeezed_to_batch_shape() -> None:
    tensors = _batch_tensors()
    tensors["rewards"] = tensors["rewards"].unsqueeze(1)
    tensors["dones"] = tensors["dones"].unsqueeze(1)

    batch = ReplayBatch(**tensors)

    assert batch.rewards.shape == (5,)
    assert batch.dones.shape == (5,)


def test_batch_size_state_dim_action_dim_properties() -> None:
    batch = ReplayBatch(**_batch_tensors())

    assert batch.batch_size == 5
    assert batch.state_dim == 3
    assert batch.action_dim == 2


def test_to_device_returns_new_batch_with_tensors_moved() -> None:
    batch = ReplayBatch(**_batch_tensors())

    moved = batch.to("cpu")

    assert moved is not batch
    assert moved.states.device.type == "cpu"
    assert moved.action_ids is not None
    assert moved.action_ids.device.type == "cpu"


def test_shape_mismatch_raises_value_error() -> None:
    tensors = _batch_tensors()
    tensors["actions"] = torch.zeros(4, 2)

    with pytest.raises(ValueError, match="batch size"):
        ReplayBatch(**tensors)


def test_non_finite_transition_tensors_raise_value_error() -> None:
    for key in ("states", "actions", "rewards", "next_states"):
        tensors = _batch_tensors()
        tensors[key] = tensors[key].clone()
        tensors[key].reshape(-1)[0] = float("nan")
        with pytest.raises(ValueError, match="finite"):
            ReplayBatch(**tensors)


def test_invalid_numeric_dones_raise_value_error() -> None:
    tensors = _batch_tensors()
    tensors["dones"] = torch.tensor([0.0, 1.0, 2.0, 0.0, 0.0])

    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        ReplayBatch(**tensors)


def test_action_ids_shape_mismatch_raises_value_error() -> None:
    tensors = _batch_tensors()
    tensors["action_ids"] = torch.arange(4)

    with pytest.raises(ValueError, match="action_ids"):
        ReplayBatch(**tensors)


def test_in_memory_replay_dataset_length() -> None:
    dataset = InMemoryReplayDataset(**_batch_tensors())

    assert len(dataset) == 5


def test_sample_returns_replay_batch_with_requested_batch_size() -> None:
    dataset = InMemoryReplayDataset(**_batch_tensors())

    sample = dataset.sample(3, generator=torch.Generator().manual_seed(0))

    assert isinstance(sample, ReplayBatch)
    assert sample.batch_size == 3


def test_sample_is_deterministic_with_fixed_generator_seed() -> None:
    dataset = InMemoryReplayDataset(**_batch_tensors())

    first = dataset.sample(4, generator=torch.Generator().manual_seed(123))
    second = dataset.sample(4, generator=torch.Generator().manual_seed(123))

    assert torch.equal(first.states, second.states)
    assert torch.equal(first.actions, second.actions)


def test_sample_without_replacement_rejects_too_large_batch() -> None:
    dataset = InMemoryReplayDataset(**_batch_tensors(), replacement=False)

    with pytest.raises(ValueError, match="without replacement"):
        dataset.sample(6)


def test_optional_ids_are_preserved_in_sampled_batches() -> None:
    dataset = InMemoryReplayDataset(**_batch_tensors())

    sample = dataset.sample(3, generator=torch.Generator().manual_seed(0))

    assert sample.action_ids is not None
    assert sample.next_logged_action_ids is not None
    assert sample.action_ids.shape == (3,)
    assert sample.next_logged_action_ids.shape == (3,)


def test_empty_dataset_raises_value_error() -> None:
    with pytest.raises(ValueError, match="at least one transition"):
        InMemoryReplayDataset(
            states=torch.empty(0, 3),
            actions=torch.empty(0, 2),
            rewards=torch.empty(0),
            next_states=torch.empty(0, 3),
            dones=torch.empty(0),
        )
