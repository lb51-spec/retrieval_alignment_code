"""Tests for tiny synthetic smoke-test data generation."""

from __future__ import annotations

import sys
from dataclasses import asdict
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from retrieval_alignment.config import SyntheticDatasetConfig
from retrieval_alignment.synthetic import build_exact_index_from_synthetic, make_synthetic_offline_data


def _config(seed: int = 123) -> SyntheticDatasetConfig:
    return SyntheticDatasetConfig(
        num_transitions=20,
        state_dim=4,
        action_dim=3,
        num_actions=7,
        seed=seed,
        reward_noise_std=0.01,
        terminal_prob=0.25,
    )


def test_generated_dataset_has_requested_length() -> None:
    data = make_synthetic_offline_data(_config())

    assert len(data.dataset) == 20


def test_action_features_shape() -> None:
    data = make_synthetic_offline_data(_config())

    assert data.action_features.shape == (7, 3)


def test_action_ids_shape() -> None:
    data = make_synthetic_offline_data(_config())

    assert data.action_ids.shape == (7,)


def test_sampled_replay_batch_has_correct_shapes() -> None:
    data = make_synthetic_offline_data(_config())
    batch = data.dataset.sample(5, generator=torch.Generator().manual_seed(0))

    assert batch.states.shape == (5, 4)
    assert batch.actions.shape == (5, 3)
    assert batch.rewards.shape == (5,)
    assert batch.next_states.shape == (5, 4)
    assert batch.dones.shape == (5,)


def test_generation_is_deterministic_with_same_seed() -> None:
    first = make_synthetic_offline_data(_config(seed=77))
    second = make_synthetic_offline_data(_config(seed=77))

    assert torch.equal(first.dataset.batch.states, second.dataset.batch.states)
    assert torch.equal(first.action_features, second.action_features)
    assert torch.equal(first.dataset.batch.rewards, second.dataset.batch.rewards)


def test_different_seeds_produce_different_data() -> None:
    first = make_synthetic_offline_data(_config(seed=1))
    second = make_synthetic_offline_data(_config(seed=2))

    assert not torch.equal(first.dataset.batch.states, second.dataset.batch.states)


def test_rewards_are_finite() -> None:
    data = make_synthetic_offline_data(_config())

    assert torch.isfinite(data.dataset.batch.rewards).all()


def test_dones_are_valid() -> None:
    data = make_synthetic_offline_data(_config())

    assert data.dataset.batch.dones.dtype == torch.bool
    assert data.dataset.batch.dones.shape == (20,)


def test_logged_action_ids_are_valid_catalogue_ids() -> None:
    data = make_synthetic_offline_data(_config())
    action_ids = data.dataset.batch.action_ids

    assert action_ids is not None
    assert action_ids.min() >= 0
    assert action_ids.max() < 7


def test_build_exact_index_from_synthetic_returns_searchable_index() -> None:
    data = make_synthetic_offline_data(_config())
    index = build_exact_index_from_synthetic(data, metric="dot")

    result = index.search(data.action_features[:2], k=3)

    assert result.ids.shape == (2, 3)
    assert result.scores.shape == (2, 3)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("num_transitions", 0, "num_transitions"),
        ("state_dim", 0, "state_dim"),
        ("action_dim", 0, "action_dim"),
        ("num_actions", 0, "num_actions"),
        ("gamma", 1.5, "gamma"),
        ("reward_noise_std", -0.1, "reward_noise_std"),
        ("terminal_prob", 1.5, "terminal_prob"),
    ],
)
def test_invalid_config_values_raise_value_error(field: str, value: object, message: str) -> None:
    kwargs = asdict(_config())
    kwargs[field] = value

    with pytest.raises(ValueError, match=message):
        make_synthetic_offline_data(SyntheticDatasetConfig(**kwargs))


def test_metadata_contains_useful_config_values() -> None:
    config = _config(seed=55)
    data = make_synthetic_offline_data(config)

    assert data.metadata["num_transitions"] == config.num_transitions
    assert data.metadata["state_dim"] == config.state_dim
    assert data.metadata["action_dim"] == config.action_dim
    assert data.metadata["num_actions"] == config.num_actions
    assert data.metadata["seed"] == config.seed
    assert data.metadata["kind"] == "smoke_test_synthetic"
