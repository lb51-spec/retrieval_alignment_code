"""Tiny deterministic synthetic data for integration smoke tests.

This module is not a benchmark and should not be used for paper claims. It
exists only to validate that early pipeline components can connect:

``dataset -> action catalogue -> exact index -> candidate sets -> policies -> losses``.

The generated transitions are simple random tensors with a stable bilinear
reward signal and optional reward noise. No real environment dynamics are
modeled beyond sampled next states.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

try:  # pragma: no cover - exercised when dependencies are installed.
    import torch
except ModuleNotFoundError:  # pragma: no cover - keeps imports clean pre-install.
    torch = None  # type: ignore[assignment]

from .ann_index import ExactActionIndex
from .config import SyntheticDatasetConfig
from .datasets import InMemoryReplayDataset


def _require_torch() -> None:
    if torch is None:
        raise ModuleNotFoundError("PyTorch is required to generate synthetic data.")


@dataclass(slots=True)
class SyntheticOfflineData:
    """Generated smoke-test offline data and action catalogue."""

    dataset: InMemoryReplayDataset
    action_features: "torch.Tensor | Any"
    action_ids: "torch.Tensor | Any"
    reward_weights: "torch.Tensor | Any"
    metadata: dict[str, Any]


def _validate_config(config: SyntheticDatasetConfig) -> None:
    if config.num_transitions <= 0:
        raise ValueError("num_transitions must be positive.")
    if config.state_dim <= 0:
        raise ValueError("state_dim must be positive.")
    if config.action_dim <= 0:
        raise ValueError("action_dim must be positive.")
    if config.num_actions <= 0:
        raise ValueError("num_actions must be positive.")
    if not 0.0 <= config.gamma <= 1.0:
        raise ValueError("gamma must be in [0, 1].")
    if config.reward_noise_std < 0:
        raise ValueError("reward_noise_std must be nonnegative.")
    if not 0.0 <= config.terminal_prob <= 1.0:
        raise ValueError("terminal_prob must be in [0, 1].")


def make_synthetic_offline_data(config: SyntheticDatasetConfig | None = None) -> SyntheticOfflineData:
    """Create deterministic offline transitions and an action catalogue.

    Args:
        config: Synthetic smoke-test configuration. The same seed produces the
            same tensors.

    Returns:
        ``SyntheticOfflineData`` containing an ``InMemoryReplayDataset``,
        action features, action ids, reward weights, and metadata.
    """

    _require_torch()
    config = config or SyntheticDatasetConfig()
    _validate_config(config)
    generator = torch.Generator().manual_seed(config.seed)

    n = config.num_transitions
    states = torch.randn(n, config.state_dim, generator=generator)
    next_states = states + 0.1 * torch.randn(n, config.state_dim, generator=generator)

    action_features = torch.randn(config.num_actions, config.action_dim, generator=generator)
    # Avoid accidental zero vectors so cosine index smoke tests are well-defined.
    action_features = action_features + 0.01
    action_ids = torch.arange(config.num_actions, dtype=torch.long)

    logged_action_ids = torch.randint(config.num_actions, (n,), generator=generator)
    next_logged_action_ids = torch.randint(config.num_actions, (n,), generator=generator)
    actions = action_features[logged_action_ids]

    reward_weights = torch.randn(config.state_dim, config.action_dim, generator=generator)
    deterministic_rewards = (states @ reward_weights * actions).sum(dim=1)
    if config.reward_noise_std > 0:
        noise = config.reward_noise_std * torch.randn(n, generator=generator)
    else:
        noise = torch.zeros(n)
    rewards = deterministic_rewards + noise

    dones = torch.rand(n, generator=generator) < config.terminal_prob
    metadata = {
        "num_transitions": config.num_transitions,
        "state_dim": config.state_dim,
        "action_dim": config.action_dim,
        "num_actions": config.num_actions,
        "gamma": config.gamma,
        "seed": config.seed,
        "reward_noise_std": config.reward_noise_std,
        "terminal_prob": config.terminal_prob,
        "kind": "smoke_test_synthetic",
    }
    dataset = InMemoryReplayDataset(
        states=states,
        actions=actions,
        rewards=rewards,
        next_states=next_states,
        dones=dones,
        action_ids=logged_action_ids,
        next_logged_action_ids=next_logged_action_ids,
        metadata=metadata,
    )
    return SyntheticOfflineData(
        dataset=dataset,
        action_features=action_features,
        action_ids=action_ids,
        reward_weights=reward_weights,
        metadata=metadata,
    )


def build_exact_index_from_synthetic(
    data: SyntheticOfflineData,
    metric: str = "dot",
) -> ExactActionIndex:
    """Build an ``ExactActionIndex`` over synthetic action features."""

    return ExactActionIndex(
        data.action_features,
        action_ids=data.action_ids,
        metric=metric,
        normalize=(metric == "cosine"),
    )
