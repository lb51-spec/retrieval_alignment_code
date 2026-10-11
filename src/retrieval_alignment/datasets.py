"""Offline replay data containers for DW-MQL+ANN.

The training flow in ``docs/algorithm_pseudocode.md`` starts from an offline
replay batch of ``(s, a, r, s', done)`` transitions. This module provides a
minimal validated ``ReplayBatch`` and an in-memory replay dataset for early
experiments. It is intentionally independent of FAISS, networks, losses, and
training loops.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

try:  # pragma: no cover - exercised when the project dependencies are installed.
    import torch
except ModuleNotFoundError:  # pragma: no cover - keeps imports clean pre-install.
    torch = None  # type: ignore[assignment]


def _require_torch() -> None:
    if torch is None:
        raise ModuleNotFoundError("PyTorch is required to use replay datasets.")


def _as_float_tensor(values: Any, *, name: str) -> "torch.Tensor":
    _require_torch()
    tensor = torch.as_tensor(values)
    if not torch.is_floating_point(tensor):
        tensor = tensor.float()
    if torch.any(~torch.isfinite(tensor)):
        raise ValueError(f"{name} must contain only finite values.")
    return tensor


def _as_feature_matrix(values: Any, *, name: str) -> "torch.Tensor":
    tensor = _as_float_tensor(values, name=name)
    if tensor.ndim != 2:
        raise ValueError(f"{name} must have shape [batch, dim].")
    if tensor.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one transition.")
    if tensor.shape[1] == 0:
        raise ValueError(f"{name} must have positive feature dimension.")
    return tensor


def _as_vector(values: Any, *, name: str, batch_size: int | None = None) -> "torch.Tensor":
    tensor = _as_float_tensor(values, name=name)
    if tensor.ndim == 2 and tensor.shape[1] == 1:
        tensor = tensor.squeeze(1)
    if tensor.ndim != 1:
        raise ValueError(f"{name} must have shape [batch] or [batch, 1].")
    if batch_size is not None and tensor.shape[0] != batch_size:
        raise ValueError(f"{name} length {tensor.shape[0]} must match batch size {batch_size}.")
    return tensor


def _as_done_vector(values: Any, *, batch_size: int) -> "torch.Tensor":
    _require_torch()
    tensor = torch.as_tensor(values)
    if tensor.ndim == 2 and tensor.shape[1] == 1:
        tensor = tensor.squeeze(1)
    if tensor.ndim != 1:
        raise ValueError("dones must have shape [batch] or [batch, 1].")
    if tensor.shape[0] != batch_size:
        raise ValueError(f"dones length {tensor.shape[0]} must match batch size {batch_size}.")
    if tensor.dtype == torch.bool:
        return tensor
    if not torch.is_floating_point(tensor):
        tensor = tensor.float()
    if torch.any(~torch.isfinite(tensor)):
        raise ValueError("dones must contain only finite values.")
    if torch.any((tensor < 0) | (tensor > 1)):
        raise ValueError("numeric dones must contain values in [0, 1].")
    return tensor


def _optional_ids(values: Any | None, *, name: str, batch_size: int, device: "torch.device") -> "torch.Tensor | None":
    if values is None:
        return None
    _require_torch()
    tensor = torch.as_tensor(values, dtype=torch.long, device=device)
    if tensor.ndim == 2 and tensor.shape[1] == 1:
        tensor = tensor.squeeze(1)
    if tensor.ndim != 1:
        raise ValueError(f"{name} must have shape [batch] or [batch, 1].")
    if tensor.shape[0] != batch_size:
        raise ValueError(f"{name} length {tensor.shape[0]} must match batch size {batch_size}.")
    return tensor


@dataclass(slots=True)
class ReplayBatch:
    """Validated mini-batch of offline RL transitions.

    Fields correspond to logged transitions needed by DW-MQL+ANN. Optional
    action ids connect continuous action features to catalogue indices used by
    candidate retrieval.
    """

    states: "torch.Tensor | Any"
    actions: "torch.Tensor | Any"
    rewards: "torch.Tensor | Any"
    next_states: "torch.Tensor | Any"
    dones: "torch.Tensor | Any"
    action_ids: "torch.Tensor | Any | None" = None
    next_logged_action_ids: "torch.Tensor | Any | None" = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        states = _as_feature_matrix(self.states, name="states")
        actions = _as_feature_matrix(self.actions, name="actions").to(device=states.device)
        batch_size = states.shape[0]
        if actions.shape[0] != batch_size:
            raise ValueError(f"actions batch size {actions.shape[0]} must match states batch size {batch_size}.")

        rewards = _as_vector(self.rewards, name="rewards", batch_size=batch_size).to(device=states.device)
        next_states = _as_feature_matrix(self.next_states, name="next_states").to(device=states.device)
        if next_states.shape[0] != batch_size:
            raise ValueError(
                f"next_states batch size {next_states.shape[0]} must match states batch size {batch_size}."
            )
        if next_states.shape[1] != states.shape[1]:
            raise ValueError("next_states feature dimension must match states feature dimension.")
        dones = _as_done_vector(self.dones, batch_size=batch_size).to(device=states.device)

        self.states = states
        self.actions = actions
        self.rewards = rewards
        self.next_states = next_states
        self.dones = dones
        self.action_ids = _optional_ids(
            self.action_ids,
            name="action_ids",
            batch_size=batch_size,
            device=states.device,
        )
        self.next_logged_action_ids = _optional_ids(
            self.next_logged_action_ids,
            name="next_logged_action_ids",
            batch_size=batch_size,
            device=states.device,
        )
        self.metadata = dict(self.metadata or {})

    @property
    def batch_size(self) -> int:
        """Number of transitions in the batch."""

        return int(self.states.shape[0])

    @property
    def state_dim(self) -> int:
        """State feature dimension."""

        return int(self.states.shape[1])

    @property
    def action_dim(self) -> int:
        """Action feature dimension."""

        return int(self.actions.shape[1])

    def to(self, device: "torch.device | str") -> "ReplayBatch":
        """Return a new ``ReplayBatch`` with tensors moved to ``device``."""

        return ReplayBatch(
            states=self.states.to(device),
            actions=self.actions.to(device),
            rewards=self.rewards.to(device),
            next_states=self.next_states.to(device),
            dones=self.dones.to(device),
            action_ids=None if self.action_ids is None else self.action_ids.to(device),
            next_logged_action_ids=(
                None if self.next_logged_action_ids is None else self.next_logged_action_ids.to(device)
            ),
            metadata=dict(self.metadata),
        )


# Backwards-compatible alias for the initial implementation.
TransitionBatch = ReplayBatch


class InMemoryReplayDataset:
    """Simple in-memory offline replay dataset.

    Sampling uses ``torch.randint`` with replacement or ``torch.randperm``
    without replacement and is deterministic when a seeded ``torch.Generator``
    is supplied.
    """

    def __init__(
        self,
        states: Any,
        actions: Any,
        rewards: Any,
        next_states: Any,
        dones: Any,
        action_ids: Any | None = None,
        next_logged_action_ids: Any | None = None,
        *,
        replacement: bool = True,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.batch = ReplayBatch(
            states=states,
            actions=actions,
            rewards=rewards,
            next_states=next_states,
            dones=dones,
            action_ids=action_ids,
            next_logged_action_ids=next_logged_action_ids,
            metadata=metadata or {},
        )
        self.replacement = bool(replacement)

    @classmethod
    def from_transitions(
        cls,
        states: Any,
        actions: Any,
        rewards: Any,
        next_states: Any,
        dones: Any,
        **kwargs: Any,
    ) -> "InMemoryReplayDataset":
        """Construct an in-memory dataset from transition tensors."""

        return cls(states, actions, rewards, next_states, dones, **kwargs)

    def __len__(self) -> int:
        return self.batch.batch_size

    def sample(
        self,
        batch_size: int,
        *,
        generator: "torch.Generator | None" = None,
        replacement: bool | None = None,
    ) -> ReplayBatch:
        """Sample a mini-batch of transitions."""

        _require_torch()
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer.")
        use_replacement = self.replacement if replacement is None else bool(replacement)
        dataset_size = len(self)
        if not use_replacement and batch_size > dataset_size:
            raise ValueError("Cannot sample without replacement with batch_size > dataset size.")

        # Draw indices on the generator's device (CPU) for reproducibility, then
        # move them to the data device. This keeps the RNG stream identical
        # regardless of whether the data lives on CPU or GPU (a CPU generator
        # cannot drive device='cuda' sampling directly).
        target_device = self.batch.states.device
        if use_replacement:
            indices = torch.randint(dataset_size, (batch_size,), generator=generator)
        else:
            indices = torch.randperm(dataset_size, generator=generator)[:batch_size]
        indices = indices.to(target_device)

        return ReplayBatch(
            states=self.batch.states[indices],
            actions=self.batch.actions[indices],
            rewards=self.batch.rewards[indices],
            next_states=self.batch.next_states[indices],
            dones=self.batch.dones[indices],
            action_ids=None if self.batch.action_ids is None else self.batch.action_ids[indices],
            next_logged_action_ids=(
                None if self.batch.next_logged_action_ids is None else self.batch.next_logged_action_ids[indices]
            ),
            metadata=dict(self.batch.metadata),
        )


# Backwards-compatible wrapper name from the earlier implementation.
OfflineDataset = InMemoryReplayDataset
