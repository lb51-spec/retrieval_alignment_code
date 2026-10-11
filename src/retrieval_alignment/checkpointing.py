"""Minimal checkpoint helpers for synthetic DW-MQL experiment scaffolds.

The checkpoint format is intentionally small: model, optional optimizer,
optional target model, trainer counters, config metadata, and synthetic-only
report metadata. FAISS indexes and benchmark artifacts are not serialized.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

try:  # pragma: no cover - dependency guard.
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


CHECKPOINT_SCHEMA_NAME = "dwmql_synthetic_checkpoint"
CHECKPOINT_SCHEMA_VERSION = 1


@dataclass(slots=True)
class CheckpointMetadata:
    """Metadata stored with synthetic experiment checkpoints."""

    schema_name: str = CHECKPOINT_SCHEMA_NAME
    schema_version: int = CHECKPOINT_SCHEMA_VERSION
    step: int = 0
    experiment_name: str = "synthetic_debug"
    synthetic_only: bool = True
    benchmark_claim: bool = False
    diagnostics: dict[str, Any] = field(default_factory=dict)


def _require_torch() -> None:
    if torch is None:
        raise ModuleNotFoundError("PyTorch is required for checkpointing.")


def _optimizer_state_dict(optimizer: Any) -> Any:
    if optimizer is None:
        return None
    if hasattr(optimizer, "state_dict"):
        return optimizer.state_dict()
    state: dict[str, Any] = {}
    for name in (
        "encoder_optimizer",
        "critic_optimizer",
        "density_optimizer",
        "dual_optimizer",
        "joint_optimizer",
    ):
        part = getattr(optimizer, name, None)
        if part is not None:
            state[name] = part.state_dict()
    if not state:
        raise TypeError("optimizer must expose state_dict() or optimizer-bundle fields.")
    return state


def _load_optimizer_state_dict(optimizer: Any, state: Any) -> None:
    if state is None:
        return
    if hasattr(optimizer, "load_state_dict") and not isinstance(state, dict):
        optimizer.load_state_dict(state)
        return
    if hasattr(optimizer, "load_state_dict") and "state" in state and "param_groups" in state:
        optimizer.load_state_dict(state)
        return
    for name, part_state in state.items():
        part = getattr(optimizer, name, None)
        if part is None:
            raise ValueError(f"Optimizer checkpoint contains unavailable part: {name}.")
        part.load_state_dict(part_state)


def _metadata_dict(metadata: CheckpointMetadata | dict[str, Any] | None) -> dict[str, Any]:
    if metadata is None:
        return asdict(CheckpointMetadata())
    if isinstance(metadata, CheckpointMetadata):
        return asdict(metadata)
    if not isinstance(metadata, dict):
        raise TypeError("metadata must be CheckpointMetadata, dict, or None.")
    return dict(metadata)


def _validate_checkpoint_dict(checkpoint: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint must contain a dictionary.")
    if checkpoint.get("schema_name") != CHECKPOINT_SCHEMA_NAME:
        raise ValueError("Checkpoint schema_name is missing or unsupported.")
    if checkpoint.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("Checkpoint schema_version is missing or unsupported.")
    if "model_state_dict" not in checkpoint:
        raise ValueError("Checkpoint is missing model_state_dict.")
    metadata = checkpoint.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("Checkpoint metadata must be present.")
    if metadata.get("benchmark_claim") is not False:
        raise ValueError("Checkpoint benchmark_claim must be false.")
    if metadata.get("synthetic_only") is not True:
        raise ValueError("Checkpoint synthetic_only must be true.")
    return checkpoint


def save_checkpoint(
    path: str | Path,
    model: Any,
    optimizer: Any | None = None,
    target_model: Any | None = None,
    trainer_state: dict[str, Any] | None = None,
    config: Any | None = None,
    metadata: CheckpointMetadata | dict[str, Any] | None = None,
) -> None:
    """Save a synthetic experiment checkpoint with model and optional states."""

    _require_torch()
    checkpoint_path = Path(path)
    if checkpoint_path.exists() and checkpoint_path.is_dir():
        raise ValueError("Checkpoint path must be a file, not a directory.")
    if checkpoint_path.name == "":
        raise ValueError("Checkpoint path must include a file name.")
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_dict = _metadata_dict(metadata)
    if metadata_dict.get("benchmark_claim") is not False:
        raise ValueError("Checkpoint benchmark_claim must be false.")
    if metadata_dict.get("synthetic_only") is not True:
        raise ValueError("Checkpoint synthetic_only must be true.")
    payload = {
        "schema_name": CHECKPOINT_SCHEMA_NAME,
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": _optimizer_state_dict(optimizer) if optimizer is not None else None,
        "target_model_state_dict": None if target_model is None else target_model.state_dict(),
        "trainer_state": {} if trainer_state is None else dict(trainer_state),
        "config": config,
        "metadata": metadata_dict,
    }
    torch.save(payload, checkpoint_path)


def load_checkpoint(path: str | Path, map_location: str | Any = "cpu") -> dict[str, Any]:
    """Load and validate a synthetic experiment checkpoint."""

    _require_torch()
    checkpoint_path = Path(path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    return _validate_checkpoint_dict(checkpoint)


def restore_model_from_checkpoint(model: Any, checkpoint: dict[str, Any]) -> None:
    """Restore a model from a loaded checkpoint dictionary."""

    validated = _validate_checkpoint_dict(checkpoint)
    model.load_state_dict(validated["model_state_dict"])


def restore_optimizer_from_checkpoint(optimizer: Any, checkpoint: dict[str, Any]) -> None:
    """Restore a plain optimizer or optimizer bundle from a checkpoint."""

    validated = _validate_checkpoint_dict(checkpoint)
    state = validated.get("optimizer_state_dict")
    if state is None:
        raise ValueError("Checkpoint does not contain optimizer_state_dict.")
    _load_optimizer_state_dict(optimizer, state)


def restore_target_model_from_checkpoint(target_model: Any, checkpoint: dict[str, Any]) -> None:
    """Restore a target model from a checkpoint when present."""

    validated = _validate_checkpoint_dict(checkpoint)
    state = validated.get("target_model_state_dict")
    if state is None:
        raise ValueError("Checkpoint does not contain target_model_state_dict.")
    target_model.load_state_dict(state)
