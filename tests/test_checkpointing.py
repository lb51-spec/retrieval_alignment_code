"""Tests for synthetic experiment checkpoint helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from retrieval_alignment.checkpointing import (  # noqa: E402
    CHECKPOINT_SCHEMA_NAME,
    CHECKPOINT_SCHEMA_VERSION,
    CheckpointMetadata,
    load_checkpoint,
    restore_model_from_checkpoint,
    restore_optimizer_from_checkpoint,
    save_checkpoint,
)
from retrieval_alignment.config import ModelConfig  # noqa: E402
from retrieval_alignment.networks import build_networks  # noqa: E402


def _model() -> object:
    torch.manual_seed(0)
    return build_networks(ModelConfig(state_dim=4, action_dim=3, embedding_dim=8))


def test_save_and_load_checkpoint(tmp_path: Path) -> None:
    model = _model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    path = tmp_path / "ckpt" / "checkpoint.pt"

    save_checkpoint(
        path,
        model,
        optimizer=optimizer,
        metadata=CheckpointMetadata(step=3, experiment_name="unit"),
    )
    checkpoint = load_checkpoint(path)

    assert path.exists()
    assert checkpoint["schema_name"] == CHECKPOINT_SCHEMA_NAME
    assert checkpoint["schema_version"] == CHECKPOINT_SCHEMA_VERSION
    assert checkpoint["metadata"]["benchmark_claim"] is False
    assert checkpoint["metadata"]["synthetic_only"] is True


def test_restore_model_from_checkpoint_restores_parameters(tmp_path: Path) -> None:
    model = _model()
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, model, metadata=CheckpointMetadata(step=1))

    restored = _model()
    with torch.no_grad():
        for parameter in restored.parameters():
            parameter.add_(1.0)

    restore_model_from_checkpoint(restored, load_checkpoint(path))

    for expected, actual in zip(model.parameters(), restored.parameters(), strict=True):
        assert torch.allclose(expected, actual)


def test_restore_optimizer_from_checkpoint_restores_state(tmp_path: Path) -> None:
    model = _model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss = sum(parameter.square().sum() for parameter in model.parameters())
    loss.backward()
    optimizer.step()
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, model, optimizer=optimizer, metadata=CheckpointMetadata(step=1))

    restored_model = _model()
    restored_optimizer = torch.optim.AdamW(restored_model.parameters(), lr=1e-3)
    restore_optimizer_from_checkpoint(restored_optimizer, load_checkpoint(path))

    assert restored_optimizer.state_dict()["state"]


def test_missing_checkpoint_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_checkpoint(tmp_path / "missing.pt")


def test_invalid_checkpoint_raises(tmp_path: Path) -> None:
    path = tmp_path / "invalid.pt"
    torch.save({"schema_name": CHECKPOINT_SCHEMA_NAME, "schema_version": CHECKPOINT_SCHEMA_VERSION}, path)

    with pytest.raises(ValueError, match="model_state_dict"):
        load_checkpoint(path)


def test_save_checkpoint_rejects_directory_path(tmp_path: Path) -> None:
    model = _model()

    with pytest.raises(ValueError, match="file"):
        save_checkpoint(tmp_path, model, metadata=CheckpointMetadata())
