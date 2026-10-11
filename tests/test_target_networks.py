"""Tests for target-network cloning and update utilities."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from retrieval_alignment.config import ModelConfig
from retrieval_alignment.networks import build_networks
from retrieval_alignment.target_networks import clone_target_network, hard_update_target, soft_update_target


def _model(seed: int = 0):
    torch.manual_seed(seed)
    return build_networks(
        ModelConfig(
            state_dim=4,
            action_dim=3,
            embedding_dim=8,
            hidden_dims=(16,),
        )
    )


def _all_params(model):
    return [parameter.detach().clone() for parameter in model.parameters()]


def test_clone_has_same_parameter_values_as_source() -> None:
    source = _model()
    target = clone_target_network(source)

    for source_param, target_param in zip(source.parameters(), target.parameters(), strict=True):
        assert torch.equal(source_param, target_param)


def test_clone_is_different_object() -> None:
    source = _model()
    target = clone_target_network(source)

    assert target is not source


def test_clone_parameters_require_grad_false() -> None:
    target = clone_target_network(_model())

    assert all(not parameter.requires_grad for parameter in target.parameters())


def test_hard_update_copies_changed_source_parameters() -> None:
    source = _model(seed=1)
    target = clone_target_network(_model(seed=2))

    hard_update_target(target, source)

    for source_param, target_param in zip(source.parameters(), target.parameters(), strict=True):
        assert torch.equal(source_param, target_param)


def test_soft_update_tau_zero_leaves_target_unchanged() -> None:
    source = _model(seed=1)
    target = clone_target_network(_model(seed=2))
    before = _all_params(target)

    soft_update_target(target, source, tau=0.0)

    for before_param, target_param in zip(before, target.parameters(), strict=True):
        assert torch.equal(before_param, target_param)


def test_soft_update_tau_one_matches_source() -> None:
    source = _model(seed=1)
    target = clone_target_network(_model(seed=2))

    soft_update_target(target, source, tau=1.0)

    for source_param, target_param in zip(source.parameters(), target.parameters(), strict=True):
        assert torch.allclose(source_param, target_param)


def test_soft_update_tau_half_gives_midpoint_values() -> None:
    source = _model(seed=1)
    target = clone_target_network(_model(seed=2))
    before = _all_params(target)

    soft_update_target(target, source, tau=0.5)

    for before_param, source_param, target_param in zip(
        before,
        source.parameters(),
        target.parameters(),
        strict=True,
    ):
        expected = 0.5 * before_param + 0.5 * source_param.detach()
        assert torch.allclose(target_param, expected)


def test_invalid_tau_raises_value_error() -> None:
    source = _model(seed=1)
    target = clone_target_network(_model(seed=2))

    with pytest.raises(ValueError, match="tau"):
        soft_update_target(target, source, tau=-0.1)

    with pytest.raises(ValueError, match="tau"):
        soft_update_target(target, source, tau=1.1)


def test_target_remains_eval_mode_and_frozen_after_updates() -> None:
    source = _model(seed=1)
    target = clone_target_network(_model(seed=2))
    target.train()

    hard_update_target(target, source)
    assert not target.training
    assert all(not parameter.requires_grad for parameter in target.parameters())

    target.train()
    soft_update_target(target, source, tau=0.5)
    assert not target.training
    assert all(not parameter.requires_grad for parameter in target.parameters())


def test_architecture_mismatch_raises_value_error() -> None:
    source = _model(seed=1)
    target = clone_target_network(
        build_networks(
            ModelConfig(
                state_dim=4,
                action_dim=3,
                embedding_dim=6,
                hidden_dims=(16,),
            )
        )
    )

    with pytest.raises(ValueError, match="state_dict"):
        hard_update_target(target, source)
