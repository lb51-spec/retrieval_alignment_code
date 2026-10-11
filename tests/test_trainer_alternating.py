"""Tests for optimizer-bundle and alternating-update smoke scaffolding."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from retrieval_alignment.config import ModelConfig, SyntheticDatasetConfig
from retrieval_alignment.networks import build_networks
from retrieval_alignment.synthetic import make_synthetic_offline_data
from retrieval_alignment.trainer import (
    OptimizerBundle,
    TrainLoopSmokeResult,
    make_optimizer_bundle,
    run_minimal_alternating_loop,
    run_minimal_train_step,
)


def _data_model(seed: int = 456):
    data = make_synthetic_offline_data(
        SyntheticDatasetConfig(
            num_transitions=40,
            state_dim=4,
            action_dim=3,
            num_actions=10,
            seed=seed,
            terminal_prob=0.2,
        )
    )
    torch.manual_seed(seed)
    model = build_networks(
        ModelConfig(
            state_dim=4,
            action_dim=3,
            embedding_dim=8,
            hidden_dims=(16,),
            pair_feature_mode="concat_product",
        )
    )
    return data, model


def _params(module):
    return [parameter.detach().clone() for parameter in module.parameters()]


def _changed(before, module) -> bool:
    return any(
        not torch.equal(before_param, after_param)
        for before_param, after_param in zip(before, module.parameters(), strict=True)
    )


def _unchanged(before, module) -> bool:
    return all(
        torch.equal(before_param, after_param)
        for before_param, after_param in zip(before, module.parameters(), strict=True)
    )


def test_make_optimizer_bundle_joint_creates_only_joint_optimizer() -> None:
    _, model = _data_model()

    bundle = make_optimizer_bundle(model, mode="joint")

    assert isinstance(bundle, OptimizerBundle)
    assert bundle.joint_optimizer is not None
    assert bundle.encoder_optimizer is None
    assert bundle.critic_optimizer is None
    assert bundle.density_optimizer is None
    assert bundle.dual_optimizer is None


def test_make_optimizer_bundle_separate_heads_creates_expected_optimizers() -> None:
    _, model = _data_model()

    bundle = make_optimizer_bundle(model, mode="separate_heads")

    assert bundle.joint_optimizer is None
    assert bundle.encoder_optimizer is not None
    assert bundle.critic_optimizer is not None
    assert bundle.density_optimizer is not None
    assert bundle.dual_optimizer is not None


def test_make_optimizer_bundle_invalid_mode_raises_value_error() -> None:
    _, model = _data_model()

    with pytest.raises(ValueError, match="mode"):
        make_optimizer_bundle(model, mode="split")


def test_optimizer_and_optimizer_bundle_together_raise_value_error() -> None:
    data, model = _data_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    bundle = make_optimizer_bundle(model, mode="joint")

    with pytest.raises(ValueError, match="either optimizer or optimizer_bundle"):
        run_minimal_train_step(
            data=data,
            model=model,
            optimizer=optimizer,
            optimizer_bundle=bundle,
            batch_size=6,
            k=4,
            gamma=0.95,
        )


def test_optimizer_or_bundle_is_required() -> None:
    data, model = _data_model()

    with pytest.raises(ValueError, match="optimizer"):
        run_minimal_train_step(
            data=data,
            model=model,
            optimizer=None,
            batch_size=6,
            k=4,
            gamma=0.95,
        )


def test_optimizer_bundle_can_run_one_train_step() -> None:
    data, model = _data_model()
    bundle = make_optimizer_bundle(model, mode="joint")

    result = run_minimal_train_step(
        data=data,
        model=model,
        optimizer=None,
        optimizer_bundle=bundle,
        batch_size=6,
        k=4,
        gamma=0.95,
        generator=torch.Generator().manual_seed(11),
    )

    assert torch.isfinite(result.total_loss)
    assert result.parameter_delta_norm.item() > 0
    assert result.diagnostics["stepped_parts"] == ["joint"]


def test_critic_update_part_changes_q_head_only_among_heads() -> None:
    data, model = _data_model()
    bundle = make_optimizer_bundle(model, mode="separate_heads")
    q_before = _params(model.q_head)
    density_before = _params(model.density_head)
    dual_before = _params(model.dual_head)

    result = run_minimal_train_step(
        data=data,
        model=model,
        optimizer=None,
        optimizer_bundle=bundle,
        update_parts=["critic"],
        batch_size=6,
        k=4,
        gamma=0.95,
        generator=torch.Generator().manual_seed(11),
    )

    assert result.diagnostics["stepped_parts"] == ["critic"]
    assert _changed(q_before, model.q_head)
    assert _unchanged(density_before, model.density_head)
    assert _unchanged(dual_before, model.dual_head)


def test_density_update_part_changes_density_head_only_among_heads() -> None:
    data, model = _data_model()
    bundle = make_optimizer_bundle(model, mode="separate_heads")
    q_before = _params(model.q_head)
    density_before = _params(model.density_head)
    dual_before = _params(model.dual_head)

    result = run_minimal_train_step(
        data=data,
        model=model,
        optimizer=None,
        optimizer_bundle=bundle,
        update_parts=["density"],
        batch_size=6,
        k=4,
        gamma=0.95,
        generator=torch.Generator().manual_seed(11),
    )

    assert result.diagnostics["stepped_parts"] == ["density"]
    assert _unchanged(q_before, model.q_head)
    assert _changed(density_before, model.density_head)
    assert _unchanged(dual_before, model.dual_head)


def test_dual_update_part_changes_dual_head_only_among_heads() -> None:
    data, model = _data_model()
    bundle = make_optimizer_bundle(model, mode="separate_heads")
    q_before = _params(model.q_head)
    density_before = _params(model.density_head)
    dual_before = _params(model.dual_head)

    result = run_minimal_train_step(
        data=data,
        model=model,
        optimizer=None,
        optimizer_bundle=bundle,
        update_parts=["dual"],
        batch_size=6,
        k=4,
        gamma=0.95,
        generator=torch.Generator().manual_seed(11),
    )

    assert result.diagnostics["stepped_parts"] == ["dual"]
    assert _unchanged(q_before, model.q_head)
    assert _unchanged(density_before, model.density_head)
    assert _changed(dual_before, model.dual_head)


def test_run_minimal_alternating_loop_runs_small_schedule() -> None:
    data, model = _data_model()
    bundle = make_optimizer_bundle(model, mode="separate_heads")

    result = run_minimal_alternating_loop(
        data=data,
        model=model,
        optimizer_bundle=bundle,
        num_steps=4,
        batch_size=6,
        k=4,
        gamma=0.95,
        update_schedule=[["dual"], ["density"], ["encoder", "critic"]],
        generator=torch.Generator().manual_seed(11),
    )

    assert isinstance(result, TrainLoopSmokeResult)
    assert result.num_steps == 4
    assert len(result.losses) == 4
    assert len(result.grad_norms) == 4
    assert all(torch.isfinite(torch.tensor(loss)) for loss in result.losses)
    assert result.diagnostics["update_schedule"] == [
        ["dual"],
        ["density"],
        ["encoder", "critic"],
        ["dual"],
    ]

