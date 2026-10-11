"""Tests for opt-in split-backward scaffolding."""

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
    SyntheticDWMQLTrainer,
    critic_loss_config,
    default_smoke_alternating_schedule,
    density_loss_config,
    dual_loss_config,
    make_optimizer_bundle,
    run_minimal_train_step,
)


def _data_model_optimizer(seed: int = 975):
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
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    return data, model, optimizer


def _params(module):
    return [parameter.detach().clone() for parameter in module.parameters()]


def _changed(before, module) -> bool:
    return any(
        not torch.equal(before_param, after_param)
        for before_param, after_param in zip(before, module.parameters(), strict=True)
    )


def test_default_split_backward_false_preserves_joint_diagnostics() -> None:
    data, model, optimizer = _data_model_optimizer()

    result = run_minimal_train_step(
        data=data,
        model=model,
        optimizer=optimizer,
        batch_size=6,
        k=4,
        gamma=0.95,
        generator=torch.Generator().manual_seed(41),
    )

    assert result.diagnostics["split_backward"] is False
    assert result.diagnostics["backward_blocks_run"] == ["joint"]


def test_split_backward_requires_optimizer_bundle() -> None:
    data, model, optimizer = _data_model_optimizer()

    with pytest.raises(ValueError, match="optimizer_bundle"):
        run_minimal_train_step(
            data=data,
            model=model,
            optimizer=optimizer,
            batch_size=6,
            k=4,
            gamma=0.95,
            split_backward=True,
        )


def test_split_backward_runs_with_separate_heads_bundle() -> None:
    data, model, _ = _data_model_optimizer()
    bundle = make_optimizer_bundle(model, mode="separate_heads")

    result = run_minimal_train_step(
        data=data,
        model=model,
        optimizer=None,
        optimizer_bundle=bundle,
        update_parts=["critic"],
        batch_size=6,
        k=4,
        gamma=0.95,
        generator=torch.Generator().manual_seed(41),
        loss_config=critic_loss_config(),
        split_backward=True,
    )

    assert torch.isfinite(result.total_loss)
    assert result.diagnostics["split_backward"] is True
    assert result.diagnostics["backward_blocks_run"] == ["bellman"]


def test_split_backward_critic_loss_updates_critic_parameters() -> None:
    data, model, _ = _data_model_optimizer()
    bundle = make_optimizer_bundle(model, mode="separate_heads")
    before = _params(model.q_head)

    run_minimal_train_step(
        data=data,
        model=model,
        optimizer=None,
        optimizer_bundle=bundle,
        update_parts=["critic"],
        batch_size=6,
        k=4,
        gamma=0.95,
        generator=torch.Generator().manual_seed(41),
        loss_config=critic_loss_config(),
        split_backward=True,
    )

    assert _changed(before, model.q_head)


def test_split_backward_density_loss_updates_density_parameters() -> None:
    data, model, _ = _data_model_optimizer()
    bundle = make_optimizer_bundle(model, mode="separate_heads")
    before = _params(model.density_head)

    result = run_minimal_train_step(
        data=data,
        model=model,
        optimizer=None,
        optimizer_bundle=bundle,
        update_parts=["density"],
        batch_size=6,
        k=4,
        gamma=0.95,
        generator=torch.Generator().manual_seed(41),
        loss_config=density_loss_config(),
        split_backward=True,
    )

    assert result.diagnostics["backward_blocks_run"] == ["stationarity", "density_regularization"]
    assert _changed(before, model.density_head)


def test_split_backward_dual_loss_updates_dual_parameters() -> None:
    data, model, _ = _data_model_optimizer()
    bundle = make_optimizer_bundle(model, mode="separate_heads")
    before = _params(model.dual_head)

    result = run_minimal_train_step(
        data=data,
        model=model,
        optimizer=None,
        optimizer_bundle=bundle,
        update_parts=["dual"],
        batch_size=6,
        k=4,
        gamma=0.95,
        generator=torch.Generator().manual_seed(41),
        loss_config=dual_loss_config(),
        split_backward=True,
    )

    assert result.diagnostics["backward_blocks_run"] == ["stationarity"]
    assert _changed(before, model.dual_head)


def test_synthetic_trainer_works_with_split_backward() -> None:
    data, model, _ = _data_model_optimizer()
    bundle = make_optimizer_bundle(model, mode="separate_heads")
    trainer = SyntheticDWMQLTrainer(
        data=data,
        model=model,
        optimizer_bundle=bundle,
        batch_size=6,
        k=4,
        gamma=0.95,
        split_backward=True,
    )

    history = trainer.train(3, update_schedule=default_smoke_alternating_schedule())

    assert len(history.losses) == 3
    assert history.diagnostics["split_backward"] is True
    assert history.diagnostics["update_parts_by_step"] == [
        ["dual"],
        ["density"],
        ["encoder", "critic"],
    ]

