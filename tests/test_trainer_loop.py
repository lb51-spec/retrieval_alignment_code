"""Tests for the tiny synthetic multi-step train-loop smoke utility."""

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
from retrieval_alignment.target_networks import clone_target_network
from retrieval_alignment.trainer import TrainLoopSmokeResult, run_minimal_train_loop


def _data_model_optimizer(seed: int = 321):
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


def test_loop_returns_result_and_collects_per_step_values() -> None:
    data, model, optimizer = _data_model_optimizer()

    result = run_minimal_train_loop(
        data=data,
        model=model,
        optimizer=optimizer,
        num_steps=3,
        batch_size=6,
        k=4,
        gamma=0.95,
        generator=torch.Generator().manual_seed(7),
    )

    assert isinstance(result, TrainLoopSmokeResult)
    assert result.num_steps == 3
    assert len(result.losses) == 3
    assert len(result.grad_norms) == 3
    assert len(result.parameter_delta_norms) == 3
    assert result.initial_total_loss == result.losses[0]
    assert result.final_total_loss == result.losses[-1]


def test_loop_losses_and_grad_norms_are_finite_and_positive() -> None:
    data, model, optimizer = _data_model_optimizer()

    result = run_minimal_train_loop(
        data=data,
        model=model,
        optimizer=optimizer,
        num_steps=3,
        batch_size=6,
        k=4,
        gamma=0.95,
        generator=torch.Generator().manual_seed(7),
    )

    assert all(torch.isfinite(torch.tensor(value)) for value in result.losses)
    assert all(torch.isfinite(torch.tensor(value)) and value > 0 for value in result.grad_norms)
    assert all(
        torch.isfinite(torch.tensor(value)) and value > 0
        for value in result.parameter_delta_norms
    )


def test_model_parameters_change_across_loop() -> None:
    data, model, optimizer = _data_model_optimizer()
    before = [parameter.detach().clone() for parameter in model.parameters()]

    run_minimal_train_loop(
        data=data,
        model=model,
        optimizer=optimizer,
        num_steps=3,
        batch_size=6,
        k=4,
        gamma=0.95,
        generator=torch.Generator().manual_seed(7),
    )

    assert any(
        not torch.equal(before_param, after_param)
        for before_param, after_param in zip(before, model.parameters(), strict=True)
    )


def test_fixed_target_model_is_not_changed_without_tau() -> None:
    data, model, optimizer = _data_model_optimizer()
    target_model = clone_target_network(model)
    target_before = [parameter.detach().clone() for parameter in target_model.parameters()]

    result = run_minimal_train_loop(
        data=data,
        model=model,
        optimizer=optimizer,
        num_steps=3,
        batch_size=6,
        k=4,
        gamma=0.95,
        generator=torch.Generator().manual_seed(7),
        target_model=target_model,
        target_tau=None,
    )

    assert result.target_update_count == 0
    assert all(
        torch.equal(before_param, after_param)
        for before_param, after_param in zip(target_before, target_model.parameters(), strict=True)
    )


def test_target_model_changes_when_tau_is_provided() -> None:
    data, model, optimizer = _data_model_optimizer()
    target_model = clone_target_network(model)
    target_before = [parameter.detach().clone() for parameter in target_model.parameters()]

    result = run_minimal_train_loop(
        data=data,
        model=model,
        optimizer=optimizer,
        num_steps=3,
        batch_size=6,
        k=4,
        gamma=0.95,
        generator=torch.Generator().manual_seed(7),
        target_model=target_model,
        target_tau=0.5,
    )

    assert result.target_update_count == 3
    assert any(
        not torch.equal(before_param, after_param)
        for before_param, after_param in zip(target_before, target_model.parameters(), strict=True)
    )
    assert all(not parameter.requires_grad for parameter in target_model.parameters())


def test_target_update_count_respects_update_period() -> None:
    data, model, optimizer = _data_model_optimizer()
    target_model = clone_target_network(model)

    result = run_minimal_train_loop(
        data=data,
        model=model,
        optimizer=optimizer,
        num_steps=5,
        batch_size=6,
        k=4,
        gamma=0.95,
        generator=torch.Generator().manual_seed(7),
        target_model=target_model,
        target_tau=0.5,
        target_update_every=2,
    )

    assert result.target_update_count == 2


def test_invalid_num_steps_raises_value_error() -> None:
    data, model, optimizer = _data_model_optimizer()

    with pytest.raises(ValueError, match="num_steps"):
        run_minimal_train_loop(data, model, optimizer, num_steps=0, batch_size=6, k=4, gamma=0.95)


def test_invalid_target_update_every_raises_value_error() -> None:
    data, model, optimizer = _data_model_optimizer()

    with pytest.raises(ValueError, match="target_update_every"):
        run_minimal_train_loop(
            data,
            model,
            optimizer,
            num_steps=3,
            batch_size=6,
            k=4,
            gamma=0.95,
            target_update_every=0,
        )


def test_target_tau_without_target_model_raises_value_error() -> None:
    data, model, optimizer = _data_model_optimizer()

    with pytest.raises(ValueError, match="target_tau"):
        run_minimal_train_loop(
            data,
            model,
            optimizer,
            num_steps=3,
            batch_size=6,
            k=4,
            gamma=0.95,
            target_tau=0.5,
        )

