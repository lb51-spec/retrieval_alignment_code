"""Tests for per-group gradient and parameter diagnostics."""

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
    critic_loss_config,
    density_loss_config,
    dual_loss_config,
    gradient_group_norms,
    make_optimizer_bundle,
    parameter_delta_group_norms,
    parameter_group_norms,
    run_minimal_alternating_loop,
    run_minimal_train_loop,
    run_minimal_train_step,
)


EXPECTED_KEYS = {"encoder", "critic", "density", "dual"}


def _data_model(seed: int = 987):
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


def _group_snapshot(model):
    return {
        "encoder": [
            *(parameter.detach().clone() for parameter in model.state_encoder.parameters()),
            *(parameter.detach().clone() for parameter in model.action_encoder.parameters()),
        ],
        "critic": [parameter.detach().clone() for parameter in model.q_head.parameters()],
        "density": [parameter.detach().clone() for parameter in model.density_head.parameters()],
        "dual": [parameter.detach().clone() for parameter in model.dual_head.parameters()],
    }


def _assert_finite_group_values(values: dict[str, float]) -> None:
    assert set(values) == EXPECTED_KEYS
    for value in values.values():
        assert isinstance(value, float)
        assert torch.isfinite(torch.tensor(value))
        assert value >= 0.0


def test_gradient_group_norms_return_expected_keys_before_backward_zero() -> None:
    _, model = _data_model()

    norms = gradient_group_norms(model)

    _assert_finite_group_values(norms)
    assert all(value == 0.0 for value in norms.values())


def test_parameter_group_norms_return_expected_finite_keys() -> None:
    _, model = _data_model()

    norms = parameter_group_norms(model)

    _assert_finite_group_values(norms)
    assert all(value > 0.0 for value in norms.values())


def test_parameter_delta_group_norms_return_expected_keys() -> None:
    _, model = _data_model()
    before = _group_snapshot(model)

    deltas = parameter_delta_group_norms(before, model)

    _assert_finite_group_values(deltas)
    assert all(value == 0.0 for value in deltas.values())


def test_train_step_diagnostics_include_group_norms_and_positive_gradient() -> None:
    data, model = _data_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    result = run_minimal_train_step(
        data=data,
        model=model,
        optimizer=optimizer,
        batch_size=6,
        k=4,
        gamma=0.95,
        generator=torch.Generator().manual_seed(23),
    )

    grad_norms = result.diagnostics["gradient_group_norms"]
    delta_norms = result.diagnostics["parameter_delta_group_norms"]
    param_norms = result.diagnostics["parameter_group_norms"]
    _assert_finite_group_values(grad_norms)
    _assert_finite_group_values(delta_norms)
    _assert_finite_group_values(param_norms)
    assert any(value > 0.0 for value in grad_norms.values())


def test_critic_only_step_has_positive_critic_gradient_and_delta() -> None:
    data, model = _data_model()
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
        generator=torch.Generator().manual_seed(23),
        loss_config=critic_loss_config(),
    )

    assert result.diagnostics["gradient_group_norms"]["critic"] > 0.0
    assert result.diagnostics["parameter_delta_group_norms"]["critic"] > 0.0
    assert result.diagnostics["parameter_delta_group_norms"]["density"] == 0.0
    assert result.diagnostics["parameter_delta_group_norms"]["dual"] == 0.0


def test_density_only_step_has_positive_density_gradient_and_delta() -> None:
    data, model = _data_model()
    bundle = make_optimizer_bundle(model, mode="separate_heads")

    result = run_minimal_train_step(
        data=data,
        model=model,
        optimizer=None,
        optimizer_bundle=bundle,
        update_parts=["density"],
        batch_size=6,
        k=4,
        gamma=0.95,
        generator=torch.Generator().manual_seed(23),
        loss_config=density_loss_config(),
    )

    assert result.diagnostics["gradient_group_norms"]["density"] > 0.0
    assert result.diagnostics["parameter_delta_group_norms"]["density"] > 0.0
    assert result.diagnostics["parameter_delta_group_norms"]["critic"] == 0.0
    assert result.diagnostics["parameter_delta_group_norms"]["dual"] == 0.0


def test_dual_only_step_has_positive_dual_gradient_and_delta() -> None:
    data, model = _data_model()
    bundle = make_optimizer_bundle(model, mode="separate_heads")

    result = run_minimal_train_step(
        data=data,
        model=model,
        optimizer=None,
        optimizer_bundle=bundle,
        update_parts=["dual"],
        batch_size=6,
        k=4,
        gamma=0.95,
        generator=torch.Generator().manual_seed(23),
        loss_config=dual_loss_config(),
    )

    assert result.diagnostics["gradient_group_norms"]["dual"] > 0.0
    assert result.diagnostics["parameter_delta_group_norms"]["dual"] > 0.0
    assert result.diagnostics["parameter_delta_group_norms"]["critic"] == 0.0
    assert result.diagnostics["parameter_delta_group_norms"]["density"] == 0.0


def test_train_loop_collects_group_norms_by_step() -> None:
    data, model = _data_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    result = run_minimal_train_loop(
        data=data,
        model=model,
        optimizer=optimizer,
        num_steps=2,
        batch_size=6,
        k=4,
        gamma=0.95,
        generator=torch.Generator().manual_seed(23),
    )

    assert len(result.diagnostics["gradient_group_norms_by_step"]) == 2
    assert len(result.diagnostics["parameter_delta_group_norms_by_step"]) == 2
    _assert_finite_group_values(result.diagnostics["gradient_group_norms_by_step"][0])


def test_alternating_loop_collects_group_norms_by_step() -> None:
    data, model = _data_model()
    bundle = make_optimizer_bundle(model, mode="separate_heads")

    result = run_minimal_alternating_loop(
        data=data,
        model=model,
        optimizer_bundle=bundle,
        num_steps=3,
        batch_size=6,
        k=4,
        gamma=0.95,
        update_schedule=[
            {"parts": ["dual"], "loss_config": dual_loss_config()},
            {"parts": ["density"], "loss_config": density_loss_config()},
            {"parts": ["critic"], "loss_config": critic_loss_config()},
        ],
        generator=torch.Generator().manual_seed(23),
    )

    assert len(result.diagnostics["gradient_group_norms_by_step"]) == 3
    assert len(result.diagnostics["parameter_delta_group_norms_by_step"]) == 3
    _assert_finite_group_values(result.diagnostics["parameter_delta_group_norms_by_step"][2])

