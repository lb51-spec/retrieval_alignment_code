"""Tests for explicit loss-component controls in the minimal trainer."""

from __future__ import annotations

import sys
from dataclasses import replace
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
    LossComponentConfig,
    make_optimizer_bundle,
    run_minimal_train_step,
)


def _data_model_optimizer(seed: int = 789):
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


def _step_with_config(loss_config: LossComponentConfig):
    data, model, optimizer = _data_model_optimizer()
    return run_minimal_train_step(
        data=data,
        model=model,
        optimizer=optimizer,
        batch_size=6,
        k=4,
        gamma=0.95,
        generator=torch.Generator().manual_seed(17),
        loss_config=loss_config,
        density_reg_weight=1.0,
    )


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


def test_default_loss_component_config_is_used() -> None:
    data, model, optimizer = _data_model_optimizer()

    result = run_minimal_train_step(
        data=data,
        model=model,
        optimizer=optimizer,
        batch_size=6,
        k=4,
        gamma=0.95,
        generator=torch.Generator().manual_seed(17),
    )

    assert result.diagnostics["loss_config"] == {
        "use_bellman_loss": True,
        "use_stationarity_loss": True,
        "use_density_regularization": True,
        "detach_density_weights_for_bellman": True,
        "detach_policy_probs_for_bellman": True,
    }
    assert result.diagnostics["active_losses"] == [
        "bellman",
        "stationarity",
        "density_regularization",
    ]
    assert result.diagnostics["bellman_weights_detached"] is True
    assert result.diagnostics["bellman_policy_probs_detached"] is True


def test_invalid_loss_config_non_bool_field_raises_type_error() -> None:
    bad_config = LossComponentConfig()
    bad_config.use_bellman_loss = "yes"  # type: ignore[assignment]

    with pytest.raises(TypeError, match="use_bellman_loss"):
        _step_with_config(bad_config)


def test_disabling_all_loss_components_raises_value_error() -> None:
    with pytest.raises(ValueError, match="At least one loss component"):
        _step_with_config(
            LossComponentConfig(
                use_bellman_loss=False,
                use_stationarity_loss=False,
                use_density_regularization=False,
            )
        )


def test_bellman_only_step_uses_only_bellman_loss() -> None:
    result = _step_with_config(
        LossComponentConfig(
            use_bellman_loss=True,
            use_stationarity_loss=False,
            use_density_regularization=False,
        )
    )

    assert result.diagnostics["active_losses"] == ["bellman"]
    assert torch.allclose(result.total_loss, result.bellman_loss)


def test_stationarity_only_step_uses_only_stationarity_loss() -> None:
    result = _step_with_config(
        LossComponentConfig(
            use_bellman_loss=False,
            use_stationarity_loss=True,
            use_density_regularization=False,
        )
    )

    assert result.diagnostics["active_losses"] == ["stationarity"]
    assert torch.allclose(result.total_loss, result.stationarity_loss)


def test_density_regularization_only_step_uses_only_density_regularization() -> None:
    result = _step_with_config(
        LossComponentConfig(
            use_bellman_loss=False,
            use_stationarity_loss=False,
            use_density_regularization=True,
        )
    )

    assert result.diagnostics["active_losses"] == ["density_regularization"]
    assert torch.allclose(result.total_loss, result.density_regularization)


def test_detached_density_weights_prevent_density_head_update_in_bellman_only_density_step() -> None:
    data, model, _ = _data_model_optimizer()
    bundle = make_optimizer_bundle(model, mode="separate_heads")
    density_before = _params(model.density_head)

    result = run_minimal_train_step(
        data=data,
        model=model,
        optimizer=None,
        optimizer_bundle=bundle,
        update_parts=["density"],
        batch_size=6,
        k=4,
        gamma=0.95,
        generator=torch.Generator().manual_seed(17),
        loss_config=LossComponentConfig(
            use_bellman_loss=True,
            use_stationarity_loss=False,
            use_density_regularization=False,
            detach_density_weights_for_bellman=True,
        ),
    )

    assert result.diagnostics["active_losses"] == ["bellman"]
    assert result.diagnostics["bellman_weights_detached"] is True
    assert _unchanged(density_before, model.density_head)


def test_non_detached_density_weights_allow_density_head_update_in_bellman_only_density_step() -> None:
    data, model, _ = _data_model_optimizer()
    bundle = make_optimizer_bundle(model, mode="separate_heads")
    density_before = _params(model.density_head)

    result = run_minimal_train_step(
        data=data,
        model=model,
        optimizer=None,
        optimizer_bundle=bundle,
        update_parts=["density"],
        batch_size=6,
        k=4,
        gamma=0.95,
        generator=torch.Generator().manual_seed(17),
        loss_config=LossComponentConfig(
            use_bellman_loss=True,
            use_stationarity_loss=False,
            use_density_regularization=False,
            detach_density_weights_for_bellman=False,
        ),
    )

    assert result.diagnostics["active_losses"] == ["bellman"]
    assert result.diagnostics["bellman_weights_detached"] is False
    assert _changed(density_before, model.density_head)


def test_policy_probability_detach_setting_is_recorded() -> None:
    config = replace(LossComponentConfig(), detach_policy_probs_for_bellman=False)
    result = _step_with_config(config)

    assert result.diagnostics["loss_config"]["detach_policy_probs_for_bellman"] is False
    assert result.diagnostics["bellman_policy_probs_detached"] is False

