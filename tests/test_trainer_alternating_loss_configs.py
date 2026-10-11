"""Tests for loss configs threaded through alternating smoke schedules."""

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
    LossComponentConfig,
    TrainLoopSmokeResult,
    critic_loss_config,
    default_smoke_alternating_schedule,
    density_loss_config,
    dual_loss_config,
    make_optimizer_bundle,
    parse_update_schedule_step,
    run_minimal_alternating_loop,
)


def _data_model(seed: int = 654):
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


def test_critic_preset_has_only_bellman_active() -> None:
    config = critic_loss_config()

    assert config.use_bellman_loss is True
    assert config.use_stationarity_loss is False
    assert config.use_density_regularization is False
    assert config.detach_density_weights_for_bellman is True
    assert config.detach_policy_probs_for_bellman is True


def test_density_preset_has_stationarity_and_density_regularization_active() -> None:
    config = density_loss_config()

    assert config.use_bellman_loss is False
    assert config.use_stationarity_loss is True
    assert config.use_density_regularization is True


def test_dual_preset_has_only_stationarity_active() -> None:
    config = dual_loss_config()

    assert config.use_bellman_loss is False
    assert config.use_stationarity_loss is True
    assert config.use_density_regularization is False


def test_parse_update_schedule_step_supports_legacy_parts() -> None:
    parts, loss_config = parse_update_schedule_step(["encoder", "critic"])

    assert parts == ["encoder", "critic"]
    assert loss_config is None


def test_parse_update_schedule_step_supports_structured_step() -> None:
    config = critic_loss_config()

    parts, loss_config = parse_update_schedule_step(
        {"parts": ["critic"], "loss_config": config}
    )

    assert parts == ["critic"]
    assert loss_config is config


def test_structured_alternating_loop_runs_and_records_active_losses() -> None:
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
            {"parts": ["encoder", "critic"], "loss_config": critic_loss_config()},
        ],
        generator=torch.Generator().manual_seed(19),
    )

    assert isinstance(result, TrainLoopSmokeResult)
    assert len(result.losses) == 3
    assert all(torch.isfinite(torch.tensor(loss)) for loss in result.losses)
    assert result.diagnostics["update_schedule"] == [
        ["dual"],
        ["density"],
        ["encoder", "critic"],
    ]
    assert result.diagnostics["active_losses_by_step"] == [
        ["stationarity"],
        ["stationarity", "density_regularization"],
        ["bellman"],
    ]


def test_structured_step_without_parts_raises_value_error() -> None:
    with pytest.raises(ValueError, match="parts"):
        parse_update_schedule_step({"loss_config": critic_loss_config()})


def test_structured_step_with_invalid_loss_config_type_raises_type_error() -> None:
    with pytest.raises(TypeError, match="loss_config"):
        parse_update_schedule_step({"parts": ["critic"], "loss_config": "critic"})


def test_structured_step_with_invalid_parts_type_raises_type_error() -> None:
    with pytest.raises(TypeError, match="parts"):
        parse_update_schedule_step({"parts": "critic", "loss_config": critic_loss_config()})


def test_structured_step_with_empty_parts_raises_value_error() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        parse_update_schedule_step({"parts": [], "loss_config": LossComponentConfig()})


def test_default_smoke_alternating_schedule_has_expected_order() -> None:
    schedule = default_smoke_alternating_schedule()

    assert len(schedule) == 3
    assert schedule[0]["parts"] == ["dual"]
    assert schedule[1]["parts"] == ["density"]
    assert schedule[2]["parts"] == ["encoder", "critic"]


def test_default_smoke_alternating_schedule_can_exclude_encoder_from_critic_step() -> None:
    schedule = default_smoke_alternating_schedule(include_encoder_on_critic=False)

    assert schedule[2]["parts"] == ["critic"]


def test_default_smoke_alternating_schedule_steps_parse() -> None:
    schedule = default_smoke_alternating_schedule()

    parsed = [parse_update_schedule_step(step) for step in schedule]

    assert [parts for parts, _ in parsed] == [["dual"], ["density"], ["encoder", "critic"]]
    assert all(loss_config is not None for _, loss_config in parsed)


def test_run_minimal_alternating_loop_works_with_default_schedule() -> None:
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
        update_schedule=default_smoke_alternating_schedule(),
        generator=torch.Generator().manual_seed(19),
    )

    assert isinstance(result, TrainLoopSmokeResult)
    assert result.diagnostics["update_schedule"] == [
        ["dual"],
        ["density"],
        ["encoder", "critic"],
    ]
    assert result.diagnostics["active_losses_by_step"] == [
        ["stationarity"],
        ["stationarity", "density_regularization"],
        ["bellman"],
    ]


def test_default_smoke_alternating_schedule_invalid_encoder_flag_type_raises() -> None:
    with pytest.raises(TypeError, match="include_encoder_on_critic"):
        default_smoke_alternating_schedule(include_encoder_on_critic="yes")  # type: ignore[arg-type]
