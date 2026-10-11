"""Tests for the minimal differentiable train-step smoke utility."""

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
from retrieval_alignment.train import main
from retrieval_alignment.trainer import TrainStepResult, run_minimal_train_step


def _data_model_optimizer(seed: int = 123):
    data = make_synthetic_offline_data(
        SyntheticDatasetConfig(
            num_transitions=32,
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


def _step(seed: int = 123) -> TrainStepResult:
    data, model, optimizer = _data_model_optimizer(seed)
    return run_minimal_train_step(
        data=data,
        model=model,
        optimizer=optimizer,
        batch_size=6,
        k=4,
        gamma=0.95,
        generator=torch.Generator().manual_seed(999),
        density_max_weight=10.0,
    )


def test_run_minimal_train_step_returns_result() -> None:
    assert isinstance(_step(), TrainStepResult)


def test_train_step_records_exact_index_backend() -> None:
    result = _step()

    assert result.diagnostics["index_backend"] == "exact"
    assert result.diagnostics["index_metric"] == "dot"


def test_train_step_runs_with_exact_cosine_backend() -> None:
    data, model, optimizer = _data_model_optimizer()

    result = run_minimal_train_step(
        data=data,
        model=model,
        optimizer=optimizer,
        batch_size=6,
        k=4,
        gamma=0.95,
        generator=torch.Generator().manual_seed(999),
        index_backend="exact",
        index_metric="cosine",
    )

    assert torch.isfinite(result.total_loss)
    assert result.diagnostics["index_backend"] == "exact"
    assert result.diagnostics["index_metric"] == "cosine"


def test_losses_are_finite_scalars() -> None:
    result = _step()

    for value in (
        result.total_loss,
        result.bellman_loss,
        result.stationarity_loss,
        result.density_regularization,
    ):
        assert value.ndim == 0
        assert torch.isfinite(value)


def test_grad_norm_and_parameter_delta_are_finite_and_positive() -> None:
    result = _step()

    assert result.grad_norm.ndim == 0
    assert result.parameter_delta_norm.ndim == 0
    assert torch.isfinite(result.grad_norm)
    assert torch.isfinite(result.parameter_delta_norm)
    assert result.grad_norm.item() > 0
    assert result.parameter_delta_norm.item() > 0


def test_default_detach_bellman_target_is_true() -> None:
    result = _step()

    assert result.diagnostics["detach_bellman_target"] is True
    assert result.diagnostics["target_model_used"] is False
    assert result.diagnostics["expected_next_q_requires_grad_before_detach"] is True
    assert result.diagnostics["bellman_target_requires_grad"] is False


def test_train_step_changes_parameters_with_detached_target_enabled() -> None:
    data, model, optimizer = _data_model_optimizer()
    before = [param.detach().clone() for param in model.parameters()]

    result = run_minimal_train_step(
        data=data,
        model=model,
        optimizer=optimizer,
        batch_size=6,
        k=4,
        gamma=0.95,
        generator=torch.Generator().manual_seed(999),
        detach_bellman_target=True,
    )

    assert result.parameter_delta_norm.item() > 0
    assert any(not torch.equal(before_param, after_param) for before_param, after_param in zip(before, model.parameters(), strict=True))


def test_non_detached_bellman_target_mode_runs_and_records_diagnostics() -> None:
    data, model, optimizer = _data_model_optimizer()

    result = run_minimal_train_step(
        data=data,
        model=model,
        optimizer=optimizer,
        batch_size=6,
        k=4,
        gamma=0.95,
        generator=torch.Generator().manual_seed(999),
        detach_bellman_target=False,
    )

    assert result.diagnostics["detach_bellman_target"] is False
    assert result.diagnostics["expected_next_q_requires_grad_before_detach"] is True
    assert result.diagnostics["bellman_target_requires_grad"] is True
    assert torch.isfinite(result.total_loss)


def test_invalid_detach_bellman_target_type_raises() -> None:
    data, model, optimizer = _data_model_optimizer()

    with pytest.raises(TypeError, match="detach_bellman_target"):
        run_minimal_train_step(
            data=data,
            model=model,
            optimizer=optimizer,
            batch_size=6,
            k=4,
            gamma=0.95,
            detach_bellman_target="yes",  # type: ignore[arg-type]
        )


def test_target_model_can_be_used_for_bellman_bootstrap() -> None:
    data, model, optimizer = _data_model_optimizer()
    target_model = clone_target_network(model)

    result = run_minimal_train_step(
        data=data,
        model=model,
        optimizer=optimizer,
        batch_size=6,
        k=4,
        gamma=0.95,
        generator=torch.Generator().manual_seed(999),
        target_model=target_model,
    )

    assert result.diagnostics["target_model_used"] is True
    assert result.diagnostics["bellman_target_requires_grad"] is False
    assert result.diagnostics["expected_next_q_requires_grad_before_detach"] is False
    assert torch.isfinite(result.total_loss)


def test_target_model_parameters_do_not_change_but_online_model_changes() -> None:
    data, model, optimizer = _data_model_optimizer()
    target_model = clone_target_network(model)
    target_before = [param.detach().clone() for param in target_model.parameters()]
    online_before = [param.detach().clone() for param in model.parameters()]

    run_minimal_train_step(
        data=data,
        model=model,
        optimizer=optimizer,
        batch_size=6,
        k=4,
        gamma=0.95,
        generator=torch.Generator().manual_seed(999),
        target_model=target_model,
    )

    assert all(
        torch.equal(before, after)
        for before, after in zip(target_before, target_model.parameters(), strict=True)
    )
    assert any(
        not torch.equal(before, after)
        for before, after in zip(online_before, model.parameters(), strict=True)
    )


def test_unfrozen_target_model_raises_value_error() -> None:
    data, model, optimizer = _data_model_optimizer()
    target_model = build_networks(
        ModelConfig(
            state_dim=4,
            action_dim=3,
            embedding_dim=8,
            hidden_dims=(16,),
            pair_feature_mode="concat_product",
        )
    )

    with pytest.raises(ValueError, match="frozen"):
        run_minimal_train_step(
            data=data,
            model=model,
            optimizer=optimizer,
            batch_size=6,
            k=4,
            gamma=0.95,
            target_model=target_model,
        )


def test_non_detached_flag_with_target_model_still_uses_no_grad_bootstrap() -> None:
    data, model, optimizer = _data_model_optimizer()
    target_model = clone_target_network(model)

    result = run_minimal_train_step(
        data=data,
        model=model,
        optimizer=optimizer,
        batch_size=6,
        k=4,
        gamma=0.95,
        generator=torch.Generator().manual_seed(999),
        detach_bellman_target=False,
        target_model=target_model,
    )

    assert result.diagnostics["detach_bellman_target"] is False
    assert result.diagnostics["target_model_used"] is True
    assert result.diagnostics["expected_next_q_requires_grad_before_detach"] is False
    assert result.diagnostics["bellman_target_requires_grad"] is False


def test_at_least_one_model_parameter_changes_after_step() -> None:
    data, model, optimizer = _data_model_optimizer()
    before = [param.detach().clone() for param in model.parameters()]

    run_minimal_train_step(
        data=data,
        model=model,
        optimizer=optimizer,
        batch_size=6,
        k=4,
        gamma=0.95,
        generator=torch.Generator().manual_seed(999),
    )

    assert any(not torch.equal(before_param, after_param) for before_param, after_param in zip(before, model.parameters(), strict=True))


def test_invalid_k_raises_value_error() -> None:
    data, model, optimizer = _data_model_optimizer()

    with pytest.raises(ValueError, match="k"):
        run_minimal_train_step(data, model, optimizer, batch_size=4, k=0, gamma=0.9)

    with pytest.raises(ValueError, match="number of actions"):
        run_minimal_train_step(data, model, optimizer, batch_size=4, k=99, gamma=0.9)


def test_invalid_gamma_raises_value_error() -> None:
    data, model, optimizer = _data_model_optimizer()

    with pytest.raises(ValueError, match="gamma"):
        run_minimal_train_step(data, model, optimizer, batch_size=4, k=3, gamma=1.1)


def test_existing_forward_only_smoke_cli_still_works(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(
        [
            "--smoke",
            "--seed",
            "3",
            "--batch-size",
            "4",
            "--num-transitions",
            "16",
            "--num-actions",
            "8",
            "--state-dim",
            "4",
            "--action-dim",
            "3",
            "--embedding-dim",
            "8",
            "--k",
            "3",
            "--gamma",
            "0.9",
        ]
    )

    assert exit_code == 0
    assert "DW-MQL smoke check passed" in capsys.readouterr().out
