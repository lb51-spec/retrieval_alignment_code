"""Tests for smoke gradient-flow assertions."""

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
    assert_expected_gradient_flow,
    default_smoke_alternating_schedule,
    expected_groups_for_update_parts,
    make_optimizer_bundle,
    run_minimal_alternating_loop,
)


def _data_model(seed: int = 135):
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


def test_assert_expected_gradient_flow_succeeds_for_positive_expected_groups() -> None:
    norms = {"encoder": 0.1, "critic": 0.2, "density": 0.0, "dual": 0.0}

    assert assert_expected_gradient_flow(norms, ["encoder", "critic"]) is True


def test_assert_expected_gradient_flow_missing_group_raises_assertion_error() -> None:
    norms = {"encoder": 0.1, "critic": 0.2, "density": 0.0}

    with pytest.raises(AssertionError, match="missing"):
        assert_expected_gradient_flow(norms, ["dual"])


def test_assert_expected_gradient_flow_zero_norm_raises_assertion_error() -> None:
    norms = {"encoder": 0.1, "critic": 0.0, "density": 0.0, "dual": 0.0}

    with pytest.raises(AssertionError, match="critic"):
        assert_expected_gradient_flow(norms, ["critic"])


def test_assert_expected_gradient_flow_unknown_group_raises_value_error() -> None:
    norms = {"encoder": 0.1, "critic": 0.2, "density": 0.0, "dual": 0.0}

    with pytest.raises(ValueError, match="Unknown expected"):
        assert_expected_gradient_flow(norms, ["policy"])


def test_assert_expected_gradient_flow_negative_or_nonfinite_norm_raises_value_error() -> None:
    with pytest.raises(ValueError, match="finite and nonnegative"):
        assert_expected_gradient_flow({"encoder": -1.0, "critic": 0.0, "density": 0.0, "dual": 0.0}, ["encoder"])

    with pytest.raises(ValueError, match="finite and nonnegative"):
        assert_expected_gradient_flow({"encoder": float("nan"), "critic": 0.0, "density": 0.0, "dual": 0.0}, ["encoder"])


def test_assert_expected_gradient_flow_invalid_min_norm_raises_value_error() -> None:
    norms = {"encoder": 0.1, "critic": 0.2, "density": 0.0, "dual": 0.0}

    with pytest.raises(ValueError, match="min_norm"):
        assert_expected_gradient_flow(norms, ["encoder"], min_norm=-1.0)


def test_expected_groups_for_update_parts_maps_known_parts() -> None:
    assert expected_groups_for_update_parts(["encoder", "critic"]) == {"encoder", "critic"}
    assert expected_groups_for_update_parts(("density", "dual")) == {"density", "dual"}
    assert expected_groups_for_update_parts({"joint"}) == {"encoder", "critic", "density", "dual"}


def test_expected_groups_for_update_parts_rejects_unknown_parts() -> None:
    with pytest.raises(ValueError, match="Unknown update"):
        expected_groups_for_update_parts(["policy"])


def test_run_minimal_alternating_loop_asserts_gradient_flow() -> None:
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
        generator=torch.Generator().manual_seed(29),
        assert_gradient_flow=True,
    )

    assert result.diagnostics["assert_gradient_flow"] is True

