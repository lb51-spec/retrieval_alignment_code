"""Tests for core DW-MQL network modules."""

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
from retrieval_alignment.networks import (
    ActionEncoder,
    DensityHead,
    DualTestHead,
    PairFeatureBuilder,
    QHead,
    StateEncoder,
    build_networks,
)


def test_state_encoder_output_shape() -> None:
    encoder = StateEncoder(state_dim=5, embedding_dim=7, hidden_dims=(11,))

    output = encoder(torch.randn(4, 5))

    assert output.shape == (4, 7)


def test_action_encoder_output_shape() -> None:
    encoder = ActionEncoder(action_dim=3, embedding_dim=7, hidden_dims=(11,))

    output = encoder(torch.randn(6, 3))

    assert output.shape == (6, 7)


def test_pair_feature_builder_concat_shape() -> None:
    builder = PairFeatureBuilder("concat")

    features = builder(torch.randn(4, 7), torch.randn(4, 7))

    assert features.shape == (4, 14)


def test_pair_feature_builder_product_shape() -> None:
    builder = PairFeatureBuilder("product")

    features = builder(torch.randn(4, 7), torch.randn(4, 7))

    assert features.shape == (4, 7)


def test_pair_feature_builder_concat_product_shape() -> None:
    builder = PairFeatureBuilder("concat_product")

    features = builder(torch.randn(4, 7), torch.randn(4, 7))

    assert features.shape == (4, 21)


def test_pair_feature_builder_supports_candidate_action_embeddings() -> None:
    builder = PairFeatureBuilder("concat_product")

    features = builder(torch.randn(4, 7), torch.randn(4, 5, 7))

    assert features.shape == (4, 5, 21)


def test_pair_feature_builder_rejects_mismatched_embedding_dims() -> None:
    builder = PairFeatureBuilder("concat")

    with pytest.raises(ValueError, match="dimensions"):
        builder(torch.randn(4, 7), torch.randn(4, 8))


def test_q_head_outputs_batch_shape() -> None:
    head = QHead(feature_dim=21, hidden_dims=(11,))

    values = head(torch.randn(4, 21))

    assert values.shape == (4,)


def test_q_head_outputs_candidate_shape() -> None:
    head = QHead(feature_dim=21, hidden_dims=(11,))

    values = head(torch.randn(4, 5, 21))

    assert values.shape == (4, 5)


def test_density_head_outputs_raw_logits_shape() -> None:
    head = DensityHead(feature_dim=21, hidden_dims=(11,))

    logits = head(torch.randn(4, 5, 21))

    assert logits.shape == (4, 5)


def test_dual_test_head_outputs_shape() -> None:
    head = DualTestHead(feature_dim=21, hidden_dims=(11,))

    values = head(torch.randn(4, 21))

    assert values.shape == (4,)


def test_combined_wrapper_produces_q_density_and_dual_outputs() -> None:
    config = ModelConfig(
        state_dim=5,
        action_dim=3,
        embedding_dim=7,
        hidden_dims=(11,),
        pair_feature_mode="concat_product",
    )
    model = build_networks(config)
    state_embeddings = model.encode_states(torch.randn(4, 5))
    action_embeddings = model.encode_actions(torch.randn(4, 3))

    assert model.q_values(state_embeddings, action_embeddings).shape == (4,)
    assert model.density_logits(state_embeddings, action_embeddings).shape == (4,)
    assert model.dual_values(state_embeddings, action_embeddings).shape == (4,)


def test_wrapper_supports_candidate_action_embeddings() -> None:
    config = ModelConfig(
        state_dim=5,
        action_dim=3,
        embedding_dim=7,
        hidden_dims=(11,),
        pair_feature_mode="concat_product",
    )
    model = build_networks(config)
    state_embeddings = torch.randn(4, 7)
    action_embeddings = torch.randn(4, 6, 7)

    assert model.q_values(state_embeddings, action_embeddings).shape == (4, 6)
    assert model.density_logits(state_embeddings, action_embeddings).shape == (4, 6)
    assert model.dual_values(state_embeddings, action_embeddings).shape == (4, 6)


def test_unsupported_pair_mode_raises_value_error() -> None:
    with pytest.raises(ValueError, match="Unsupported pair feature mode"):
        PairFeatureBuilder("bad_mode")


def test_non_finite_input_raises_value_error() -> None:
    encoder = StateEncoder(state_dim=2, embedding_dim=3, hidden_dims=(5,))
    bad_states = torch.tensor([[1.0, float("nan")]])

    with pytest.raises(ValueError, match="finite"):
        encoder(bad_states)
