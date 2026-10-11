"""Basic tensor shape tests for network modules."""

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
    DualHead,
    PairFeatureBuilder,
    QHead,
    StateEncoder,
    build_networks,
)


def test_encoders_return_embedding_shapes() -> None:
    config = ModelConfig(state_dim=5, action_dim=3, embedding_dim=7, hidden_dim=11)
    batch_size = 4

    state_encoder = StateEncoder(config.state_dim, config.embedding_dim, config.hidden_dim)
    action_encoder = ActionEncoder(config.action_dim, config.embedding_dim, config.hidden_dim)

    states = torch.randn(batch_size, config.state_dim)
    actions = torch.randn(batch_size, config.action_dim)

    assert state_encoder(states).shape == (batch_size, config.embedding_dim)
    assert action_encoder(actions).shape == (batch_size, config.embedding_dim)


def test_pair_heads_return_batch_scores() -> None:
    embedding_dim = 7
    hidden_dim = 11
    batch_size = 4
    state_embeddings = torch.randn(batch_size, embedding_dim)
    action_embeddings = torch.randn(batch_size, embedding_dim)
    pair_features = PairFeatureBuilder("concat_product")(state_embeddings, action_embeddings)

    for head_cls in (QHead, DensityHead, DualHead):
        head = head_cls(embedding_dim * 3, hidden_dim)
        scores = head(pair_features)
        assert scores.shape == (batch_size,)


def test_network_bundle_shapes() -> None:
    config = ModelConfig(state_dim=5, action_dim=3, embedding_dim=7, hidden_dim=11)
    batch_size = 4
    networks = build_networks(config)

    states = torch.randn(batch_size, config.state_dim)
    actions = torch.randn(batch_size, config.action_dim)
    state_embeddings, action_embeddings = networks.encode_pair(states, actions)

    assert state_embeddings.shape == (batch_size, config.embedding_dim)
    assert action_embeddings.shape == (batch_size, config.embedding_dim)
    pair_features = networks.pair_features(state_embeddings, action_embeddings)
    assert networks.q_head(pair_features).shape == (batch_size,)
    assert networks.density_head(pair_features).shape == (batch_size,)
    assert networks.dual_head(pair_features).shape == (batch_size,)
