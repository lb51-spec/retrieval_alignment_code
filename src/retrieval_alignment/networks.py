"""Core PyTorch modules for DW-MQL+ANN.

The model decomposition follows ``docs/code_architecture.md`` and
``docs/algorithm_pseudocode.md``:

- ``StateEncoder`` implements a state/query tower.
- ``ActionEncoder`` implements an action/item tower.
- ``PairFeatureBuilder`` builds state-action features.
- ``QHead`` represents ``q_theta(s,a)``.
- ``DensityHead`` emits raw density logits for ``w_eta(s,a)``; positivity and
  normalization stay in ``src/retrieval_alignment/density.py``.
- ``DualTestHead`` represents the stationarity test function ``f_phi(s,a)``.

These modules are intentionally independent of replay buffers, FAISS, losses,
and training loops.
"""

from __future__ import annotations

from collections.abc import Sequence

try:  # pragma: no cover - exercised when project dependencies are installed.
    import torch
    from torch import nn
except ModuleNotFoundError:  # pragma: no cover - keeps imports clean pre-install.
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]

from .config import ModelConfig


def _require_torch() -> None:
    if torch is None or nn is None:
        raise ModuleNotFoundError("PyTorch is required to construct retrieval_alignment.networks modules.")


class _TorchModuleBase(nn.Module if nn is not None else object):  # type: ignore[misc]
    """Base class that allows importing this module before PyTorch is installed."""


def _activation(name: str) -> "_TorchModuleBase":
    _require_torch()
    normalized = name.lower()
    if normalized == "relu":
        return nn.ReLU()
    if normalized == "gelu":
        return nn.GELU()
    if normalized == "tanh":
        return nn.Tanh()
    raise ValueError(f"Unsupported activation: {name!r}.")


def _validate_dims(input_dim: int, hidden_dims: Sequence[int], output_dim: int) -> None:
    dims = [input_dim, *hidden_dims, output_dim]
    if any(int(dim) <= 0 for dim in dims):
        raise ValueError("MLP dimensions must all be positive.")


def build_mlp(
    input_dim: int,
    hidden_dims: Sequence[int] | int,
    output_dim: int,
    activation: str = "relu",
) -> "_TorchModuleBase":
    """Build a deterministic feed-forward MLP.

    Args:
        input_dim: Last-dimension input size.
        hidden_dims: Hidden layer widths, or a single integer width.
        output_dim: Last-dimension output size.
        activation: ``"relu"`` by default. ``"gelu"`` and ``"tanh"`` are also
            supported for small experiments.
    """

    _require_torch()
    if isinstance(hidden_dims, int):
        hidden = (hidden_dims,)
    else:
        hidden = tuple(hidden_dims)
    _validate_dims(input_dim, hidden, output_dim)

    layers: list["_TorchModuleBase"] = []
    prev_dim = int(input_dim)
    for hidden_dim in hidden:
        layers.append(nn.Linear(prev_dim, int(hidden_dim)))
        layers.append(_activation(activation))
        prev_dim = int(hidden_dim)
    layers.append(nn.Linear(prev_dim, int(output_dim)))
    return nn.Sequential(*layers)


def _as_float_tensor(values: "torch.Tensor", *, name: str) -> "torch.Tensor":
    _require_torch()
    if not torch.is_tensor(values):
        raise TypeError(f"{name} must be a torch.Tensor.")
    if not torch.is_floating_point(values):
        values = values.float()
    if torch.any(~torch.isfinite(values)):
        raise ValueError(f"{name} must contain only finite values.")
    return values


class StateEncoder(_TorchModuleBase):
    """Encode state/query features into embeddings shaped ``[batch, dim]``."""

    def __init__(
        self,
        state_dim: int,
        embedding_dim: int,
        hidden_dims: Sequence[int] | int = 32,
        activation: str = "relu",
    ) -> None:
        _require_torch()
        super().__init__()
        self.state_dim = int(state_dim)
        self.embedding_dim = int(embedding_dim)
        self.net = build_mlp(self.state_dim, hidden_dims, self.embedding_dim, activation)

    def forward(self, states: "torch.Tensor") -> "torch.Tensor":
        states = _as_float_tensor(states, name="states")
        if states.ndim != 2 or states.shape[-1] != self.state_dim:
            raise ValueError(f"states must have shape [batch, {self.state_dim}].")
        return self.net(states)


class ActionEncoder(_TorchModuleBase):
    """Encode action/item features into embeddings.

    Accepts ``[batch, action_dim]`` or ``[num_actions, action_dim]`` and returns
    the same leading dimension with ``embedding_dim`` as the last dimension.
    """

    def __init__(
        self,
        action_dim: int,
        embedding_dim: int,
        hidden_dims: Sequence[int] | int = 32,
        activation: str = "relu",
    ) -> None:
        _require_torch()
        super().__init__()
        self.action_dim = int(action_dim)
        self.embedding_dim = int(embedding_dim)
        self.net = build_mlp(self.action_dim, hidden_dims, self.embedding_dim, activation)

    def forward(self, actions: "torch.Tensor") -> "torch.Tensor":
        actions = _as_float_tensor(actions, name="actions")
        if actions.ndim != 2 or actions.shape[-1] != self.action_dim:
            raise ValueError(f"actions must have shape [batch, {self.action_dim}].")
        return self.net(actions)


class PairFeatureBuilder:
    """Build pair features for ``(s,a)`` scoring heads.

    Supported modes:
    - ``concat``: ``[s_emb, a_emb]``
    - ``product``: ``[s_emb * a_emb]``
    - ``concat_product``: ``[s_emb, a_emb, s_emb * a_emb]``

    Candidate action embeddings shaped ``[batch, k, dim]`` are supported by
    broadcasting state embeddings shaped ``[batch, dim]``.
    """

    def __init__(self, mode: str = "concat_product") -> None:
        if mode not in {"concat", "product", "concat_product"}:
            raise ValueError(f"Unsupported pair feature mode: {mode!r}.")
        self.mode = mode

    def feature_dim(self, embedding_dim: int) -> int:
        """Return output feature width for a given embedding dimension."""

        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive.")
        if self.mode == "concat":
            return 2 * embedding_dim
        if self.mode == "product":
            return embedding_dim
        return 3 * embedding_dim

    def __call__(
        self,
        state_embeddings: "torch.Tensor",
        action_embeddings: "torch.Tensor",
    ) -> "torch.Tensor":
        state_embeddings = _as_float_tensor(state_embeddings, name="state_embeddings")
        action_embeddings = _as_float_tensor(action_embeddings, name="action_embeddings")
        if state_embeddings.ndim != 2:
            raise ValueError("state_embeddings must have shape [batch, dim].")
        if action_embeddings.ndim not in {2, 3}:
            raise ValueError("action_embeddings must have shape [batch, dim] or [batch, k, dim].")
        if state_embeddings.shape[-1] != action_embeddings.shape[-1]:
            raise ValueError("state and action embedding dimensions must match.")
        if state_embeddings.shape[0] != action_embeddings.shape[0]:
            raise ValueError("state and action batch dimensions must match.")

        if action_embeddings.ndim == 3:
            state_for_pair = state_embeddings.unsqueeze(1).expand(
                -1,
                action_embeddings.shape[1],
                -1,
            )
        else:
            state_for_pair = state_embeddings

        product = state_for_pair * action_embeddings
        if self.mode == "concat":
            return torch.cat([state_for_pair, action_embeddings], dim=-1)
        if self.mode == "product":
            return product
        return torch.cat([state_for_pair, action_embeddings, product], dim=-1)


class ScalarPairHead(_TorchModuleBase):
    """Base scalar head for pair features."""

    def __init__(
        self,
        feature_dim: int,
        hidden_dims: Sequence[int] | int = 32,
        activation: str = "relu",
    ) -> None:
        _require_torch()
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.net = build_mlp(self.feature_dim, hidden_dims, 1, activation)

    def forward(self, pair_features: "torch.Tensor") -> "torch.Tensor":
        pair_features = _as_float_tensor(pair_features, name="pair_features")
        if pair_features.ndim not in {2, 3}:
            raise ValueError("pair_features must have shape [batch, feature_dim] or [batch, k, feature_dim].")
        if pair_features.shape[-1] != self.feature_dim:
            raise ValueError(f"pair_features last dimension must be {self.feature_dim}.")
        return self.net(pair_features).squeeze(-1)


class QHead(ScalarPairHead):
    """Critic head representing ``q_theta(s,a)``."""


class DensityHead(ScalarPairHead):
    """Density head returning raw logits for ``w_eta(s,a)``."""


class DualTestHead(ScalarPairHead):
    """Dual test-function head representing ``f_phi(s,a)``."""


# Backwards-compatible alias for the earlier implementation and tests.
DualHead = DualTestHead


class DWMQLNetworks(_TorchModuleBase):
    """Simple wrapper containing encoders, pair builder, and scalar heads."""

    def __init__(self, config: ModelConfig) -> None:
        _require_torch()
        super().__init__()
        hidden_dims = config.hidden_dims if config.hidden_dims is not None else (config.hidden_dim,)
        self.config = config
        self.state_encoder = StateEncoder(
            config.state_dim,
            config.embedding_dim,
            hidden_dims,
            config.activation,
        )
        self.action_encoder = ActionEncoder(
            config.action_dim,
            config.embedding_dim,
            hidden_dims,
            config.activation,
        )
        self.pair_builder = PairFeatureBuilder(config.pair_feature_mode)
        pair_dim = self.pair_builder.feature_dim(config.embedding_dim)
        self.q_head = QHead(pair_dim, hidden_dims, config.activation)
        self.density_head = DensityHead(pair_dim, hidden_dims, config.activation)
        self.dual_head = DualTestHead(pair_dim, hidden_dims, config.activation)
        # B2 retrieval projection heads (optional). When enabled these are the
        # only trainable parameters in the retrieval path; they read DETACHED
        # encoder outputs so the alignment loss cannot reach the Q/Bellman path.
        if getattr(config, "retrieval_projection", False):
            self.retrieval_state_proj = build_mlp(
                config.embedding_dim, hidden_dims, config.embedding_dim, config.activation
            )
            self.retrieval_action_proj = build_mlp(
                config.embedding_dim, hidden_dims, config.embedding_dim, config.activation
            )
        else:
            self.retrieval_state_proj = None
            self.retrieval_action_proj = None
        # Wolpertinger proto-action head: maps detached state embedding to action
        # embedding space. Trained by BC loss; used as ANN query key instead of
        # the raw state embedding. Separate from B2 projection heads.
        if getattr(config, "proto_action", False):
            self.proto_action_head = build_mlp(
                config.embedding_dim, hidden_dims, config.embedding_dim, config.activation
            )
        else:
            self.proto_action_head = None

    @property
    def has_retrieval_projection(self) -> bool:
        """True when B2 retrieval projection heads are present."""

        return self.retrieval_action_proj is not None

    @property
    def has_proto_action(self) -> bool:
        """True when the Wolpertinger proto-action head is present."""

        return self.proto_action_head is not None

    def encode_states(self, states: "torch.Tensor") -> "torch.Tensor":
        """Encode state/query features."""

        return self.state_encoder(states)

    def encode_actions(self, actions: "torch.Tensor") -> "torch.Tensor":
        """Encode action/item features."""

        return self.action_encoder(actions)

    def encode_state_query(self, states: "torch.Tensor") -> "torch.Tensor":
        """Encode states for the retrieval/ANN path.

        Priority order: Wolpertinger proto-action head > B2 retrieval projection > base encoder.
        Both alternative heads read DETACHED base embeddings, so no gradient from
        the retrieval path can reach the Q/Bellman encoder.
        """

        if self.proto_action_head is not None:
            return self.proto_action_head(self.state_encoder(states).detach())
        if self.retrieval_state_proj is not None:
            return self.retrieval_state_proj(self.state_encoder(states).detach())
        return self.state_encoder(states)

    def encode_action_index(self, actions: "torch.Tensor") -> "torch.Tensor":
        """Encode actions for the retrieval/ANN index path.

        With projection enabled, applies the retrieval projection to the
        DETACHED base action embedding (firewall). Otherwise falls back to the
        base action encoder.
        """

        if self.retrieval_action_proj is None:
            return self.action_encoder(actions)
        return self.retrieval_action_proj(self.action_encoder(actions).detach())

    def encode_pair(self, states: "torch.Tensor", actions: "torch.Tensor") -> tuple["torch.Tensor", "torch.Tensor"]:
        """Return encoded state and action tensors."""

        return self.encode_states(states), self.encode_actions(actions)

    def pair_features(
        self,
        state_embeddings: "torch.Tensor",
        action_embeddings: "torch.Tensor",
    ) -> "torch.Tensor":
        """Build state-action pair features from embeddings."""

        return self.pair_builder(state_embeddings, action_embeddings)

    def q_values(self, state_embeddings: "torch.Tensor", action_embeddings: "torch.Tensor") -> "torch.Tensor":
        """Compute critic values ``q_theta(s,a)``."""

        return self.q_head(self.pair_features(state_embeddings, action_embeddings))

    def density_logits(self, state_embeddings: "torch.Tensor", action_embeddings: "torch.Tensor") -> "torch.Tensor":
        """Compute raw density logits before positivity/clipping."""

        return self.density_head(self.pair_features(state_embeddings, action_embeddings))

    def dual_values(self, state_embeddings: "torch.Tensor", action_embeddings: "torch.Tensor") -> "torch.Tensor":
        """Compute dual test-function values ``f_phi(s,a)``."""

        return self.dual_head(self.pair_features(state_embeddings, action_embeddings))


def build_networks(config: ModelConfig | None = None) -> DWMQLNetworks:
    """Build the minimal DW-MQL network bundle."""

    return DWMQLNetworks(config or ModelConfig())
