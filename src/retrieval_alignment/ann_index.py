"""Action-retrieval backends for development, audits, and optional FAISS use.

DW-MQL+ANN needs an exact-search path before approximate indexes are introduced.
This module implements a brute-force in-memory action index and a matching
optional FAISS backend behind a shared interface. The exact backend is used for:

- development without FAISS;
- exact-search audits for retrieval bias;
- testing candidate-set construction;
- validating ANN-truncated policies against exhaustive retrieval.

FAISS remains optional and is imported lazily by FAISS-backed classes. The
implementation is intentionally independent of training logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

try:  # pragma: no cover - exercised when the project dependencies are installed.
    import torch
except ModuleNotFoundError:  # pragma: no cover - keeps imports clean pre-install.
    torch = None  # type: ignore[assignment]


@dataclass(slots=True)
class IndexSearchResult:
    """Search result returned by future ANN index implementations."""

    ids: "torch.Tensor | Any"
    scores: "torch.Tensor | Any"


@runtime_checkable
class ActionIndex(Protocol):
    """Protocol shared by exact and approximate action-retrieval backends."""

    @property
    def num_actions(self) -> int:
        """Number of indexed actions."""

    @property
    def embedding_dim(self) -> int:
        """Dimensionality of indexed action embeddings."""

    def search(self, query_embeddings: "torch.Tensor | Any", k: int) -> IndexSearchResult:
        """Return top-k action ids and scores for each query."""

    def update(
        self,
        action_embeddings: "torch.Tensor | Any",
        action_ids: "torch.Tensor | Any | None" = None,
    ) -> None:
        """Replace indexed embeddings and optional action ids."""


class ANNIndex:
    """Minimal interface for exact or approximate action retrieval."""

    def build(self, embeddings: "torch.Tensor | Any", ids: "torch.Tensor | Any | None" = None) -> None:
        raise NotImplementedError("The base ANN index does not implement build().")

    def search(self, queries: "torch.Tensor | Any", k: int) -> IndexSearchResult:
        raise NotImplementedError("The base ANN index does not implement search().")


def _require_torch() -> None:
    if torch is None:
        raise ModuleNotFoundError("PyTorch is required to use action indexes.")


def _as_embedding_matrix(values: Any, *, name: str) -> "torch.Tensor":
    _require_torch()
    tensor = torch.as_tensor(values)
    if not torch.is_floating_point(tensor):
        tensor = tensor.float()
    if tensor.ndim != 2:
        raise ValueError(f"{name} must have shape [num_items, embedding_dim].")
    if tensor.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one row.")
    if tensor.shape[1] == 0:
        raise ValueError(f"{name} must have positive embedding dimension.")
    if torch.any(~torch.isfinite(tensor)):
        raise ValueError(f"{name} must contain only finite values.")
    return tensor.detach().clone()


def _validate_metric(metric: str) -> str:
    if metric not in {"dot", "cosine"}:
        raise ValueError("metric must be either 'dot' or 'cosine'.")
    return metric


def _safe_l2_normalize(values: "torch.Tensor", *, name: str) -> "torch.Tensor":
    norms = torch.linalg.vector_norm(values, dim=1, keepdim=True)
    if torch.any(norms == 0):
        raise ValueError(f"{name} contains zero-norm embeddings, which are invalid for cosine search.")
    return values / norms


def _validate_k(k: int, *, num_actions: int) -> None:
    if not isinstance(k, int) or k <= 0:
        raise ValueError("k must be a positive integer.")
    if k > num_actions:
        raise ValueError(f"k={k} cannot exceed num_actions={num_actions}.")


def _default_ids(num_actions: int, *, device: "torch.device") -> "torch.Tensor":
    return torch.arange(num_actions, dtype=torch.long, device=device)


def _validate_action_ids(action_ids: Any, *, num_actions: int, device: "torch.device") -> "torch.Tensor":
    ids = torch.as_tensor(action_ids, dtype=torch.long, device=device)
    if ids.ndim != 1:
        raise ValueError("action_ids must have shape [num_actions].")
    if ids.shape[0] != num_actions:
        raise ValueError(f"action_ids length {ids.shape[0]} must match num_actions {num_actions}.")
    return ids.detach().clone()


class ExactActionIndex(ANNIndex):
    """Brute-force exact top-k action index.

    Args:
        action_embeddings: Tensor shaped ``[num_actions, embedding_dim]``.
        action_ids: Optional tensor shaped ``[num_actions]``. If omitted, row
            indices ``0..num_actions-1`` are returned.
        metric: Either ``"dot"`` or ``"cosine"``.
        normalize: If true, store normalized embeddings for cosine search. For
            cosine metric, embeddings are always normalized safely and zero-norm
            vectors are rejected.

    Tie behavior:
        Search uses stable sorting over scores in descending order, so ties are
        resolved by the stored action order.
    """

    def __init__(
        self,
        action_embeddings: "torch.Tensor | Any",
        action_ids: "torch.Tensor | Any | None" = None,
        metric: str = "dot",
        normalize: bool = True,
    ) -> None:
        self.metric = _validate_metric(metric)
        self.normalize = normalize
        self._embeddings: "torch.Tensor | None" = None
        self._ids: "torch.Tensor | None" = None
        self.update(action_embeddings, action_ids=action_ids)

    @property
    def num_actions(self) -> int:
        """Number of indexed actions."""

        if self._embeddings is None:
            return 0
        return int(self._embeddings.shape[0])

    @property
    def embedding_dim(self) -> int:
        """Dimensionality of indexed action embeddings."""

        if self._embeddings is None:
            return 0
        return int(self._embeddings.shape[1])

    def build(self, embeddings: "torch.Tensor | Any", ids: "torch.Tensor | Any | None" = None) -> None:
        """Alias for ``update`` to match future ANN backends."""

        self.update(embeddings, action_ids=ids)

    def update(
        self,
        action_embeddings: "torch.Tensor | Any",
        action_ids: "torch.Tensor | Any | None" = None,
    ) -> None:
        """Replace indexed embeddings and optional action ids."""

        embeddings = _as_embedding_matrix(action_embeddings, name="action_embeddings")
        if self.metric == "cosine" or self.normalize:
            embeddings = _safe_l2_normalize(embeddings, name="action_embeddings")

        if action_ids is None:
            ids = _default_ids(embeddings.shape[0], device=embeddings.device)
        else:
            ids = _validate_action_ids(action_ids, num_actions=embeddings.shape[0], device=embeddings.device)

        self._embeddings = embeddings
        self._ids = ids.detach().clone()

    def _validate_queries(self, query_embeddings: Any) -> "torch.Tensor":
        queries = torch.as_tensor(query_embeddings)
        if not torch.is_floating_point(queries):
            queries = queries.float()
        if queries.ndim == 1:
            queries = queries.unsqueeze(0)
        if queries.ndim != 2:
            raise ValueError("query_embeddings must have shape [batch, embedding_dim] or [embedding_dim].")
        if queries.shape[1] != self.embedding_dim:
            raise ValueError(
                f"query embedding dimension {queries.shape[1]} does not match index dimension {self.embedding_dim}."
            )
        if torch.any(~torch.isfinite(queries)):
            raise ValueError("query_embeddings must contain only finite values.")
        queries = queries.to(device=self._embeddings.device, dtype=self._embeddings.dtype)
        if self.metric == "cosine" or self.normalize:
            queries = _safe_l2_normalize(queries, name="query_embeddings")
        return queries

    def search(self, queries: "torch.Tensor | Any", k: int) -> IndexSearchResult:
        """Return exact top-k action ids and scores for each query.

        Args:
            queries: Tensor shaped ``[batch, embedding_dim]`` or
                ``[embedding_dim]``.
            k: Number of actions to return. Must satisfy
                ``1 <= k <= num_actions``.
        """

        if self._embeddings is None or self._ids is None or self.num_actions == 0:
            raise ValueError("The action index is empty.")
        _validate_k(k, num_actions=self.num_actions)

        query_matrix = self._validate_queries(queries)
        scores = query_matrix @ self._embeddings.T
        order = torch.argsort(scores, dim=1, descending=True, stable=True)
        top_positions = order[:, :k]
        top_scores = scores.gather(1, top_positions)
        top_ids = self._ids.gather(0, top_positions.reshape(-1)).reshape(top_positions.shape)
        return IndexSearchResult(ids=top_ids, scores=top_scores)


class FaissActionIndex(ANNIndex):
    """Optional exact FAISS action index with the ``ActionIndex`` interface.

    This backend uses ``faiss.IndexFlatIP`` for both dot-product and cosine
    search. Cosine mode normalizes indexed actions and queries before inner
    product search and rejects zero-norm embeddings, matching
    ``ExactActionIndex`` behavior.

    FAISS is imported lazily during construction. If it is not installed,
    constructing this class raises an ``ImportError`` with installation context.
    """

    def __init__(
        self,
        action_embeddings: "torch.Tensor | Any",
        action_ids: "torch.Tensor | Any | None" = None,
        metric: str = "dot",
        normalize: bool = True,
    ) -> None:
        self._faiss = self._import_faiss()
        self.metric = _validate_metric(metric)
        self.normalize = normalize
        self._embeddings: "torch.Tensor | None" = None
        self._ids: "torch.Tensor | None" = None
        self._index: Any | None = None
        self.update(action_embeddings, action_ids=action_ids)

    @staticmethod
    def _import_faiss() -> Any:
        try:
            import faiss  # type: ignore[import-not-found]
        except ModuleNotFoundError as exc:
            raise ImportError(
                "FAISS is optional and is not installed. Install a compatible "
                "faiss package to use FaissActionIndex, or use backend='exact'."
            ) from exc
        return faiss

    @property
    def num_actions(self) -> int:
        """Number of indexed actions."""

        if self._embeddings is None:
            return 0
        return int(self._embeddings.shape[0])

    @property
    def embedding_dim(self) -> int:
        """Dimensionality of indexed action embeddings."""

        if self._embeddings is None:
            return 0
        return int(self._embeddings.shape[1])

    def build(self, embeddings: "torch.Tensor | Any", ids: "torch.Tensor | Any | None" = None) -> None:
        """Alias for ``update`` to match exact and future ANN backends."""

        self.update(embeddings, action_ids=ids)

    def update(
        self,
        action_embeddings: "torch.Tensor | Any",
        action_ids: "torch.Tensor | Any | None" = None,
    ) -> None:
        """Replace indexed embeddings, optional ids, and the FAISS index."""

        embeddings = _as_embedding_matrix(action_embeddings, name="action_embeddings")
        if self.metric == "cosine" or self.normalize:
            embeddings = _safe_l2_normalize(embeddings, name="action_embeddings")

        if action_ids is None:
            ids = _default_ids(embeddings.shape[0], device=embeddings.device)
        else:
            ids = _validate_action_ids(action_ids, num_actions=embeddings.shape[0], device=embeddings.device)

        self._embeddings = embeddings
        self._ids = ids
        self._index = self._faiss.IndexFlatIP(int(embeddings.shape[1]))
        self._index.add(self._as_faiss_matrix(embeddings))

    def _as_faiss_matrix(self, values: "torch.Tensor") -> Any:
        return values.detach().to(device="cpu", dtype=torch.float32).contiguous().numpy()

    def _validate_queries(self, query_embeddings: Any) -> "torch.Tensor":
        queries = torch.as_tensor(query_embeddings)
        if not torch.is_floating_point(queries):
            queries = queries.float()
        if queries.ndim == 1:
            queries = queries.unsqueeze(0)
        if queries.ndim != 2:
            raise ValueError("query_embeddings must have shape [batch, embedding_dim] or [embedding_dim].")
        if queries.shape[1] != self.embedding_dim:
            raise ValueError(
                f"query embedding dimension {queries.shape[1]} does not match index dimension {self.embedding_dim}."
            )
        if torch.any(~torch.isfinite(queries)):
            raise ValueError("query_embeddings must contain only finite values.")
        queries = queries.to(dtype=self._embeddings.dtype)
        if self.metric == "cosine" or self.normalize:
            queries = _safe_l2_normalize(queries, name="query_embeddings")
        return queries

    def search(self, queries: "torch.Tensor | Any", k: int) -> IndexSearchResult:
        """Return FAISS top-k action ids and scores for each query."""

        if self._index is None or self._embeddings is None or self._ids is None or self.num_actions == 0:
            raise ValueError("The action index is empty.")
        _validate_k(k, num_actions=self.num_actions)

        query_matrix = self._validate_queries(queries)
        scores_np, positions_np = self._index.search(self._as_faiss_matrix(query_matrix), k)
        positions = torch.as_tensor(positions_np, dtype=torch.long, device=self._ids.device)
        scores = torch.as_tensor(scores_np, dtype=self._embeddings.dtype, device=self._embeddings.device)
        ids = self._ids.gather(0, positions.reshape(-1)).reshape(positions.shape)
        return IndexSearchResult(ids=ids, scores=scores)


class FaissIVFActionIndex(FaissActionIndex):
    """Optional approximate FAISS IVF-flat action index scaffold.

    This backend uses ``faiss.IndexIVFFlat`` with inner-product search. Cosine
    mode follows the flat FAISS backend by normalizing indexed actions and
    queries first. The index is trained during construction and update.

    ``nlist`` and ``nprobe`` are optional. When omitted, conservative small-data
    defaults are chosen, with ``nprobe`` defaulting to the effective ``nlist`` so
    smoke paths avoid empty result slots. This is scaffolding, not a benchmarked
    approximate retrieval configuration.
    """

    def __init__(
        self,
        action_embeddings: "torch.Tensor | Any",
        action_ids: "torch.Tensor | Any | None" = None,
        metric: str = "dot",
        normalize: bool = True,
        nlist: int | None = None,
        nprobe: int | None = None,
    ) -> None:
        self._requested_nlist = self._validate_optional_positive_int(nlist, name="nlist")
        self._requested_nprobe = self._validate_optional_positive_int(nprobe, name="nprobe")
        self.nlist = 0
        self.nprobe = 0
        super().__init__(
            action_embeddings,
            action_ids=action_ids,
            metric=metric,
            normalize=normalize,
        )

    @staticmethod
    def _validate_optional_positive_int(value: int | None, *, name: str) -> int | None:
        if value is None:
            return None
        if not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer when provided.")
        return value

    @staticmethod
    def _default_nlist(num_actions: int) -> int:
        return max(1, min(num_actions, int(num_actions**0.5) or 1))

    def _effective_nlist(self, num_actions: int) -> int:
        requested = self._requested_nlist or self._default_nlist(num_actions)
        return max(1, min(requested, num_actions))

    def _effective_nprobe(self, nlist: int) -> int:
        requested = self._requested_nprobe or nlist
        return max(1, min(requested, nlist))

    def update(
        self,
        action_embeddings: "torch.Tensor | Any",
        action_ids: "torch.Tensor | Any | None" = None,
    ) -> None:
        """Replace indexed embeddings, ids, and retrain the IVF-flat index."""

        embeddings = _as_embedding_matrix(action_embeddings, name="action_embeddings")
        if self.metric == "cosine" or self.normalize:
            embeddings = _safe_l2_normalize(embeddings, name="action_embeddings")

        if action_ids is None:
            ids = _default_ids(embeddings.shape[0], device=embeddings.device)
        else:
            ids = _validate_action_ids(action_ids, num_actions=embeddings.shape[0], device=embeddings.device)

        self._embeddings = embeddings
        self._ids = ids
        self.nlist = self._effective_nlist(int(embeddings.shape[0]))
        self.nprobe = self._effective_nprobe(self.nlist)

        quantizer = self._faiss.IndexFlatIP(int(embeddings.shape[1]))
        self._index = self._faiss.IndexIVFFlat(
            quantizer,
            int(embeddings.shape[1]),
            int(self.nlist),
            self._faiss.METRIC_INNER_PRODUCT,
        )
        matrix = self._as_faiss_matrix(embeddings)
        self._index.train(matrix)
        self._index.add(matrix)
        self._index.nprobe = int(self.nprobe)

    def search(self, queries: "torch.Tensor | Any", k: int) -> IndexSearchResult:
        """Return IVF-flat top-k action ids and scores for each query."""

        if self._index is None or self._embeddings is None or self._ids is None or self.num_actions == 0:
            raise ValueError("The action index is empty.")
        _validate_k(k, num_actions=self.num_actions)

        query_matrix = self._validate_queries(queries)
        scores_np, positions_np = self._index.search(self._as_faiss_matrix(query_matrix), k)
        positions = torch.as_tensor(positions_np, dtype=torch.long, device=self._ids.device)
        if torch.any(positions < 0):
            raise ValueError("FAISS IVF returned empty result slots; increase nprobe or reduce k.")
        scores = torch.as_tensor(scores_np, dtype=self._embeddings.dtype, device=self._embeddings.device)
        ids = self._ids.gather(0, positions.reshape(-1)).reshape(positions.shape)
        return IndexSearchResult(ids=ids, scores=scores)


class FaissHNSWActionIndex(FaissActionIndex):
    """Optional approximate FAISS HNSW (graph) action index.

    Uses ``faiss.IndexHNSWFlat`` with inner-product search. Unlike IVF, HNSW is a
    navigable-small-world graph and needs no training and no Voronoi partition,
    which makes it robust to embedding geometries that cluster poorly under
    k-means (the audit's preferred approximate index).

    Graph connectivity ``M`` and build effort ``ef_construction`` are fixed
    sensible defaults. Query effort ``ef_search`` controls the speed/recall
    tradeoff (HNSW's analogue of IVF ``nprobe``); when ``nprobe`` is supplied via
    :func:`make_action_index` it is reused as ``ef_search``. ``ef_search`` is
    always clamped to at least ``k`` at query time to avoid empty result slots.
    Scaffolding, not a tuned approximate configuration.
    """

    def __init__(
        self,
        action_embeddings: "torch.Tensor | Any",
        action_ids: "torch.Tensor | Any | None" = None,
        metric: str = "dot",
        normalize: bool = True,
        m: int | None = None,
        ef_construction: int | None = None,
        ef_search: int | None = None,
    ) -> None:
        self._m = self._validate_optional_positive_int(m, name="m") or 32
        self._ef_construction = self._validate_optional_positive_int(ef_construction, name="ef_construction") or 40
        self._requested_ef_search = self._validate_optional_positive_int(ef_search, name="ef_search")
        self.ef_search = self._requested_ef_search or 64
        super().__init__(action_embeddings, action_ids=action_ids, metric=metric, normalize=normalize)

    @staticmethod
    def _validate_optional_positive_int(value: int | None, *, name: str) -> int | None:
        if value is None:
            return None
        if not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer when provided.")
        return value

    def update(
        self,
        action_embeddings: "torch.Tensor | Any",
        action_ids: "torch.Tensor | Any | None" = None,
    ) -> None:
        """Replace indexed embeddings, ids, and rebuild the HNSW graph index."""

        embeddings = _as_embedding_matrix(action_embeddings, name="action_embeddings")
        if self.metric == "cosine" or self.normalize:
            embeddings = _safe_l2_normalize(embeddings, name="action_embeddings")

        if action_ids is None:
            ids = _default_ids(embeddings.shape[0], device=embeddings.device)
        else:
            ids = _validate_action_ids(action_ids, num_actions=embeddings.shape[0], device=embeddings.device)

        self._embeddings = embeddings
        self._ids = ids
        self._index = self._faiss.IndexHNSWFlat(
            int(embeddings.shape[1]), int(self._m), self._faiss.METRIC_INNER_PRODUCT
        )
        self._index.hnsw.efConstruction = int(self._ef_construction)
        self._index.add(self._as_faiss_matrix(embeddings))
        self._index.hnsw.efSearch = int(self.ef_search)

    def search(self, queries: "torch.Tensor | Any", k: int) -> IndexSearchResult:
        """Return HNSW top-k action ids and scores for each query."""

        if self._index is None or self._embeddings is None or self._ids is None or self.num_actions == 0:
            raise ValueError("The action index is empty.")
        _validate_k(k, num_actions=self.num_actions)

        self._index.hnsw.efSearch = int(max(self.ef_search, k))
        query_matrix = self._validate_queries(queries)
        scores_np, positions_np = self._index.search(self._as_faiss_matrix(query_matrix), k)
        positions = torch.as_tensor(positions_np, dtype=torch.long, device=self._ids.device)
        if torch.any(positions < 0):
            raise ValueError("FAISS HNSW returned empty result slots; increase ef_search or reduce k.")
        scores = torch.as_tensor(scores_np, dtype=self._embeddings.dtype, device=self._embeddings.device)
        ids = self._ids.gather(0, positions.reshape(-1)).reshape(positions.shape)
        return IndexSearchResult(ids=ids, scores=scores)


def make_action_index(
    action_embeddings: "torch.Tensor | Any",
    action_ids: "torch.Tensor | Any | None" = None,
    backend: str = "exact",
    metric: str = "dot",
    normalize: bool = True,
    nlist: int | None = None,
    nprobe: int | None = None,
) -> ActionIndex:
    """Create an action index backend.

    ``backend='exact'`` remains the default and requires no optional
    dependencies. ``backend='faiss'`` constructs exact flat FAISS search.
    ``backend='faiss_ivf'`` constructs an IVF-flat scaffold. ``backend='faiss_hnsw'``
    constructs an HNSW graph scaffold (reusing ``nprobe`` as ``ef_search``).
    FAISS-backed backends raise a clear ``ImportError`` if FAISS is unavailable.
    """

    if backend == "exact":
        return ExactActionIndex(action_embeddings, action_ids=action_ids, metric=metric, normalize=normalize)
    if backend == "faiss":
        return FaissActionIndex(action_embeddings, action_ids=action_ids, metric=metric, normalize=normalize)
    if backend == "faiss_ivf":
        return FaissIVFActionIndex(
            action_embeddings,
            action_ids=action_ids,
            metric=metric,
            normalize=normalize,
            nlist=nlist,
            nprobe=nprobe,
        )
    if backend == "faiss_hnsw":
        return FaissHNSWActionIndex(
            action_embeddings,
            action_ids=action_ids,
            metric=metric,
            normalize=normalize,
            ef_search=nprobe,
        )
    raise ValueError("backend must be one of 'exact', 'faiss', 'faiss_ivf', or 'faiss_hnsw'.")
