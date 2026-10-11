"""Tests for exact and optional FAISS action indexes."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from retrieval_alignment.ann_index import ExactActionIndex, FaissActionIndex, FaissHNSWActionIndex, FaissIVFActionIndex, make_action_index


def test_construction_with_valid_embeddings() -> None:
    index = ExactActionIndex(torch.eye(3), metric="dot", normalize=False)

    assert index.num_actions == 3


def test_num_actions_and_embedding_dim_properties() -> None:
    index = ExactActionIndex(torch.zeros(5, 7), metric="dot", normalize=False)

    assert index.num_actions == 5
    assert index.embedding_dim == 7


def test_dot_product_search_returns_known_nearest_actions() -> None:
    embeddings = torch.tensor([[1.0, 0.0], [0.0, 1.0], [2.0, 0.0]])
    index = ExactActionIndex(embeddings, metric="dot", normalize=False)

    result = index.search(torch.tensor([[1.0, 0.0]]), k=2)

    assert result.ids.tolist() == [[2, 0]]
    assert result.scores.tolist()[0] == pytest.approx([2.0, 1.0])


def test_cosine_search_returns_known_nearest_actions() -> None:
    embeddings = torch.tensor([[1.0, 0.0], [0.0, 1.0], [2.0, 0.0]])
    index = ExactActionIndex(embeddings, metric="cosine")

    result = index.search(torch.tensor([[1.0, 0.0]]), k=2)

    assert result.ids.tolist() == [[0, 2]]
    assert result.scores.tolist()[0] == pytest.approx([1.0, 1.0])


def test_single_query_input_returns_batched_output() -> None:
    index = ExactActionIndex(torch.eye(3), metric="dot", normalize=False)

    result = index.search(torch.tensor([0.0, 1.0, 0.0]), k=1)

    assert result.ids.shape == (1, 1)
    assert result.scores.shape == (1, 1)
    assert result.ids.tolist() == [[1]]


def test_action_ids_are_returned_instead_of_row_indices() -> None:
    index = ExactActionIndex(
        torch.eye(3),
        action_ids=torch.tensor([10, 20, 30]),
        metric="dot",
        normalize=False,
    )

    result = index.search(torch.tensor([[0.0, 0.0, 1.0]]), k=1)

    assert result.ids.tolist() == [[30]]


def test_update_replaces_embeddings_and_ids() -> None:
    index = ExactActionIndex(torch.eye(2), metric="dot", normalize=False)

    index.update(
        torch.tensor([[0.0, 1.0], [3.0, 0.0]]),
        action_ids=torch.tensor([100, 200]),
    )
    result = index.search(torch.tensor([[1.0, 0.0]]), k=1)

    assert index.num_actions == 2
    assert result.ids.tolist() == [[200]]


def test_invalid_metric_raises_value_error() -> None:
    with pytest.raises(ValueError, match="metric"):
        ExactActionIndex(torch.eye(2), metric="l2")


def test_invalid_k_raises_value_error() -> None:
    index = ExactActionIndex(torch.eye(2), metric="dot", normalize=False)

    with pytest.raises(ValueError, match="k"):
        index.search(torch.tensor([[1.0, 0.0]]), k=0)


def test_k_greater_than_num_actions_raises_value_error() -> None:
    index = ExactActionIndex(torch.eye(2), metric="dot", normalize=False)

    with pytest.raises(ValueError, match="cannot exceed"):
        index.search(torch.tensor([[1.0, 0.0]]), k=3)


def test_query_dimension_mismatch_raises_value_error() -> None:
    index = ExactActionIndex(torch.eye(2), metric="dot", normalize=False)

    with pytest.raises(ValueError, match="dimension"):
        index.search(torch.tensor([[1.0, 0.0, 0.0]]), k=1)


def test_non_finite_embeddings_raise_value_error() -> None:
    with pytest.raises(ValueError, match="finite"):
        ExactActionIndex(torch.tensor([[1.0, float("nan")]]), metric="dot", normalize=False)


def test_zero_norm_cosine_embeddings_are_rejected() -> None:
    with pytest.raises(ValueError, match="zero-norm"):
        ExactActionIndex(torch.tensor([[0.0, 0.0], [1.0, 0.0]]), metric="cosine")


def test_zero_norm_cosine_queries_are_rejected() -> None:
    index = ExactActionIndex(torch.eye(2), metric="cosine")

    with pytest.raises(ValueError, match="zero-norm"):
        index.search(torch.tensor([[0.0, 0.0]]), k=1)


def test_deterministic_tie_behavior_preserves_index_order() -> None:
    embeddings = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    ids = torch.tensor([30, 10, 20])
    index = ExactActionIndex(embeddings, action_ids=ids, metric="dot", normalize=False)

    result = index.search(torch.tensor([[1.0, 0.0]]), k=2)

    assert result.ids.tolist() == [[30, 10]]


def test_make_action_index_exact_returns_exact_action_index() -> None:
    index = make_action_index(torch.eye(3), backend="exact", metric="dot")

    assert isinstance(index, ExactActionIndex)


def test_make_action_index_unknown_backend_raises_value_error() -> None:
    with pytest.raises(ValueError, match="backend"):
        make_action_index(torch.eye(3), backend="unknown", metric="dot")


def test_make_action_index_faiss_missing_or_constructs() -> None:
    try:
        __import__("faiss")
    except ModuleNotFoundError:
        with pytest.raises(ImportError, match="FAISS"):
            make_action_index(torch.eye(3), backend="faiss", metric="dot")
    else:
        index = make_action_index(torch.eye(3), backend="faiss", metric="dot")
        assert isinstance(index, FaissActionIndex)


def test_make_action_index_faiss_ivf_missing_or_constructs() -> None:
    try:
        __import__("faiss")
    except ModuleNotFoundError:
        with pytest.raises(ImportError, match="FAISS"):
            make_action_index(torch.eye(4), backend="faiss_ivf", metric="dot")
    else:
        index = make_action_index(torch.eye(4), backend="faiss_ivf", metric="dot", nlist=2, nprobe=2)
        assert isinstance(index, FaissIVFActionIndex)


def test_faiss_ivf_invalid_nlist_and_nprobe_raise_value_error() -> None:
    with pytest.raises(ValueError, match="nlist"):
        make_action_index(torch.eye(4), backend="faiss_ivf", nlist=0)

    with pytest.raises(ValueError, match="nprobe"):
        make_action_index(torch.eye(4), backend="faiss_ivf", nprobe=0)


def test_make_action_index_faiss_hnsw_missing_or_constructs() -> None:
    try:
        __import__("faiss")
    except ModuleNotFoundError:
        with pytest.raises(ImportError, match="FAISS"):
            make_action_index(torch.eye(4), backend="faiss_hnsw", metric="dot")
    else:
        index = make_action_index(torch.eye(4), backend="faiss_hnsw", metric="dot", nprobe=32)
        assert isinstance(index, FaissHNSWActionIndex)
        assert index.ef_search == 32  # nprobe is reused as ef_search


def test_faiss_hnsw_search_matches_exact_when_available() -> None:
    pytest.importorskip("faiss")
    torch.manual_seed(0)
    embeddings = torch.randn(500, 8)
    queries = torch.randn(20, 8)
    exact = make_action_index(embeddings, backend="exact", metric="cosine")
    hnsw = make_action_index(embeddings, backend="faiss_hnsw", metric="cosine", nprobe=128)
    k = 10
    exact_ids = exact.search(queries, k=k).ids
    hnsw_ids = hnsw.search(queries, k=k).ids
    overlap = sum(
        len(set(exact_ids[i].tolist()) & set(hnsw_ids[i].tolist())) for i in range(queries.shape[0])
    ) / (queries.shape[0] * k)
    # HNSW with generous ef_search recovers nearly all exact neighbours.
    assert overlap >= 0.9


def test_faiss_dot_search_matches_exact_when_available() -> None:
    pytest.importorskip("faiss")
    embeddings = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 2.0],
            [3.0, 0.0],
            [-1.0, 0.0],
        ]
    )
    queries = torch.tensor([[1.0, 0.1], [0.1, 1.0]])
    exact = ExactActionIndex(embeddings, metric="dot", normalize=False)
    faiss_index = FaissActionIndex(embeddings, metric="dot", normalize=False)

    exact_result = exact.search(queries, k=2)
    faiss_result = faiss_index.search(queries, k=2)

    assert faiss_result.ids.tolist() == exact_result.ids.tolist()
    assert torch.allclose(faiss_result.scores, exact_result.scores)


def test_faiss_cosine_search_matches_exact_when_available() -> None:
    pytest.importorskip("faiss")
    embeddings = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [-1.0, 0.0],
            [0.0, -1.0],
        ]
    )
    queries = torch.tensor([[3.0, 1.0], [1.0, 3.0]])
    exact = ExactActionIndex(embeddings, metric="cosine")
    faiss_index = FaissActionIndex(embeddings, metric="cosine")

    exact_result = exact.search(queries, k=2)
    faiss_result = faiss_index.search(queries, k=2)

    assert faiss_result.ids.tolist() == exact_result.ids.tolist()
    assert torch.allclose(faiss_result.scores, exact_result.scores, atol=1e-6)


def test_faiss_custom_action_ids_when_available() -> None:
    pytest.importorskip("faiss")
    embeddings = torch.tensor([[1.0, 0.0], [0.0, 2.0], [3.0, 0.0]])
    ids = torch.tensor([10, 20, 30])
    index = FaissActionIndex(embeddings, action_ids=ids, metric="dot", normalize=False)

    result = index.search(torch.tensor([[1.0, 0.0]]), k=2)

    assert result.ids.tolist() == [[30, 10]]


def test_faiss_ivf_search_shapes_and_catalogue_ids_when_available() -> None:
    pytest.importorskip("faiss")
    embeddings = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [2.0, 0.0],
            [0.0, 2.0],
            [-1.0, 0.0],
            [0.0, -1.0],
        ]
    )
    ids = torch.tensor([10, 11, 12, 13, 14, 15])
    index = FaissIVFActionIndex(
        embeddings,
        action_ids=ids,
        metric="dot",
        normalize=False,
        nlist=2,
        nprobe=2,
    )

    result = index.search(torch.tensor([[1.0, 0.0], [0.0, 1.0]]), k=2)

    assert result.ids.shape == (2, 2)
    assert result.scores.shape == (2, 2)
    assert set(result.ids.reshape(-1).tolist()).issubset(set(ids.tolist()))
    assert index.nlist == 2
    assert index.nprobe == 2


def test_faiss_ivf_cosine_mode_when_available() -> None:
    pytest.importorskip("faiss")
    embeddings = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [-1.0, 0.0],
            [0.0, -1.0],
        ]
    )
    index = FaissIVFActionIndex(embeddings, metric="cosine", nlist=2, nprobe=2)

    result = index.search(torch.tensor([[3.0, 1.0]]), k=2)

    assert result.ids.shape == (1, 2)
    assert result.scores.shape == (1, 2)
    assert torch.isfinite(result.scores).all()
