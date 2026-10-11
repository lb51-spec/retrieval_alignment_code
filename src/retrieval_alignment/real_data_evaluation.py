"""Independent held-out and retrieval metrics for real_data experiments."""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from typing import Any

import torch

from .ann_index import make_action_index
from .benchmarks import (
    KuaiRandLocalTask,
    MovieLensLocalTask,
    _movielens_states_for_rows,
    _movielens_target_indices,
    score_movielens_catalogue_with_model,
)
from .metrics import (
    candidate_constrained_ranking_metrics,
    fixed_reference_retrieval_metrics,
    single_positive_ranking_metrics,
)


@dataclass(frozen=True, slots=True)
class RealDataEvaluationConfig:
    """Settings for deterministic held-out real_data evaluation."""

    split: str = "test"
    k: int = 20
    index_backend: str = "exact"
    index_metric: str = "dot"
    temperature: float = 1.0
    max_eval_rows: int | None = None
    seed: int = 0
    query_batch_size: int = 512
    ef_search: int = 64

    def __post_init__(self) -> None:
        if self.split not in {"train", "validation", "test"}:
            raise ValueError("split must be one of: train, validation, test.")
        if not isinstance(self.k, int) or self.k <= 0:
            raise ValueError("k must be a positive integer.")
        if self.index_backend not in {"exact", "faiss", "faiss_ivf", "faiss_hnsw"}:
            raise ValueError("index_backend must be exact, faiss, faiss_ivf, or faiss_hnsw.")
        if self.index_metric not in {"dot", "cosine"}:
            raise ValueError("index_metric must be dot or cosine.")
        if not isinstance(self.temperature, int | float) or float(self.temperature) <= 0:
            raise ValueError("temperature must be positive.")
        if self.max_eval_rows is not None and (
            not isinstance(self.max_eval_rows, int) or self.max_eval_rows <= 0
        ):
            raise ValueError("max_eval_rows must be None or a positive integer.")
        if not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer.")
        if not isinstance(self.query_batch_size, int) or self.query_batch_size <= 0:
            raise ValueError("query_batch_size must be a positive integer.")
        if not isinstance(self.ef_search, int) or self.ef_search <= 0:
            raise ValueError("ef_search must be a positive integer.")


def stable_action_id_hash(action_ids: Any) -> str:
    """Return a stable digest for a one-dimensional catalogue id tensor."""

    ids = torch.as_tensor(action_ids, dtype=torch.int64).detach().cpu().contiguous()
    if ids.ndim != 1 or ids.numel() == 0:
        raise ValueError("action_ids must be a non-empty one-dimensional tensor.")
    return hashlib.sha256(ids.numpy().tobytes()).hexdigest()


def _select_positive_rows(
    task: MovieLensLocalTask | KuaiRandLocalTask,
    config: RealDataEvaluationConfig,
) -> list[dict[str, Any]]:
    rows = [row for row in task.get_split_rows(config.split) if bool(row["positive"])]
    if not rows:
        raise ValueError(f"{config.split} split contains no positive evaluation rows.")
    if config.max_eval_rows is not None and len(rows) > config.max_eval_rows:
        generator = torch.Generator().manual_seed(config.seed)
        selected = torch.randperm(len(rows), generator=generator)[: config.max_eval_rows].tolist()
        rows = [rows[index] for index in sorted(selected)]
    return rows


def _latency_summary(values: list[float]) -> dict[str, float | int]:
    if not values:
        raise ValueError("query latency values must be non-empty.")
    ordered = sorted(float(value) for value in values)
    p95_index = min(len(ordered) - 1, int(0.95 * len(ordered)))
    return {
        "num_query_batches": len(ordered),
        "query_latency_median_ms": ordered[len(ordered) // 2] * 1000.0,
        "query_latency_p95_ms": ordered[p95_index] * 1000.0,
    }


def _retrieve_candidates(
    task: MovieLensLocalTask | KuaiRandLocalTask,
    model: Any,
    rows: list[dict[str, Any]],
    config: RealDataEvaluationConfig,
) -> tuple[torch.Tensor, dict[str, Any]]:
    action_features, action_ids = task.make_action_catalogue()
    states = _movielens_states_for_rows(task, rows)
    model.eval()
    with torch.no_grad():
        build_started = time.perf_counter()
        action_embeddings = model.encode_action_index(action_features).detach()
        compact_ids = torch.arange(
            task.num_actions,
            dtype=torch.long,
            device=action_embeddings.device,
        )
        index = make_action_index(
            action_embeddings,
            action_ids=compact_ids,
            backend=config.index_backend,
            metric=config.index_metric,
            normalize=config.index_metric == "cosine",
            nprobe=config.ef_search if config.index_backend == "faiss_hnsw" else None,
        )
        build_seconds = time.perf_counter() - build_started
        candidate_batches: list[torch.Tensor] = []
        query_latencies: list[float] = []
        for start in range(0, len(rows), config.query_batch_size):
            batch_states = states[start : start + config.query_batch_size]
            queries = model.encode_state_query(batch_states).detach()
            query_started = time.perf_counter()
            candidate_batches.append(index.search(queries, k=min(config.k, task.num_actions)).ids)
            query_latencies.append(time.perf_counter() - query_started)
    parameters: dict[str, int] = {}
    if config.index_backend == "faiss_hnsw":
        parameters = {"M": 32, "ef_construction": 40, "ef_search": config.ef_search}
    metadata = {
        "index_build_seconds": build_seconds,
        "index_parameters": parameters,
        "action_id_hash": stable_action_id_hash(action_ids),
        **_latency_summary(query_latencies),
    }
    return torch.cat(candidate_batches, dim=0).cpu(), metadata


def evaluate_real_data_metrics(
    task: MovieLensLocalTask | KuaiRandLocalTask,
    model: Any,
    config: RealDataEvaluationConfig,
    reference_model: Any | None = None,
) -> dict[str, Any]:
    """Evaluate full-catalogue and deployed-candidate held-out quality."""

    if not isinstance(task, (MovieLensLocalTask, KuaiRandLocalTask)):
        raise TypeError("task must be a MovieLensLocalTask or KuaiRandLocalTask.")
    if not isinstance(config, RealDataEvaluationConfig):
        raise TypeError("config must be a RealDataEvaluationConfig.")
    rows = _select_positive_rows(task, config)
    scores = score_movielens_catalogue_with_model(task, model, rows)
    positive_ids = _movielens_target_indices(task, rows)
    candidate_ids, index_metadata = _retrieve_candidates(task, model, rows, config)
    result: dict[str, Any] = {
        "full_catalogue": single_positive_ranking_metrics(scores, positive_ids, config.k),
        "deployed_candidates": candidate_constrained_ranking_metrics(
            scores,
            positive_ids,
            candidate_ids,
            config.k,
        ),
        "self_critic_reference": fixed_reference_retrieval_metrics(
            scores,
            candidate_ids,
            temperature=config.temperature,
        ),
        "metadata": {
            "split": config.split,
            "index_backend": config.index_backend,
            "index_metric": config.index_metric,
            "fixed_reference_used": reference_model is not None,
            **index_metadata,
        },
    }
    if reference_model is not None:
        reference_scores = score_movielens_catalogue_with_model(task, reference_model, rows)
        result["fixed_reference"] = fixed_reference_retrieval_metrics(
            reference_scores,
            candidate_ids,
            temperature=config.temperature,
        )
    return result
