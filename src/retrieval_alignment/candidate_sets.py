"""Candidate-set construction for ANN-truncated policies.

DW-MQL+ANN learns and evaluates policies on retrieved action support rather
than the full action catalogue. As described in ``docs/algorithm_pseudocode.md``
and ``docs/code_architecture.md``, this module builds the per-state union

``U(s') = {logged action} union ANN_top_k(s') union random negatives union hard negatives``.

The output ``CandidateSet`` contains padded action ids and a boolean mask. Later
policy and Bellman-target code can use that mask to renormalize a target policy
over only valid candidates, making the ANN truncation explicit.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Any

try:  # pragma: no cover - exercised when the project dependencies are installed.
    import torch
except ModuleNotFoundError:  # pragma: no cover - keeps imports clean pre-install.
    torch = None  # type: ignore[assignment]


class CandidateSource(IntEnum):
    """Integer source labels for candidate ids."""

    PAD = 0
    LOGGED = 1
    ANN = 2
    RANDOM = 3
    HARD = 4


@dataclass(slots=True)
class CandidateSet:
    """A batched, padded candidate-action tensor.

    Attributes:
        ids: Integer action ids with shape ``[batch, width]``.
        mask: Boolean validity mask with shape ``[batch, width]``.
        scores: Optional retrieval scores aligned with ``ids`` where available.
        sources: Optional integer labels indicating whether each id came from
            logged, ANN, random-negative, hard-negative, or padding inputs.
    """

    ids: "torch.Tensor | Any"
    mask: "torch.Tensor | Any"
    scores: "torch.Tensor | Any | None" = None
    sources: "torch.Tensor | Any | None" = None

    @property
    def batch_size(self) -> int:
        return int(self.ids.shape[0])

    @property
    def width(self) -> int:
        return int(self.ids.shape[1])


def _require_torch() -> None:
    if torch is None:
        raise ModuleNotFoundError("PyTorch is required to build candidate sets.")


def _as_2d_long_tensor(values: Any, *, name: str, batch_size: int | None = None) -> "torch.Tensor":
    """Convert candidate-like input into a two-dimensional long tensor."""

    _require_torch()
    if values is None:
        if batch_size is None:
            raise ValueError(f"{name} cannot be None when batch_size is unknown.")
        return torch.empty((batch_size, 0), dtype=torch.long)

    tensor = torch.as_tensor(values, dtype=torch.long)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(1)
    if tensor.ndim != 2:
        raise ValueError(f"{name} must be a 1D or 2D tensor-like object.")
    if batch_size is not None and tensor.shape[0] != batch_size:
        raise ValueError(
            f"{name} batch size {tensor.shape[0]} does not match expected batch size {batch_size}."
        )
    return tensor


def _optional_2d(values: Any, *, name: str, batch_size: int, device: "torch.device") -> "torch.Tensor":
    tensor = _as_2d_long_tensor(values, name=name, batch_size=batch_size)
    return tensor.to(device=device)


def build_candidate_set(
    ann_indices: Any,
    *,
    logged_action_indices: Any | None = None,
    random_negative_indices: Any | None = None,
    hard_negative_indices: Any | None = None,
    max_candidates: int | None = None,
    pad_value: int = -1,
) -> CandidateSet:
    """Build a deterministic batched candidate union.

    Args:
        ann_indices: ANN top-k action ids shaped ``[batch, k]`` or ``[batch]``.
            These are accepted as input so the function remains independent of
            FAISS and any specific ANN backend.
        logged_action_indices: Optional logged action ids shaped ``[batch]`` or
            ``[batch, 1]``. When provided, they are placed first and therefore
            survive truncation unless ``max_candidates`` is zero.
        random_negative_indices: Optional random negative action ids shaped
            ``[batch, r]``.
        hard_negative_indices: Optional hard negative action ids shaped
            ``[batch, h]``.
        max_candidates: Optional fixed output width. Rows are truncated or
            padded to this width. If omitted, the width is the largest
            deduplicated row length in the batch.
        pad_value: Action id used in padded positions. Padded positions always
            have ``mask=False`` and source ``CandidateSource.PAD``.

    Returns:
        ``CandidateSet`` with action ids, boolean mask, and source labels.

    Design:
        Deduplication is per row and keeps the first occurrence in the order
        logged, ANN, random negatives, hard negatives. This order makes logged
        behavior support explicit while still preserving ANN retrieval as the
        main candidate source for the truncated target policy.
    """

    _require_torch()
    ann = _as_2d_long_tensor(ann_indices, name="ann_indices")
    batch_size = ann.shape[0]
    device = ann.device

    logged = _optional_2d(
        logged_action_indices,
        name="logged_action_indices",
        batch_size=batch_size,
        device=device,
    )
    random_negatives = _optional_2d(
        random_negative_indices,
        name="random_negative_indices",
        batch_size=batch_size,
        device=device,
    )
    hard_negatives = _optional_2d(
        hard_negative_indices,
        name="hard_negative_indices",
        batch_size=batch_size,
        device=device,
    )

    if max_candidates is not None and max_candidates < 0:
        raise ValueError("max_candidates must be non-negative or None.")

    row_ids: list[list[int]] = []
    row_sources: list[list[int]] = []
    for row in range(batch_size):
        seen: set[int] = set()
        ids: list[int] = []
        sources: list[int] = []

        for tensor, source in (
            (logged, CandidateSource.LOGGED),
            (ann, CandidateSource.ANN),
            (random_negatives, CandidateSource.RANDOM),
            (hard_negatives, CandidateSource.HARD),
        ):
            for value in tensor[row].tolist():
                action_id = int(value)
                if action_id in seen:
                    continue
                seen.add(action_id)
                ids.append(action_id)
                sources.append(int(source))

        if max_candidates is not None:
            ids = ids[:max_candidates]
            sources = sources[:max_candidates]

        row_ids.append(ids)
        row_sources.append(sources)

    if max_candidates is None:
        width = max((len(ids) for ids in row_ids), default=0)
    else:
        width = max_candidates

    candidate_ids = torch.full((batch_size, width), pad_value, dtype=torch.long, device=device)
    mask = torch.zeros((batch_size, width), dtype=torch.bool, device=device)
    source_tensor = torch.full(
        (batch_size, width),
        int(CandidateSource.PAD),
        dtype=torch.long,
        device=device,
    )

    for row, ids in enumerate(row_ids):
        valid_count = min(len(ids), width)
        if valid_count == 0:
            continue
        candidate_ids[row, :valid_count] = torch.tensor(ids[:valid_count], dtype=torch.long, device=device)
        mask[row, :valid_count] = True
        source_tensor[row, :valid_count] = torch.tensor(
            row_sources[row][:valid_count],
            dtype=torch.long,
            device=device,
        )

    return CandidateSet(ids=candidate_ids, mask=mask, sources=source_tensor)


def build_next_state_candidates(
    ann_indices: Any,
    *,
    logged_action_indices: Any | None = None,
    random_negative_indices: Any | None = None,
    hard_negative_indices: Any | None = None,
    max_candidates: int | None = None,
    pad_value: int = -1,
) -> CandidateSet:
    """Alias for candidate construction at Bellman next states."""

    return build_candidate_set(
        ann_indices,
        logged_action_indices=logged_action_indices,
        random_negative_indices=random_negative_indices,
        hard_negative_indices=hard_negative_indices,
        max_candidates=max_candidates,
        pad_value=pad_value,
    )
