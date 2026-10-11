"""Tests for deterministic candidate-set construction."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from retrieval_alignment.candidate_sets import CandidateSource, build_candidate_set


def test_logged_action_is_included_first() -> None:
    candidates = build_candidate_set(
        torch.tensor([[2, 3, 4]]),
        logged_action_indices=torch.tensor([9]),
    )

    assert candidates.ids.tolist() == [[9, 2, 3, 4]]
    assert candidates.mask.tolist() == [[True, True, True, True]]
    assert candidates.sources.tolist()[0][0] == int(CandidateSource.LOGGED)


def test_deduplicates_preserving_deterministic_order() -> None:
    candidates = build_candidate_set(
        torch.tensor([[2, 3, 2, 4]]),
        logged_action_indices=torch.tensor([3]),
        random_negative_indices=torch.tensor([[4, 5, 3]]),
        hard_negative_indices=torch.tensor([[5, 6]]),
    )

    assert candidates.ids.tolist() == [[3, 2, 4, 5, 6]]
    assert candidates.sources.tolist() == [[
        int(CandidateSource.LOGGED),
        int(CandidateSource.ANN),
        int(CandidateSource.ANN),
        int(CandidateSource.RANDOM),
        int(CandidateSource.HARD),
    ]]


def test_padding_to_fixed_width() -> None:
    candidates = build_candidate_set(
        torch.tensor([[2, 3]]),
        logged_action_indices=torch.tensor([1]),
        max_candidates=5,
        pad_value=-99,
    )

    assert candidates.ids.tolist() == [[1, 2, 3, -99, -99]]
    assert candidates.mask.tolist() == [[True, True, True, False, False]]
    assert candidates.sources.tolist()[0][-2:] == [int(CandidateSource.PAD), int(CandidateSource.PAD)]


def test_truncation_keeps_logged_then_ann_order() -> None:
    candidates = build_candidate_set(
        torch.tensor([[2, 3, 4, 5]]),
        logged_action_indices=torch.tensor([1]),
        random_negative_indices=torch.tensor([[6, 7]]),
        max_candidates=3,
    )

    assert candidates.ids.tolist() == [[1, 2, 3]]
    assert candidates.mask.tolist() == [[True, True, True]]


def test_batched_construction_and_mask_correctness() -> None:
    candidates = build_candidate_set(
        torch.tensor([[2, 3], [4, 4], [8, 9]]),
        logged_action_indices=torch.tensor([1, 4, 8]),
        random_negative_indices=torch.tensor([[5], [6], [9]]),
        hard_negative_indices=torch.tensor([[7], [6], [10]]),
        max_candidates=4,
    )

    assert candidates.ids.tolist() == [
        [1, 2, 3, 5],
        [4, 6, -1, -1],
        [8, 9, 10, -1],
    ]
    assert candidates.mask.tolist() == [
        [True, True, True, True],
        [True, True, False, False],
        [True, True, True, False],
    ]


def test_empty_optional_negatives_are_supported() -> None:
    candidates = build_candidate_set(
        torch.tensor([[2, 3], [4, 5]]),
        logged_action_indices=torch.tensor([1, 4]),
        random_negative_indices=None,
        hard_negative_indices=torch.empty((2, 0), dtype=torch.long),
    )

    assert candidates.ids.tolist() == [[1, 2, 3], [4, 5, -1]]
    assert candidates.mask.tolist() == [[True, True, True], [True, True, False]]


def test_deterministic_output_for_same_inputs_and_seed() -> None:
    torch.manual_seed(123)
    ann = torch.tensor([[2, 3, 2], [4, 5, 6]])
    logged = torch.tensor([1, 4])
    random_negatives = torch.tensor([[7, 8], [6, 9]])
    hard_negatives = torch.tensor([[8, 10], [10, 11]])

    first = build_candidate_set(
        ann,
        logged_action_indices=logged,
        random_negative_indices=random_negatives,
        hard_negative_indices=hard_negatives,
        max_candidates=5,
    )
    torch.manual_seed(123)
    second = build_candidate_set(
        ann,
        logged_action_indices=logged,
        random_negative_indices=random_negatives,
        hard_negative_indices=hard_negatives,
        max_candidates=5,
    )

    assert torch.equal(first.ids, second.ids)
    assert torch.equal(first.mask, second.mask)
    assert torch.equal(first.sources, second.sources)


def test_mismatched_batch_size_raises() -> None:
    with pytest.raises(ValueError, match="batch size"):
        build_candidate_set(
            torch.tensor([[1, 2], [3, 4]]),
            logged_action_indices=torch.tensor([9]),
        )
