from __future__ import annotations

import sys
from pathlib import Path

import torch


EVAL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EVAL_ROOT))

from bbox_teacher_audit.utility import (  # noqa: E402
    deletion_sets,
    grad_input_utility,
    retained_indices,
)


def test_grad_input_uses_relu_not_absolute_value():
    features = torch.tensor([[1.0, 2.0], [1.0, 1.0]])
    gradient = torch.tensor([[-2.0, 0.0], [2.0, -1.0]])
    assert torch.equal(grad_input_utility(gradient, features), torch.tensor([0.0, 1.0]))


def test_top_bottom_random_delete_equal_counts_and_random_is_reproducible():
    utility = torch.arange(10, dtype=torch.float32)
    first = deletion_sets(utility, "sample", 42)
    second = deletion_sets(utility, "sample", 42)
    assert first["top"].numel() == first["bottom"].numel() == 2
    assert all(indices.numel() == 2 for indices in first["random"])
    assert all(torch.equal(a, b) for a, b in zip(first["random"], second["random"]))
    assert first["top"].tolist() == [8, 9]
    assert first["bottom"].tolist() == [0, 1]


def test_retained_indices_are_sorted_and_have_expected_count():
    assert retained_indices(7, [5, 1]).tolist() == [0, 2, 3, 4, 6]
