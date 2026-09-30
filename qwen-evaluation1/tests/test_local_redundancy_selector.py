"""CPU tests for inference-only Local-GAZE focal redundancy."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from token_compression.selector_model import (
    _build_merged_token_grid_metadata,
    _compute_focal_redundancy_diagnostics,
    _compute_local_max_similarity,
    _select_focal_indices,
    _select_indices_with_cosine_penalty,
    _select_indices_with_local_cosine_penalty,
)


def _local_select(
    scores,
    features,
    positions,
    k,
    *,
    groups=None,
    radius=1,
    weight=2.0,
    threshold=0.7,
):
    score_tensor = torch.tensor(scores, dtype=torch.float32)
    feature_tensor = torch.tensor(features, dtype=torch.float32)
    position_tensor = torch.tensor(positions, dtype=torch.long)
    if groups is None:
        groups = [0] * len(scores)
    return _select_indices_with_local_cosine_penalty(
        score_tensor,
        feature_tensor,
        position_tensor,
        torch.tensor(groups, dtype=torch.long),
        k,
        weight,
        threshold,
        radius,
    )


def test_local_max_masks_nonlocal_before_max_and_preserves_negative_cosine():
    similarity = torch.tensor([[-0.5, 0.9], [0.7, 0.8]])
    values, has_neighbor = _compute_local_max_similarity(
        similarity,
        torch.tensor([[0, 0], [5, 5]]),
        torch.tensor([[0, 1], [10, 10]]),
        torch.zeros(2, dtype=torch.long),
        torch.zeros(2, dtype=torch.long),
        radius=1,
    )
    torch.testing.assert_close(values, torch.tensor([-0.5, 0.0]))
    assert has_neighbor.tolist() == [True, False]
    assert torch.isfinite(values).all()


def test_empty_selected_set_returns_zero_penalty_input():
    values, has_neighbor = _compute_local_max_similarity(
        torch.empty((2, 0)),
        torch.tensor([[0, 0], [1, 1]]),
        torch.empty((0, 2), dtype=torch.long),
        torch.zeros(2, dtype=torch.long),
        torch.empty((0,), dtype=torch.long),
        radius=1,
    )
    assert values.tolist() == [0.0, 0.0]
    assert not has_neighbor.any()


def test_empty_candidate_set_is_safe():
    values, has_neighbor = _compute_local_max_similarity(
        torch.empty((0, 2)),
        torch.empty((0, 2), dtype=torch.long),
        torch.tensor([[0, 0], [1, 1]]),
        torch.empty((0,), dtype=torch.long),
        torch.zeros(2, dtype=torch.long),
        radius=1,
    )
    assert values.shape == has_neighbor.shape == (0,)


def test_radius_one_is_exact_three_by_three_chebyshev_neighborhood():
    candidate = torch.tensor([[4, 4]] * 4)
    selected = torch.tensor([[3, 3], [4, 5], [5, 4], [6, 4]])
    _, has_neighbor = _compute_local_max_similarity(
        torch.ones((4, 4)),
        candidate,
        selected,
        torch.arange(4),
        torch.arange(4),
        radius=1,
    )
    assert has_neighbor.tolist() == [True, True, True, False]


def test_local_neighbors_do_not_cross_group_boundaries():
    values, has_neighbor = _compute_local_max_similarity(
        torch.tensor([[0.99]]),
        torch.tensor([[0, 0]]),
        torch.tensor([[0, 0]]),
        torch.tensor([0]),
        torch.tensor([1]),
        radius=1,
    )
    assert values.item() == 0.0
    assert not has_neighbor.item()


@pytest.mark.parametrize("radius,error", [(-1, ValueError), (1.5, TypeError), (True, TypeError)])
def test_invalid_radius_is_rejected(radius, error):
    with pytest.raises(error):
        _compute_local_max_similarity(
            torch.ones((1, 1)),
            torch.zeros((1, 2), dtype=torch.long),
            torch.zeros((1, 2), dtype=torch.long),
            torch.zeros(1, dtype=torch.long),
            torch.zeros(1, dtype=torch.long),
            radius=radius,
        )


def test_near_high_cosine_is_penalized_but_far_high_cosine_is_not():
    selected = _local_select(
        [1.0, 0.99, 0.98, 0.2],
        [[1, 0], [1, 0], [1, 0], [0, 1]],
        [[0, 0], [0, 1], [0, 3], [1, 0]],
        2,
    )
    assert selected.tolist() == [0, 2]


def test_near_below_threshold_is_not_penalized():
    selected = _local_select(
        [1.0, 0.99, 0.5],
        [[1, 0], [0, 1], [1, 0]],
        [[0, 0], [0, 1], [0, 3]],
        2,
    )
    assert selected.tolist() == [0, 1]


def test_local_selection_has_exact_unique_budget_and_can_be_sorted_for_reconstruction():
    selected = _local_select(
        [5, 4, 3, 2, 1],
        [[1, 0], [1, 0], [0, 1], [1, 0], [0, 1]],
        [[0, i] for i in range(5)],
        3,
    )
    assert len(selected) == len(set(selected.tolist())) == 3
    assert torch.sort(selected).values.tolist() == sorted(selected.tolist())


def test_large_radius_matches_global_selection_and_penalty_behavior():
    generator = torch.Generator().manual_seed(20260929)
    scores = torch.randn(20, generator=generator)
    features = torch.randn(20, 7, generator=generator)
    positions = torch.tensor([[row, col] for row in range(4) for col in range(5)])
    groups = torch.zeros(20, dtype=torch.long)
    global_indices = _select_indices_with_cosine_penalty(
        scores, features, 7, 1.0, 0.7
    )
    local_indices = _select_indices_with_local_cosine_penalty(
        scores, features, positions, groups, 7, 1.0, 0.7, radius=4
    )
    assert torch.equal(global_indices, local_indices)


def test_global_dispatch_remains_exact_legacy_call():
    generator = torch.Generator().manual_seed(7)
    scores = torch.randn(17, generator=generator)
    features = torch.randn(17, 5, generator=generator)
    expected = _select_indices_with_cosine_penalty(scores, features, 6, 1.0, 0.7)
    actual = _select_focal_indices(scores, features, 6, 1.0, 0.7)
    assert torch.equal(expected, actual)


def test_none_dispatch_is_exact_importance_topk():
    scores = torch.tensor([0.2, 0.9, 0.5, 0.8, 0.1])
    features = torch.randn(5, 3, generator=torch.Generator().manual_seed(11))
    expected = torch.topk(scores, k=3, largest=True).indices
    actual = _select_focal_indices(
        scores,
        features,
        3,
        penalty_weight=999.0,
        similarity_threshold=-1.0,
        focal_redundancy_mode="none",
    )
    assert torch.equal(expected, actual)


def test_none_diagnostics_have_zero_effective_redundancy():
    scores = torch.tensor([0.9, 0.8, 0.7, 0.6])
    features = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
    diagnostics = _compute_focal_redundancy_diagnostics(
        scores,
        features,
        torch.tensor([0, 1]),
        torch.tensor([[0, 0], [0, 1], [0, 2], [0, 3]]),
        torch.zeros(4, dtype=torch.long),
        mode="none",
        radius=1,
        penalty_weight=1.0,
        similarity_threshold=-1.0,
    )
    assert diagnostics["mode"] == "none"
    assert diagnostics["adaptive_alpha"] == 0.0
    assert diagnostics["lambda_eff"] == 0.0
    assert diagnostics["max_similarity"] == [0.0, 0.0]
    assert diagnostics["penalty_base"] == [0.0, 0.0]


def test_none_and_distance_aware_modes_are_mutually_exclusive():
    with pytest.raises(ValueError, match="mutually exclusive"):
        _select_focal_indices(
            torch.tensor([1.0, 0.5]),
            torch.eye(2),
            1,
            1.0,
            0.7,
            distance_aware_redundancy=True,
            focal_redundancy_mode="none",
        )


def test_local_and_distance_aware_modes_are_mutually_exclusive():
    with pytest.raises(ValueError, match="mutually exclusive"):
        _select_focal_indices(
            torch.tensor([1.0, 0.5]),
            torch.eye(2),
            1,
            1.0,
            0.7,
            distance_aware_redundancy=True,
            focal_redundancy_mode="local",
            local_radius=1,
            token_positions=torch.tensor([[0, 0], [0, 1]]),
            token_group_ids=torch.zeros(2, dtype=torch.long),
        )


def test_post_merge_metadata_matches_qwen_block_flatten_order():
    grid = torch.tensor([[2, 4, 6], [1, 2, 4]])
    positions, groups, temporal_ids = _build_merged_token_grid_metadata(
        grid, merge_size=2, device=torch.device("cpu")
    )
    assert positions.shape == (14, 2)
    assert groups.tolist() == [0] * 6 + [1] * 6 + [2] * 2
    assert temporal_ids.tolist() == [0] * 6 + [1] * 6 + [0] * 2

    offset = 0
    for image_id, (t, h, w) in enumerate(grid.tolist()):
        unique = torch.arange(t * h * w).reshape(t, h, w)
        qwen_blocks = (
            unique.reshape(t, h // 2, 2, w // 2, 2)
            .permute(0, 1, 3, 2, 4)
            .reshape(-1, 4)
        )
        for sequence_index, block in enumerate(qwen_blocks):
            temporal_id = sequence_index // ((h // 2) * (w // 2))
            within_frame = sequence_index % ((h // 2) * (w // 2))
            row, col = divmod(within_frame, w // 2)
            absolute = offset + sequence_index
            assert positions[absolute].tolist() == [row, col]
            assert temporal_ids[absolute].item() == temporal_id
            expected = unique[
                temporal_id,
                row * 2 : (row + 1) * 2,
                col * 2 : (col + 1) * 2,
            ].reshape(-1)
            assert torch.equal(block, expected)
        offset += len(qwen_blocks)


def test_multiple_images_and_frames_never_share_group_ids():
    positions, groups, temporal_ids = _build_merged_token_grid_metadata(
        torch.tensor([[2, 2, 2], [1, 2, 2]]),
        merge_size=2,
        device=torch.device("cpu"),
    )
    assert positions.tolist() == [[0, 0], [0, 0], [0, 0]]
    assert groups.tolist() == [0, 1, 2]
    assert temporal_ids.tolist() == [0, 1, 0]



def test_redundancy_diagnostics_are_finite_and_report_global_distance():
    from token_compression.selector_model import _compute_focal_redundancy_diagnostics

    diagnostics = _compute_focal_redundancy_diagnostics(
        torch.tensor([1.0, 0.9, 0.8]),
        torch.tensor([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]),
        torch.tensor([0]),
        torch.tensor([[0, 0], [0, 1], [0, 3]]),
        torch.zeros(3, dtype=torch.long),
        mode="local",
        radius=1,
        penalty_weight=1.0,
        similarity_threshold=0.7,
    )
    assert diagnostics["global_argmax_grid_distances"] == [1, 3]
    assert diagnostics["global_argmax_outside_local_ratio"] == 0.5
    assert diagnostics["empty_local_neighbor_ratio"] == 0.5
    assert diagnostics["nonzero_penalty_ratio"] == 0.5
    assert all(torch.isfinite(torch.tensor(diagnostics[key])).all() for key in (
        "max_similarity", "penalty_base",
    ))



def test_local_mode_is_rejected_in_training_before_any_selection():
    from token_compression.selector_model import (
        Qwen2_5_VisionTransformerPretrainedModel_Selector,
    )

    model = object.__new__(Qwen2_5_VisionTransformerPretrainedModel_Selector)
    torch.nn.Module.__init__(model)
    model.focal_redundancy_mode = "local"
    model.train()
    with pytest.raises(RuntimeError, match="inference-only"):
        model.forward(torch.empty((0, 1)), torch.tensor([[1, 2, 2]]))
