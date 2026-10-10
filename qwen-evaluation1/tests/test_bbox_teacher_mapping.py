from __future__ import annotations

import sys
from pathlib import Path

import torch


EVAL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EVAL_ROOT))

from bbox_teacher_audit.token_mapping import (  # noqa: E402
    foreground_indices,
    pixel_bboxs_to_merged_mask,
    validate_mapping,
)


def test_pixel_bbox_mapping_matches_gaze_endpoint_rule():
    mask = pixel_bboxs_to_merged_mask(
        [[0, 0, 49, 49]], width=100, height=100, grid_thw=torch.tensor([1, 8, 8]), spatial_merge_size=2
    )
    expected = torch.zeros(4, 4, dtype=torch.bool)
    expected[0:2, 0:2] = True
    assert torch.equal(mask, expected)
    validate_mapping(mask, torch.tensor([1, 8, 8]), 2)


def test_multiple_bboxes_form_a_union():
    mask = pixel_bboxs_to_merged_mask(
        [[0, 0, 1, 1], [99, 99, 100, 100]],
        width=100,
        height=100,
        grid_thw=torch.tensor([1, 8, 8]),
    )
    assert int(mask.sum()) == 2
    assert mask[0, 0] and mask[-1, -1]


def test_foreground_indices_preserve_row_major_order():
    mask = torch.tensor([[False, True, False], [True, False, True]])
    assert foreground_indices(mask).tolist() == [1, 3, 5]


def test_mapping_is_exactly_compatible_with_training_implementation():
    training_root = Path("/home/ruimeng/gaze/qwen-vl-finetune")
    sys.path.insert(0, str(training_root))
    from qwenvl.data.data_qwen import pixel_bboxs_to_ref_token_mask  # noqa: E402

    bboxs = [[7, 11, 58, 67], [73, 3, 99, 21]]
    grid = torch.tensor([1, 10, 14])
    reference = pixel_bboxs_to_ref_token_mask(bboxs, 100, 80, grid, spatial_merge_size=2)
    audited = pixel_bboxs_to_merged_mask(bboxs, 100, 80, grid, spatial_merge_size=2)
    assert torch.equal(audited, reference)
