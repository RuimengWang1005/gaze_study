from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch


EVAL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EVAL_ROOT))

from head_audit.token_mapping import (  # noqa: E402
    TokenMapping,
    bbox_masks_from_pixels,
    build_window_index,
)
from qwen25vl.modeling_qwen2_5_vl import Qwen2_5_VLPatchMerger  # noqa: E402


def _mapping() -> TokenMapping:
    grid = torch.tensor([[1, 4, 4]], dtype=torch.long)
    window_index, _ = build_window_index(grid, spatial_merge_size=2, patch_size=1, window_size=2)
    return TokenMapping((1, 4, 4), spatial_merge_size=2, window_index=window_index)


def test_synthetic_mapping_restores_row_major_patch_grid():
    mapping = _mapping()
    expected = torch.arange(16).reshape(1, 4, 4)
    attention_order = mapping.grid_to_attention(expected)
    restored = mapping.attention_to_grid(attention_order)
    assert torch.equal(restored, expected)

    # These explicit wrong transforms ensure this test detects orientation errors.
    assert not torch.equal(restored.transpose(-1, -2), expected)
    assert not torch.equal(restored.flip(-1), expected)
    assert not torch.equal(restored.flip(-2), expected)


def test_each_postmerge_token_owns_the_correct_two_by_two_group():
    mapping = _mapping()
    expected = torch.arange(16, dtype=torch.float32).reshape(1, 4, 4)
    attention_order = mapping.grid_to_attention(expected)
    post_sum = mapping.premerge_to_postmerge(attention_order, reduction="sum")
    assert torch.equal(post_sum, torch.tensor([10.0, 18.0, 42.0, 50.0]))
    assert mapping.num_pre_tokens == mapping.num_post_tokens * mapping.spatial_merge_unit


def test_wrong_reverse_index_is_detected_by_synthetic_values():
    grid = torch.tensor([[1, 4, 8]], dtype=torch.long)
    window_index, _ = build_window_index(grid, spatial_merge_size=2, patch_size=1, window_size=2)
    mapping = TokenMapping((1, 4, 8), 2, window_index)
    expected = torch.arange(32).reshape(1, 4, 8)
    attention_order = mapping.grid_to_attention(expected)
    wrong = attention_order.reshape(mapping.num_post_tokens, 4).reshape(1, 4, 8)
    assert not torch.equal(wrong, expected)
    assert torch.equal(mapping.attention_to_grid(attention_order), expected)


def test_bbox_masks_match_lengths_and_reject_degenerate_masks():
    mapping = _mapping()
    masks = bbox_masks_from_pixels([[0, 0, 50, 50]], 100, 100, mapping)
    assert masks.valid
    assert masks.premerge_attention.numel() == mapping.num_pre_tokens
    assert masks.postmerge_original.numel() == mapping.num_post_tokens
    assert 0 < masks.premerge_attention.sum() < mapping.num_pre_tokens

    full = bbox_masks_from_pixels([[0, 0, 100, 100]], 100, 100, mapping)
    assert not full.valid
    assert full.reason == "bbox leaves no background token"


def test_patch_merger_length_is_premerge_divided_by_merge_unit():
    merger = Qwen2_5_VLPatchMerger(dim=32, context_dim=16, spatial_merge_size=2).eval()
    hidden = torch.randn(16, 16)
    with torch.no_grad():
        output = merger(hidden)
    assert output.shape == (4, 32)


def test_mapping_rejects_non_permutation_window_index():
    with pytest.raises(ValueError, match="not a permutation"):
        TokenMapping((1, 4, 4), 2, torch.tensor([0, 0, 2, 3]))
