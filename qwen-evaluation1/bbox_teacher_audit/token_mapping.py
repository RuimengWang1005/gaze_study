"""BBox-to-merged-token mapping compatible with GAZE Visual CoT training."""

from __future__ import annotations

from typing import Iterable, Optional, Sequence

import torch


MAPPING_PROVENANCE = (
    "/home/ruimeng/gaze/qwen-vl-finetune/qwenvl/data/data_qwen.py:"
    "pixel_bboxs_to_ref_token_mask"
)


def pixel_bboxs_to_merged_mask(
    bboxs: Optional[Iterable[Sequence[float]]],
    width: Optional[float],
    height: Optional[float],
    grid_thw: Optional[torch.Tensor],
    spatial_merge_size: int = 2,
) -> Optional[torch.Tensor]:
    """Reproduce GAZE's pixel-bbox mask on row-major Patch Merger tokens.

    The base Qwen vision tower restores row-major order after windowed ViT
    processing, so flattening this union mask matches the merged feature order.
    Endpoint inclusion intentionally matches the training implementation.
    """

    bboxs = list(bboxs) if bboxs is not None else []
    if not bboxs or width is None or height is None:
        return None
    if grid_thw is None or not isinstance(grid_thw, torch.Tensor) or grid_thw.numel() != 3:
        return None
    try:
        width_f, height_f = float(width), float(height)
    except (TypeError, ValueError):
        return None
    if width_f <= 0 or height_f <= 0:
        return None

    t, h, w = grid_thw.to(dtype=torch.long).tolist()
    h_tokens = h // spatial_merge_size
    w_tokens = w // spatial_merge_size
    if t != 1 or h_tokens <= 0 or w_tokens <= 0:
        return None
    mask = torch.zeros((h_tokens, w_tokens), dtype=torch.bool)
    valid = 0
    for bbox in bboxs:
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            continue
        try:
            x1, y1, x2, y2 = (float(value) for value in bbox)
        except (TypeError, ValueError):
            continue
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1
        x1 = max(0.0, min(1.0, x1 / width_f))
        x2 = max(0.0, min(1.0, x2 / width_f))
        y1 = max(0.0, min(1.0, y1 / height_f))
        y2 = max(0.0, min(1.0, y2 / height_f))
        col_start = max(0, min(w_tokens - 1, int(x1 * w_tokens)))
        row_start = max(0, min(h_tokens - 1, int(y1 * h_tokens)))
        col_end = min(int(x2 * w_tokens), w_tokens - 1)
        row_end = min(int(y2 * h_tokens), h_tokens - 1)
        col_end = max(col_start, col_end)
        row_end = max(row_start, row_end)
        mask[row_start : row_end + 1, col_start : col_end + 1] = True
        valid += 1
    return mask if valid and bool(mask.any()) else None


def foreground_indices(mask: torch.Tensor) -> torch.Tensor:
    """Return sorted row-major merged-token indices for a union mask."""

    if mask.ndim != 2 or mask.dtype != torch.bool:
        raise ValueError("foreground mask must be a 2D BoolTensor")
    return mask.flatten().nonzero(as_tuple=False).flatten()


def validate_mapping(mask: torch.Tensor, grid_thw: torch.Tensor, merge_size: int) -> None:
    t, h, w = (int(value) for value in grid_thw.tolist())
    expected = t * (h // merge_size) * (w // merge_size)
    if mask.numel() != expected:
        raise ValueError(f"bbox mask has {mask.numel()} tokens; expected {expected}")
    indices = foreground_indices(mask)
    if not len(indices) or not bool(torch.all(indices[1:] > indices[:-1])):
        raise ValueError("bbox foreground indices are empty or not strictly sorted")
