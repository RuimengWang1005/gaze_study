"""Exact Qwen2.5-VL pre/post-merger token-order mappings."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple

import torch
import torch.nn.functional as F


def build_window_index(
    grid_thw: torch.Tensor,
    spatial_merge_size: int,
    patch_size: int,
    window_size: int,
) -> Tuple[torch.Tensor, list[int]]:
    """Reproduce ``Qwen2_5_VisionTransformerPretrainedModel.get_window_index``."""

    window_indices = []
    cu_window_seqlens = [0]
    window_index_id = 0
    vit_merger_window_size = window_size // spatial_merge_size // patch_size
    if vit_merger_window_size <= 0:
        raise ValueError("invalid ViT merger window size")

    for grid_t, grid_h, grid_w in grid_thw.to(dtype=torch.long):
        llm_grid_h = grid_h // spatial_merge_size
        llm_grid_w = grid_w // spatial_merge_size
        index = torch.arange(grid_t * llm_grid_h * llm_grid_w).reshape(grid_t, llm_grid_h, llm_grid_w)
        pad_h = vit_merger_window_size - llm_grid_h % vit_merger_window_size
        pad_w = vit_merger_window_size - llm_grid_w % vit_merger_window_size
        num_windows_h = (llm_grid_h + pad_h) // vit_merger_window_size
        num_windows_w = (llm_grid_w + pad_w) // vit_merger_window_size
        padded = F.pad(index, (0, int(pad_w), 0, int(pad_h)), "constant", -100)
        padded = padded.reshape(
            grid_t,
            num_windows_h,
            vit_merger_window_size,
            num_windows_w,
            vit_merger_window_size,
        )
        padded = padded.permute(0, 1, 3, 2, 4).reshape(
            grid_t, num_windows_h * num_windows_w, vit_merger_window_size, vit_merger_window_size
        )
        seqlens = (padded != -100).sum((2, 3)).reshape(-1)
        flattened = padded.reshape(-1)
        index_new = flattened[flattened != -100]
        window_indices.append(index_new + window_index_id)
        cumulative = seqlens.cumsum(0) * (spatial_merge_size**2) + cu_window_seqlens[-1]
        cu_window_seqlens.extend(cumulative.tolist())
        window_index_id += int(grid_t * llm_grid_h * llm_grid_w)
    return torch.cat(window_indices), cu_window_seqlens


def _validate_grid(grid_thw: Sequence[int], merge_size: int) -> Tuple[int, int, int]:
    if len(grid_thw) != 3:
        raise ValueError(f"grid_thw must have three values, got {grid_thw}")
    t, h, w = (int(v) for v in grid_thw)
    if min(t, h, w) <= 0 or h % merge_size or w % merge_size:
        raise ValueError(f"grid {grid_thw} is incompatible with merge size {merge_size}")
    return t, h, w


@dataclass(frozen=True)
class TokenMapping:
    """Mapping for one image between attention, patch-grid, and merger order."""

    grid_thw: Tuple[int, int, int]
    spatial_merge_size: int
    window_index: torch.Tensor

    def __post_init__(self) -> None:
        t, h, w = _validate_grid(self.grid_thw, self.spatial_merge_size)
        expected_post = t * (h // self.spatial_merge_size) * (w // self.spatial_merge_size)
        if self.window_index.ndim != 1 or self.window_index.numel() != expected_post:
            raise ValueError(
                f"window_index length {self.window_index.numel()} != post-merge token count {expected_post}"
            )
        values = self.window_index.detach().cpu().to(torch.long)
        if not torch.equal(values.sort().values, torch.arange(expected_post)):
            raise ValueError("window_index is not a permutation of post-merge indices")

    @property
    def spatial_merge_unit(self) -> int:
        return self.spatial_merge_size**2

    @property
    def num_pre_tokens(self) -> int:
        t, h, w = self.grid_thw
        return t * h * w

    @property
    def num_post_tokens(self) -> int:
        return self.num_pre_tokens // self.spatial_merge_unit

    @property
    def reverse_indices(self) -> torch.Tensor:
        return torch.argsort(self.window_index)

    def _check_pre(self, scores: torch.Tensor) -> None:
        if scores.shape[-1] != self.num_pre_tokens:
            raise ValueError(f"last dimension {scores.shape[-1]} != N_pre {self.num_pre_tokens}")

    def attention_to_grouped_original(self, scores: torch.Tensor) -> torch.Tensor:
        """Undo window reorder while preserving each merger unit's inner order."""

        self._check_pre(scores)
        leading = scores.shape[:-1]
        grouped_attention = scores.reshape(*leading, self.num_post_tokens, self.spatial_merge_unit)
        reverse = self.reverse_indices.to(scores.device)
        return grouped_attention.index_select(-2, reverse)

    def attention_to_grid(self, scores: torch.Tensor) -> torch.Tensor:
        """Map attention-order pre-merge values to ``[..., T, H, W]``."""

        grouped = self.attention_to_grouped_original(scores)
        t, h, w = self.grid_thw
        m = self.spatial_merge_size
        leading_count = scores.ndim - 1
        grouped = grouped.reshape(*scores.shape[:-1], t, h // m, w // m, m, m)
        dims = list(range(leading_count)) + [leading_count, leading_count + 1, leading_count + 3,
                                             leading_count + 2, leading_count + 4]
        return grouped.permute(*dims).reshape(*scores.shape[:-1], t, h, w)

    def grid_to_attention(self, grid_values: torch.Tensor) -> torch.Tensor:
        """Map ``[..., T,H,W]`` values into the exact target-attention order."""

        t, h, w = self.grid_thw
        if tuple(grid_values.shape[-3:]) != (t, h, w):
            raise ValueError(f"grid shape {tuple(grid_values.shape[-3:])} != {(t, h, w)}")
        m = self.spatial_merge_size
        leading = grid_values.shape[:-3]
        grouped = grid_values.reshape(*leading, t, h // m, m, w // m, m)
        leading_count = len(leading)
        dims = list(range(leading_count)) + [leading_count, leading_count + 1, leading_count + 3,
                                             leading_count + 2, leading_count + 4]
        grouped = grouped.permute(*dims).reshape(*leading, self.num_post_tokens, self.spatial_merge_unit)
        index = self.window_index.to(grid_values.device)
        return grouped.index_select(-2, index).reshape(*leading, self.num_pre_tokens)

    def premerge_to_postmerge(self, scores: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
        """Aggregate merger units and return post-merge values in original spatial order."""

        grouped = self.attention_to_grouped_original(scores)
        if reduction == "mean":
            return grouped.mean(dim=-1)
        if reduction == "sum":
            return grouped.sum(dim=-1)
        raise ValueError(f"unsupported reduction: {reduction}")

    def postmerge_to_grid(self, scores: torch.Tensor) -> torch.Tensor:
        t, h, w = self.grid_thw
        expected = self.num_post_tokens
        if scores.shape[-1] != expected:
            raise ValueError(f"last dimension {scores.shape[-1]} != N_post {expected}")
        return scores.reshape(*scores.shape[:-1], t, h // self.spatial_merge_size, w // self.spatial_merge_size)


@dataclass(frozen=True)
class BBoxMasks:
    premerge_attention: torch.Tensor
    premerge_grid: torch.Tensor
    postmerge_original: torch.Tensor
    valid: bool
    reason: str


def bbox_masks_from_pixels(
    bboxs: Iterable[Sequence[float]],
    source_width: float,
    source_height: float,
    mapping: TokenMapping,
) -> BBoxMasks:
    """Rasterize original-image pixel bboxes onto Qwen's resized patch grid.

    Qwen's image processor performs aspect-preserving ``smart_resize`` to a
    multiple of ``patch_size * merge_size`` and performs no crop or padding.
    Normalizing by the original width/height therefore exactly composes with
    that resize.
    """

    t, h, w = mapping.grid_thw
    grid = torch.zeros((t, h, w), dtype=torch.bool)
    if source_width <= 0 or source_height <= 0:
        return BBoxMasks(torch.empty(0, dtype=torch.bool), grid, torch.empty(0, dtype=torch.bool), False,
                         "non-positive source dimensions")

    valid_box_count = 0
    for bbox in bboxs:
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            continue
        x1, y1, x2, y2 = (float(value) for value in bbox)
        x1, x2 = sorted((x1, x2))
        y1, y2 = sorted((y1, y2))
        x1, x2 = max(0.0, x1), min(float(source_width), x2)
        y1, y2 = max(0.0, y1), min(float(source_height), y2)
        if x2 <= x1 or y2 <= y1:
            continue
        col_start = max(0, min(w - 1, math.floor(x1 / source_width * w)))
        col_end = max(col_start + 1, min(w, math.ceil(x2 / source_width * w)))
        row_start = max(0, min(h - 1, math.floor(y1 / source_height * h)))
        row_end = max(row_start + 1, min(h, math.ceil(y2 / source_height * h)))
        grid[:, row_start:row_end, col_start:col_end] = True
        valid_box_count += 1

    pre = mapping.grid_to_attention(grid)
    grouped = mapping.attention_to_grouped_original(pre)
    post = grouped.any(dim=-1)
    if valid_box_count == 0 or not bool(pre.any()):
        return BBoxMasks(pre, grid, post, False, "bbox maps to no foreground token")
    if bool(pre.all()):
        return BBoxMasks(pre, grid, post, False, "bbox leaves no background token")
    return BBoxMasks(pre, grid, post, True, "ok")
