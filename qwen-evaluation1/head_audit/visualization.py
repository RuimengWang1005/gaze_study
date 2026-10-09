"""Contact-sheet visualization for per-head spatial attention."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from PIL import Image


def _draw_bbox(ax, bboxs: Iterable[Sequence[float]]) -> None:
    for bbox in bboxs:
        if len(bbox) != 4:
            continue
        x1, y1, x2, y2 = (float(v) for v in bbox)
        ax.add_patch(
            Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                fill=False,
                edgecolor="#00ff66",
                linewidth=2.2,
            )
        )


def _overlay(ax, image: Image.Image, heat: np.ndarray, bboxs, vmax: float, title: str) -> None:
    ax.imshow(image)
    ax.imshow(
        heat,
        cmap="magma",
        interpolation="bilinear",
        alpha=0.58,
        extent=(0, image.width, image.height, 0),
        vmin=0.0,
        vmax=vmax,
    )
    _draw_bbox(ax, bboxs)
    ax.set_title(title, fontsize=9, pad=5)
    ax.set_axis_off()


def save_contact_sheet(
    image: Image.Image,
    bboxs: Iterable[Sequence[float]],
    premerge_grids: np.ndarray,
    postmerge_mean_grid: np.ndarray,
    layer_id: int,
    output_path: Path,
) -> None:
    """Save original+bbox, every head, all-head mean, and post-merge mean."""

    if premerge_grids.ndim == 4:
        premerge_grids = premerge_grids.mean(axis=1)
    if postmerge_mean_grid.ndim == 3:
        postmerge_mean_grid = postmerge_mean_grid.mean(axis=0)
    num_heads, grid_h, grid_w = premerge_grids.shape
    all_head_mean = premerge_grids.mean(axis=0)
    common_vmax = float(max(premerge_grids.max(), all_head_mean.max(), 1e-12))

    panel_count = num_heads + 3
    cols = 5
    rows = math.ceil(panel_count / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.8 * rows), constrained_layout=True)
    axes = np.asarray(axes).reshape(-1)

    axes[0].imshow(image)
    _draw_bbox(axes[0], bboxs)
    axes[0].set_title(f"Original + bbox\npre-grid={grid_h}x{grid_w}", fontsize=9, pad=5)
    axes[0].set_axis_off()
    for head_id in range(num_heads):
        _overlay(
            axes[head_id + 1], image, premerge_grids[head_id], bboxs, common_vmax,
            f"Layer {layer_id} / Head {head_id}",
        )
    _overlay(axes[num_heads + 1], image, all_head_mean, bboxs, common_vmax, "All-head mean (pre-merge)")
    _overlay(
        axes[num_heads + 2], image, postmerge_mean_grid, bboxs, common_vmax,
        "All-head mean (post-merge)",
    )
    for ax in axes[panel_count:]:
        ax.set_visible(False)
    fig.suptitle(
        f"Qwen2.5-VL attention head audit — layer {layer_id} — shared scale [0, {common_vmax:.3g}]",
        fontsize=14,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
