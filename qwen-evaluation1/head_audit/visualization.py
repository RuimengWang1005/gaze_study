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


def _overlay(
    ax,
    image: Image.Image,
    heat: np.ndarray,
    bboxs,
    vmax: float,
    title: str,
    vmin: float = 0.0,
    cmap: str = "magma",
) -> None:
    ax.imshow(image)
    ax.imshow(
        heat,
        cmap=cmap,
        interpolation="bilinear",
        alpha=0.58,
        extent=(0, image.width, image.height, 0),
        vmin=vmin,
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


def save_head_diagnostic_sheet(
    image: Image.Image,
    bboxs: Iterable[Sequence[float]],
    grids: np.ndarray,
    layer_id: int,
    output_path: Path,
    mode_title: str,
    cmap: str = "magma",
    signed: bool = False,
    fixed_range: tuple[float, float] | None = None,
) -> None:
    """Save all heads using either a shared robust range or an explicit range."""

    if grids.ndim == 4:
        grids = grids.mean(axis=1)
    num_heads, grid_h, grid_w = grids.shape
    mean_grid = grids.mean(axis=0)
    if fixed_range is not None:
        vmin, vmax = fixed_range
    elif signed:
        limit = float(max(np.quantile(np.abs(grids), 0.995), 1e-12))
        vmin, vmax = -limit, limit
    else:
        vmin, vmax = 0.0, float(max(np.quantile(grids, 0.995), 1e-12))

    panel_count = num_heads + 2
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
            axes[head_id + 1], image, grids[head_id], bboxs, vmax,
            f"Layer {layer_id} / Head {head_id}", vmin=vmin, cmap=cmap,
        )
    _overlay(
        axes[num_heads + 1], image, mean_grid, bboxs, vmax,
        "All-head mean", vmin=vmin, cmap=cmap,
    )
    for ax in axes[panel_count:]:
        ax.set_visible(False)
    fig.suptitle(f"{mode_title} — shared display range [{vmin:.3g}, {vmax:.3g}]", fontsize=14)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=110)
    plt.close(fig)


def save_selector_comparison(
    image: Image.Image,
    bboxs: Iterable[Sequence[float]],
    selector_grid: np.ndarray,
    incoming_mean_grid: np.ndarray,
    value_mean_grid: np.ndarray,
    query_delta_mean_grid: np.ndarray,
    best_relative_grid: np.ndarray,
    best_head_id: int,
    layer_id: int,
    output_path: Path,
) -> None:
    """Save a compact selector/head comparison without implying head selection."""

    arrays = [incoming_mean_grid, value_mean_grid, query_delta_mean_grid, best_relative_grid]
    arrays = [a.mean(axis=0) if a.ndim == 3 else a for a in arrays]
    selector = selector_grid.mean(axis=0) if selector_grid.ndim == 3 else selector_grid
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    axes = axes.reshape(-1)
    axes[0].imshow(image)
    _draw_bbox(axes[0], bboxs)
    axes[0].set_title("Original + bbox")
    axes[0].set_axis_off()

    selector_vmax = float(max(np.quantile(selector, 0.995), 1e-12))
    incoming_vmax = float(max(np.quantile(arrays[0], 0.995), 1e-12))
    value_vmax = float(max(np.quantile(arrays[1], 0.995), 1e-12))
    delta_limit = float(max(np.quantile(np.abs(arrays[2]), 0.995), 1e-12))
    relative_limit = float(max(np.quantile(np.abs(arrays[3]), 0.995), 1e-12))
    _overlay(axes[1], image, selector, bboxs, selector_vmax, "GAZE selector percentile", cmap="magma")
    _overlay(axes[2], image, arrays[0], bboxs, incoming_vmax, "All-head incoming mean")
    _overlay(axes[3], image, arrays[1], bboxs, value_vmax, "Value-weighted all-head mean")
    _overlay(
        axes[4], image, arrays[2], bboxs, delta_limit,
        "BBox-query minus background-query", vmin=-delta_limit, cmap="coolwarm",
    )
    _overlay(
        axes[5], image, arrays[3], bboxs, relative_limit,
        f"Diagnostic max bbox-mass head {best_head_id}\nlog2 ratio to uniform",
        vmin=-relative_limit, cmap="coolwarm",
    )
    fig.suptitle(f"ViT head signals vs trained GAZE selector — layer {layer_id}", fontsize=14)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
