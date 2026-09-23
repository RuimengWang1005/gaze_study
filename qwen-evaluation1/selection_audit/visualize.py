"""Standalone overlays; no feature vectors needed."""
from pathlib import Path
import numpy as np
from PIL import Image


def save_overlay(record, scores, image_path, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    with Image.open(image_path) as source:
        image = source.convert("RGB")
    w, h = image.size
    gh, gw = record["Hm"], record["Wm"]
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    titles = ["Original + visible CUB parts", "Raw score (same scorer)",
              "Raw Top-K_focal", "GAZE: focal orange / spatial blue"]
    for ax, title in zip(axes, titles):
        ax.imshow(image)
        ax.set_title(title)
        ax.set_xlim(0, w)
        ax.set_ylim(h, 0)
        ax.axis("off")
    heat = axes[1].imshow(np.asarray(scores).reshape(gh, gw), extent=(0, w, h, 0),
                          alpha=.6, cmap="inferno", interpolation="nearest")
    fig.colorbar(heat, ax=axes[1], fraction=.046)
    for ax, ids, color in ((axes[2], record["raw_topk_indices"], "gold"),
                           (axes[3], record["focal_indices"], "darkorange"),
                           (axes[3], record["spatial_indices"], "deepskyblue")):
        for i in ids:
            row, col = divmod(i, gw)
            ax.add_patch(Rectangle((col * w / gw, row * h / gh), w / gw, h / gh,
                                   fill=True, facecolor=color, edgecolor=color, alpha=.45))
    for j, part in enumerate(record["parts"]):
        for panel, ax in enumerate(axes):
            branch = "raw_topk" if panel == 2 else "union"
            color = "lime" if part["coverage"][branch]["1"] else "magenta"
            ax.scatter(part["x"], part["y"], s=30, c=color, edgecolors="black", linewidths=.5)
            if panel in (0, 3):
                ax.annotate(str(part["part_id"]), (part["x"], part["y"]), color=color,
                            xytext=(3, 3), textcoords="offset points", fontsize=8)
    clusters = record["metrics"]["focal"]["duplicates"]["1"]["clusters"]
    if clusters:
        rows, cols = np.divmod(clusters[0], gw)
        axes[3].add_patch(Rectangle((cols.min() * w / gw, rows.min() * h / gh),
            (cols.max() - cols.min() + 1) * w / gw, (rows.max() - rows.min() + 1) * h / gh,
            fill=False, edgecolor="red", linewidth=2.5, linestyle="--"))
    legend = "; ".join(f"{p['part_id']} {p['name']}" for p in record["parts"])
    fig.suptitle(f"ID {record['image_id']} | class {record['class_id']} | grid {gh}x{gw} | "
                 f"K={record['K']} | actionable={record['ActionableRedundancyCase']}\n"
                 "Parts: green=covered, magenta=missed (1-cell); dashed red=largest local duplicate cluster", fontsize=11)
    fig.text(.5, .02, legend, ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .06, 1, .87))
    fig.savefig(Path(output), dpi=140)
    plt.close(fig)
