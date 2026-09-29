"""Paired summaries and aligned visualizations for distance-aware ablations."""
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image


def load_records(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def mean(records, getter):
    values = [getter(record) for record in records]
    values = [value for value in values if value is not None]
    return float(np.mean(values)) if values else None


def summarize_run(records, config):
    return {
        "num_samples": len(records),
        "accuracy": mean(records, lambda r: float(r["classification_correct"])),
        "bbox_inside_selected_ratio": mean(records, lambda r: 1 - r["bbox_outside_ratio"]["union"]),
        "bbox_outside_background_ratio": mean(records, lambda r: r["bbox_outside_ratio"]["union"]),
        "visible_part_recall_radius1": mean(records, lambda r: r["coverage"]["1"]["union_PartRecall"]),
        "visible_part_recall_radius2": mean(records, lambda r: r["coverage"]["2"]["union_PartRecall"]),
        "max_3x3_concentration": mean(records, lambda r: r["metrics"]["union"]["spatial"]["window_3x3_share"]),
        "largest_connected_component_share": mean(records, lambda r: r["metrics"]["union"]["spatial"]["largest_component_share"]),
        "mean_nearest_neighbor_distance": mean(records, lambda r: r["metrics"]["union"]["distance"]["mean_nearest_neighbor_distance"]),
        "local_high_similarity_pair_ratio": mean(records, lambda r: r["metrics"]["union"]["distance"]["local_high_similarity_pair_ratio"]),
        "mean_inference_seconds": mean(records, lambda r: r["elapsed_seconds"]),
        "total_inference_and_metrics_seconds": config["inference_and_metrics_seconds"],
        "peak_cuda_allocated_gib": config["peak_cuda_allocated_gib"],
        "fallback_count": int(sum(r["fallback_count"] for r in records)),
        "duplicate_or_oob_samples": int(sum(r["duplicate_indices"] or r["out_of_bounds_indices"] for r in records)),
    }


def add_paired_overlap(summary, records, original, topk):
    def overlap(a, b):
        left, right = set(a), set(b)
        return len(left & right) / len(left) if left else None
    summary["selected_overlap_with_original"] = mean(
        list(zip(records, original)), lambda pair: overlap(pair[0]["selected_indices"], pair[1]["selected_indices"])
    )
    summary["selected_overlap_with_topk"] = mean(
        list(zip(records, topk)), lambda pair: overlap(pair[0]["selected_indices"], pair[1]["selected_indices"])
    )


def write_table(path, rows):
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _draw_base(ax, image, record):
    from matplotlib.patches import Rectangle
    width, height = image.size
    ax.imshow(image)
    bx, by, bw, bh = record["bbox"]
    ax.add_patch(Rectangle((bx, by), bw, bh, fill=False, edgecolor="white", linewidth=2,
                           label="CUB ground-truth bbox"))
    for part in record["parts"]:
        ax.scatter(part["x"], part["y"], c="yellow", s=20, edgecolors="black",
                   linewidths=.4, label="Visible CUB part")
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.axis("off")


def _patch(ax, index, record, color, label, alpha=.55):
    from matplotlib.patches import Rectangle
    width, height = record["original_width"], record["original_height"]
    row, col = divmod(index, record["Wm"])
    ax.add_patch(Rectangle((col * width / record["Wm"], row * height / record["Hm"]),
                           width / record["Wm"], height / record["Hm"],
                           facecolor=color, edgecolor=color, alpha=alpha, label=label))


def save_method_overlay(record, image_path, output, method):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    fig, ax = plt.subplots(figsize=(7, 7))
    _draw_base(ax, image, record)
    for index in record["selected_indices"]:
        _patch(ax, index, record, "deepskyblue", "Selected token")
    ax.set_title(f"{method} | image {record['image_id']} | K={record['K']}")
    ax.legend(handles=[Patch(facecolor="deepskyblue", alpha=.55, label="Selected token"),
                       Line2D([], [], marker="o", color="yellow", linestyle="None", label="Visible CUB part"),
                       Patch(facecolor="none", edgecolor="white", label="CUB ground-truth bbox")],
              loc="lower left", facecolor="black", labelcolor="white")
    fig.tight_layout()
    fig.savefig(output, dpi=140)
    plt.close(fig)


def save_difference_overlay(original, distance, image_path, output, metadata_path, method):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    fig, ax = plt.subplots(figsize=(7, 7))
    _draw_base(ax, image, original)
    old, new = set(original["selected_indices"]), set(distance["selected_indices"])
    groups = ((old & new, "limegreen", "Original ∩ Distance-aware"),
              (new - old, "dodgerblue", "Distance-aware only"),
              (old - new, "red", "Original only"))
    for indices, color, label in groups:
        for index in indices:
            _patch(ax, index, original, color, label)
    ax.set_title(f"Difference: Original GAZE vs {method} | image {original['image_id']}")
    ax.legend(handles=[Patch(facecolor=color, alpha=.55, label=label) for _, color, label in groups] +
              [Line2D([], [], marker="o", color="yellow", linestyle="None", label="Visible CUB part"),
               Patch(facecolor="none", edgecolor="white", label="CUB ground-truth bbox"),
               Patch(facecolor="black", alpha=.45, label="Neither selected (image remains visible)")],
              loc="lower left", facecolor="black", labelcolor="white", fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=140)
    plt.close(fig)
    metadata = {
        "image_id": original["image_id"], "method": method,
        "colors": {"green": "selected by Original GAZE and Distance-aware",
                   "blue": "selected only by Distance-aware", "red": "selected only by Original GAZE",
                   "yellow_points": "visible=1 entries from CUB parts/part_locs.txt",
                   "white_box": "CUB bounding_boxes.txt ground-truth bbox",
                   "dark_unmarked": "selected by neither method"},
        "common_indices": sorted(old & new), "distance_only_indices": sorted(new - old),
        "original_only_indices": sorted(old - new), "bbox": original["bbox"],
        "largest_selected_cluster_box": None,
    }
    Path(metadata_path).write_text(json.dumps(metadata, indent=2) + "\n")
