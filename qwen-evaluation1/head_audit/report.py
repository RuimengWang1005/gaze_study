"""Small review-pack writers for the ViT head audit."""

from __future__ import annotations

import csv
import inspect
import json
import shutil
from pathlib import Path
from typing import Dict, Iterable, List


METRIC_FIELDS = [
    "sample_id", "layer_id", "head_id", "num_tokens_premerge", "bbox_ratio",
    "foreground_mass", "raw_excess", "normalized_excess", "entropy",
    "kl_to_uniform", "precision_10", "precision_20", "valid",
]


def write_csv(path: Path, rows: Iterable[Dict[str, object]], fields: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def source_location(obj) -> str:
    try:
        filename = inspect.getsourcefile(obj) or "unknown"
        _, line = inspect.getsourcelines(obj)
        return f"{filename}:{line}"
    except (OSError, TypeError):
        return "unknown"


def write_review_readme(path: Path, summary: Dict[str, object]) -> None:
    text = f"""# Qwen2.5-VL ViT Head Audit v1

This review pack validates attention capture and spatial token mapping only.

- No model was trained or modified.
- No FGVC dataset was used.
- No final good heads were selected.
- The GAZE selector was not modified.
- Heatmaps are diagnostic and are not final evidence that a head is effective.

## Run summary

- Target full-attention layer: `{summary.get('target_layer')}`
- Number of heads: `{summary.get('num_heads')}`
- Samples requested/completed: `{summary.get('samples_requested')}` / `{summary.get('samples_completed')}`
- All required validation checks passed: `{summary.get('all_required_checks_passed')}`
- Maximum capture output error: `{summary.get('max_capture_output_error')}`
- Maximum attention row-sum error: `{summary.get('max_attention_row_sum_error')}`
- Maximum incoming-score sum error: `{summary.get('max_incoming_sum_error')}`

The heatmaps share one color scale within each image. Review bbox alignment,
orientation, head diversity, and agreement between pre/post-merge mean maps.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def assemble_review_pack(output_dir: Path, summary: Dict[str, object]) -> Path:
    review = output_dir / "review_pack"
    review.mkdir(parents=True, exist_ok=True)
    for name in (
        "model_introspection.txt", "run_config.json", "validation_summary.json",
        "head_metrics.csv", "sample_manifest.csv", "test_report.txt", "git_diff_summary.txt",
    ):
        source = output_dir / name
        if source.exists():
            shutil.copy2(source, review / name)
    source_heatmaps = output_dir / "heatmaps"
    target_heatmaps = review / "heatmaps"
    target_heatmaps.mkdir(exist_ok=True)
    for source in source_heatmaps.glob("*.png"):
        shutil.copy2(source, target_heatmaps / source.name)
    write_review_readme(review / "README.md", summary)
    return review
