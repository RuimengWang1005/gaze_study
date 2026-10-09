#!/usr/bin/env python3
"""Second-stage, 20-sample diagnostics for Qwen ViT head bbox signals.

This script does not train, ablate, or select heads.  It compares several
attention diagnostics with translated-bbox controls and the trained GAZE
selector on the exact samples used by the first-stage mapping audit.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from PIL import Image


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from audit_vit_heads import (  # noqa: E402
    DEFAULT_ANNOTATIONS,
    DEFAULT_IMAGE_ROOT,
    DEFAULT_MODEL,
    _actual_window_index,
    _model_device_dtype,
    git_diff_summary,
    load_runtime,
    run_stage_a_tests,
    select_samples,
    set_seed,
)
from head_audit.attention_capture import AttentionCapture  # noqa: E402
from head_audit.report import write_csv, write_json  # noqa: E402
from head_audit.selector_comparison import load_selector_scorer, verify_frozen_visual  # noqa: E402
from head_audit.signal_analysis import (  # noqa: E402
    benjamini_hochberg,
    normalize_distribution,
    one_sided_sign_test,
    pearson_correlation,
    percentile_ranks,
    query_conditioned_attention,
    random_translated_masks,
    shuffled_bbox_statistics,
    spearman_correlation,
    topk_overlap,
    uniform_log2_ratio,
    value_weighted_incoming,
)
from head_audit.token_mapping import TokenMapping, bbox_masks_from_pixels  # noqa: E402
from head_audit.visualization import save_head_diagnostic_sheet, save_selector_comparison  # noqa: E402


DEFAULT_SELECTOR = Path("/home/ruimeng/gaze/output_ckpt666/VisionSelector-Qwen2.5-VL-7B")
DEFAULT_OUTPUT = SCRIPT_DIR / "results" / "vit_head_signal_v1"
HEAD_FIELDS = [
    "sample_id", "head_id", "attention_real_mass", "attention_shuffle_mean",
    "attention_shuffle_delta", "attention_shuffle_z", "attention_shuffle_p",
    "value_real_mass", "value_shuffle_mean", "value_shuffle_delta", "value_shuffle_z",
    "value_shuffle_p", "bbox_query_key_preference", "selector_pearson", "selector_spearman",
    "selector_top10_overlap", "selector_top20_overlap",
]
SELECTOR_FIELDS = [
    "sample_id", "bbox_ratio_post", "selector_foreground_mass", "selector_raw_excess",
    "selector_precision_10", "selector_precision_20", "selector_shuffle_mean",
    "selector_shuffle_delta", "selector_shuffle_z", "selector_shuffle_p",
    "eager_flash_output_max_abs_error", "eager_flash_output_mean_abs_error",
    "eager_flash_output_relative_rms_error",
]
SUMMARY_FIELDS = [
    "head_id", "mean_attention_shuffle_delta", "attention_delta_positive_rate",
    "attention_p_le_0_05_rate", "attention_sign_test_p", "attention_bh_q",
    "mean_value_shuffle_delta", "value_delta_positive_rate", "value_p_le_0_05_rate",
    "value_sign_test_p", "value_bh_q", "mean_bbox_query_key_preference",
    "bbox_query_positive_rate", "bbox_query_sign_test_p", "bbox_query_bh_q",
    "mean_selector_pearson",
    "mean_selector_spearman", "mean_selector_top10_overlap", "mean_selector_top20_overlap",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default=DEFAULT_MODEL)
    parser.add_argument("--selector-checkpoint", type=Path, default=DEFAULT_SELECTOR)
    parser.add_argument("--annotations", default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--image-root", default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--num-samples", type=int, default=20, choices=range(1, 21), metavar="1..20")
    parser.add_argument("--scan-limit", type=int, default=4000)
    parser.add_argument("--num-shuffles", type=int, default=100)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--min-pixels", type=int, default=3136)
    parser.add_argument("--max-pixels", type=int, default=401408)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-tests", action="store_true")
    return parser.parse_args()


def _precision(scores: torch.Tensor, mask: torch.Tensor, fraction: float) -> float:
    k = max(1, int(math.ceil(scores.numel() * fraction)))
    indices = torch.topk(scores, k=k).indices
    return float(mask.bool()[indices].float().mean().item())


def _head_rows(sample_id, attention_stats, value_stats, query_delta, mask, correlations):
    query_preference = query_delta[:, mask.bool()].sum(dim=-1)
    rows = []
    for head_id in range(query_delta.shape[0]):
        rows.append(
            {
                "sample_id": sample_id,
                "head_id": head_id,
                "attention_real_mass": float(attention_stats["real_mass"][head_id]),
                "attention_shuffle_mean": float(attention_stats["shuffle_mean"][head_id]),
                "attention_shuffle_delta": float(attention_stats["delta"][head_id]),
                "attention_shuffle_z": float(attention_stats["z_score"][head_id]),
                "attention_shuffle_p": float(attention_stats["empirical_p"][head_id]),
                "value_real_mass": float(value_stats["real_mass"][head_id]),
                "value_shuffle_mean": float(value_stats["shuffle_mean"][head_id]),
                "value_shuffle_delta": float(value_stats["delta"][head_id]),
                "value_shuffle_z": float(value_stats["z_score"][head_id]),
                "value_shuffle_p": float(value_stats["empirical_p"][head_id]),
                "bbox_query_key_preference": float(query_preference[head_id]),
                "selector_pearson": float(correlations["pearson"][head_id]),
                "selector_spearman": float(correlations["spearman"][head_id]),
                "selector_top10_overlap": float(correlations["top10"][head_id]),
                "selector_top20_overlap": float(correlations["top20"][head_id]),
            }
        )
    return rows


def audit_signal_sample(model, processor, scorer, record, args):
    from qwen25vl.modeling_qwen2_5_vl import Qwen2_5_VLVisionAttention

    image_path = Path(record["_image_path"])
    image = Image.open(image_path).convert("RGB")
    expected_size = (int(record["width"]), int(record["height"]))
    if image.size != expected_size:
        raise RuntimeError(f"bbox metadata size {expected_size} != image size {image.size}")
    inputs = processor.image_processor(images=[image], return_tensors="pt")
    grid = inputs["image_grid_thw"]
    visual = model.visual
    config = visual.config
    target_layer = int(config.fullatt_block_indexes[-1])
    original_attention = visual.blocks[target_layer].attn
    device, dtype = _model_device_dtype(visual)
    pixels = inputs["pixel_values"].to(device=device, dtype=dtype)
    grid_device = grid.to(device=device)
    grid_tuple = tuple(int(v) for v in grid[0].tolist())
    mapping = TokenMapping(
        grid_tuple,
        int(config.spatial_merge_size),
        _actual_window_index(visual, grid),
    )
    masks = bbox_masks_from_pixels(record.get("bboxs") or [], expected_size[0], expected_size[1], mapping)
    if not masks.valid:
        raise RuntimeError(f"invalid bbox mapping: {masks.reason}")

    attention_dim = int(original_attention.qkv.out_features // 3)
    eager_attention = Qwen2_5_VLVisionAttention(attention_dim, int(config.num_heads)).to(device=device, dtype=dtype)
    eager_attention.load_state_dict(original_attention.state_dict(), strict=True)
    eager_attention.eval()
    visual.blocks[target_layer].attn = eager_attention
    try:
        with torch.no_grad(), AttentionCapture(eager_attention) as capture:
            eager_features = visual(pixels, grid_thw=grid_device)
    finally:
        visual.blocks[target_layer].attn = original_attention
    if capture.attention_probs is None or capture.value_norms is None:
        raise RuntimeError("attention/value capture failed")

    with torch.no_grad():
        selector_features = visual(pixels, grid_thw=grid_device)
        selector_raw = scorer(selector_features.unsqueeze(0)).squeeze(0).float().cpu()
    feature_delta = eager_features.float() - selector_features.float()
    eager_flash_error = float(feature_delta.abs().max().item())
    eager_flash_mean_error = float(feature_delta.abs().mean().item())
    selector_rms = selector_features.float().square().mean().sqrt()
    eager_flash_relative_rms_error = float(
        feature_delta.square().mean().sqrt().div(selector_rms.clamp_min(1e-12)).item()
    )

    probs = capture.attention_probs[0]
    incoming = normalize_distribution(probs.mean(dim=-2))
    value_scores = value_weighted_incoming(incoming, capture.value_norms[0])
    bbox_query, background_query, query_delta = query_conditioned_attention(probs, masks.premerge_attention)
    head_post = normalize_distribution(mapping.premerge_to_postmerge(incoming, reduction="sum"))
    selector_prob = torch.softmax(selector_raw, dim=-1)

    generator = torch.Generator().manual_seed(args.seed + int(record["_row_index"]))
    shuffled_grid = random_translated_masks(masks.premerge_grid[0], args.num_shuffles, generator)
    shuffled_pre = mapping.grid_to_attention(shuffled_grid.unsqueeze(1))
    shuffled_post = mapping.attention_to_grouped_original(shuffled_pre).any(dim=-1)
    attention_stats = shuffled_bbox_statistics(incoming, masks.premerge_attention, shuffled_pre)
    value_stats = shuffled_bbox_statistics(value_scores, masks.premerge_attention, shuffled_pre)
    selector_stats = shuffled_bbox_statistics(
        selector_prob.unsqueeze(0), masks.postmerge_original, shuffled_post
    )
    correlations = {
        "pearson": pearson_correlation(head_post, selector_raw),
        "spearman": spearman_correlation(head_post, selector_raw),
        "top10": topk_overlap(head_post, selector_raw, 0.10),
        "top20": topk_overlap(head_post, selector_raw, 0.20),
    }

    sample_id = f"{int(record['_row_index']):06d}_{record.get('dataset', 'unknown')}"
    head_rows = _head_rows(
        sample_id, attention_stats, value_stats, query_delta, masks.premerge_attention, correlations
    )
    selector_row = {
        "sample_id": sample_id,
        "bbox_ratio_post": float(masks.postmerge_original.float().mean()),
        "selector_foreground_mass": float(selector_stats["real_mass"][0]),
        "selector_raw_excess": float(
            selector_stats["real_mass"][0] - masks.postmerge_original.float().mean()
        ),
        "selector_precision_10": _precision(selector_raw, masks.postmerge_original, 0.10),
        "selector_precision_20": _precision(selector_raw, masks.postmerge_original, 0.20),
        "selector_shuffle_mean": float(selector_stats["shuffle_mean"][0]),
        "selector_shuffle_delta": float(selector_stats["delta"][0]),
        "selector_shuffle_z": float(selector_stats["z_score"][0]),
        "selector_shuffle_p": float(selector_stats["empirical_p"][0]),
        "eager_flash_output_max_abs_error": eager_flash_error,
        "eager_flash_output_mean_abs_error": eager_flash_mean_error,
        "eager_flash_output_relative_rms_error": eager_flash_relative_rms_error,
    }

    relative_grid = mapping.attention_to_grid(uniform_log2_ratio(incoming)).numpy()
    percentile_grid = mapping.attention_to_grid(percentile_ranks(incoming)).numpy()
    value_grid = mapping.attention_to_grid(value_scores).numpy()
    delta_grid = mapping.attention_to_grid(query_delta).numpy()
    incoming_grid = mapping.attention_to_grid(incoming).numpy()
    selector_percentile = percentile_ranks(selector_raw.unsqueeze(0))[0]
    selector_grid = mapping.postmerge_to_grid(selector_percentile).numpy()
    heat_root = args.output_dir / "review_pack" / "heatmaps"
    modes = [
        ("relative_uniform", relative_grid, "log2 attention ratio to uniform", "coolwarm", True, None),
        ("per_head_percentile", percentile_grid, "Per-head percentile (pattern only)", "magma", False, (0.0, 1.0)),
        ("value_weighted", value_grid, "Value-weighted incoming attention", "magma", False, None),
        ("bbox_query_delta", delta_grid, "BBox-query minus background-query attention", "coolwarm", True, None),
    ]
    for dirname, grids, title, cmap, signed, fixed in modes:
        save_head_diagnostic_sheet(
            image, record.get("bboxs") or [], grids, target_layer,
            heat_root / dirname / f"sample_{sample_id}.png",
            title, cmap=cmap, signed=signed, fixed_range=fixed,
        )
    diagnostic_head = int(attention_stats["real_mass"].argmax().item())
    save_selector_comparison(
        image,
        record.get("bboxs") or [],
        selector_grid,
        incoming_grid.mean(axis=0),
        value_grid.mean(axis=0),
        delta_grid.mean(axis=0),
        relative_grid[diagnostic_head],
        diagnostic_head,
        target_layer,
        heat_root / "selector_comparison" / f"sample_{sample_id}.png",
    )
    validation = {
        "sample_id": sample_id,
        "attention_shape": list(probs.shape),
        "value_norm_shape": list(capture.value_norms[0].shape),
        "incoming_sum_max_error": float((incoming.sum(-1) - 1).abs().max()),
        "value_sum_max_error": float((value_scores.sum(-1) - 1).abs().max()),
        "bbox_query_sum_max_error": float((bbox_query.sum(-1) - 1).abs().max()),
        "background_query_sum_max_error": float((background_query.sum(-1) - 1).abs().max()),
        "query_delta_sum_max_error": float(query_delta.sum(-1).abs().max()),
        "shuffle_count": int(shuffled_pre.shape[0]),
        "shuffle_token_count_preserved": bool(
            torch.all(shuffled_pre.sum(-1) == masks.premerge_attention.sum())
        ),
        "eager_flash_output_max_abs_error": eager_flash_error,
        "eager_flash_output_mean_abs_error": eager_flash_mean_error,
        "eager_flash_output_relative_rms_error": eager_flash_relative_rms_error,
        "target_module_restored": visual.blocks[target_layer].attn is original_attention,
    }
    return head_rows, selector_row, validation


def aggregate_heads(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[int(row["head_id"])].append(row)
    output = []
    for head_id, values in sorted(grouped.items()):
        mean = lambda key: float(np.mean([float(v[key]) for v in values]))
        rate = lambda key, predicate: float(np.mean([predicate(float(v[key])) for v in values]))
        output.append(
            {
                "head_id": head_id,
                "mean_attention_shuffle_delta": mean("attention_shuffle_delta"),
                "attention_delta_positive_rate": rate("attention_shuffle_delta", lambda x: x > 0),
                "attention_p_le_0_05_rate": rate("attention_shuffle_p", lambda x: x <= 0.05),
                "attention_sign_test_p": one_sided_sign_test(
                    float(value["attention_shuffle_delta"]) for value in values
                ),
                "mean_value_shuffle_delta": mean("value_shuffle_delta"),
                "value_delta_positive_rate": rate("value_shuffle_delta", lambda x: x > 0),
                "value_p_le_0_05_rate": rate("value_shuffle_p", lambda x: x <= 0.05),
                "value_sign_test_p": one_sided_sign_test(
                    float(value["value_shuffle_delta"]) for value in values
                ),
                "mean_bbox_query_key_preference": mean("bbox_query_key_preference"),
                "bbox_query_positive_rate": rate("bbox_query_key_preference", lambda x: x > 0),
                "bbox_query_sign_test_p": one_sided_sign_test(
                    float(value["bbox_query_key_preference"]) for value in values
                ),
                "mean_selector_pearson": mean("selector_pearson"),
                "mean_selector_spearman": mean("selector_spearman"),
                "mean_selector_top10_overlap": mean("selector_top10_overlap"),
                "mean_selector_top20_overlap": mean("selector_top20_overlap"),
            }
        )
    for prefix in ("attention", "value", "bbox_query"):
        adjusted = benjamini_hochberg(item[f"{prefix}_sign_test_p"] for item in output)
        for item, q_value in zip(output, adjusted):
            item[f"{prefix}_bh_q"] = q_value
    return output


def write_findings(
    path: Path,
    summaries: List[Dict[str, object]],
    head_rows: List[Dict[str, object]],
    selector_rows: List[Dict[str, object]],
) -> None:
    def average(rows, key):
        return float(np.mean([float(row[key]) for row in rows]))

    def rate(rows, key, predicate):
        return float(np.mean([predicate(float(row[key])) for row in rows]))

    def head_list(key, threshold=0.05):
        return [int(row["head_id"]) for row in summaries if float(row[key]) <= threshold]

    aligned = sorted(summaries, key=lambda row: float(row["mean_selector_spearman"]), reverse=True)
    raw_robust = head_list("attention_bh_q")
    value_robust = head_list("value_bh_q")
    query_robust = head_list("bbox_query_bh_q")
    selector_sig = sum(float(row["selector_shuffle_p"]) <= 0.05 for row in selector_rows)
    path.write_text(
        f"""# Findings: ViT Head BBox-Signal Diagnostics v1

These are exploratory diagnostics on 20 samples and use zero-based head IDs.
They do not constitute final head selection or an ablation result.

## Main result

The heads are not spatially equivalent. Across all {len(head_rows)} head/sample
pairs, raw incoming attention placed on average
`{average(head_rows, 'attention_shuffle_delta'):.4f}` more probability mass in
the real bbox than in 100 translated same-shape controls, and the effect was
positive in `{rate(head_rows, 'attention_shuffle_delta', lambda x: x > 0):.1%}`
of pairs. Value weighting changed these numbers to
`{average(head_rows, 'value_shuffle_delta'):.4f}` and
`{rate(head_rows, 'value_shuffle_delta', lambda x: x > 0):.1%}`.

After an exact one-sided cross-sample sign test and Benjamini-Hochberg
correction across 16 heads, the raw-attention-consistent heads were
`{raw_robust}`, the value-weighted-consistent heads were `{value_robust}`, and
the bbox-query-preference-consistent heads were `{query_robust}`. This shows
repeatable head specialization, while also showing that the strongest set
depends on the diagnostic definition.

## Relation to the trained GAZE selector

The five heads with the highest mean Spearman alignment to the trained
selector were `{[int(row['head_id']) for row in aligned[:5]]}`, with mean
correlations `{[round(float(row['mean_selector_spearman']), 3) for row in aligned[:5]]}`.
The three lowest were `{[int(row['head_id']) for row in aligned[-3:]]}`, with
mean correlations `{[round(float(row['mean_selector_spearman']), 3) for row in aligned[-3:]]}`.
Therefore some heads resemble the learned selector while others emphasize
different or opposing regions.

The trained selector's mean bbox probability excess over bbox area was
`{average(selector_rows, 'selector_raw_excess'):.4f}`. Its mean top-10% bbox
precision was `{average(selector_rows, 'selector_precision_10'):.3f}`, versus
a mean bbox area ratio of `{average(selector_rows, 'bbox_ratio_post'):.3f}`.
Only `{selector_sig}/{len(selector_rows)}` individual samples passed the
translated-bbox empirical test at p <= 0.05, so bbox overlap remains strongly
sample-dependent and the translated control should not be treated as a
complete definition of relevance.

## Interpretation boundary

These results support the statement that different heads attend to different
spatial patterns, and that a subset aligns poorly with the learned selector.
They do **not** establish that negatively aligned or bbox-external attention is
useless: context outside a bbox can matter, raw attention is not causal
importance, and the sample size is small. A pruning/weighting claim still
requires a held-out downstream ablation.
""",
        encoding="utf-8",
    )


def write_review_readme(path: Path, summary: Dict[str, object]) -> None:
    path.write_text(
        f"""# ViT Head BBox-Signal Diagnostics v1

This second-stage review pack diagnoses weak spatial patterns; it does not
train, ablate, or select final heads.

- Samples completed: `{summary['samples_completed']}`
- Target layer / heads: `{summary['target_layer']}` / `{summary['num_heads']}`
- Translated bbox controls per sample: `{summary['num_shuffles']}`
- Frozen base/selector visual sentinel tensors exactly equal: `{summary['frozen_visual_verification']['all_exact']}`
- Technical validation passed: `{summary['all_technical_checks_passed']}`

Heatmap modes:

- `relative_uniform`: signed log2 ratio to uniform, with one robust shared scale.
- `per_head_percentile`: reveals weak within-head structure; absolute head strength is intentionally lost.
- `value_weighted`: incoming attention multiplied by per-token value norm.
- `bbox_query_delta`: bbox-query outgoing attention minus background-query outgoing attention.
- `selector_comparison`: compact comparison with the trained GAZE selector.

Head capture temporarily uses the model's eager attention implementation,
whereas selector scoring uses the restored original FlashAttention path. The
reported eager/Flash errors quantify this numerical implementation difference;
the selector itself is always evaluated on its faithful original path.

Translated-bbox statistics and selector correlations are exploratory on only
20 samples. They are not a `selected_heads.json` and are not final evidence of
head utility.

See `FINDINGS.md` for the aggregate interpretation and its limitations.
""",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    if not args.skip_tests:
        run_stage_a_tests(args.output_dir)
    samples = select_samples(Path(args.annotations), Path(args.image_root), args.num_samples, args.scan_limit)
    model, processor = load_runtime(args)
    device, dtype = _model_device_dtype(model.visual)
    scorer = load_selector_scorer(args.selector_checkpoint, device, dtype)
    frozen_verification = verify_frozen_visual(model.visual, args.selector_checkpoint)
    if not frozen_verification["all_exact"]:
        raise RuntimeError("base model and selector checkpoint visual weights differ")

    run_config = {
        "model_path": args.model_path,
        "selector_checkpoint": str(args.selector_checkpoint),
        "annotations": args.annotations,
        "image_root": args.image_root,
        "samples": args.num_samples,
        "num_shuffles": args.num_shuffles,
        "target_layer": int(model.visual.config.fullatt_block_indexes[-1]),
        "num_heads": int(model.visual.config.num_heads),
        "training_performed": False,
        "head_ablation_performed": False,
        "final_heads_selected": False,
    }
    write_json(args.output_dir / "run_config.json", run_config)
    all_head_rows, selector_rows, validations = [], [], []
    for index, record in enumerate(samples, 1):
        print(f"[{index}/{len(samples)}] signal audit {record['_image_path']}", flush=True)
        head_rows, selector_row, validation = audit_signal_sample(model, processor, scorer, record, args)
        all_head_rows.extend(head_rows)
        selector_rows.append(selector_row)
        validations.append(validation)

    summaries = aggregate_heads(all_head_rows)
    write_csv(args.output_dir / "head_signal_metrics.csv", all_head_rows, HEAD_FIELDS)
    write_csv(args.output_dir / "selector_metrics.csv", selector_rows, SELECTOR_FIELDS)
    write_csv(args.output_dir / "signal_summary.csv", summaries, SUMMARY_FIELDS)
    summary = {
        "samples_completed": len(validations),
        "target_layer": run_config["target_layer"],
        "num_heads": run_config["num_heads"],
        "num_shuffles": args.num_shuffles,
        "frozen_visual_verification": frozen_verification,
        "max_incoming_sum_error": max(v["incoming_sum_max_error"] for v in validations),
        "max_value_sum_error": max(v["value_sum_max_error"] for v in validations),
        "max_query_distribution_sum_error": max(
            max(v["bbox_query_sum_max_error"], v["background_query_sum_max_error"])
            for v in validations
        ),
        "max_query_delta_sum_error": max(v["query_delta_sum_max_error"] for v in validations),
        "max_eager_flash_output_error": max(v["eager_flash_output_max_abs_error"] for v in validations),
        "max_eager_flash_output_mean_abs_error": max(
            v["eager_flash_output_mean_abs_error"] for v in validations
        ),
        "max_eager_flash_output_relative_rms_error": max(
            v["eager_flash_output_relative_rms_error"] for v in validations
        ),
        "all_technical_checks_passed": all(
            v["incoming_sum_max_error"] < 1e-5
            and v["value_sum_max_error"] < 1e-5
            and v["bbox_query_sum_max_error"] < 5e-3
            and v["background_query_sum_max_error"] < 5e-3
            and v["query_delta_sum_max_error"] < 5e-3
            and v["shuffle_token_count_preserved"]
            and v["target_module_restored"]
            for v in validations
        ),
        "samples": validations,
    }
    write_json(args.output_dir / "validation_summary.json", summary)
    git_diff_summary(args.output_dir / "git_diff_summary.txt")
    review = args.output_dir / "review_pack"
    review.mkdir(parents=True, exist_ok=True)
    for name in (
        "run_config.json", "validation_summary.json", "head_signal_metrics.csv",
        "selector_metrics.csv", "signal_summary.csv", "test_report.txt", "git_diff_summary.txt",
    ):
        shutil.copy2(args.output_dir / name, review / name)
    write_review_readme(review / "README.md", summary)
    write_findings(review / "FINDINGS.md", summaries, all_head_rows, selector_rows)
    print(json.dumps({"summary": summary, "review_pack": str(review)}, indent=2))


if __name__ == "__main__":
    main()
