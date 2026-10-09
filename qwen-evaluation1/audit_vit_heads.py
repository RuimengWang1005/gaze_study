#!/usr/bin/env python3
"""Audit Qwen2.5-VL per-head vision attention on bbox-labelled Visual-CoT.

This is intentionally independent from the GAZE selector.  It performs no
training, no token selection, and no head selection.
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
import transformers
from PIL import Image


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from head_audit.attention_capture import AttentionCapture  # noqa: E402
from head_audit.metrics import compute_head_metrics  # noqa: E402
from head_audit.report import (  # noqa: E402
    METRIC_FIELDS,
    assemble_review_pack,
    source_location,
    write_csv,
    write_json,
)
from head_audit.token_mapping import (  # noqa: E402
    TokenMapping,
    bbox_masks_from_pixels,
    build_window_index,
)
from head_audit.visualization import save_contact_sheet  # noqa: E402


DEFAULT_MODEL = "/data/ModelZoo/LVLM/Qwen-Qwen2.5-VL-7B-Instruct"
DEFAULT_ANNOTATIONS = "/home/ruimeng/gaze/viscot_dataset/viscot_mix_20k_30-10-20-10-10_seedww42.jsonl"
DEFAULT_IMAGE_ROOT = "/data/DataSet/vlm_dataset/cot_image_data"
DEFAULT_OUTPUT = SCRIPT_DIR / "results" / "vit_head_audit_v1"
MANIFEST_FIELDS = [
    "sample_id", "row_index", "dataset", "image_path", "width", "height",
    "bbox_count", "bbox_area_ratio", "bbox_size", "bbox_position", "orientation",
    "grid_t", "grid_h", "grid_w", "num_tokens_premerge", "num_tokens_postmerge", "valid",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default=DEFAULT_MODEL)
    parser.add_argument("--annotations", default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--image-root", default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--num-samples", type=int, default=1, choices=range(1, 21), metavar="1..20")
    parser.add_argument("--scan-limit", type=int, default=4000)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--min-pixels", type=int, default=3136)
    parser.add_argument("--max-pixels", type=int, default=401408)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-tests", action="store_true")
    return parser.parse_args()


def _run(command: Sequence[str], cwd: Path = REPO_ROOT) -> subprocess.CompletedProcess:
    return subprocess.run(command, cwd=cwd, text=True, capture_output=True, check=False)


def run_stage_a_tests(output_dir: Path) -> None:
    tests = [
        str(SCRIPT_DIR / "tests" / "test_head_attention_capture.py"),
        str(SCRIPT_DIR / "tests" / "test_head_token_mapping.py"),
        str(SCRIPT_DIR / "tests" / "test_head_signal_analysis.py"),
    ]
    completed = _run([sys.executable, "-m", "pytest", "-q", *tests])
    report = completed.stdout + ("\n" + completed.stderr if completed.stderr else "")
    (output_dir / "test_report.txt").write_text(report, encoding="utf-8")
    if completed.returncode:
        raise RuntimeError(f"Stage A tests failed; see {output_dir / 'test_report.txt'}")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _image_path(image_root: Path, record: Dict[str, object]) -> Path:
    images = record.get("images")
    if not isinstance(images, list) or len(images) != 1:
        raise ValueError("audit accepts exactly one image per sample")
    relative = str(images[0])
    if relative.startswith("cot/"):
        relative = relative[4:]
    return image_root / relative


def _bbox_features(record: Dict[str, object]) -> Dict[str, object]:
    width, height = float(record["width"]), float(record["height"])
    boxes = record.get("bboxs") or []
    area = 0.0
    centers = []
    for box in boxes:
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            continue
        x1, y1, x2, y2 = (float(v) for v in box)
        x1, x2 = sorted((max(0.0, x1), min(width, x2)))
        y1, y2 = sorted((max(0.0, y1), min(height, y2)))
        if x2 > x1 and y2 > y1:
            area += (x2 - x1) * (y2 - y1)
            centers.append(((x1 + x2) / (2 * width), (y1 + y2) / (2 * height)))
    ratio = min(area / max(width * height, 1.0), 1.0)
    size = "small" if ratio < 0.05 else "medium" if ratio < 0.25 else "large"
    if centers:
        cx = sum(v[0] for v in centers) / len(centers)
        cy = sum(v[1] for v in centers) / len(centers)
        position = "center" if 0.25 <= cx <= 0.75 and 0.25 <= cy <= 0.75 else "edge"
    else:
        position = "invalid"
    orientation = "landscape" if width > height else "portrait" if height > width else "square"
    return {
        "bbox_area_ratio": ratio,
        "bbox_size": size,
        "bbox_position": position,
        "orientation": orientation,
        "bbox_count": len(boxes),
    }


def select_samples(annotation_path: Path, image_root: Path, count: int, scan_limit: int) -> List[Dict[str, object]]:
    candidates: List[Dict[str, object]] = []
    with annotation_path.open(encoding="utf-8") as handle:
        for row_index, line in enumerate(handle):
            if row_index >= scan_limit:
                break
            record = json.loads(line)
            try:
                path = _image_path(image_root, record)
                features = _bbox_features(record)
                if not path.is_file() or features["bbox_position"] == "invalid":
                    continue
                with Image.open(path) as image:
                    image.verify()
            except (OSError, ValueError, KeyError, TypeError):
                continue
            record = dict(record)
            record["_row_index"] = row_index
            record["_image_path"] = str(path)
            record["_features"] = features
            candidates.append(record)
    if len(candidates) < count:
        raise RuntimeError(f"only found {len(candidates)} usable samples, requested {count}")

    selected: List[Dict[str, object]] = []
    covered = set()
    remaining = candidates
    while len(selected) < count:
        def rank(record):
            f = record["_features"]
            tags = {
                "size:" + str(f["bbox_size"]),
                "position:" + str(f["bbox_position"]),
                "orientation:" + str(f["orientation"]),
                "objects:" + ("multi" if int(f["bbox_count"]) > 1 else "single"),
                "dataset:" + str(record.get("dataset", "unknown")),
            }
            novelty = len(tags - covered)
            smoke_bonus = 0.5 if f["bbox_size"] == "medium" else 0.0
            return novelty + smoke_bonus, -int(record["_row_index"])

        best = max(remaining, key=rank)
        selected.append(best)
        f = best["_features"]
        covered.update(
            {
                "size:" + str(f["bbox_size"]),
                "position:" + str(f["bbox_position"]),
                "orientation:" + str(f["orientation"]),
                "objects:" + ("multi" if int(f["bbox_count"]) > 1 else "single"),
                "dataset:" + str(best.get("dataset", "unknown")),
            }
        )
        remaining = [record for record in remaining if record is not best]
    return selected


def configure_processor(processor, min_pixels: int, max_pixels: int) -> None:
    image_processor = processor.image_processor
    image_processor.min_pixels = int(min_pixels)
    image_processor.max_pixels = int(max_pixels)
    if isinstance(getattr(image_processor, "size", None), dict):
        image_processor.size["shortest_edge"] = int(min_pixels)
        image_processor.size["longest_edge"] = int(max_pixels)


def load_runtime(args: argparse.Namespace):
    from qwen25vl.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
    from qwen25vl.processing_qwen2_5_vl import Qwen2_5_VLProcessor

    dtype = getattr(torch, args.dtype)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        attn_implementation="flash_attention_2",
        device_map=args.device,
        low_cpu_mem_usage=True,
    ).eval()
    processor = Qwen2_5_VLProcessor.from_pretrained(args.model_path)
    configure_processor(processor, args.min_pixels, args.max_pixels)
    return model, processor


def _source_line(obj) -> str:
    return source_location(obj)


def write_model_introspection(path: Path, model) -> None:
    from qwen25vl import modeling_qwen2_5_vl as module

    visual = model.visual
    config = visual.config
    lines = [
        "Qwen2.5-VL ViT head audit model introspection",
        f"torch={torch.__version__}",
        f"transformers={transformers.__version__} ({transformers.__file__})",
        f"model_class={type(model).__module__}.{type(model).__name__}",
        f"model_class_source={_source_line(type(model))}",
        "",
        f"model.visual.config.depth={config.depth}",
        f"model.visual.config.num_heads={config.num_heads}",
        f"model.visual.config.fullatt_block_indexes={config.fullatt_block_indexes}",
        f"model.visual.config.spatial_merge_size={config.spatial_merge_size}",
        f"model.visual.config.patch_size={config.patch_size}",
        f"model.visual.config.temporal_patch_size={config.temporal_patch_size}",
        f"model.visual.config._attn_implementation={config._attn_implementation}",
        f"target_layer={config.fullatt_block_indexes[-1]}",
        "audit_target_attention_implementation=eager monkeypatch (temporary; restored on exit)",
        "",
        "Implementation locations:",
        f"Qwen2_5_VLVisionBlock={_source_line(module.Qwen2_5_VLVisionBlock)}",
        f"Qwen2_5_VLVisionAttention={_source_line(module.Qwen2_5_VLVisionAttention)}",
        f"Qwen2_5_VLVisionFlashAttention2={_source_line(module.Qwen2_5_VLVisionFlashAttention2)}",
        f"QKV/eager forward={_source_line(module.Qwen2_5_VLVisionAttention.forward)}",
        f"visual RoPE={_source_line(module.apply_rotary_pos_emb_vision)}",
        f"vision full/window forward={_source_line(module.Qwen2_5_VisionTransformerPretrainedModel.forward)}",
        f"get_window_index={_source_line(module.Qwen2_5_VisionTransformerPretrainedModel.get_window_index)}",
        f"Patch Merger forward={_source_line(module.Qwen2_5_VLPatchMerger.forward)}",
        "",
        "Visual attention modules:",
    ]
    for name, child in visual.named_modules():
        if "attn" in name.lower():
            lines.append(f"{name}: {type(child).__module__}.{type(child).__name__}")
    target = visual.blocks[config.fullatt_block_indexes[-1]].attn
    lines.extend(
        [
            "",
            f"target_qkv={target.qkv!r}",
            f"target_proj={target.proj!r}",
            f"patch_merger={visual.merger!r}",
            "",
            "Token-order trace:",
            "1. image processor smart-resizes with no crop/padding",
            "2. processor flattens patches in spatial_merge_size x spatial_merge_size groups",
            "3. vision forward reshapes into merger groups and applies window_index",
            "4. RoPE receives the identical window_index reorder",
            "5. full-attention blocks use image/frame cu_seqlens; window blocks use cu_window_seqlens",
            "6. Patch Merger consumes contiguous merger groups",
            "7. reverse_indices=argsort(window_index) restores post-merge spatial order",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _actual_window_index(visual, grid_thw: torch.Tensor) -> torch.Tensor:
    actual, _ = visual.get_window_index(grid_thw.detach().cpu())
    reproduced, _ = build_window_index(
        grid_thw.detach().cpu(),
        spatial_merge_size=int(visual.config.spatial_merge_size),
        patch_size=int(visual.config.patch_size),
        window_size=int(visual.config.window_size),
    )
    if not torch.equal(actual.cpu(), reproduced.cpu()):
        raise RuntimeError("independent window_index reproduction differs from model.get_window_index")
    return actual.cpu()


def _model_device_dtype(visual) -> Tuple[torch.device, torch.dtype]:
    parameter = next(visual.parameters())
    return parameter.device, parameter.dtype


def audit_sample(model, processor, record: Dict[str, object], args: argparse.Namespace):
    from qwen25vl.modeling_qwen2_5_vl import Qwen2_5_VLVisionAttention

    image_path = Path(record["_image_path"])
    image = Image.open(image_path).convert("RGB")
    metadata_size = (int(record["width"]), int(record["height"]))
    if image.size != metadata_size:
        raise RuntimeError(f"bbox metadata size {metadata_size} != actual image size {image.size}")
    # This audit runs the vision tower only.  Calling the exact image processor
    # avoids fabricating text merely to satisfy the multimodal wrapper API.
    inputs = processor.image_processor(images=[image], return_tensors="pt")
    grid = inputs["image_grid_thw"]
    if grid.shape != (1, 3):
        raise RuntimeError(f"single-image audit expected grid [1,3], got {tuple(grid.shape)}")

    visual = model.visual
    config = visual.config
    target_layer = int(config.fullatt_block_indexes[-1])
    target_attention = visual.blocks[target_layer].attn
    device, dtype = _model_device_dtype(visual)
    pixel_values = inputs["pixel_values"].to(device=device, dtype=dtype)
    grid_device = grid.to(device=device)
    window_index = _actual_window_index(visual, grid)
    grid_tuple = tuple(int(v) for v in grid[0].tolist())
    mapping = TokenMapping(grid_tuple, int(config.spatial_merge_size), window_index)
    if grid_tuple[0] != 1:
        raise RuntimeError(f"still-image audit expected grid_t=1, got {grid_tuple[0]}")

    if pixel_values.shape[0] != mapping.num_pre_tokens:
        raise RuntimeError(f"pixel patches {pixel_values.shape[0]} != N_pre {mapping.num_pre_tokens}")
    if mapping.num_pre_tokens != mapping.num_post_tokens * mapping.spatial_merge_unit:
        raise RuntimeError("Patch Merger length invariant failed")

    # Temporarily use Qwen's own eager attention at the target block.  First run
    # its untouched forward, then install capture on that exact module.  Finally
    # restore the originally loaded FlashAttention module.
    attention_dim = int(target_attention.qkv.out_features // 3)
    eager_attention = Qwen2_5_VLVisionAttention(attention_dim, int(config.num_heads)).to(device=device, dtype=dtype)
    eager_attention.load_state_dict(target_attention.state_dict(), strict=True)
    eager_attention.eval()
    target_block = visual.blocks[target_layer]
    target_block.attn = eager_attention
    original_eager_forward_func = eager_attention.forward.__func__
    try:
        with torch.no_grad():
            reference_output = visual(pixel_values, grid_thw=grid_device)
        with torch.no_grad(), AttentionCapture(eager_attention, capture=True) as capture:
            captured_output = visual(pixel_values, grid_thw=grid_device)
        if eager_attention.forward.__func__ is not original_eager_forward_func:
            raise RuntimeError("capture context did not restore Qwen's original eager forward")
    finally:
        target_block.attn = target_attention
    if target_block.attn is not target_attention:
        raise RuntimeError("audit did not restore the originally loaded target attention module")
    if reference_output.shape != captured_output.shape:
        raise RuntimeError("capture changed visual output shape")
    max_abs_error = float((reference_output.float() - captured_output.float()).abs().max().item())
    tolerance = 5e-3 if dtype in (torch.float16, torch.bfloat16) else 1e-5
    if max_abs_error > tolerance:
        raise RuntimeError(f"capture output error {max_abs_error} exceeds tolerance {tolerance}")

    probs = capture.attention_probs
    if probs is None or capture.metadata is None:
        raise RuntimeError("target attention probabilities were not captured")
    expected_shape = (1, int(config.num_heads), mapping.num_pre_tokens, mapping.num_pre_tokens)
    if tuple(probs.shape) != expected_shape:
        raise RuntimeError(f"attention shape {tuple(probs.shape)} != {expected_shape}")
    normalization_tolerance = 5e-3 if dtype in (torch.float16, torch.bfloat16) else 1e-5
    if (capture.metadata.row_sum_max_error > normalization_tolerance
            or capture.metadata.incoming_sum_max_error > normalization_tolerance):
        raise RuntimeError(f"attention normalization failed: {capture.metadata}")

    incoming = probs.mean(dim=-2)[0]
    masks = bbox_masks_from_pixels(
        record.get("bboxs") or [],
        float(record["width"]),
        float(record["height"]),
        mapping,
    )
    if masks.premerge_attention.numel() != mapping.num_pre_tokens:
        raise RuntimeError("bbox pre-merge mask length mismatch")
    if masks.postmerge_original.numel() != mapping.num_post_tokens:
        raise RuntimeError("bbox post-merge mask length mismatch")
    if not masks.valid:
        raise RuntimeError(f"invalid bbox mapping: {masks.reason}")

    pre_grids = mapping.attention_to_grid(incoming).numpy()
    post_scores = mapping.premerge_to_postmerge(incoming, reduction="mean")
    post_grids = mapping.postmerge_to_grid(post_scores).numpy()
    sample_id = f"{int(record['_row_index']):06d}_{record.get('dataset', 'unknown')}"
    metrics = compute_head_metrics(incoming, masks.premerge_attention, sample_id, target_layer, masks.valid)
    heatmap_path = args.output_dir / "heatmaps" / f"sample_{sample_id}_layer_{target_layer}_all_heads.png"
    save_contact_sheet(
        image,
        record.get("bboxs") or [],
        pre_grids,
        post_grids.mean(axis=0),
        target_layer,
        heatmap_path,
    )

    features = record["_features"]
    manifest = {
        "sample_id": sample_id,
        "row_index": int(record["_row_index"]),
        "dataset": record.get("dataset", "unknown"),
        "image_path": str(image_path),
        "width": int(record["width"]),
        "height": int(record["height"]),
        **features,
        "grid_t": grid_tuple[0],
        "grid_h": grid_tuple[1],
        "grid_w": grid_tuple[2],
        "num_tokens_premerge": mapping.num_pre_tokens,
        "num_tokens_postmerge": mapping.num_post_tokens,
        "valid": masks.valid,
    }
    validation = {
        "sample_id": sample_id,
        "target_layer": target_layer,
        "num_heads": int(config.num_heads),
        "attention_shape": list(probs.shape),
        "attention_dtype": capture.metadata.dtype,
        "premerge_grid": list(grid_tuple),
        "num_tokens_premerge": mapping.num_pre_tokens,
        "num_tokens_postmerge": mapping.num_post_tokens,
        "spatial_merge_unit": mapping.spatial_merge_unit,
        "patch_merger_length_valid": mapping.num_pre_tokens == mapping.num_post_tokens * mapping.spatial_merge_unit,
        "window_index_matches_model": True,
        "bbox_premerge_length_valid": masks.premerge_attention.numel() == mapping.num_pre_tokens,
        "bbox_postmerge_length_valid": masks.postmerge_original.numel() == mapping.num_post_tokens,
        "bbox_mapping_valid": masks.valid,
        "bbox_mapping_reason": masks.reason,
        "capture_output_shape_equal": reference_output.shape == captured_output.shape,
        "capture_output_max_abs_error": max_abs_error,
        "capture_output_tolerance": tolerance,
        "attention_row_sum_max_error": capture.metadata.row_sum_max_error,
        "incoming_score_sum_max_error": capture.metadata.incoming_sum_max_error,
        "normalization_tolerance": normalization_tolerance,
        "attention_rows_normalized": capture.metadata.row_sum_max_error <= normalization_tolerance,
        "incoming_scores_normalized": capture.metadata.incoming_sum_max_error <= normalization_tolerance,
        "head_pairwise_max_difference": float(
            max((incoming[i] - incoming[j]).abs().max().item()
                for i in range(incoming.shape[0]) for j in range(i))
        ),
        "original_eager_forward_restored": eager_attention.forward.__func__ is original_eager_forward_func,
        "loaded_attention_module_restored": target_block.attn is target_attention,
        "heatmap": str(heatmap_path),
    }
    if validation["head_pairwise_max_difference"] <= 1e-8:
        raise RuntimeError("all captured heads appear identical")
    return metrics, manifest, validation


def git_diff_summary(path: Path) -> None:
    commands = [
        ["git", "status", "--short", "--branch"],
        ["git", "diff", "--stat"],
        ["git", "diff", "--cached", "--stat"],
        ["git", "diff", "--check"],
        ["git", "diff", "--cached", "--check"],
    ]
    chunks = []
    for command in commands:
        completed = _run(command)
        chunks.append(f"$ {' '.join(command)}\n{completed.stdout}{completed.stderr}")
    path.write_text("\n".join(chunks), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    if not args.skip_tests:
        run_stage_a_tests(args.output_dir)

    annotation_path = Path(args.annotations)
    image_root = Path(args.image_root)
    if not annotation_path.is_file() or not image_root.is_dir():
        raise FileNotFoundError(f"missing Visual-CoT input: {annotation_path} or {image_root}")

    samples = select_samples(annotation_path, image_root, args.num_samples, args.scan_limit)
    model, processor = load_runtime(args)
    write_model_introspection(args.output_dir / "model_introspection.txt", model)

    config = model.visual.config
    run_config = {
        "stage": "B-smoke" if args.num_samples == 1 else "C-small-mapping-audit",
        "model_path": args.model_path,
        "annotations": str(annotation_path),
        "image_root": str(image_root),
        "output_dir": str(args.output_dir),
        "samples_requested": args.num_samples,
        "scan_limit": args.scan_limit,
        "seed": args.seed,
        "device": args.device,
        "dtype": args.dtype,
        "min_pixels": args.min_pixels,
        "max_pixels": args.max_pixels,
        "target_layer": int(config.fullatt_block_indexes[-1]),
        "num_heads": int(config.num_heads),
        "loaded_attention_implementation": config._attn_implementation,
        "target_attention_audit_implementation": "temporary eager monkeypatch",
        "training_performed": False,
        "fgvc_used": False,
        "good_heads_selected": False,
        "selector_modified": False,
    }
    write_json(args.output_dir / "run_config.json", run_config)

    all_metrics: List[Dict[str, object]] = []
    manifests: List[Dict[str, object]] = []
    validations: List[Dict[str, object]] = []
    for index, record in enumerate(samples, 1):
        print(f"[{index}/{len(samples)}] auditing {record['_image_path']}", flush=True)
        metrics, manifest, validation = audit_sample(model, processor, record, args)
        all_metrics.extend(metrics)
        manifests.append(manifest)
        validations.append(validation)

    write_csv(args.output_dir / "head_metrics.csv", all_metrics, METRIC_FIELDS)
    write_csv(args.output_dir / "sample_manifest.csv", manifests, MANIFEST_FIELDS)
    summary = {
        "target_layer": run_config["target_layer"],
        "num_heads": run_config["num_heads"],
        "samples_requested": args.num_samples,
        "samples_completed": len(validations),
        "max_capture_output_error": max(v["capture_output_max_abs_error"] for v in validations),
        "max_attention_row_sum_error": max(v["attention_row_sum_max_error"] for v in validations),
        "max_incoming_sum_error": max(v["incoming_score_sum_max_error"] for v in validations),
        "min_head_pairwise_max_difference": min(v["head_pairwise_max_difference"] for v in validations),
        "all_required_checks_passed": all(
            v["patch_merger_length_valid"]
            and v["window_index_matches_model"]
            and v["bbox_premerge_length_valid"]
            and v["bbox_postmerge_length_valid"]
            and v["bbox_mapping_valid"]
            and v["capture_output_shape_equal"]
            and v["capture_output_max_abs_error"] <= v["capture_output_tolerance"]
            and v["attention_rows_normalized"]
            and v["incoming_scores_normalized"]
            and v["original_eager_forward_restored"]
            and v["loaded_attention_module_restored"]
            and v["head_pairwise_max_difference"] > 1e-8
            for v in validations
        ),
        "samples": validations,
    }
    write_json(args.output_dir / "validation_summary.json", summary)
    git_diff_summary(args.output_dir / "git_diff_summary.txt")
    review = assemble_review_pack(args.output_dir, summary)
    print(json.dumps({"summary": summary, "review_pack": str(review)}, indent=2))


if __name__ == "__main__":
    main()
