"""Observe the existing GAZE selector, with explicit reproduced-baseline settings."""
import argparse
import gc
import hashlib
import inspect
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
from datetime import datetime, timezone
from uuid import uuid4

import numpy as np
from PIL import Image

from selection_audit.cub_annotations import CUBAnnotations
from selection_audit.metrics import analyze, coordinates, validate_selection
from selection_audit.report import write_json, write_reports
from selection_audit.visualize import save_overlay

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
CACHE_NAMES = ("last_grid_thw", "last_token_features", "last_selected_indices",
               "last_focal_indices", "last_spatial_indices", "last_selected_scores")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline-result", required=True, type=Path,
                   help="Existing reproduced FGVC result JSON; all model args come from its config.")
    p.add_argument("--mode", choices=("fgvc-options", "cub-train"), default="cub-train")
    p.add_argument("--budget", type=float, required=True)
    p.add_argument("--limit", type=int)
    p.add_argument("--manifest", type=Path, help="Reuse an existing cub-train manifest exactly")
    p.add_argument("--run-id", help="Must be a new directory name under audit_outputs")
    p.add_argument("--equivalence", action="store_true", help="Run original eval first, then compare exactly 5 examples")
    p.add_argument("--duplicate-threshold", type=float, default=.7,
                   help="Diagnostic cosine threshold only; does not affect GAZE")
    p.add_argument("--output-root", type=Path, default=HERE / "audit_outputs")
    p.add_argument("--spatial-ratio-override", type=float)
    p.add_argument("--cosine-penalty-weight-override", type=float)
    p.add_argument("--distance-aware-redundancy", action="store_true")
    p.add_argument("--distance-sigma", type=float, default=.1)
    p.add_argument("--distance-floor", type=float, default=0.)
    p.add_argument("--distance-apply-to", choices=("focal",), default="focal")
    args = p.parse_args()
    if not 0 < args.budget <= 1 or not -1 <= args.duplicate_threshold <= 1:
        p.error("Invalid budget or diagnostic threshold")
    if args.limit is not None and args.limit <= 0:
        p.error("limit must be positive")
    if args.equivalence and (args.mode != "fgvc-options" or args.budget != .1 or args.limit != 5):
        p.error("equivalence requires --mode fgvc-options --budget 0.1 --limit 5")
    if args.manifest and (args.mode != "cub-train" or args.limit is not None):
        p.error("manifest requires cub-train and no limit")
    if args.distance_sigma <= 0 or not 0 <= args.distance_floor <= 1:
        p.error("distance sigma/floor out of range")
    return args


def baseline_args(cli, output):
    payload = json.loads(cli.baseline_result.read_text())
    cfg = dict(payload["config"])
    required = ("cosine_penalty_weight", "cosine_penalty_threshold", "spatial_ratio", "min_pixels",
                "max_pixels", "seed", "budgets", "model_path", "image_folder", "options_root",
                "attn_implementation", "torch_dtype", "device", "device_map", "temperature",
                "top_p", "num_beams", "max_new_tokens", "method", "choice_enumeration",
                "max_examples_per_class", "prompt_query", "contextual_ratio")
    for key in required:
        if key not in cfg:
            raise ValueError(f"Baseline config missing {key}; refusing an implicit default")
    if cfg["method"] != "selector" or cfg["dataset"] != "bird-200":
        raise ValueError("Expected a reproduced bird-200 selector baseline")
    if cfg["contextual_ratio"] != cfg["spatial_ratio"]:
        raise ValueError("Ambiguous baseline branch ratio")
    for key in ("model_path", "image_folder", "options_root"):
        value = Path(cfg[key])
        cfg[key] = str((HERE / value).resolve() if not value.is_absolute() else value.resolve())
        if not Path(cfg[key]).exists():
            raise FileNotFoundError(cfg[key])
    cfg.setdefault("distance_aware_redundancy", False)
    cfg.setdefault("distance_sigma", .1)
    cfg.setdefault("distance_floor", 0.)
    cfg.setdefault("distance_apply_to", "focal")
    cfg.update(budgets=cli.budget, limit=cli.limit, results_output_folder=str(output / "original_eval"),
               distance_aware_redundancy=cli.distance_aware_redundancy,
               distance_sigma=cli.distance_sigma, distance_floor=cli.distance_floor,
               distance_apply_to=cli.distance_apply_to)
    if cli.spatial_ratio_override is not None:
        cfg["spatial_ratio"] = cfg["contextual_ratio"] = cli.spatial_ratio_override
    if cli.cosine_penalty_weight_override is not None:
        cfg["cosine_penalty_weight"] = cli.cosine_penalty_weight_override
    return argparse.Namespace(**cfg), payload


def check_processor(processor):
    from qwen_vl_utils.vision_process import fetch_image
    ip = processor.image_processor
    if type(ip).__name__ != "Qwen2VLImageProcessor":
        raise ValueError("Unreviewed image processor; verify real geometry before running")
    for flag in ("do_center_crop", "do_crop", "do_pad", "do_thumbnail", "do_align_long_axis"):
        if getattr(ip, flag, False):
            raise ValueError(f"Unsupported geometry transform: {flag}")
    sources = {"fetch_image": inspect.getsource(fetch_image),
               "image_preprocess": inspect.getsource(type(ip)._preprocess)}
    # The supported implementation resizes the entire image twice; reject added
    # geometric operations rather than silently treating a crop as a resize.
    for source in sources.values():
        if any(word in source for word in (".crop(", "center_crop(", "letterbox(", "np.pad(")):
            raise ValueError("Detected unsupported spatial transform; mapping needs review")
    return {"mapping": "full-image resize only; no crop, spatial padding or letterbox",
            "processor_class": type(ip).__name__, "processor_config": ip.to_dict(),
            "sources": {name: hashlib.sha256(source.encode()).hexdigest() for name, source in sources.items()},
            "processor_source": inspect.getsourcefile(type(ip)),
            "fetch_image_source": inspect.getsourcefile(fetch_image)}


def original_eval(args, baseline):
    """Execute the unchanged eval_model; hook only its returned visual indices."""
    import torch
    captured = []
    original_loader = baseline.load_model
    def loader(cfg):
        model, processor, tokenizer = original_loader(cfg)
        def capture(module, inputs, result):
            captured.append({"selected_indices": result[1].detach().cpu().tolist(),
                             "N": int(result[2]), "K": int(result[1].numel())})
        model.visual.register_forward_hook(capture)
        return model, processor, tokenizer
    os.environ["SELECTOR_STORE_VIS_ATTRS"] = "0"
    baseline.load_model = loader
    try:
        baseline.set_seed(args.seed)
        with torch.inference_mode():
            result_path = baseline.eval_model(args)
    finally:
        baseline.load_model = original_loader
    result = json.loads(result_path.read_text())
    assert len(captured) == len(result["predictions"]) == 5
    write_json(result_path.parent / "visual_selection.json", captured)
    gc.collect()
    torch.cuda.empty_cache()
    return result, captured


def release_visual_cache(visual):
    for name in vars(visual):
        if name.startswith("last_") and name not in ("last_vision_encoder_time_ms", "last_selector_score_select_time_ms"):
            setattr(visual, name, None)


def snapshot(model, item, args, threshold):
    import torch
    visual = model.visual
    for name in CACHE_NAMES:
        assert getattr(visual, name, None) is not None, f"Missing {name}"
    features = visual.last_token_features
    raw = visual.importance_scorer(features.unsqueeze(0)).squeeze(0)
    assert raw.ndim == 1 and torch.isfinite(raw).all() and torch.isfinite(features).all()
    stored = visual.last_selected_scores
    selected_tensor = visual.last_selected_indices
    assert torch.isfinite(stored).all()
    torch.testing.assert_close(raw[selected_tensor], stored, rtol=1e-4, atol=1e-5)
    grid = visual.last_grid_thw.detach().cpu().reshape(-1, 3).tolist()
    assert len(grid) == 1
    t, h, w = grid[0]
    merge = int(visual.spatial_merge_size)
    assert t == 1 and h % merge == w % merge == 0
    hm, wm = h // merge, w // merge
    n = len(features)
    assert n == hm * wm == len(raw)
    selected, focal, spatial = [getattr(visual, name).detach().cpu().tolist() for name in
                                ("last_selected_indices", "last_focal_indices", "last_spatial_indices")]
    k = min(n, max(1, int(n * args.budgets)))
    validate_selection(n, selected, focal, spatial, k)
    if k < n:
        expected_spatial = min(k - 1, max(0, min(round(k * args.spatial_ratio), k)))
        assert len(spatial) == expected_spatial and len(focal) == k - expected_spatial
    with Image.open(Path(args.image_folder) / item["relative_path"]) as image:
        ow, oh = image.size
        # CUB includes orientation=0 (unspecified). PIL/fetch_image does not
        # rotate it; both 0 and 1 use the stored pixel coordinate frame.
        assert image.getexif().get(274, 1) in (0, 1), "Unreviewed EXIF orientation"
    scores = raw.detach().float().cpu().numpy()
    cpu_features = features.detach().float().cpu().numpy()
    analysis, ranks = analyze(scores, cpu_features, selected, focal, spatial,
        item["parts"], item["bbox"], ow, oh, hm, wm, threshold)
    tokens = {}
    for name, ids in (("selected", selected), ("focal", focal), ("spatial", spatial)):
        tokens[name] = {"indices": ids, "coordinates": coordinates(ids, wm).tolist(),
                        "raw_scores": scores[ids].tolist()}
    record = {key: item[key] for key in ("image_id", "relative_path", "class_id", "is_train", "bbox")}
    record.update(original_width=ow, original_height=oh, grid_thw=grid[0], Hm=hm, Wm=wm,
                  merge_size=merge, N=n, budget=args.budgets, K=k, K_focal=len(focal), K_spatial=len(spatial),
                  selected_indices=selected, focal_indices=focal, spatial_indices=spatial, tokens=tokens,
                  fallback_count=int(visual.last_selector_fallback_count),
                  distance_sigma=float(args.distance_sigma),
                  duplicate_indices=len(selected) != len(set(selected)),
                  out_of_bounds_indices=any(i < 0 or i >= n for i in selected),
                  raw_score_max_abs_error=float((raw[selected_tensor] - stored).abs().max()), **analysis)
    selection_type = np.zeros(n, dtype=np.uint8)
    selection_type[focal] = 1
    selection_type[spatial] = 2
    local_cosine = np.full(n, -2., dtype=np.float32)  # -2 = no selected local neighbor
    for idx, value in zip(selected, analysis["metrics"]["union"]["duplicates"]["1"]["max_local_cosine"]):
        if value is not None:
            local_cosine[idx] = value
    arrays = {"index": np.arange(n), "row": np.arange(n) // wm, "col": np.arange(n) % wm,
              "raw_score": scores, "raw_rank": ranks, "selection_type": selection_type,
              "max_selected_local_cosine": local_cosine}
    release_visual_cache(visual)
    del features, raw, stored, selected_tensor, cpu_features
    return record, arrays


def main():
    cli = parse_args()
    cli.baseline_result = cli.baseline_result.resolve()
    run_id = cli.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:8]
    if Path(run_id).name != run_id or run_id in (".", ".."):
        raise ValueError("run-id must be a single directory name")
    output = cli.output_root.resolve() / run_id
    args, previous = baseline_args(cli, output)
    cub = CUBAnnotations(args.image_folder)
    import torch
    import eval_fgvc_selector as baseline
    for key in ("SELECTOR_FOCAL_REDUNDANCY_LIMIT", "SELECTOR_SPATIAL_RATIO", "SELECTOR_CONTEXTUAL_RATIO",
                "SELECTOR_COSINE_PENALTY_WEIGHT", "SELECTOR_COSINE_PENALTY_THRESHOLD"):
        if key in os.environ:
            raise ValueError(f"Unset {key}: reproduced baseline did not record an override")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    status = subprocess.check_output(["git", "status", "--short"], cwd=REPO, text=True)
    baseline.set_seed(args.seed)
    if cli.mode == "fgvc-options":
        examples, options_path = baseline.load_examples(args)
        ids = [cub.by_path[e["image"]] for e in examples]
    else:
        options_path = str(Path(args.options_root) / baseline.DATASET_CONFIGS["bird-200"]["options_file"])
        examples = None
        if cli.manifest:
            manifest = json.loads(cli.manifest.read_text())
            assert manifest["mode"] == "cub-train" and manifest["seed"] == args.seed
            assert manifest["annotation_hashes"] == annotation_hashes(cub.root)
            ids = manifest["image_ids"]
        else:
            ids = cub.sample_train(args.seed, cli.limit)
        assert all(cub.images[i]["is_train"] == 1 for i in ids)
    assert ids and len(ids) == len(set(ids))
    # No existing run can be overwritten, including partially completed runs.
    output.mkdir(parents=True, exist_ok=False)
    (output / "tokens").mkdir()
    (output / "overlays").mkdir()
    manifest = {"mode": cli.mode, "seed": args.seed, "image_ids": ids,
                "images": [{k: cub.images[i][k] for k in ("image_id", "relative_path", "class_id", "is_train")} for i in ids],
                "annotation_hashes": annotation_hashes(cub.root)}
    write_json(output / "manifest.json", manifest)
    checkpoint = Path(args.model_path)
    config = {"run_id": run_id, "git_commit": commit, "git_status": status,
              "command": sys.argv, "audit_args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(cli).items()},
              "model_args": vars(args), "reproduced_baseline_config": previous["config"],
              "baseline_result": str(cli.baseline_result), "baseline_sha256": sha256(cli.baseline_result),
              "checkpoint": {"path": str(checkpoint), "config_sha256": sha256(checkpoint / "config.json"),
                "weight_files": [{"name": p.name, "size": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns}
                                 for p in sorted(checkpoint.glob("*.safetensors"))]},
              "options_path": options_path, "options_sha256": sha256(options_path),
              "environment": {k: os.environ[k] for k in (
                  "CUDA_VISIBLE_DEVICES", "HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "HF_HUB_DISABLE_TELEMETRY",
                  "TRANSFORMERS_OFFLINE", "NCCL_P2P_DISABLE", "NCCL_IB_DISABLE", "SELECTOR_STORE_VIS_ATTRS")
                  if k in os.environ},
              "versions": {"python": sys.version, "torch": torch.__version__, "numpy": np.__version__},
              "source_hashes": {str(p.relative_to(REPO)): sha256(p) for p in
                  [Path(__file__), HERE / "eval_fgvc_selector.py", HERE / "token_compression/selector_model.py",
                   *sorted((HERE / "selection_audit").glob("*.py"))]},
              "status": "started"}
    write_json(output / "run_config.json", config)
    original = original_eval(args, baseline) if cli.equivalence else None
    baseline.set_seed(args.seed)
    os.environ["SELECTOR_STORE_VIS_ATTRS"] = "1"
    model, processor, tokenizer = baseline.load_model(args)
    baseline.replace_layers(args, model)
    config["geometry"] = check_processor(processor)
    config["effective_selector"] = {name: getattr(model.visual, name) for name in
        ("budgets", "spatial_ratio", "cosine_penalty_weight", "cosine_penalty_threshold",
         "selector_focal_redundancy_limit", "distance_aware_redundancy", "distance_sigma",
         "distance_floor", "distance_apply_to")}
    config["environment"]["SELECTOR_STORE_VIS_ATTRS"] = "1"
    assert model.visual.selector_focal_redundancy_limit == 0
    write_json(output / "run_config.json", config)
    records = []
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    with (output / "records.jsonl").open("w") as stream, torch.inference_mode():
        for j, image_id in enumerate(ids):
            tick = time.perf_counter()
            item = cub.images[image_id]
            if examples is not None:
                prompt, gold, choices = baseline.prepare_prompt_mc(examples[j]["options"],
                    args.prompt_query or baseline.DATASET_CONFIGS["bird-200"]["prompt_query"], args.choice_enumeration)
            else:
                prompt, gold, choices = "Describe the bird in this image briefly.", None, None
            image_path = str(cub.root / item["relative_path"])
            if examples is not None:
                answer = baseline.generate_answer(model, processor, tokenizer, image_path, prompt, args)
            else:
                messages = baseline.build_messages(image_path, prompt, args)
                texts = [processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in messages]
                images, videos = baseline.process_vision_info(messages)
                inputs = processor(text=texts, images=images, videos=videos, padding=True, return_tensors="pt")
                inputs = inputs.to("cuda" if args.device_map == "auto" else args.device)
                result = model(**inputs, use_cache=False)
                del result, inputs, images, videos
                answer = None
            record, arrays = snapshot(model, item, args, cli.duplicate_threshold)
            torch.cuda.synchronize()
            record.update(elapsed_seconds=time.perf_counter() - tick, prompt=prompt,
                          prediction_raw=answer, prediction_parsed=baseline.parse_prediction(answer, args.choice_enumeration) if answer is not None else None,
                          gold_label=gold, options=choices, git_commit=commit, checkpoint=str(checkpoint), seed=args.seed,
                          run_config="run_config.json")
            record["classification_correct"] = (
                record["prediction_parsed"] == gold if gold is not None else None
            )
            np.savez_compressed(output / "tokens" / f"{image_id:05d}.npz", **arrays)
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()
            records.append(record)
            del arrays
            print(
                f"[{j + 1}/{len(ids)}] image={image_id} N={record['N']} K={record['K']} "
                f"focal={record['K_focal']} spatial={record['K_spatial']} "
                f"sigma={record['distance_sigma']} fallback={record['fallback_count']} "
                f"duplicate={record['duplicate_indices']} out_of_bounds={record['out_of_bounds_indices']}",
                flush=True,
            )
    if original:
        result, selections = original
        assert result["config"] == vars(args), "Baseline/audit config mismatch"
        for record, prediction, selection in zip(records, result["predictions"], selections):
            assert record["relative_path"] == prediction["image"]
            for key in ("prediction_raw", "prediction_parsed", "gold_label", "options"):
                assert record[key] == prediction[key], f"Prediction/prompt mismatch: {key}"
            for key in ("K", "N", "selected_indices"):
                assert record[key] == selection[key], f"Visual selection mismatch: {key}"
        write_json(output / "equivalence.json", {"passed": True, "num_images": 5,
            "prediction_equal": True, "selected_indices_equal": True, "selected_count_equal": True,
            "model_config_equal": True, "baseline_cache_enabled": False, "audit_cache_enabled": True})
    config.update(status="writing_reports", inference_and_metrics_seconds=time.perf_counter() - start,
                  peak_cuda_allocated_gib=torch.cuda.max_memory_allocated() / 1024**3,
                  peak_cuda_reserved_gib=torch.cuda.max_memory_reserved() / 1024**3)
    write_json(output / "run_config.json", config)
    summary = write_reports(output, records, config)
    random_ids = random.Random(args.seed).sample(ids, min(20, len(ids)))
    ranked = sorted(records, key=lambda r: (-r["actionable_inspection_score"], r["image_id"]))[:20]
    top_ids = [r["image_id"] for r in ranked]
    write_json(output / "overlays" / "selection.json", {"random_image_ids": random_ids, "top_actionable_image_ids": top_ids})
    for record in records:
        if record["image_id"] in set(random_ids + top_ids):
            with np.load(output / "tokens" / f"{record['image_id']:05d}.npz") as arrays:
                save_overlay(record, arrays["raw_score"], cub.root / record["relative_path"],
                             output / "overlays" / f"{record['image_id']:05d}.png")
    config.update(status="complete", total_audit_seconds=time.perf_counter() - start)
    write_json(output / "run_config.json", config)
    print(json.dumps({"output": str(output), "ActionableRedundancyRate": summary["ActionableRedundancyRate"]}, indent=2))


def annotation_hashes(root):
    return {name: sha256(root / name) for name in ("images.txt", "image_class_labels.txt", "train_test_split.txt",
                                                 "parts/parts.txt", "parts/part_locs.txt", "bounding_boxes.txt")}


if __name__ == "__main__":
    main()
