#!/usr/bin/env python3
"""Run the independent BBox-only sufficiency and teacher-utility audit."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import torch


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from bbox_teacher_audit.data import (  # noqa: E402
    TRAINING_SOURCE_COUNTS,
    build_candidate_pools,
    proportional_quotas,
    select_utility_subset,
    select_valid_samples,
)
from bbox_teacher_audit.io_utils import (  # noqa: E402
    environment_snapshot,
    load_sample_records,
    save_sample_record,
    write_json,
    write_jsonl,
)
from bbox_teacher_audit.report import build_review_zip, finalize_outputs  # noqa: E402
from bbox_teacher_audit.teacher_forward import (  # noqa: E402
    TeacherRuntime,
    model_sentinels,
    parameter_grads_absent,
    parameters_are_frozen,
)
from bbox_teacher_audit.token_mapping import MAPPING_PROVENANCE  # noqa: E402


DEFAULT_MODEL = "/data/ModelZoo/LVLM/Qwen-Qwen2.5-VL-7B-Instruct"
DEFAULT_DATA_DIR = "/home/ruimeng/gaze/viscot_dataset"
DEFAULT_TRAINING_MANIFEST = "/home/bbhan/gaze/viscot_mix_20k_30-15-15-10-5-5-5-5-5_seed42.jsonl"
DEFAULT_IMAGE_ROOT = "/data/DataSet/vlm_dataset/cot_image_data"
DEFAULT_OUTPUT = SCRIPT_DIR / "results" / "bbox_teacher_audit_v1"
DEFAULT_ZIP = SCRIPT_DIR / "review_pack_bbox_teacher_audit_v1.zip"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("sufficiency", "utility", "all"), default="all")
    parser.add_argument("--num-sufficiency", type=int, default=200)
    parser.add_argument("--num-utility", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--skip-tests", action="store_true")
    parser.add_argument("--model-path", default=DEFAULT_MODEL)
    parser.add_argument("--data-dir", type=Path, default=Path(DEFAULT_DATA_DIR))
    parser.add_argument("--training-manifest", type=Path, default=Path(DEFAULT_TRAINING_MANIFEST))
    parser.add_argument("--image-root", type=Path, default=Path(DEFAULT_IMAGE_ROOT))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--min-pixels", type=int, default=3136)
    parser.add_argument("--max-pixels", type=int, default=401408)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--review-zip", type=Path, default=DEFAULT_ZIP)
    args = parser.parse_args()
    if args.num_sufficiency <= 0 or args.num_utility <= 0:
        parser.error("sample counts must be positive")
    if args.num_utility > args.num_sufficiency:
        parser.error("--num-utility cannot exceed --num-sufficiency")
    return args


class RunLogger:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path

    def __call__(self, message: str) -> None:
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
        print(line, flush=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
            handle.flush()
            os.fsync(handle.fileno())


def run_tests(output_dir: Path) -> None:
    tests = [
        SCRIPT_DIR / "tests" / "test_bbox_teacher_mapping.py",
        SCRIPT_DIR / "tests" / "test_bbox_teacher_utility.py",
        SCRIPT_DIR / "tests" / "test_bbox_teacher_forward.py",
        SCRIPT_DIR / "tests" / "test_bbox_teacher_resume.py",
    ]
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", *(str(path) for path in tests)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    (output_dir / "test_report.txt").write_text(
        completed.stdout + ("\n" + completed.stderr if completed.stderr else ""), encoding="utf-8"
    )
    if completed.returncode:
        raise RuntimeError(f"unit tests failed; see {output_dir / 'test_report.txt'}")


def load_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def prepare_manifests(args: argparse.Namespace, processor, log: RunLogger):
    sufficiency_path = args.output_dir / "sample_manifest_200.jsonl"
    utility_path = args.output_dir / "utility_manifest_100.jsonl"
    selection_path = args.output_dir / "selection_summary.json"
    if args.resume and sufficiency_path.exists() and utility_path.exists() and selection_path.exists():
        log("resume: reusing precommitted manifests")
        return load_jsonl(sufficiency_path), load_jsonl(utility_path), json.loads(selection_path.read_text())
    pools, basic_exclusions, accepted_counts, raw_counts = build_candidate_pools(
        args.data_dir, args.image_root, args.training_manifest, args.seed
    )
    sufficiency_quotas = proportional_quotas(args.num_sufficiency)
    samples, validation_exclusions, examined = select_valid_samples(
        pools,
        processor.image_processor,
        merge_size=int(processor.image_processor.merge_size),
        quotas=sufficiency_quotas,
        min_bbox_tokens=3,
    )
    utility_quotas = proportional_quotas(args.num_utility)
    utility_samples = select_utility_subset(samples, args.seed, quotas=utility_quotas)
    exclusions = Counter(basic_exclusions)
    exclusions.update(validation_exclusions)
    selection = {
        "deterministic_seed": args.seed,
        "training_manifest": str(args.training_manifest),
        "training_manifest_rows": 20000,
        "training_source_counts": TRAINING_SOURCE_COUNTS,
        "held_out_verified": True,
        "held_out_verification": "canonical dataset/image/question/answer key excluded against the registered 20K manifest",
        "raw_candidate_counts": raw_counts,
        "raw_candidate_total": sum(raw_counts.values()),
        "candidate_counts_after_basic_validation": accepted_counts,
        "deterministic_validation_pool_sizes": {key: len(value) for key, value in pools.items()},
        "candidates_examined_for_valid_quota": examined,
        "sufficiency_quotas": sufficiency_quotas,
        "utility_quotas": utility_quotas,
        "sufficiency_selected": len(samples),
        "utility_selected": len(utility_samples),
        "excluded_from_utility_for_too_few_bbox_tokens": sum(
            int(record["bbox_visual_tokens"]) < 3 for record in samples
        ),
        "exclusions": dict(sorted(exclusions.items())),
    }
    write_jsonl(sufficiency_path, samples)
    write_jsonl(utility_path, utility_samples)
    write_json(selection_path, selection)
    log(f"precommitted manifests: sufficiency={len(samples)}, utility={len(utility_samples)}")
    return samples, utility_samples, selection


def main() -> None:
    args = parse_args()
    args.output_dir = args.output_dir.resolve()
    args.review_zip = args.review_zip.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    log = RunLogger(args.output_dir / "run.log")
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    config = {
        "command": shlex.join(sys.argv),
        "stage": args.stage,
        "num_sufficiency": args.num_sufficiency,
        "num_utility": args.num_utility,
        "seed": args.seed,
        "batch_size": 1,
        "model_path": args.model_path,
        "data_dir": str(args.data_dir),
        "training_manifest": str(args.training_manifest),
        "image_root": str(args.image_root),
        "device": args.device,
        "dtype": args.dtype,
        "min_pixels": args.min_pixels,
        "max_pixels": args.max_pixels,
        "max_new_tokens": args.max_new_tokens,
        "bbox_mapping_provenance": MAPPING_PROVENANCE,
        "training_performed": False,
        "optimizer_created": False,
        "production_selector_modified": False,
        "attention_heads_used": False,
        "official_visual_cot_evaluator": None,
        "normalized_exact_match_is_auxiliary": True,
        "random20_repeats": 3,
        "random20_seed_rule": "sha256(sample_id:base_seed:repeat), repeats 0/1/2",
    }
    write_json(args.output_dir / "config.json", config)
    write_json(args.output_dir / "environment.json", environment_snapshot(REPO_ROOT))
    if not args.skip_tests:
        log("running unit tests")
        run_tests(args.output_dir)

    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(
        args.model_path,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
        local_files_only=True,
    )
    sufficiency_manifest, utility_manifest, selection = prepare_manifests(args, processor, log)
    if args.dry_run:
        log("dry run complete; model was not loaded")
        return

    log(f"loading frozen model {args.model_path}")
    runtime = TeacherRuntime.load(
        args.model_path,
        device=args.device,
        dtype=args.dtype,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
        max_new_tokens=args.max_new_tokens,
    )
    before = model_sentinels(runtime.model)
    if not parameters_are_frozen(runtime.model):
        raise RuntimeError("model parameters are not frozen")

    sufficiency_records = load_sample_records(args.output_dir, "sufficiency") if args.resume else {}
    if args.stage in ("sufficiency", "all"):
        for index, record in enumerate(sufficiency_manifest, 1):
            identifier = str(record["sample_id"])
            if identifier in sufficiency_records:
                log(f"sufficiency {index}/{len(sufficiency_manifest)} {identifier}: resume skip")
                continue
            log(f"sufficiency {index}/{len(sufficiency_manifest)} {identifier}")
            prepared = runtime.prepare(record)
            result = runtime.evaluate_sufficiency(prepared, smoke_reference=args.smoke_test)
            if args.smoke_test and float(result["injected_full_nll_abs_error"]) > 1e-4:
                raise RuntimeError("injected full-token path differs from the original model path")
            save_sample_record(args.output_dir, "sufficiency", identifier, result)
            sufficiency_records[identifier] = result
            del prepared
            if runtime.device.type == "cuda":
                torch.cuda.empty_cache()

    utility_records = load_sample_records(args.output_dir, "utility") if args.resume else {}
    if args.stage in ("utility", "all"):
        for index, record in enumerate(utility_manifest, 1):
            identifier = str(record["sample_id"])
            if identifier in utility_records:
                log(f"utility {index}/{len(utility_manifest)} {identifier}: resume skip")
                continue
            log(f"utility {index}/{len(utility_manifest)} {identifier}")
            prepared = runtime.prepare(record)
            sufficiency_record = sufficiency_records.get(identifier)
            if sufficiency_record is None:
                generation = runtime.generate(prepared, prepared.visual_features, prepared.foreground_indices)
                bbox_correct = bool(generation["correct"])
            else:
                bbox_correct = bool(sufficiency_record["bbox_correct"])
            result = runtime.evaluate_utility(prepared, args.seed, bbox_correct)
            if not bool(result["parameter_grads_absent"]):
                raise RuntimeError("a frozen model parameter unexpectedly received a gradient")
            save_sample_record(args.output_dir, "utility", identifier, result)
            utility_records[identifier] = result
            del prepared
            if runtime.device.type == "cuda":
                torch.cuda.empty_cache()

    after = model_sentinels(runtime.model)
    model_unchanged = before == after
    if not model_unchanged:
        raise RuntimeError("model sentinel values changed during the audit")
    selection = {**selection, "model_frozen": True, "model_sentinels_unchanged": model_unchanged}
    write_json(args.output_dir / "selection_summary.json", selection)
    ordered_sufficiency = [sufficiency_records[str(row["sample_id"])] for row in sufficiency_manifest if str(row["sample_id"]) in sufficiency_records]
    ordered_utility = [utility_records[str(row["sample_id"])] for row in utility_manifest if str(row["sample_id"]) in utility_records]
    summary = finalize_outputs(args.output_dir, ordered_sufficiency, ordered_utility, selection, args.seed)
    build_review_zip(args.output_dir, args.review_zip, REPO_ROOT)
    log(f"complete: {json.dumps({'stage_a_passed': summary['stage_a_passed'], 'stage_b_passed': summary['stage_b_passed'], 'go': summary['go_for_2000_sample_distillation']})}")
    log(f"review pack: {args.review_zip}")


if __name__ == "__main__":
    main()
