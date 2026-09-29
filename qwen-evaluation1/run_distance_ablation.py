"""Run paired distance-aware GAZE smoke or pilot configurations sequentially."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from selection_audit.distance_ablation import (add_paired_overlap, load_records,
                                                save_difference_overlay,
                                                save_method_overlay, summarize_run,
                                                write_table)

HERE = Path(__file__).resolve().parent
DEFAULT_BASELINE = HERE / "resultrw/fgvc_eval/bird-200/0.1/VisionSelector-Qwen2.5-VL-7B_selector_0.1/20260902_153802_results.json"


def methods():
    result = [("topk", ["--spatial-ratio-override", "0", "--cosine-penalty-weight-override", "0"]),
              ("gaze_original", [])]
    for sigma, suffix in ((.05, "005"), (.1, "010"), (.2, "020")):
        common = ["--distance-aware-redundancy", "--distance-sigma", str(sigma),
                  "--distance-floor", "0", "--distance-apply-to", "focal"]
        result.append((f"gaze_distance_sigma_{suffix}", common))
        result.append((f"distance_focal_only_sigma_{suffix}",
                       common + ["--spatial-ratio-override", "0"]))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("smoke", "pilot", "full"), required=True)
    parser.add_argument("--root", type=Path, default=HERE / "results/distance_ablation")
    parser.add_argument("--baseline-result", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()
    root = args.root.resolve()
    stage = root / args.stage
    stage.mkdir(parents=True, exist_ok=False)
    limit = 5 if args.stage == "smoke" else (200 if args.stage == "pilot" else None)
    stage_methods = methods()
    if args.stage == "full":
        stage_methods = [(name, extra) for name, extra in stage_methods
                         if name in {"topk", "gaze_original", "gaze_distance_sigma_020",
                                     "distance_focal_only_sigma_020"}]
    configs = []
    env = os.environ.copy()
    for budget in (.1, .2):
        for method, extra in stage_methods:
            run_id = f"{method}_b{int(budget * 10):02d}"
            command = [args.python, str(HERE / "audit_fgvc_selection.py"),
                "--baseline-result", str(args.baseline_result.resolve()), "--mode", "fgvc-options",
                "--budget", str(budget), "--run-id", run_id,
                "--output-root", str(stage), *extra]
            if limit is not None:
                command.extend(["--limit", str(limit)])
            log_path = stage / f"{run_id}.log"
            with log_path.open("w") as log:
                process = subprocess.run(command, cwd=HERE, env=env, stdout=log,
                                         stderr=subprocess.STDOUT, text=True)
            if process.returncode:
                raise subprocess.CalledProcessError(process.returncode, command)
            output = stage / run_id
            (output / "command.txt").write_text(" ".join(command) + "\n")
            (output / "stdout_stderr.log").write_text(log_path.read_text())
            configs.append({"run_id": run_id, "method": method, "budget": budget,
                            "command": command, "output": str(output)})

    loaded = {cfg["run_id"]: load_records(stage / cfg["run_id"] / "records.jsonl") for cfg in configs}
    canonical = loaded[configs[0]["run_id"]]
    sample_ids = [record["image_id"] for record in canonical]
    for records in loaded.values():
        assert [record["image_id"] for record in records] == sample_ids
    rows = []
    for cfg in configs:
        records = loaded[cfg["run_id"]]
        config = json.loads((stage / cfg["run_id"] / "run_config.json").read_text())
        summary = summarize_run(records, config)
        budget_suffix = f"b{int(cfg['budget'] * 10):02d}"
        add_paired_overlap(summary, records, loaded[f"gaze_original_{budget_suffix}"],
                           loaded[f"topk_{budget_suffix}"])
        summary.update(method=cfg["method"], budget=cfg["budget"], run_id=cfg["run_id"])
        (stage / cfg["run_id"] / "ablation_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        rows.append(summary)
    write_table(stage / "summary.csv", rows)
    (stage / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")
    manifest = {"stage": args.stage, "sample_ids": sample_ids, "num_samples": len(sample_ids),
                "configs": configs, "paired": True}
    (stage / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    root.mkdir(parents=True, exist_ok=True)
    root_manifest = root / "manifest.json"
    existing = json.loads(root_manifest.read_text()) if root_manifest.exists() else {"stages": {}}
    existing["stages"][args.stage] = str(stage / "manifest.json")
    root_manifest.write_text(json.dumps(existing, indent=2) + "\n")

    visual_ids = sample_ids if args.stage == "smoke" else sample_ids[:10]
    visual_root = stage / "paired_visualizations"
    visual_root.mkdir()
    cub_root = Path(json.loads((stage / configs[0]["run_id"] / "run_config.json").read_text())["model_args"]["image_folder"])
    for budget in (.1, .2):
        suffix = f"b{int(budget * 10):02d}"
        original = {r["image_id"]: r for r in loaded[f"gaze_original_{suffix}"]}
        for method, _ in stage_methods:
            records = {r["image_id"]: r for r in loaded[f"{method}_{suffix}"]}
            method_dir = visual_root / f"{method}_{suffix}"
            method_dir.mkdir()
            for image_id in visual_ids:
                image_path = cub_root / records[image_id]["relative_path"]
                save_method_overlay(records[image_id], image_path,
                                    method_dir / f"{image_id:05d}_method.png", method)
                if method.startswith(("gaze_distance", "distance_focal")):
                    save_difference_overlay(original[image_id], records[image_id], image_path,
                        method_dir / f"{image_id:05d}_difference.png",
                        method_dir / f"{image_id:05d}_difference.json", method)
    print(json.dumps({"stage": args.stage, "output": str(stage), "samples": len(sample_ids)}, indent=2))


if __name__ == "__main__":
    main()
