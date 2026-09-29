"""Paired analysis for Global GAZE and Local-GAZE audit runs."""

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_records(run_dir: Path) -> List[dict]:
    return [
        json.loads(line)
        for line in (run_dir / "records.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def jaccard(left: Sequence[int], right: Sequence[int]) -> float:
    left_set, right_set = set(left), set(right)
    union = left_set | right_set
    return len(left_set & right_set) / len(union) if union else 1.0


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return float(np.mean(values)) if values else 0.0


def quantiles(values: Sequence[float]) -> dict:
    if not values:
        return {key: None for key in ("mean", "median", "p10", "p25", "p50", "p75", "p90")}
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p10": float(np.quantile(array, 0.10)),
        "p25": float(np.quantile(array, 0.25)),
        "p50": float(np.quantile(array, 0.50)),
        "p75": float(np.quantile(array, 0.75)),
        "p90": float(np.quantile(array, 0.90)),
    }


def exact_mcnemar_p(global_wins: int, local_wins: int) -> float:
    discordant = global_wins + local_wins
    if discordant == 0:
        return 1.0
    tail = min(global_wins, local_wins)
    probability = sum(
        math.comb(discordant, index) for index in range(tail + 1)
    ) / (2 ** discordant)
    return min(1.0, 2.0 * probability)


def paired_bootstrap_ci(
    global_correct: np.ndarray,
    local_correct: np.ndarray,
    seed: int,
    samples: int,
) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    paired_delta = local_correct.astype(np.float64) - global_correct.astype(np.float64)
    indices = rng.integers(0, len(paired_delta), size=(samples, len(paired_delta)))
    deltas = paired_delta[indices].mean(axis=1)
    lower, upper = np.quantile(deltas, [0.025, 0.975])
    return float(lower), float(upper)


def validate_run(records: List[dict], label: str) -> None:
    if not records:
        raise ValueError(f"{label}: empty records")
    for record in records:
        diagnostics = record.get("redundancy_diagnostics")
        if diagnostics is None:
            raise ValueError(f"{label}: missing redundancy diagnostics")
        if record["K"] != record["target_token_count"]:
            raise ValueError(f"{label}: target budget mismatch")
        if record["K"] != len(record["selected_indices"]):
            raise ValueError(f"{label}: actual budget mismatch")
        if record["duplicate_token_count"] != 0 or record["duplicate_indices"]:
            raise ValueError(f"{label}: duplicate token")
        if record["out_of_bounds_indices"]:
            raise ValueError(f"{label}: out-of-bounds token")
        for key in ("max_similarity", "penalty_base"):
            if not all(math.isfinite(value) for value in diagnostics[key]):
                raise ValueError(f"{label}: non-finite {key}")


def redundancy_summary(records: List[dict]) -> dict:
    max_similarity: List[float] = []
    penalties: List[float] = []
    distances: List[int] = []
    candidate_count = 0
    empty_count = 0.0
    outside_count = 0.0
    for record in records:
        diagnostics = record["redundancy_diagnostics"]
        count = int(diagnostics["num_candidates"])
        candidate_count += count
        max_similarity.extend(diagnostics["max_similarity"])
        penalties.extend(diagnostics["penalty_base"])
        distances.extend(
            value
            for value in diagnostics["global_argmax_grid_distances"]
            if value is not None
        )
        empty_count += diagnostics["empty_local_neighbor_ratio"] * count
        outside_count += diagnostics["global_argmax_outside_local_ratio"] * count
    nonzero = sum(value > 0 for value in penalties)
    weighted_penalty = sum(
        record["redundancy_diagnostics"]["mean_penalty"]
        * record["redundancy_diagnostics"]["num_candidates"]
        for record in records
    )
    return {
        "max_similarity": quantiles(max_similarity),
        "global_argmax_grid_distance": quantiles(distances),
        "nonzero_penalty_candidate_ratio": nonzero / candidate_count,
        "mean_penalty": weighted_penalty / candidate_count,
        "mean_score_iqr": mean(
            record["redundancy_diagnostics"]["score_iqr"] for record in records
        ),
        "mean_penalty_iqr": mean(
            record["redundancy_diagnostics"]["penalty_iqr"] for record in records
        ),
        "mean_lambda_eff": mean(
            record["redundancy_diagnostics"]["lambda_eff"] for record in records
        ),
        "adaptive_alpha": mean(
            record["redundancy_diagnostics"]["adaptive_alpha"] for record in records
        ),
        "empty_local_neighbor_candidate_ratio": empty_count / candidate_count,
        "global_argmax_outside_local_ratio": outside_count / candidate_count,
        "num_candidates": candidate_count,
    }


def run_summary(records: List[dict], warmup: int) -> dict:
    timed = records[min(warmup, len(records)) :]
    return {
        "accuracy": mean(record["classification_correct"] for record in records),
        "mean_focal_tokens": mean(record["K_focal"] for record in records),
        "mean_spatial_tokens": mean(record["K_spatial"] for record in records),
        "mean_final_tokens": mean(record["K"] for record in records),
        "mean_target_tokens": mean(record["target_token_count"] for record in records),
        "mean_actual_retention_ratio": mean(
            record["actual_retention_ratio"] for record in records
        ),
        "total_duplicate_tokens": sum(record["duplicate_token_count"] for record in records),
        "total_fallback_count": sum(record["fallback_count"] for record in records),
        "mean_selector_ms": mean(
            record["selector_score_select_time_ms"] for record in timed
        ),
        "peak_cuda_allocated_gib": None,
        "redundancy": redundancy_summary(records),
    }


def paired_summary(
    global_records: List[dict],
    local_records: List[dict],
    seed: int,
    bootstrap_samples: int,
) -> Tuple[dict, List[dict]]:
    rows = []
    for global_record, local_record in zip(global_records, local_records):
        if global_record["image_id"] != local_record["image_id"]:
            raise ValueError("sample order mismatch")
        for key in ("gold_label", "N", "K", "K_focal", "K_spatial"):
            if global_record[key] != local_record[key]:
                raise ValueError(f"paired field mismatch: {key}")
        focal_global, focal_local = (
            global_record["focal_indices"],
            local_record["focal_indices"],
        )
        spatial_global, spatial_local = (
            global_record["spatial_indices"],
            local_record["spatial_indices"],
        )
        final_global, final_local = (
            global_record["selected_indices"],
            local_record["selected_indices"],
        )
        replaced = len(set(final_global) - set(final_local))
        symmetric_difference = len(set(final_global) ^ set(final_local))
        if symmetric_difference != 2 * replaced:
            raise ValueError("symmetric difference invariant failed")
        rows.append(
            {
                "image_id": global_record["image_id"],
                "relative_path": global_record["relative_path"],
                "global_correct": bool(global_record["classification_correct"]),
                "local_correct": bool(local_record["classification_correct"]),
                "focal_jaccard": jaccard(focal_global, focal_local),
                "spatial_jaccard": jaccard(spatial_global, spatial_local),
                "final_jaccard": jaccard(final_global, final_local),
                "sym_diff_size": symmetric_difference,
                "num_replaced": replaced,
                "focal_unchanged": focal_global == focal_local,
                "final_unchanged": final_global == final_local,
            }
        )

    global_correct = np.asarray([row["global_correct"] for row in rows], dtype=bool)
    local_correct = np.asarray([row["local_correct"] for row in rows], dtype=bool)
    global_wins = int(np.sum(global_correct & ~local_correct))
    local_wins = int(np.sum(~global_correct & local_correct))
    both_correct = int(np.sum(global_correct & local_correct))
    both_wrong = int(np.sum(~global_correct & ~local_correct))
    ci = paired_bootstrap_ci(
        global_correct, local_correct, seed=seed, samples=bootstrap_samples
    )
    return {
        "accuracy_delta": float(local_correct.mean() - global_correct.mean()),
        "mean_focal_jaccard": mean(row["focal_jaccard"] for row in rows),
        "mean_spatial_jaccard": mean(row["spatial_jaccard"] for row in rows),
        "mean_final_jaccard": mean(row["final_jaccard"] for row in rows),
        "mean_sym_diff_size": mean(row["sym_diff_size"] for row in rows),
        "mean_num_replaced": mean(row["num_replaced"] for row in rows),
        "focal_unchanged_ratio": mean(row["focal_unchanged"] for row in rows),
        "final_unchanged_ratio": mean(row["final_unchanged"] for row in rows),
        "local_wins": local_wins,
        "global_wins": global_wins,
        "both_correct": both_correct,
        "both_wrong": both_wrong,
        "net_corrections": local_wins - global_wins,
        "paired_bootstrap_95ci": list(ci),
        "exact_mcnemar_p": exact_mcnemar_p(global_wins, local_wins),
    }, rows


def markdown_report(summary: dict) -> str:
    lines = [
        "# Local-GAZE paired CUB/Bird-200 results",
        "",
        "Pilot results were not used to select a radius. Radius 1 is the pre-specified main result; radii 2 and 3 are sensitivity analyses.",
        "",
        "| Method | Radius | Accuracy | Delta Accuracy | Focal Jaccard | Final Jaccard | Avg Replaced Tokens |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    global_run = summary["runs"]["global"]
    lines.append(
        f"| Original GAZE | Global | {global_run['accuracy']:.6f} | 0.000000 | 1.000000 | 1.000000 | 0 |"
    )
    for label in summary["local_labels"]:
        run = summary["runs"][label]
        paired = summary["paired"][label]
        lines.append(
            f"| Local-GAZE | {summary['radii'][label]} | {run['accuracy']:.6f} | "
            f"{paired['accuracy_delta']:+.6f} | {paired['mean_focal_jaccard']:.6f} | "
            f"{paired['mean_final_jaccard']:.6f} | {paired['mean_num_replaced']:.3f} |"
        )
    lines += [
        "",
        "| Method | Local Wins | Global Wins | Both Correct | Both Wrong | 95% CI | McNemar p |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label in summary["local_labels"]:
        paired = summary["paired"][label]
        ci = paired["paired_bootstrap_95ci"]
        lines.append(
            f"| Local-GAZE-{label} | {paired['local_wins']} | {paired['global_wins']} | "
            f"{paired['both_correct']} | {paired['both_wrong']} | "
            f"[{ci[0]:+.6f}, {ci[1]:+.6f}] | {paired['exact_mcnemar_p']:.6g} |"
        )
    lines += [
        "",
        "Accuracy differences are paired observations on the same test samples. The observed best test radius is exploratory, not a validation-selected configuration.",
        "",
    ]
    return "\n".join(lines)


def parse_local(value: str) -> Tuple[str, Path]:
    label, separator, path = value.partition("=")
    if not separator or not label or not path:
        raise argparse.ArgumentTypeError("--local must use LABEL=RUN_DIR")
    return label, Path(path).resolve()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--global-run", required=True, type=Path)
    parser.add_argument("--local", required=True, action="append", type=parse_local)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup", type=int, default=1)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")
    if args.bootstrap_samples <= 0 or args.warmup < 0:
        parser.error("invalid bootstrap sample or warmup count")

    global_dir = args.global_run.resolve()
    global_records = load_records(global_dir)
    validate_run(global_records, "global")
    global_config = load_json(global_dir / "run_config.json")
    if global_config["status"] != "complete":
        raise ValueError("global run is incomplete")

    local_runs: Dict[str, Path] = dict(args.local)
    if len(local_runs) != len(args.local):
        raise ValueError("duplicate local label")
    runs = {"global": run_summary(global_records, args.warmup)}
    paired, paired_rows, radii = {}, {}, {}
    for label, run_dir in local_runs.items():
        records = load_records(run_dir)
        validate_run(records, label)
        config = load_json(run_dir / "run_config.json")
        if config["status"] != "complete":
            raise ValueError(f"{label}: incomplete run")
        if len(records) != len(global_records):
            raise ValueError(f"{label}: sample count mismatch")
        runs[label] = run_summary(records, args.warmup)
        runs[label]["peak_cuda_allocated_gib"] = config["peak_cuda_allocated_gib"]
        paired[label], paired_rows[label] = paired_summary(
            global_records, records, args.seed, args.bootstrap_samples
        )
        paired[label]["selector_extra_ms"] = (
            runs[label]["mean_selector_ms"] - runs["global"]["mean_selector_ms"]
        )
        radii[label] = int(config["effective_selector"]["local_radius"])
    runs["global"]["peak_cuda_allocated_gib"] = global_config["peak_cuda_allocated_gib"]

    summary = {
        "num_samples": len(global_records),
        "seed": args.seed,
        "bootstrap_samples": args.bootstrap_samples,
        "warmup_samples_excluded_from_timing": args.warmup,
        "global_run": str(global_dir),
        "local_runs": {label: str(path) for label, path in local_runs.items()},
        "local_labels": list(local_runs),
        "radii": radii,
        "runs": runs,
        "paired": paired,
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    fields = [
        "method", "image_id", "relative_path", "global_correct", "local_correct",
        "focal_jaccard", "spatial_jaccard", "final_jaccard", "sym_diff_size",
        "num_replaced", "focal_unchanged", "final_unchanged",
    ]
    with (args.output_dir / "paired_records.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for label, rows in paired_rows.items():
            for row in rows:
                writer.writerow({"method": label, **row})
    (args.output_dir / "REPORT.md").write_text(
        markdown_report(summary), encoding="utf-8"
    )
    print(json.dumps({"output": str(args.output_dir), "paired": paired}, indent=2))


if __name__ == "__main__":
    main()
