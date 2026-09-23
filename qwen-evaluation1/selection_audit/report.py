"""JSON/CSV summaries with explicit denominators and absent-part handling."""
from collections import Counter
import csv
import json
from pathlib import Path
import numpy as np


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def flatten(value, prefix=""):
    result = {}
    for key, item in value.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(item, dict):
            result.update(flatten(item, name))
        elif isinstance(item, (str, int, float, bool)) or item is None:
            result[name] = item
    return result


def write_csv(path, rows):
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def summarize(records):
    parts = [p for record in records for p in record["parts"]]
    reasons = Counter(p["miss_reason"] for p in parts if p["miss_reason"])
    missed = sum(reasons.values())
    summary = {"num_images": len(records), "visible_parts": len(parts), "union_missed_parts_radius1": missed,
               "ActionableRedundancyRate": sum(r["ActionableRedundancyCase"] for r in records) / len(records),
               "miss_reason_counts": dict(reasons),
               "miss_reason_fraction_of_union_misses": {k: reasons[k] / missed if missed else None
                    for k in ("scorer_miss", "selection_suppression", "near_candidate")},
               "miss_reason_fraction_of_visible_parts": {k: reasons[k] / len(parts) if parts else None
                    for k in ("scorer_miss", "selection_suppression", "near_candidate")}}
    micro = {}
    for radius in (0, 1):
        key = str(radius)
        rescued = sum(not p["coverage"]["focal"][key] and p["coverage"]["spatial"][key] for p in parts)
        focal_misses = sum(not p["coverage"]["focal"][key] for p in parts)
        micro[key] = {name + "_PartRecall": sum(p["coverage"][name][key] for p in parts) / len(parts)
                      if parts else None for name in ("focal", "spatial", "union", "raw_topk")}
        micro[key].update(SpatialRescue=rescued / len(parts) if parts else None,
                          SpatialRescueConditional=rescued / focal_misses if focal_misses else None,
                          rescued_count=rescued, focal_miss_count=focal_misses,
                          raw_topk_lost_count=sum(p["coverage"]["raw_topk"][key] and
                              not p["coverage"]["focal"][key] for p in parts),
                          focal_gained_count=sum(p["coverage"]["focal"][key] and
                              not p["coverage"]["raw_topk"][key] for p in parts))
    summary["part_micro"] = micro
    summary["macro_parts"] = {}
    for macro in ("head", "torso", "wings", "legs", "tail"):
        present = [r["macro_coverage"][macro] for r in records if macro in r["macro_coverage"]]
        macro_parts = [p for p in parts if p["macro"] == macro]
        summary["macro_parts"][macro] = {"visible_images": len(present), "visible_parts": len(macro_parts),
            "any_part_recall": {name: {str(radius): sum(c[name][str(radius)] for c in present) / len(present)
                if present else None for radius in (0, 1)} for name in ("focal", "spatial", "union", "raw_topk")},
            "part_recall": {name: {str(radius): sum(p["coverage"][name][str(radius)] for p in macro_parts) / len(macro_parts)
                if macro_parts else None for radius in (0, 1)} for name in ("focal", "spatial", "union", "raw_topk")}}
    rows = [flatten({"metrics": r["metrics"], "coverage": r["coverage"],
                     "bbox_outside_ratio": r["bbox_outside_ratio"], "K": r["K"],
                     "K_focal": r["K_focal"], "K_spatial": r["K_spatial"],
                     "elapsed_seconds": r["elapsed_seconds"]}) for r in records]
    summary["image_means"] = {key: float(np.mean([row[key] for row in rows if row.get(key) is not None]))
                              for key in rows[0] if any(row.get(key) is not None for row in rows)}
    return summary


def write_reports(output, records, config):
    output = Path(output)
    summary = summarize(records)
    write_json(output / "summary.json", summary)
    write_csv(output / "summary.csv", [flatten(summary)])
    write_csv(output / "per_sample.csv", [flatten(r) for r in records])
    write_csv(output / "per_part.csv", [{"image_id": r["image_id"], **flatten(p),
               "grid_row": p["grid"][0], "grid_col": p["grid"][1]} for r in records for p in r["parts"]])
    cov = summary["part_micro"]["1"]
    text = ["# CUB selection audit", "", "CUB part coverage 只是解剖覆盖代理，不等于真实类别判别 utility。", "",
            f"Images: {len(records)}; budget: {config['model_args']['budgets']}; seed: {config['model_args']['seed']}.",
            "", "1-cell coverage (visible part micro average):", "",
            *[f"- {k}: {v}" for k, v in cov.items()], "",
            f"ActionableRedundancyRate: {summary['ActionableRedundancyRate']}", "",
            "Miss reasons (denominator = all union misses at radius 1):", "",
            *[f"- {k}: {v}" for k, v in summary['miss_reason_fraction_of_union_misses'].items()], "",
            "See summary.json for exact-cell coverage, both branch spatial/duplicate metrics, radius-2 sensitivity,",
            "macro-part coverage and per-image means. Absent visible parts have null recall; empty selections have zero duplicate rates.",
            "", "SpatialRescue denominator is all visible parts; SpatialRescueConditional uses focal-missed visible parts.",
            "A macro-part is covered if any of its visible landmarks is covered. Bbox membership uses token centers.",
            "Raw-score ties are broken by ascending token index; raw rank is 1-based.",
            "ActionableRedundancy uses radius 1, the configured diagnostic cosine threshold, and entirely missed macro-parts.",
            "Overlay ranking uses candidate missed macro count × focal excess members only for inspection ordering.",
            "Random overlays are sampled independently with fixed seed; top-ranked overlays are also included.",
            "These diagnostics are observational; scorer/selection labels do not establish causal utility.", ""]
    (output / "report.md").write_text("\n".join(text))
    return summary
