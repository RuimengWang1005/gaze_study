"""Pre-registered metrics and paired uncertainty estimates."""

from __future__ import annotations

import math
import re
import string
from collections import defaultdict
from typing import Dict, Iterable, List, Mapping, Sequence

import numpy as np
from scipy.stats import spearmanr, wilcoxon


def normalize_answer(text: object) -> str:
    value = str(text or "").lower().strip()
    value = value.replace("<|im_end|>", " ")
    value = value.translate(str.maketrans("", "", string.punctuation))
    value = re.sub(r"\b(a|an|the)\b", " ", value)
    return " ".join(value.split())


def normalized_exact_match(prediction: object, answer: object) -> bool:
    return normalize_answer(prediction) == normalize_answer(answer)


def bootstrap_ci(
    values: Sequence[float],
    seed: int = 42,
    confidence: float = 0.95,
    repetitions: int = 10000,
) -> List[float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return [float("nan"), float("nan")]
    generator = np.random.default_rng(seed)
    draws = generator.choice(array, size=(repetitions, array.size), replace=True).mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    return [float(np.quantile(draws, alpha)), float(np.quantile(draws, 1.0 - alpha))]


def bootstrap_upper_bound(
    values: Sequence[float],
    seed: int = 42,
    confidence: float = 0.95,
    repetitions: int = 10000,
) -> float:
    """Percentile upper confidence bound for a paired mean."""

    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return float("nan")
    generator = np.random.default_rng(seed)
    draws = generator.choice(array, size=(repetitions, array.size), replace=True).mean(axis=1)
    return float(np.quantile(draws, confidence))


def safe_mean(values: Iterable[float]) -> float:
    values = [float(value) for value in values]
    return float(np.mean(values)) if values else float("nan")


def safe_median(values: Iterable[float]) -> float:
    values = [float(value) for value in values]
    return float(np.median(values)) if values else float("nan")


def signed_rank(values: Sequence[float]) -> Dict[str, float | int | None]:
    finite = np.asarray([value for value in values if math.isfinite(float(value))], dtype=np.float64)
    nonzero = finite[finite != 0]
    if not nonzero.size:
        return {"statistic": None, "p_value_two_sided": None, "n_nonzero": 0}
    result = wilcoxon(nonzero, alternative="two-sided", method="auto")
    return {
        "statistic": float(result.statistic),
        "p_value_two_sided": float(result.pvalue),
        "n_nonzero": int(nonzero.size),
    }


def correlation(x: Sequence[float], y: Sequence[float]) -> Dict[str, float | None]:
    pairs = [(float(a), float(b)) for a, b in zip(x, y) if math.isfinite(float(a)) and math.isfinite(float(b))]
    if len(pairs) < 3 or len({a for a, _ in pairs}) < 2 or len({b for _, b in pairs}) < 2:
        return {"spearman_r": None, "p_value": None, "n": len(pairs)}
    result = spearmanr([a for a, _ in pairs], [b for _, b in pairs])
    return {"spearman_r": float(result.statistic), "p_value": float(result.pvalue), "n": len(pairs)}


def summarize_sufficiency(rows: Sequence[Mapping[str, object]], seed: int = 42) -> Dict[str, object]:
    deltas = [float(row["delta_nll"]) for row in rows]
    accuracy_deltas = [float(row["bbox_correct"]) - float(row["full_correct"]) for row in rows]
    full_correct = [row for row in rows if bool(row["full_correct"])]
    summary = {
        "n": len(rows),
        "full_mean_nll": safe_mean(row["full_mean_nll"] for row in rows),
        "bbox_mean_nll": safe_mean(row["bbox_mean_nll"] for row in rows),
        "delta_nll_mean": safe_mean(deltas),
        "delta_nll_median": safe_median(deltas),
        "delta_nll_bootstrap_ci_95": bootstrap_ci(deltas, seed),
        "delta_nll_bootstrap_upper_95_one_sided": bootstrap_upper_bound(deltas, seed),
        "full_normalized_em": safe_mean(float(row["full_correct"]) for row in rows),
        "bbox_normalized_em": safe_mean(float(row["bbox_correct"]) for row in rows),
        "accuracy_delta": safe_mean(accuracy_deltas),
        "accuracy_delta_bootstrap_ci_95": bootstrap_ci(accuracy_deltas, seed + 1),
        "conditional_sufficiency": safe_mean(float(row["bbox_correct"]) for row in full_correct),
        "full_correct_count": len(full_correct),
        "mean_visual_token_retention": safe_mean(row["visual_token_retention"] for row in rows),
        "mean_visual_token_reduction": 1.0 - safe_mean(row["visual_token_retention"] for row in rows),
        "mean_full_visual_tokens": safe_mean(row["full_visual_tokens"] for row in rows),
        "mean_bbox_visual_tokens": safe_mean(row["bbox_visual_tokens"] for row in rows),
        "mean_full_sequence_length": safe_mean(row["full_sequence_length"] for row in rows),
        "mean_bbox_sequence_length": safe_mean(row["bbox_sequence_length"] for row in rows),
        "mean_full_forward_ms": safe_mean(row["full_forward_ms"] for row in rows),
        "mean_bbox_forward_ms": safe_mean(row["bbox_forward_ms"] for row in rows),
        "mean_full_generation_ms": safe_mean(row["full_generation_ms"] for row in rows),
        "mean_bbox_generation_ms": safe_mean(row["bbox_generation_ms"] for row in rows),
        "mean_latency_ratio": safe_mean(row["latency_ratio"] for row in rows),
        "mean_full_peak_memory_mb": safe_mean(row["full_peak_memory_mb"] for row in rows),
        "mean_bbox_peak_memory_mb": safe_mean(row["bbox_peak_memory_mb"] for row in rows),
        "mean_peak_memory_ratio": safe_mean(
            float(row["bbox_peak_memory_mb"]) / float(row["full_peak_memory_mb"]) for row in rows
        ),
        "bbox_coverage_delta_nll_correlation": correlation(
            [float(row["visual_token_retention"]) for row in rows], deltas
        ),
    }
    upper = summary["delta_nll_bootstrap_upper_95_one_sided"]
    summary["criteria"] = {
        "accuracy_drop_at_most_0_03": summary["accuracy_delta"] >= -0.03,
        "delta_nll_upper_below_0_10": upper < 0.10,
        "conditional_sufficiency_at_least_0_90": summary["conditional_sufficiency"] >= 0.90,
    }
    summary["passed"] = all(summary["criteria"].values())
    by_source = {}
    grouped = defaultdict(list)
    for row in rows:
        grouped[str(row["source"])].append(row)
    for source, source_rows in sorted(grouped.items()):
        by_source[source] = {
            "n": len(source_rows),
            "delta_nll_mean": safe_mean(row["delta_nll"] for row in source_rows),
            "full_em": safe_mean(float(row["full_correct"]) for row in source_rows),
            "bbox_em": safe_mean(float(row["bbox_correct"]) for row in source_rows),
            "mean_retention": safe_mean(row["visual_token_retention"] for row in source_rows),
        }
    summary["by_source"] = by_source
    return summary


def _effect_summary(values: Sequence[float], seed: int) -> Dict[str, object]:
    return {
        "mean": safe_mean(values),
        "median": safe_median(values),
        "bootstrap_ci_95": bootstrap_ci(values, seed),
        "wilcoxon": signed_rank(values),
    }


def summarize_utility(rows: Sequence[Mapping[str, object]], seed: int = 42) -> Dict[str, object]:
    valid = [row for row in rows if not bool(row.get("degenerate"))]
    summary = {
        "n_requested": len(rows),
        "n_valid": len(valid),
        "n_degenerate": len(rows) - len(valid),
        "degenerate_rate": (len(rows) - len(valid)) / len(rows) if rows else float("nan"),
    }
    for offset, key in enumerate(("delta_top", "delta_random", "delta_bottom", "causal_excess", "top_vs_bottom")):
        summary[key] = _effect_summary([float(row[key]) for row in valid], seed + offset)
    summary["causal_excess_positive_rate"] = safe_mean(float(row["causal_excess"] > 0) for row in valid)
    summary["criteria"] = {
        "causal_excess_ci_lower_above_zero": summary["causal_excess"]["bootstrap_ci_95"][0] > 0,
        "top_vs_bottom_ci_lower_above_zero": summary["top_vs_bottom"]["bootstrap_ci_95"][0] > 0,
    }
    summary["passed"] = bool(valid) and all(summary["criteria"].values())
    for group_name, group_value in (("bbox_correct", True), ("bbox_incorrect", False)):
        group = [row for row in valid if bool(row.get("bbox_base_correct")) is group_value]
        summary[group_name] = {
            "n": len(group),
            "causal_excess_mean": safe_mean(row["causal_excess"] for row in group),
            "top_vs_bottom_mean": safe_mean(row["top_vs_bottom"] for row in group),
        }
    return summary
