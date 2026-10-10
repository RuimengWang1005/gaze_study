"""Summaries, plots, report, and privacy-scoped review bundle."""

from __future__ import annotations

import json
import shutil
import zipfile
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .io_utils import atomic_write_text, command_output, write_csv, write_json
from .metrics import summarize_sufficiency, summarize_utility


def _save_figure(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def make_plots(output_dir: Path, sufficiency_rows: Sequence[Mapping], utility_rows: Sequence[Mapping]) -> None:
    plots = output_dir / "plots"
    full = np.asarray([float(row["full_mean_nll"]) for row in sufficiency_rows])
    bbox = np.asarray([float(row["bbox_mean_nll"]) for row in sufficiency_rows])
    delta = bbox - full
    retention = np.asarray([float(row["visual_token_retention"]) for row in sufficiency_rows])

    plt.figure(figsize=(5.5, 5))
    plt.scatter(full, bbox, alpha=0.55, s=18)
    low, high = float(min(full.min(), bbox.min())), float(max(full.max(), bbox.max()))
    plt.plot([low, high], [low, high], "k--", linewidth=1)
    plt.xlabel("Full-token mean NLL")
    plt.ylabel("BBox-only mean NLL")
    _save_figure(plots / "nll_full_vs_bbox.png")

    plt.figure(figsize=(6, 4))
    plt.hist(delta, bins=30)
    plt.axvline(0, color="black", linestyle="--")
    plt.xlabel("BBox-only minus Full mean NLL")
    _save_figure(plots / "delta_nll_hist.png")

    plt.figure(figsize=(6, 4))
    plt.hist(retention, bins=25)
    plt.xlabel("BBox visual-token retention")
    _save_figure(plots / "token_retention_hist.png")

    valid = [row for row in utility_rows if not bool(row.get("degenerate"))]
    if valid:
        plt.figure(figsize=(6, 4))
        plt.boxplot(
            [
                [float(row["delta_top"]) for row in valid],
                [float(row["delta_random"]) for row in valid],
                [float(row["delta_bottom"]) for row in valid],
            ],
            tick_labels=["Top20", "Random20", "Bottom20"],
        )
        plt.axhline(0, color="black", linestyle="--")
        plt.ylabel("NLL increase from BBox-only base")
        _save_figure(plots / "utility_condition_boxplot.png")

        plt.figure(figsize=(6, 4))
        plt.hist([float(row["causal_excess"]) for row in valid], bins=25)
        plt.axvline(0, color="black", linestyle="--")
        plt.xlabel("Top20 damage minus Random20 damage")
        _save_figure(plots / "causal_excess_hist.png")

    sources = sorted({str(row["source"]) for row in sufficiency_rows})
    source_delta = [np.mean([float(row["delta_nll"]) for row in sufficiency_rows if row["source"] == source]) for source in sources]
    plt.figure(figsize=(7, 4))
    plt.bar(sources, source_delta)
    plt.axhline(0, color="black", linestyle="--")
    plt.ylabel("Mean delta NLL")
    plt.xticks(rotation=25)
    _save_figure(plots / "results_by_source.png")


def write_report(
    output_dir: Path,
    sufficiency: Mapping[str, object],
    utility: Mapping[str, object],
    selection: Mapping[str, object],
) -> None:
    stage_a = bool(sufficiency.get("passed"))
    stage_b = bool(utility.get("passed"))
    go = stage_a and stage_b
    by_source = sufficiency.get("by_source", {})
    source_lines = "\n".join(
        f"- `{source}`: n={values['n']}, mean delta NLL={values['delta_nll_mean']:.4f}, "
        f"Full/BBox EM={values['full_em']:.3f}/{values['bbox_em']:.3f}, retention={values['mean_retention']:.3f}"
        for source, values in by_source.items()
    )
    stage_a_interpretation = (
        "BBox-only met all three pre-registered engineering sufficiency criteria."
        if stage_a
        else "BBox-only did not meet all three pre-registered engineering sufficiency criteria."
    )
    top_mean = utility.get("delta_top", {}).get("mean", float("nan"))
    random_mean = utility.get("delta_random", {}).get("mean", float("nan"))
    bottom_mean = utility.get("delta_bottom", {}).get("mean", float("nan"))
    top_interpretation = (
        "Top-token deletion caused more mean NLL damage than random deletion."
        if top_mean > random_mean
        else "Top-token deletion did not cause more mean NLL damage than random deletion."
    )
    bottom_interpretation = (
        "Bottom-token deletion was safer on mean NLL than Top-token deletion."
        if bottom_mean < top_mean
        else "Bottom-token deletion was not safer on mean NLL than Top-token deletion."
    )
    worst_source = max(by_source, key=lambda name: by_source[name]["delta_nll_mean"]) if by_source else None
    worst_source_text = (
        f"The largest mean NLL degradation was in `{worst_source}` "
        f"({by_source[worst_source]['delta_nll_mean']:.4f} nats/token; n={by_source[worst_source]['n']})."
        if worst_source is not None
        else "No source-level result was available."
    )
    text = f"""# BBox-only Deep Teacher Feasibility Audit v1

This is a frozen-model diagnostic. No model was trained, no FGVC benchmark was
run, no attention head was selected, and the production GAZE selector was not
modified. Normalized exact match below is an auxiliary diagnostic, not an
official Visual CoT metric.

## Dataset and validity

- Sufficiency samples: `{sufficiency.get('n', 0)}`
- Utility requested / valid / degenerate: `{utility.get('n_requested', 0)}` / `{utility.get('n_valid', 0)}` / `{utility.get('n_degenerate', 0)}`
- Held out from the recorded 20K training manifest: `{selection.get('held_out_verified')}`
- Raw candidates scanned: `{selection.get('raw_candidate_total')}`; by source: `{json.dumps(selection.get('raw_candidate_counts', {}), ensure_ascii=False)}`
- Candidate exclusions: `{json.dumps(selection.get('exclusions', {}), ensure_ascii=False)}`
- Selected-sample runtime failures / mapping anomalies: `0` / `0` (a completed run aborts on either condition)
- No consistent separate task-type field was present across all source shards; stratification and breakdown therefore use `source`.

## Stage A — BBox-only sufficiency

- Full-token mean NLL: `{sufficiency.get('full_mean_nll', float('nan')):.4f}`
- BBox-only mean NLL: `{sufficiency.get('bbox_mean_nll', float('nan')):.4f}`
- Paired mean / median delta NLL: `{sufficiency.get('delta_nll_mean', float('nan')):.4f}` / `{sufficiency.get('delta_nll_median', float('nan')):.4f}`
- Delta NLL 95% paired bootstrap CI: `{sufficiency.get('delta_nll_bootstrap_ci_95')}`
- Delta NLL one-sided 95% bootstrap upper bound: `{sufficiency.get('delta_nll_bootstrap_upper_95_one_sided')}`
- Full / BBox normalized EM: `{sufficiency.get('full_normalized_em', float('nan')):.3f}` / `{sufficiency.get('bbox_normalized_em', float('nan')):.3f}`
- Accuracy delta and 95% CI: `{sufficiency.get('accuracy_delta', float('nan')):.4f}`, `{sufficiency.get('accuracy_delta_bootstrap_ci_95')}`
- P(BBox correct | Full correct): `{sufficiency.get('conditional_sufficiency', float('nan')):.3f}`
- Mean visual-token retention: `{sufficiency.get('mean_visual_token_retention', float('nan')):.3f}`
- Mean visual-token reduction: `{sufficiency.get('mean_visual_token_reduction', float('nan')):.3f}`
- Mean Full / BBox visual tokens: `{sufficiency.get('mean_full_visual_tokens', float('nan')):.1f}` / `{sufficiency.get('mean_bbox_visual_tokens', float('nan')):.1f}`
- Mean Full / BBox total sequence length: `{sufficiency.get('mean_full_sequence_length', float('nan')):.1f}` / `{sufficiency.get('mean_bbox_sequence_length', float('nan')):.1f}`
- Mean Full / BBox teacher-forcing latency (ms): `{sufficiency.get('mean_full_forward_ms', float('nan')):.2f}` / `{sufficiency.get('mean_bbox_forward_ms', float('nan')):.2f}`
- Mean BBox/Full teacher-forcing latency ratio: `{sufficiency.get('mean_latency_ratio', float('nan')):.3f}`
- Mean Full / BBox greedy-generation latency (ms): `{sufficiency.get('mean_full_generation_ms', float('nan')):.2f}` / `{sufficiency.get('mean_bbox_generation_ms', float('nan')):.2f}`
- Mean Full / BBox peak allocated memory (MiB, including resident model): `{sufficiency.get('mean_full_peak_memory_mb', float('nan')):.1f}` / `{sufficiency.get('mean_bbox_peak_memory_mb', float('nan')):.1f}`
- Mean BBox/Full peak-memory ratio: `{sufficiency.get('mean_peak_memory_ratio', float('nan')):.3f}`
- Token-retention/delta-NLL Spearman result: `{sufficiency.get('bbox_coverage_delta_nll_correlation')}`
- Pre-registered Stage A criteria: `{sufficiency.get('criteria')}`
- **Stage A passed: `{stage_a}`**

{stage_a_interpretation}

### By source

{source_lines}

{worst_source_text}

## Stage B — teacher Top-token causal utility

- Top20 delta NLL: `{utility.get('delta_top')}`
- Random20 delta NLL: `{utility.get('delta_random')}`
- Bottom20 delta NLL: `{utility.get('delta_bottom')}`
- Causal excess: `{utility.get('causal_excess')}`
- Top versus Bottom: `{utility.get('top_vs_bottom')}`
- Causal-excess positive rate: `{utility.get('causal_excess_positive_rate')}`
- BBox-correct / BBox-incorrect subgroup diagnostics: `{utility.get('bbox_correct')}` / `{utility.get('bbox_incorrect')}`
- Degenerate gradients: `{utility.get('n_degenerate', 0)}` of `{utility.get('n_requested', 0)}`
- Pre-registered Stage B criteria: `{utility.get('criteria')}`
- **Stage B passed: `{stage_b}`**

{top_interpretation} {bottom_interpretation}

## Decision

**{'GO: both stages passed; a 2,000-sample distillation study is supported.' if go else 'NO-GO: at least one pre-registered stage failed; do not proceed to the 2,000-sample distillation stage.'}**

## What this audit does not establish

- It does not show that bbox-external context is universally unnecessary.
- It does not validate a deployable student or a complete distillation loss.
- It does not establish attention-head utility or justify head pruning.
- Gradient-times-input is a local teacher attribution; Stage B deletion is the
  causal check for this audit, not a universal token-importance theorem.
- Normalized EM is not an official Visual CoT score.
- This fixed 200/100-sample audit does not establish universal performance
  across every Visual CoT source, question type, or bounding-box convention.
"""
    atomic_write_text(output_dir / "report.md", text)


def finalize_outputs(
    output_dir: Path,
    sufficiency_rows: Sequence[Mapping],
    utility_rows: Sequence[Mapping],
    selection_summary: Mapping[str, object],
    seed: int,
) -> Mapping[str, object]:
    sufficiency = summarize_sufficiency(sufficiency_rows, seed) if sufficiency_rows else {}
    utility = summarize_utility(utility_rows, seed) if utility_rows else {}
    if sufficiency_rows:
        write_csv(output_dir / "sufficiency_samples.csv", sufficiency_rows)
        write_json(output_dir / "sufficiency_summary.json", sufficiency)
    if utility_rows:
        write_csv(output_dir / "utility_samples.csv", utility_rows)
        write_json(output_dir / "utility_summary.json", utility)
    summary = {
        "stage_a": sufficiency,
        "stage_b": utility,
        "stage_a_passed": bool(sufficiency.get("passed")),
        "stage_b_passed": bool(utility.get("passed")),
        "go_for_2000_sample_distillation": bool(sufficiency.get("passed")) and bool(utility.get("passed")),
        "selection": selection_summary,
    }
    write_json(output_dir / "summary.json", summary)
    if sufficiency_rows:
        make_plots(output_dir, sufficiency_rows, utility_rows)
    write_report(output_dir, sufficiency, utility, selection_summary)
    return summary


def build_review_zip(output_dir: Path, zip_path: Path, repo_root: Path) -> None:
    patch_path = output_dir / "source.patch"
    patch = command_output(["git", "diff", "959ace1..HEAD"], repo_root)
    if not patch:
        patch = command_output(["git", "diff"], repo_root)
    atomic_write_text(patch_path, patch + "\n")
    commit_info = command_output(["git", "log", "-1", "--decorate=short", "--format=fuller"], repo_root)
    atomic_write_text(output_dir / "git_info.txt", commit_info + "\n")
    names = [
        "config.json", "environment.json", "sample_manifest_200.jsonl",
        "utility_manifest_100.jsonl", "selection_summary.json",
        "sufficiency_samples.csv", "utility_samples.csv",
        "sufficiency_summary.json", "utility_summary.json", "summary.json",
        "report.md", "run.log", "source.patch", "test_report.txt", "git_info.txt",
    ]
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in names:
            path = output_dir / name
            if path.exists():
                archive.write(path, arcname=name)
        plots = output_dir / "plots"
        if plots.exists():
            for path in sorted(plots.glob("*.png")):
                archive.write(path, arcname=f"plots/{path.name}")
