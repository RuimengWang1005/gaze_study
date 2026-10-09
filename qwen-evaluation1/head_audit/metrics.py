"""Per-head metrics for the attention head mapping audit."""

from __future__ import annotations

import math
from typing import Dict, List

import torch


def _precision_at(scores: torch.Tensor, foreground: torch.Tensor, fraction: float) -> float:
    count = max(1, int(math.ceil(scores.numel() * fraction)))
    indices = torch.topk(scores, k=count, largest=True, sorted=False).indices
    return float(foreground[indices].float().mean().item())


def compute_head_metrics(
    incoming_scores: torch.Tensor,
    foreground_mask: torch.Tensor,
    sample_id: str,
    layer_id: int,
    valid: bool,
    epsilon: float = 1e-12,
) -> List[Dict[str, object]]:
    """Compute bbox enrichment and distribution metrics for every head."""

    if incoming_scores.ndim != 2:
        raise ValueError(f"incoming_scores must be [H,N], got {tuple(incoming_scores.shape)}")
    if foreground_mask.ndim != 1 or foreground_mask.numel() != incoming_scores.shape[-1]:
        raise ValueError("foreground mask shape does not match incoming scores")
    scores = incoming_scores.detach().float().cpu()
    foreground = foreground_mask.detach().bool().cpu()
    bbox_ratio = float(foreground.float().mean().item())
    records: List[Dict[str, object]] = []
    for head_id, head_scores in enumerate(scores):
        probabilities = head_scores.clamp_min(0)
        probabilities = probabilities / probabilities.sum().clamp_min(epsilon)
        foreground_mass = float(probabilities[foreground].sum().item())
        entropy = float((-(probabilities * probabilities.clamp_min(epsilon).log()).sum()).item())
        raw_excess = foreground_mass - bbox_ratio
        normalized_excess = raw_excess / max(1.0 - bbox_ratio, epsilon)
        records.append(
            {
                "sample_id": sample_id,
                "layer_id": int(layer_id),
                "head_id": int(head_id),
                "num_tokens_premerge": int(probabilities.numel()),
                "bbox_ratio": bbox_ratio,
                "foreground_mass": foreground_mass,
                "raw_excess": raw_excess,
                "normalized_excess": normalized_excess,
                "entropy": entropy,
                "kl_to_uniform": float(math.log(probabilities.numel()) - entropy),
                "precision_10": _precision_at(probabilities, foreground, 0.10),
                "precision_20": _precision_at(probabilities, foreground, 0.20),
                "valid": bool(valid),
            }
        )
    return records
