"""Pure analysis helpers for the second-stage ViT head signal audit."""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Tuple

import torch


def normalize_distribution(scores: torch.Tensor, epsilon: float = 1e-12) -> torch.Tensor:
    scores = scores.float().clamp_min(0)
    return scores / scores.sum(dim=-1, keepdim=True).clamp_min(epsilon)


def value_weighted_incoming(
    incoming_scores: torch.Tensor,
    value_norms: torch.Tensor,
) -> torch.Tensor:
    """Normalize ``incoming * ||V||`` into one distribution per head."""

    if incoming_scores.shape != value_norms.shape:
        raise ValueError(f"incoming shape {incoming_scores.shape} != value norm shape {value_norms.shape}")
    return normalize_distribution(incoming_scores.float() * value_norms.float())


def query_conditioned_attention(
    attention_probs: torch.Tensor,
    foreground_mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return foreground-query, background-query, and foreground-minus-background maps."""

    if attention_probs.ndim != 3:
        raise ValueError("attention_probs must be [H,Q,K]")
    mask = foreground_mask.bool()
    if mask.ndim != 1 or mask.numel() != attention_probs.shape[-2]:
        raise ValueError("foreground mask does not match query dimension")
    if not bool(mask.any()) or bool(mask.all()):
        raise ValueError("query conditioning needs both foreground and background tokens")
    foreground = attention_probs[:, mask, :].float().mean(dim=1)
    background = attention_probs[:, ~mask, :].float().mean(dim=1)
    return foreground, background, foreground - background


def uniform_log2_ratio(scores: torch.Tensor, epsilon: float = 1e-12) -> torch.Tensor:
    """Signed log2 enrichment relative to the uniform probability ``1/N``."""

    probabilities = normalize_distribution(scores, epsilon)
    uniform = 1.0 / probabilities.shape[-1]
    return torch.log2((probabilities + epsilon) / uniform)


def percentile_ranks(scores: torch.Tensor) -> torch.Tensor:
    """Convert every row to deterministic percentile ranks in [0,1]."""

    flat = scores.reshape(-1, scores.shape[-1]).float()
    order = flat.argsort(dim=-1, stable=True)
    ranks = torch.empty_like(flat)
    values = torch.linspace(0.0, 1.0, flat.shape[-1], device=flat.device)
    ranks.scatter_(1, order, values.expand_as(flat))
    return ranks.reshape_as(scores)


def random_translated_masks(
    mask_grid: torch.Tensor,
    count: int,
    generator: torch.Generator,
) -> torch.Tensor:
    """Translate a mask without wraparound while preserving its exact shape/count."""

    if mask_grid.ndim != 2:
        raise ValueError("mask_grid must be [H,W]")
    mask = mask_grid.bool()
    coords = mask.nonzero(as_tuple=False)
    if coords.numel() == 0:
        raise ValueError("cannot translate an empty mask")
    h, w = mask.shape
    y0, x0 = coords.min(dim=0).values.tolist()
    y1, x1 = coords.max(dim=0).values.tolist()
    box_h, box_w = y1 - y0 + 1, x1 - x0 + 1
    relative = coords - torch.tensor([y0, x0])
    masks = []
    for _ in range(count):
        top = int(torch.randint(0, h - box_h + 1, (1,), generator=generator).item())
        left = int(torch.randint(0, w - box_w + 1, (1,), generator=generator).item())
        shifted = relative + torch.tensor([top, left])
        candidate = torch.zeros_like(mask)
        candidate[shifted[:, 0], shifted[:, 1]] = True
        masks.append(candidate)
    return torch.stack(masks)


def shuffled_bbox_statistics(
    scores: torch.Tensor,
    real_mask: torch.Tensor,
    shuffled_masks: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Compare per-head probability mass in real versus translated masks."""

    probabilities = normalize_distribution(scores)
    real = real_mask.bool()
    shuffled = shuffled_masks.bool()
    if probabilities.shape[-1] != real.numel() or shuffled.shape[-1] != real.numel():
        raise ValueError("score/mask length mismatch")
    real_mass = probabilities[:, real].sum(dim=-1)
    shuffle_mass = torch.einsum("hn,sn->hs", probabilities, shuffled.float())
    mean = shuffle_mass.mean(dim=-1)
    std = shuffle_mass.std(dim=-1, unbiased=False)
    p_value = (1.0 + (shuffle_mass >= real_mass[:, None]).sum(dim=-1).float()) / (shuffled.shape[0] + 1.0)
    return {
        "real_mass": real_mass,
        "shuffle_mean": mean,
        "shuffle_std": std,
        "delta": real_mass - mean,
        "z_score": (real_mass - mean) / std.clamp_min(1e-12),
        "empirical_p": p_value,
    }


def pearson_correlation(a: torch.Tensor, b: torch.Tensor, epsilon: float = 1e-12) -> torch.Tensor:
    """Row-wise Pearson correlation of ``a[H,N]`` against ``b[N]``."""

    if a.ndim != 2 or b.ndim != 1 or a.shape[-1] != b.numel():
        raise ValueError("correlation expects [H,N] and [N]")
    a = a.float() - a.float().mean(dim=-1, keepdim=True)
    b = b.float() - b.float().mean()
    numerator = (a * b).sum(dim=-1)
    denominator = a.square().sum(dim=-1).sqrt() * b.square().sum().sqrt()
    return numerator / denominator.clamp_min(epsilon)


def spearman_correlation(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return pearson_correlation(percentile_ranks(a), percentile_ranks(b.unsqueeze(0))[0])


def topk_overlap(a: torch.Tensor, b: torch.Tensor, fraction: float) -> torch.Tensor:
    if a.ndim != 2 or b.ndim != 1 or a.shape[-1] != b.numel():
        raise ValueError("topk overlap expects [H,N] and [N]")
    k = max(1, int(math.ceil(b.numel() * fraction)))
    b_top = torch.topk(b, k=k).indices
    overlaps = []
    for row in a:
        a_top = torch.topk(row, k=k).indices
        overlaps.append(torch.isin(a_top, b_top).float().mean())
    return torch.stack(overlaps)


def one_sided_sign_test(values: Iterable[float]) -> float:
    """Exact P(X >= positives) for non-zero paired effects under p=0.5."""

    nonzero = [float(value) for value in values if float(value) != 0.0]
    if not nonzero:
        return 1.0
    positives = sum(value > 0.0 for value in nonzero)
    numerator = sum(math.comb(len(nonzero), k) for k in range(positives, len(nonzero) + 1))
    return numerator / (2 ** len(nonzero))


def benjamini_hochberg(p_values: Iterable[float]) -> List[float]:
    """Return false-discovery-rate adjusted q-values in input order."""

    values = [min(1.0, max(0.0, float(value))) for value in p_values]
    if not values:
        return []
    order = sorted(range(len(values)), key=values.__getitem__)
    adjusted = [1.0] * len(values)
    running = 1.0
    for rank_index in range(len(values) - 1, -1, -1):
        original_index = order[rank_index]
        rank = rank_index + 1
        running = min(running, values[original_index] * len(values) / rank)
        adjusted[original_index] = min(1.0, running)
    return adjusted
