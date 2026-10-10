"""Teacher utility and equal-cardinality token deletion helpers."""

from __future__ import annotations

import hashlib
import math
from typing import Dict, Iterable, List

import torch


def grad_input_utility(gradient: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
    """Compute the pre-registered ReLU(gradient dot input) token utility."""

    if gradient.shape != features.shape or gradient.ndim != 2:
        raise ValueError("gradient and features must have the same [N,D] shape")
    return torch.relu((gradient.float() * features.detach().float()).sum(dim=-1))


def deletion_count(num_tokens: int, fraction: float = 0.20) -> int:
    if num_tokens < 3:
        raise ValueError("at least three bbox tokens are required")
    count = max(1, int(math.ceil(fraction * num_tokens)))
    if num_tokens - count < 2:
        raise ValueError("deletion would leave fewer than two visual tokens")
    return count


def _seed_for(sample_id: str, base_seed: int, repeat: int) -> int:
    digest = hashlib.sha256(f"{sample_id}:{base_seed}:{repeat}".encode()).digest()
    return int.from_bytes(digest[:8], "little") % (2**31)


def deletion_sets(
    utility: torch.Tensor,
    sample_id: str,
    base_seed: int,
    random_repeats: int = 3,
) -> Dict[str, List[torch.Tensor] | torch.Tensor]:
    """Build fixed Top/Bottom/Random deletion indices with equal counts."""

    if utility.ndim != 1 or not torch.isfinite(utility).all():
        raise ValueError("utility must be a finite vector")
    count = deletion_count(int(utility.numel()))
    top = torch.topk(utility, k=count, largest=True, sorted=True).indices.sort().values
    bottom = torch.topk(utility, k=count, largest=False, sorted=True).indices.sort().values
    random_sets = []
    for repeat in range(random_repeats):
        generator = torch.Generator(device="cpu").manual_seed(_seed_for(sample_id, base_seed, repeat))
        random_sets.append(torch.randperm(utility.numel(), generator=generator)[:count].sort().values)
    return {"top": top, "bottom": bottom, "random": random_sets}


def retained_indices(num_tokens: int, deleted: Iterable[int]) -> torch.Tensor:
    keep = torch.ones(num_tokens, dtype=torch.bool)
    deleted = torch.as_tensor(list(deleted), dtype=torch.long)
    if deleted.numel():
        if int(deleted.min()) < 0 or int(deleted.max()) >= num_tokens:
            raise IndexError("deleted token index out of range")
        keep[deleted] = False
    return keep.nonzero(as_tuple=False).flatten()
