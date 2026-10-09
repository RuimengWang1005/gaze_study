"""Temporary eager-attention capture for a single Qwen vision attention module.

The context manager changes only the selected module instance and restores its
original ``forward`` on exit.  It intentionally mirrors the eager Qwen2.5-VL
vision attention calculation instead of relying on an API that may average
heads or omit attention probabilities.
"""

from __future__ import annotations

import math
import types
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import torch


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def apply_vision_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply the same rotary embedding layout as Qwen eager vision attention."""

    q_dtype, k_dtype = q.dtype, k.dtype
    q_float, k_float = q.float(), k.float()
    cos = cos.unsqueeze(-2).float()
    sin = sin.unsqueeze(-2).float()
    q_embed = q_float * cos + _rotate_half(q_float) * sin
    k_embed = k_float * cos + _rotate_half(k_float) * sin
    return q_embed.to(q_dtype), k_embed.to(k_dtype)


def _position_cos_sin(
    rotary_pos_emb: Optional[torch.Tensor],
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]],
) -> Tuple[torch.Tensor, torch.Tensor]:
    if position_embeddings is not None:
        return position_embeddings
    if rotary_pos_emb is None:
        raise ValueError("Qwen vision attention requires rotary position embeddings")
    emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
    return emb.cos(), emb.sin()


def eager_attention_forward(
    module: torch.nn.Module,
    hidden_states: torch.Tensor,
    cu_seqlens: torch.Tensor,
    rotary_pos_emb: Optional[torch.Tensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    capture_callback: Optional[Callable[[torch.Tensor], None]] = None,
) -> torch.Tensor:
    """Run exact per-head eager attention and optionally capture probabilities.

    Qwen vision attention uses an unbatched token tensor ``[N, D]``.  Captured
    probabilities receive an explicit batch dimension and therefore have shape
    ``[1, H, N, N]`` for one image.
    """

    seq_length = hidden_states.shape[0]
    num_heads = int(module.num_heads)
    qkv_dim = int(module.qkv.out_features // 3)
    if qkv_dim % num_heads:
        raise ValueError(f"qkv dimension {qkv_dim} is not divisible by {num_heads} heads")
    head_dim = qkv_dim // num_heads

    q, k, v = (
        module.qkv(hidden_states)
        .reshape(seq_length, 3, num_heads, head_dim)
        .permute(1, 0, 2, 3)
        .unbind(0)
    )
    cos, sin = _position_cos_sin(rotary_pos_emb, position_embeddings)
    q, k = apply_vision_rope(q, k, cos, sin)

    attention_mask = torch.full(
        (1, seq_length, seq_length),
        torch.finfo(q.dtype).min,
        device=q.device,
        dtype=q.dtype,
    )
    boundaries = cu_seqlens.detach().to(device="cpu", dtype=torch.long).tolist()
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        attention_mask[..., start:end, start:end] = 0

    q = q.transpose(0, 1)
    k = k.transpose(0, 1)
    v = v.transpose(0, 1)
    attention_probs = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(head_dim)
    attention_probs = torch.softmax(attention_probs + attention_mask, dim=-1, dtype=torch.float32).to(q.dtype)
    if capture_callback is not None:
        capture_callback(attention_probs.unsqueeze(0))

    output = torch.matmul(attention_probs, v)
    output = output.transpose(0, 1).reshape(seq_length, qkv_dim)
    return module.proj(output)


@dataclass
class CaptureMetadata:
    shape: Tuple[int, ...]
    dtype: str
    row_sum_max_error: float
    incoming_sum_max_error: float


class AttentionCapture:
    """Audit-only context that temporarily installs eager attention forward."""

    def __init__(self, module: torch.nn.Module, capture: bool = True) -> None:
        self.module = module
        self.capture = capture
        self.attention_probs: Optional[torch.Tensor] = None
        self.metadata: Optional[CaptureMetadata] = None
        self._original_forward = None

    def _store(self, probs: torch.Tensor) -> None:
        detached = probs.detach().float().cpu()
        row_error = (detached.sum(dim=-1) - 1.0).abs().max().item()
        incoming = detached.mean(dim=-2)
        incoming_error = (incoming.sum(dim=-1) - 1.0).abs().max().item()
        self.attention_probs = detached
        self.metadata = CaptureMetadata(
            shape=tuple(detached.shape),
            dtype=str(probs.dtype),
            row_sum_max_error=float(row_error),
            incoming_sum_max_error=float(incoming_error),
        )

    def __enter__(self) -> "AttentionCapture":
        if self._original_forward is not None:
            raise RuntimeError("AttentionCapture cannot be entered twice")
        self._original_forward = self.module.forward
        owner = self

        def patched_forward(
            module_self,
            hidden_states,
            cu_seqlens,
            rotary_pos_emb=None,
            position_embeddings=None,
        ):
            return eager_attention_forward(
                module_self,
                hidden_states,
                cu_seqlens,
                rotary_pos_emb=rotary_pos_emb,
                position_embeddings=position_embeddings,
                capture_callback=owner._store if owner.capture else None,
            )

        self.module.forward = types.MethodType(patched_forward, self.module)
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self._original_forward is not None:
            self.module.forward = self._original_forward
            self._original_forward = None
