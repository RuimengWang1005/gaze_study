"""Audit-only helpers for Qwen2.5-VL vision attention heads."""

from .attention_capture import AttentionCapture, eager_attention_forward
from .metrics import compute_head_metrics
from .signal_analysis import query_conditioned_attention, value_weighted_incoming
from .token_mapping import TokenMapping, build_window_index

__all__ = [
    "AttentionCapture",
    "TokenMapping",
    "build_window_index",
    "compute_head_metrics",
    "eager_attention_forward",
    "query_conditioned_attention",
    "value_weighted_incoming",
]
