from __future__ import annotations

import sys
from pathlib import Path

import torch


EVAL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EVAL_ROOT))

from head_audit.attention_capture import AttentionCapture  # noqa: E402
from qwen25vl.modeling_qwen2_5_vl import Qwen2_5_VLVisionAttention  # noqa: E402


def _inputs(num_tokens: int = 8, dim: int = 32):
    generator = torch.Generator().manual_seed(7)
    hidden = torch.randn(num_tokens, dim, generator=generator)
    # Qwen duplicates a half-head rotary embedding to one full head dimension.
    angles = torch.randn(num_tokens, dim // 8, generator=generator)
    emb = torch.cat((angles, angles), dim=-1)
    position_embeddings = (emb.cos(), emb.sin())
    cu_seqlens = torch.tensor([0, num_tokens], dtype=torch.int32)
    return hidden, cu_seqlens, position_embeddings


def test_capture_matches_original_eager_attention_and_restores_forward():
    module = Qwen2_5_VLVisionAttention(dim=32, num_heads=4).eval()
    hidden, cu_seqlens, position_embeddings = _inputs()
    original_func = module.forward.__func__

    with torch.no_grad():
        reference = module(hidden, cu_seqlens=cu_seqlens, position_embeddings=position_embeddings)
        with AttentionCapture(module) as capture:
            captured_output = module(hidden, cu_seqlens=cu_seqlens, position_embeddings=position_embeddings)

    assert module.forward.__func__ is original_func
    assert captured_output.shape == reference.shape
    assert torch.allclose(captured_output, reference, atol=1e-6, rtol=1e-6)
    assert capture.attention_probs is not None
    assert capture.attention_probs.shape == (1, 4, 8, 8)


def test_attention_and_incoming_scores_are_normalized_per_head():
    module = Qwen2_5_VLVisionAttention(dim=32, num_heads=4).eval()
    hidden, cu_seqlens, position_embeddings = _inputs()
    with torch.no_grad(), AttentionCapture(module) as capture:
        module(hidden, cu_seqlens=cu_seqlens, position_embeddings=position_embeddings)

    probs = capture.attention_probs
    assert probs is not None
    assert torch.allclose(probs.sum(dim=-1), torch.ones((1, 4, 8)), atol=1e-6, rtol=1e-6)
    incoming = probs.mean(dim=-2)
    assert torch.allclose(incoming.sum(dim=-1), torch.ones((1, 4)), atol=1e-6, rtol=1e-6)
    assert capture.metadata.row_sum_max_error < 1e-6
    assert capture.metadata.incoming_sum_max_error < 1e-6


def test_segment_mask_prevents_cross_image_attention():
    module = Qwen2_5_VLVisionAttention(dim=32, num_heads=4).eval()
    hidden, _, position_embeddings = _inputs()
    cu_seqlens = torch.tensor([0, 4, 8], dtype=torch.int32)
    with torch.no_grad(), AttentionCapture(module) as capture:
        module(hidden, cu_seqlens=cu_seqlens, position_embeddings=position_embeddings)
    probs = capture.attention_probs
    assert torch.count_nonzero(probs[..., :4, 4:]) == 0
    assert torch.count_nonzero(probs[..., 4:, :4]) == 0
