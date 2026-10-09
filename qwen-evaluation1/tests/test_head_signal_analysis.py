from __future__ import annotations

import sys
from pathlib import Path

import torch


EVAL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EVAL_ROOT))

from head_audit.signal_analysis import (  # noqa: E402
    benjamini_hochberg,
    normalize_distribution,
    one_sided_sign_test,
    percentile_ranks,
    query_conditioned_attention,
    random_translated_masks,
    shuffled_bbox_statistics,
    spearman_correlation,
    topk_overlap,
    uniform_log2_ratio,
    value_weighted_incoming,
)


def test_value_weighted_scores_are_normalized_and_change_with_value_norm():
    incoming = torch.tensor([[0.25, 0.25, 0.25, 0.25]])
    norms = torch.tensor([[1.0, 1.0, 2.0, 4.0]])
    weighted = value_weighted_incoming(incoming, norms)
    assert torch.allclose(weighted.sum(-1), torch.ones(1))
    assert weighted.argmax(-1).item() == 3


def test_query_conditioned_maps_sum_to_one_and_delta_sums_to_zero():
    probs = torch.tensor([[[0.8, 0.2], [0.1, 0.9]]])
    foreground, background, delta = query_conditioned_attention(probs, torch.tensor([True, False]))
    assert torch.allclose(foreground.sum(-1), torch.ones(1))
    assert torch.allclose(background.sum(-1), torch.ones(1))
    assert torch.allclose(delta.sum(-1), torch.zeros(1), atol=1e-7)


def test_uniform_relative_and_percentile_maps_behave_as_expected():
    uniform = torch.ones(2, 4)
    assert torch.allclose(uniform_log2_ratio(uniform), torch.zeros_like(uniform))
    ranks = percentile_ranks(torch.tensor([[4.0, 1.0, 3.0, 2.0]]))
    assert torch.equal(ranks.argmax(-1), torch.tensor([0]))
    assert torch.allclose(ranks.min(-1).values, torch.zeros(1))
    assert torch.allclose(ranks.max(-1).values, torch.ones(1))


def test_random_bbox_control_preserves_shape_and_token_count():
    mask = torch.zeros(6, 8, dtype=torch.bool)
    mask[1:3, 2:5] = True
    generator = torch.Generator().manual_seed(42)
    shuffled = random_translated_masks(mask, 25, generator)
    assert shuffled.shape == (25, 6, 8)
    assert torch.all(shuffled.sum((-2, -1)) == mask.sum())


def test_shuffled_bbox_statistics_identify_real_enrichment():
    scores = torch.tensor([[0.7, 0.1, 0.1, 0.1]])
    real = torch.tensor([True, False, False, False])
    shuffled = torch.tensor([
        [False, True, False, False],
        [False, False, True, False],
        [False, False, False, True],
    ])
    stats = shuffled_bbox_statistics(scores, real, shuffled)
    assert stats["delta"].item() > 0
    assert stats["empirical_p"].item() == 0.25


def test_rank_correlation_and_topk_overlap():
    heads = torch.tensor([[1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]])
    selector = torch.tensor([1.0, 2.0, 3.0, 4.0])
    corr = spearman_correlation(heads, selector)
    assert torch.allclose(corr, torch.tensor([1.0, -1.0]), atol=1e-6)
    overlap = topk_overlap(heads, selector, 0.5)
    assert torch.allclose(overlap, torch.tensor([1.0, 0.0]))


def test_one_sided_sign_test_uses_nonzero_effects():
    assert one_sided_sign_test([1, 2, 3, 0]) == 0.125
    assert one_sided_sign_test([-1, -2, 0]) == 1.0
    assert one_sided_sign_test([0, 0]) == 1.0


def test_benjamini_hochberg_preserves_input_order():
    result = benjamini_hochberg([0.01, 0.04, 0.03])
    assert torch.allclose(torch.tensor(result), torch.tensor([0.03, 0.04, 0.04]))
