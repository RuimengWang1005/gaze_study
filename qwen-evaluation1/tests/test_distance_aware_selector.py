"""CPU tests for training-free distance-aware focal redundancy."""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from token_compression.selector_model import (
    _build_merged_token_coordinates,
    _select_focal_indices,
    _select_indices_with_cosine_penalty,
    _select_indices_with_distance_aware_penalty,
)


def select(scores, features, coords, k, dtype=torch.float32, sigma=.2, weight=1.):
    diagnostics = {}
    result = _select_indices_with_distance_aware_penalty(
        torch.tensor(scores, dtype=dtype), torch.tensor(features, dtype=dtype),
        torch.tensor(coords, dtype=torch.float32), torch.zeros(len(scores), dtype=torch.long),
        k, weight, .7, sigma, 0., diagnostics,
    )
    return result, diagnostics


def test_disabled_is_exact_legacy_selection():
    torch.manual_seed(7)
    scores = torch.randn(17)
    features = torch.randn(17, 8)
    legacy = _select_indices_with_cosine_penalty(scores, features, 6, 1., .7)
    routed = _select_focal_indices(scores, features, 6, 1., .7, False)
    assert torch.equal(legacy, routed)


def test_zero_weight_degenerates_to_pure_topk():
    scores = torch.tensor([.1, .8, .2, .7])
    features = torch.ones(4, 3)
    coords = torch.tensor([[0., 0.], [0., .3], [0., .6], [0., 1.]])
    result = _select_focal_indices(scores, features, 2, 0., .7, True, coords,
                                   torch.zeros(4, dtype=torch.long))
    assert set(result.tolist()) == set(torch.topk(scores, 2).indices.tolist())


def test_near_similar_token_is_penalized_more_than_far_token():
    result, _ = select([2., 1.01, 0., 1.], [[1., 0.]] * 4,
                       [[0., 0.], [0., .05], [0., .5], [0., 1.]], 2, sigma=.1)
    assert result.tolist() == [0, 3]


@pytest.mark.parametrize("shape", [(1, 1), (1, 7), (7, 1), (2, 2), (3, 5)])
@pytest.mark.parametrize("fraction", [.01, .2, .9, 1.])
def test_budget_and_grid_boundaries(shape, fraction):
    h, w = shape
    n = h * w
    coords, image_ids = _build_merged_token_coordinates(
        torch.tensor([[1, h * 2, w * 2]]), 2, torch.device("cpu")
    )
    assert coords.shape == (n, 2) and torch.isfinite(coords).all()
    k = min(n, max(1, int(n * fraction)))
    scores = torch.linspace(1, 0, n)
    features = torch.eye(n)
    result = _select_focal_indices(scores, features, k, 1., .7, True,
                                   coords, image_ids, .1, 0.)
    assert len(result) == k == len(set(result.tolist()))
    assert all(0 <= index < n for index in result.tolist())


def test_cross_image_tokens_do_not_receive_distance_penalty():
    coords, image_ids = _build_merged_token_coordinates(
        torch.tensor([[1, 2, 4], [1, 2, 4]]), 2, torch.device("cpu")
    )
    scores = torch.tensor([2., 1.1, 1., .9])
    features = torch.ones(4, 2)
    result = _select_focal_indices(scores, features, 2, 1., .7, True,
                                   coords, image_ids, .5, 0.)
    # Index 2 shares coordinates with the seed but belongs to another image.
    assert result.tolist() == [0, 2]


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_low_precision_is_finite(dtype):
    result, diagnostics = select([2., 1., .9], [[1., 0.], [1., 0.], [0., 1.]],
                                 [[0., 0.], [0., .1], [0., 1.]], 2, dtype=dtype)
    assert result.dtype == torch.long and len(result) == 2
    assert diagnostics["fallback_count"] == 0


def test_invalid_distance_configuration_uses_recorded_topk_fallback():
    result, diagnostics = select([.1, .3, .2], [[1., 0.]] * 3,
                                 [[0., 0.], [0., .5], [0., 1.]], 2, sigma=0.)
    assert set(result.tolist()) == {1, 2}
    assert diagnostics["fallback_count"] == 1
