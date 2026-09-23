"""CPU synthetic checks for geometry, redundancy, coverage and attribution."""
import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from selection_audit.metrics import (analyze, duplicate_metrics, hit, part_cell,
                                     spatial_metrics, validate_selection)
from selection_audit.cub_annotations import CUBAnnotations
from selection_audit.report import summarize


def test_concentrated_and_uniform():
    concentrated = [r * 16 + c for r in range(3) for c in range(3)]
    uniform = [r * 16 + c for r in (0, 7, 15) for c in (0, 7, 15)]
    a, b = [spatial_metrics(ids, 16, 16) for ids in (concentrated, uniform)]
    assert a["components_8"] == 1 and b["components_8"] == 9
    assert a["largest_component_share"] == 1
    assert a["window_3x3_max"] == 9 and b["window_3x3_max"] == 1
    assert a["adjacent_token_ratio"] == 1 and b["adjacent_token_ratio"] == 0
    assert a["macro_grid_4x4_coverage"] == 1 / 16
    assert b["macro_grid_4x4_coverage"] == 9 / 16


def test_local_similarity_and_radius_sensitivity():
    features = np.eye(16)
    features[1] = features[0]
    features[3] = features[0]
    a = duplicate_metrics([0, 1, 3], features, 4, radius=1, threshold=.9)
    assert a["LocalDuplicateRate"] == 2 / 3
    assert a["ExcessDuplicateRatio"] == 1 / 3
    assert a["LargestLocalClusterShare"] == 2 / 3
    b = duplicate_metrics([0, 1, 3], features, 4, radius=2, threshold=.9)
    assert b["ExcessDuplicateRatio"] == 2 / 3
    assert b["LocalDuplicateRate"] == 1
    assert duplicate_metrics([0, 5], np.eye(16), 4)["LocalDuplicateRate"] == 0


def test_mapping_hit_miss_and_boundary_clamp():
    assert part_cell(200, 100, 200, 100, 5, 10) == [4, 9]
    assert part_cell(-1, -1, 200, 100, 5, 10) == [0, 0]
    assert part_cell(100, 50, 200, 100, 5, 10) == [2, 5]
    assert hit([25], [2, 5], 10, 0)
    assert not hit([24], [2, 5], 10, 0)
    assert hit([24], [2, 5], 10, 1)
    assert not hit([], [2, 5], 10, 1)


def part(name="beak", macro="head", x=95, y=95, visible=1):
    return dict(part_id=1, name=name, macro=macro, x=x, y=y, visible=visible)


def audit(scores, focal, spatial, parts, features=None):
    if features is None:
        features = np.eye(100)
    return analyze(scores, features, sorted(focal + spatial), focal, spatial, parts,
                   [0, 0, 100, 100], 100, 100, 10, 10)


def test_spatial_rescue():
    result, _ = audit(np.arange(100), [0, 1], [99], [part(), part(visible=0)])
    assert len(result["parts"]) == 1
    for radius in ("0", "1"):
        cov = result["coverage"][radius]
        assert cov["focal_PartRecall"] == 0 and cov["union_PartRecall"] == 1
        assert cov["spatial_PartRecall"] == cov["SpatialRescue"] == 1
        assert cov["SpatialRescueConditional"] == 1
    assert result["parts"][0]["miss_reason"] is None


@pytest.mark.parametrize("rank,expected", [(1, "selection_suppression"), (3, "near_candidate"), (5, "scorer_miss")])
def test_miss_attribution_and_actionable(rank, expected):
    scores = -np.arange(100, dtype=float)
    scores[99] = -(rank - 1) + .5
    features = np.eye(100)
    features[1] = features[0]
    result, ranks = audit(scores, [0, 1], [], [part()], features)
    assert ranks[99] == rank
    assert result["parts"][0]["miss_reason"] == expected
    assert result["ActionableRedundancyCase"] == (rank <= 4)
    assert result["missed_macros"] == ["head"]


def test_macro_requires_all_visible_landmarks_missed():
    scores = -np.arange(100, dtype=float)
    scores[99] = 1
    features = np.ones((100, 4))
    result, _ = audit(scores, [0, 1], [], [part(), part(name="crown", x=5, y=5)], features)
    assert result["parts"][0]["miss_reason"] == "selection_suppression"
    assert result["macro_coverage"]["head"]["union"]["1"]
    assert not result["ActionableRedundancyCase"]


def test_ties_empty_and_finite():
    result, ranks = audit(np.zeros(100), [0], [], [])
    assert ranks.tolist() == list(range(1, 101))
    assert result["coverage"]["1"]["union_PartRecall"] is None
    assert duplicate_metrics([], np.eye(100), 10)["LocalDuplicateRate"] == 0
    assert spatial_metrics([], 1, 1)["window_5x5_max"] == 0
    with pytest.raises(AssertionError):
        audit(np.full(100, np.nan), [0], [], [])


def test_selection_invariants():
    validate_selection(10, [0, 3, 5], [0, 3], [5], 3)
    for selected, focal, spatial, k in (([0, 0], [0], [0], 2), ([0, 10], [0], [10], 2),
                                       ([0, 1], [0], [], 2), ([0], [0], [], 2)):
        with pytest.raises(AssertionError):
            validate_selection(10, selected, focal, spatial, k)


def test_annotation_sampling(tmp_path):
    (tmp_path / "parts").mkdir()
    (tmp_path / "images.txt").write_text("1 a.jpg\n2 b.jpg\n3 c.jpg\n4 d.jpg\n")
    (tmp_path / "image_class_labels.txt").write_text("1 1\n2 1\n3 2\n4 2\n")
    (tmp_path / "train_test_split.txt").write_text("1 1\n2 0\n3 1\n4 1\n")
    (tmp_path / "bounding_boxes.txt").write_text("".join(f"{i} 0 0 10 10\n" for i in range(1, 5)))
    (tmp_path / "parts/parts.txt").write_text("1 beak\n")
    (tmp_path / "parts/part_locs.txt").write_text("".join(f"{i} 1 5 5 1\n" for i in range(1, 5)))
    cub = CUBAnnotations(tmp_path)
    ids = cub.sample_train(42)
    assert ids == cub.sample_train(42) and 2 not in ids and 1 in ids
    assert len({cub.images[i]["class_id"] for i in ids}) == 2
    assert cub.sample_train(42, 1) == ids[:1]


def test_summary_denominators():
    r, _ = audit(np.arange(100), [0, 1], [99], [part()])
    r.update(K=3, K_focal=2, K_spatial=1, elapsed_seconds=1)
    summary = summarize([r])
    assert summary["part_micro"]["1"]["SpatialRescue"] == 1
    assert summary["miss_reason_fraction_of_union_misses"]["scorer_miss"] is None
    assert summary["macro_parts"]["legs"]["visible_images"] == 0
