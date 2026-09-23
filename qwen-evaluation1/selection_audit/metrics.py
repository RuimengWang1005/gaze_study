"""Pure NumPy CPU metrics. All ranks are 1-based, with index-ascending ties."""
import numpy as np


def coordinates(indices, width):
    ids = np.asarray(indices, dtype=np.int64)
    return np.column_stack((ids // width, ids % width))


def part_cell(x, y, original_width, original_height, height, width):
    return [int(np.clip(np.floor(y / original_height * height), 0, height - 1)),
            int(np.clip(np.floor(x / original_width * width), 0, width - 1))]


def components(adjacency):
    unseen = set(range(len(adjacency)))
    result = []
    while unseen:
        stack = [min(unseen)]
        unseen.remove(stack[0])
        group = []
        while stack:
            current = stack.pop()
            group.append(current)
            neighbors = unseen.intersection(np.flatnonzero(adjacency[current]).tolist())
            unseen.difference_update(neighbors)
            stack.extend(sorted(neighbors, reverse=True))
        result.append(sorted(group))
    return sorted(result, key=lambda g: (-len(g), g))


def distances(coords):
    return np.abs(coords[:, None, :] - coords[None, :, :]).max(axis=-1)


def spatial_metrics(indices, height, width):
    coords = coordinates(indices, width)
    n = len(coords)
    adjacency = distances(coords) <= 1
    np.fill_diagonal(adjacency, False)
    groups = components(adjacency)
    result = {"components_8": len(groups),
              "largest_component_size": len(groups[0]) if n else 0,
              "largest_component_share": len(groups[0]) / n if n else 0.,
              "adjacent_token_ratio": float(adjacency.any(axis=1).mean()) if n else 0.,
              "macro_grid_4x4_coverage": len({(int(r * 4 // height), int(c * 4 // width))
                                               for r, c in coords}) / 16}
    mask = np.zeros((height, width), dtype=int)
    if n:
        mask[coords[:, 0], coords[:, 1]] = 1
    for size in (3, 5):
        padded = np.pad(mask, size // 2)
        windows = np.lib.stride_tricks.sliding_window_view(padded, (size, size))
        count = int(windows.sum(axis=(-1, -2)).max())
        result[f"window_{size}x{size}_max"] = count
        result[f"window_{size}x{size}_share"] = count / n if n else 0.
    return result


def duplicate_metrics(indices, features, width, radius=1, threshold=0.7):
    ids = np.asarray(indices, dtype=int)
    feats = np.asarray(features[ids], dtype=np.float32)
    norms = np.maximum(np.linalg.norm(feats, axis=1, keepdims=True), 1e-6)
    unit = feats / norms
    cosine = np.clip(unit @ unit.T, -1, 1)
    local = distances(coordinates(ids, width)) <= radius
    np.fill_diagonal(local, False)
    adjacency = local & (cosine >= threshold)
    groups = components(adjacency)
    n = len(ids)
    clusters = [[int(ids[i]) for i in group] for group in groups if len(group) > 1]
    return {"LocalDuplicateRate": float(adjacency.any(axis=1).mean()) if n else 0.,
            "ExcessDuplicateRatio": (n - len(groups)) / n if n else 0.,
            "LargestLocalClusterShare": len(clusters[0]) / n if clusters else 0.,
            "excess_members": n - len(groups), "clusters": clusters,
            "max_local_cosine": [float(cosine[i, local[i]].max())
                                 if local[i].any() else None for i in range(n)]}


def hit(indices, cell, width, radius):
    coords = coordinates(indices, width)
    return bool(len(coords) and (np.abs(coords - cell).max(axis=1) <= radius).any())


def validate_selection(n, selected, focal, spatial, expected_k):
    for ids in (selected, focal, spatial):
        assert len(ids) == len(set(ids)), "Duplicate indices"
        assert all(0 <= i < n for i in ids), "Index out of bounds"
    assert not set(focal).intersection(spatial), "Branches overlap"
    assert set(selected) == set(focal).union(spatial), "Union differs from selected"
    assert len(selected) == expected_k, "Budget mismatch"


def analyze(scores, features, selected, focal, spatial, parts, bbox,
            original_width, original_height, height, width, threshold=0.7):
    scores = np.asarray(scores)
    assert np.isfinite(scores).all() and np.isfinite(features).all()
    n, kf = len(scores), len(focal)
    order = np.argsort(-scores, kind="stable")
    ranks = np.empty(n, dtype=int)
    ranks[order] = np.arange(1, n + 1)
    top = order[:kf].tolist()
    branches = {"focal": focal, "spatial": spatial, "union": selected, "raw_topk": top}
    metrics = {}
    duplicates = {}
    for name in ("focal", "union"):
        metrics[name] = {"spatial": spatial_metrics(branches[name], height, width)}
        duplicates[name] = {str(r): duplicate_metrics(branches[name], features, width, r, threshold)
                            for r in (1, 2)}
        metrics[name]["duplicates"] = duplicates[name]
    visible = []
    all_coords = coordinates(np.arange(n), width)
    for part in parts:
        if part["visible"] != 1:
            continue
        part = dict(part)
        cell = part_cell(part["x"], part["y"], original_width, original_height, height, width)
        part["grid"] = cell
        part["coverage"] = {name: {str(r): hit(ids, cell, width, r) for r in (0, 1)}
                            for name, ids in branches.items()}
        nearby = np.flatnonzero(np.abs(all_coords - cell).max(axis=1) <= 1)
        q = int(nearby[np.argmin(ranks[nearby])])
        rank = int(ranks[q])
        part.update(best_neighbor_index=q, best_neighbor_raw_rank=rank,
                    best_neighbor_raw_score=float(scores[q]), miss_reason=None)
        if not part["coverage"]["union"]["1"]:
            if rank > 2 * kf:
                part["miss_reason"] = "scorer_miss"
            elif rank <= kf and q not in focal:
                part["miss_reason"] = "selection_suppression"
            elif kf < rank <= 2 * kf:
                part["miss_reason"] = "near_candidate"
            else:
                raise AssertionError("A selected neighborhood token cannot be a union miss")
        visible.append(part)
    coverage = {}
    for r in (0, 1):
        key = str(r)
        count = len(visible)
        rescued = [p["name"] for p in visible if not p["coverage"]["focal"][key]
                   and p["coverage"]["spatial"][key]]
        focal_misses = sum(not p["coverage"]["focal"][key] for p in visible)
        coverage[key] = {name + "_PartRecall": sum(p["coverage"][name][key] for p in visible) / count
                        if count else None for name in branches}
        coverage[key].update(SpatialRescue=len(rescued) / count if count else None,
                             SpatialRescueConditional=len(rescued) / focal_misses if focal_misses else None,
                             spatial_rescued_parts=rescued,
                             raw_topk_lost_parts=[p["name"] for p in visible if p["coverage"]["raw_topk"][key]
                                                 and not p["coverage"]["focal"][key]],
                             focal_gained_parts=[p["name"] for p in visible if p["coverage"]["focal"][key]
                                                 and not p["coverage"]["raw_topk"][key]])
    macros = {}
    for macro in sorted({p["macro"] for p in visible}):
        group = [p for p in visible if p["macro"] == macro]
        macros[macro] = {name: {str(r): any(p["coverage"][name][str(r)] for p in group)
                                for r in (0, 1)} for name in branches}
    missed_macros = [m for m, cov in macros.items() if not cov["union"]["1"]]
    candidate_macros = [m for m in missed_macros if any(p["macro"] == m and
                        p["best_neighbor_raw_rank"] <= 2 * kf for p in visible)]
    excess = duplicates["focal"]["1"]["excess_members"]
    # Ranking is an inspection ordering only, not a new scientific metric.
    actionable = bool(candidate_macros and excess)
    outside = {}
    bx, by, bw, bh = bbox
    for name, ids in branches.items():
        coords = coordinates(ids, width)
        xs = (coords[:, 1] + .5) / width * original_width
        ys = (coords[:, 0] + .5) / height * original_height
        outside[name] = float(((xs < bx) | (xs > bx + bw) | (ys < by) | (ys > by + bh)).mean()) if len(ids) else None
    return {"metrics": metrics, "parts": visible, "coverage": coverage, "macro_coverage": macros,
            "missed_macros": missed_macros, "candidate_missed_macros": candidate_macros,
            "ActionableRedundancyCase": actionable,
            "actionable_inspection_score": len(candidate_macros) * excess if actionable else 0,
            "bbox_outside_ratio": outside, "raw_topk_indices": top,
            "raw_topk_not_in_actual_focal": sorted(set(top) - set(focal)),
            "actual_focal_not_in_raw_topk": sorted(set(focal) - set(top))}, ranks
