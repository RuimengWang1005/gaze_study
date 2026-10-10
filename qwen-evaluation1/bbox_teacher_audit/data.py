"""Deterministic held-out Visual CoT sampling for the BBox teacher audit."""

from __future__ import annotations

import hashlib
import heapq
import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Mapping, Sequence, Tuple

import torch
from PIL import Image

from .token_mapping import foreground_indices, pixel_bboxs_to_merged_mask, validate_mapping


SOURCE_FILES = {
    "gqa": "gqa_cot_train.jsonl",
    "docvqa": "docvqa_cot_train.jsonl",
    "textvqa": "textvqa_cot_train.jsonl",
    "vsr": "vsr_cot_train.jsonl",
    "v7w": "visual7w_cot_train.jsonl",
    "infographicsvqa": "infographicsvqa_cot_train.jsonl",
    "sroie": "sroie_cot_train.jsonl",
    "flickr30k": "flickr30k_cot_train.jsonl",
    "dude": "dude_cot_train.jsonl",
}
TRAINING_SOURCE_COUNTS = {
    "gqa": 6316,
    "docvqa": 3158,
    "textvqa": 3158,
    "vsr": 2105,
    "v7w": 1053,
    "infographicsvqa": 1053,
    "sroie": 1053,
    "flickr30k": 1052,
    "dude": 1052,
}
SHORT_ANSWER_SUFFIX = "\nAnswer the question using a single word or phrase."


def proportional_quotas(total: int, weights: Mapping[str, int] = TRAINING_SOURCE_COUNTS) -> Dict[str, int]:
    """Largest-remainder allocation in stable source order."""

    weight_sum = sum(weights.values())
    raw = {source: total * weight / weight_sum for source, weight in weights.items()}
    quotas = {source: int(value) for source, value in raw.items()}
    remainder = total - sum(quotas.values())
    order = {source: index for index, source in enumerate(weights)}
    ranked = sorted(weights, key=lambda source: (-(raw[source] - quotas[source]), order[source]))
    for source in ranked[:remainder]:
        quotas[source] += 1
    return quotas


def _clean_question(value: object) -> str:
    text = str(value or "").strip()
    if text.startswith("<image>"):
        text = text[len("<image>") :].lstrip("\n")
    if text.endswith(SHORT_ANSWER_SUFFIX.strip("\n")):
        text = text[: -len(SHORT_ANSWER_SUFFIX.strip("\n"))].rstrip()
    return text


def canonical_key(record: Mapping[str, object]) -> Tuple[str, str, str, str]:
    images = record.get("images")
    image = images[0] if isinstance(images, list) and images else record.get("image", "")
    image = str(image).removeprefix("cot/")
    question = record.get("question") or record.get("instruction") or ""
    answer = record.get("answer")
    if answer is None:
        answer = record.get("output", "")
    return (
        str(record.get("dataset") or record.get("sub_dataset") or "unknown").strip(),
        image.strip(),
        _clean_question(question),
        str(answer).strip(),
    )


def sample_id(record: Mapping[str, object]) -> str:
    key = json.dumps(canonical_key(record), ensure_ascii=False, separators=(",", ":"))
    return f"{canonical_key(record)[0]}-{hashlib.sha256(key.encode()).hexdigest()[:16]}"


def _hash_rank(identifier: str, seed: int, purpose: str) -> str:
    return hashlib.sha256(f"{purpose}:{seed}:{identifier}".encode()).hexdigest()


def iter_jsonl(path: Path) -> Iterator[Tuple[int, Dict[str, object]]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                yield line_number, json.loads(line)


def load_training_keys(training_manifest: Path) -> set[Tuple[str, str, str, str]]:
    return {canonical_key(record) for _, record in iter_jsonl(training_manifest)}


def _candidate_record(
    raw: Mapping[str, object],
    source: str,
    source_path: Path,
    line_number: int,
    image_root: Path,
) -> Dict[str, object]:
    image = raw.get("image")
    if image is None:
        images = raw.get("images")
        image = images[0] if isinstance(images, list) and images else ""
    relative = str(image).removeprefix("cot/")
    if "/" not in relative:
        relative = f"{source}/{relative}"
    answer = raw.get("answer")
    if answer is None:
        answer = raw.get("output", "")
    record = {
        "dataset": str(raw.get("dataset") or raw.get("sub_dataset") or source),
        "source": source,
        "source_file": str(source_path),
        "source_line": line_number,
        "question": _clean_question(raw.get("question") or raw.get("instruction") or ""),
        "answer": str(answer).strip(),
        "image": relative,
        "image_path": str(image_root / relative),
        "bboxs": raw.get("bboxs"),
        "width": raw.get("width"),
        "height": raw.get("height"),
        "split": str(raw.get("split") or "unknown"),
    }
    record["sample_id"] = sample_id(record)
    return record


def _basic_reason(record: Mapping[str, object], training_keys: set) -> str | None:
    if not record["question"] or not record["answer"]:
        return "missing_question_or_answer"
    if canonical_key(record) in training_keys:
        return "overlap_with_training_20k"
    bboxs = record.get("bboxs")
    if not isinstance(bboxs, list) or not bboxs:
        return "missing_bbox"
    if record.get("width") is None or record.get("height") is None:
        return "missing_dimensions"
    return None


def build_candidate_pools(
    data_dir: Path,
    image_root: Path,
    training_manifest: Path,
    seed: int,
    pool_limit_per_source: int = 5000,
) -> Tuple[Dict[str, List[Dict[str, object]]], Counter, Dict[str, int], Dict[str, int]]:
    training_keys = load_training_keys(training_manifest)
    pools: Dict[str, List[Dict[str, object]]] = {}
    accepted_counts: Dict[str, int] = {}
    raw_counts: Dict[str, int] = {}
    exclusions: Counter = Counter()
    for source, filename in SOURCE_FILES.items():
        source_path = data_dir / filename
        accepted_heap, seen = [], set()
        accepted_count = 0
        raw_count = 0
        for line_number, raw in iter_jsonl(source_path):
            raw_count += 1
            record = _candidate_record(raw, source, source_path, line_number, image_root)
            reason = _basic_reason(record, training_keys)
            if reason:
                exclusions[reason] += 1
                continue
            identifier = str(record["sample_id"])
            if identifier in seen:
                exclusions["duplicate_candidate"] += 1
                continue
            seen.add(identifier)
            accepted_count += 1
            rank = int(_hash_rank(identifier, seed, "sufficiency"), 16)
            entry = (-rank, identifier, record)
            if len(accepted_heap) < pool_limit_per_source:
                heapq.heappush(accepted_heap, entry)
            elif rank < -accepted_heap[0][0]:
                heapq.heapreplace(accepted_heap, entry)
        accepted = [entry[2] for entry in accepted_heap]
        accepted.sort(key=lambda item: _hash_rank(str(item["sample_id"]), seed, "sufficiency"))
        pools[source] = accepted
        accepted_counts[source] = accepted_count
        raw_counts[source] = raw_count
    return pools, exclusions, accepted_counts, raw_counts


def validate_candidate(
    record: Mapping[str, object],
    image_processor,
    merge_size: int,
    min_bbox_tokens: int = 1,
) -> Tuple[Dict[str, object] | None, str | None]:
    if not Path(str(record["image_path"])).is_file():
        return None, "missing_image"
    try:
        with Image.open(str(record["image_path"])) as opened:
            image = opened.convert("RGB")
        expected = (int(record["width"]), int(record["height"]))
        if image.size != expected:
            return None, "metadata_image_size_mismatch"
        processed = image_processor(images=[image], return_tensors="pt")
        grid = processed["image_grid_thw"][0].cpu()
        mask = pixel_bboxs_to_merged_mask(
            record.get("bboxs"), record.get("width"), record.get("height"), grid, merge_size
        )
        if mask is None:
            return None, "bbox_mapping_empty"
        validate_mapping(mask, grid, merge_size)
        indices = foreground_indices(mask)
        if indices.numel() < min_bbox_tokens:
            return None, "bbox_token_count_too_small"
        output = dict(record)
        output.update(
            {
                "grid_thw": [int(value) for value in grid.tolist()],
                "full_visual_tokens": int(mask.numel()),
                "bbox_visual_tokens": int(indices.numel()),
                "bbox_token_indices": [int(value) for value in indices.tolist()],
                "held_out_from_training_20k": True,
            }
        )
        return output, None
    except Exception as error:  # recorded and replenished deterministically
        return None, f"image_or_processor_error:{type(error).__name__}"


def select_valid_samples(
    pools: Mapping[str, Sequence[Dict[str, object]]],
    image_processor,
    merge_size: int,
    quotas: Mapping[str, int] | None = None,
    min_bbox_tokens: int = 1,
) -> Tuple[List[Dict[str, object]], Counter, Dict[str, int]]:
    quotas = proportional_quotas(200) if quotas is None else quotas
    selected: List[Dict[str, object]] = []
    exclusions: Counter = Counter()
    examined: Dict[str, int] = {}
    for source, quota in quotas.items():
        if quota == 0:
            examined[source] = 0
            continue
        source_selected = 0
        for index, record in enumerate(pools[source], 1):
            validated, reason = validate_candidate(record, image_processor, merge_size, min_bbox_tokens)
            if reason:
                exclusions[reason] += 1
            else:
                selected.append(validated)
                source_selected += 1
            if source_selected == quota:
                examined[source] = index
                break
        if source_selected != quota:
            raise RuntimeError(f"only {source_selected}/{quota} valid samples for {source}")
    selected.sort(key=lambda item: (str(item["source"]), str(item["sample_id"])))
    return selected, exclusions, examined


def select_utility_subset(
    sufficiency_samples: Sequence[Dict[str, object]],
    seed: int,
    quotas: Mapping[str, int] | None = None,
) -> List[Dict[str, object]]:
    quotas = proportional_quotas(100) if quotas is None else quotas
    selected = []
    for source, quota in quotas.items():
        candidates = [
            record
            for record in sufficiency_samples
            if record["source"] == source and int(record["bbox_visual_tokens"]) >= 3
        ]
        candidates.sort(key=lambda item: _hash_rank(str(item["sample_id"]), seed, "utility"))
        if len(candidates) < quota:
            raise RuntimeError(f"only {len(candidates)}/{quota} utility-valid samples for {source}")
        selected.extend(candidates[:quota])
    selected.sort(key=lambda item: str(item["sample_id"]))
    if len(selected) != sum(quotas.values()):
        raise RuntimeError("utility subset quota could not be satisfied")
    return selected
