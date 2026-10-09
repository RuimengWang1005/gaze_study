"""Load only the trained GAZE scorer and verify its frozen visual backbone."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Tuple

import torch
from safetensors import safe_open


SCORER_PREFIX = "visual.importance_scorer."
VISUAL_SENTINELS = (
    "visual.patch_embed.proj.weight",
    "visual.blocks.31.attn.qkv.weight",
    "visual.merger.mlp.2.weight",
)


def _weight_map(checkpoint: Path) -> Dict[str, str]:
    index = json.loads((checkpoint / "model.safetensors.index.json").read_text(encoding="utf-8"))
    return index["weight_map"]


def load_named_tensors(checkpoint: Path, names: Iterable[str]) -> Dict[str, torch.Tensor]:
    weight_map = _weight_map(checkpoint)
    names = list(names)
    by_file: Dict[str, list[str]] = {}
    for name in names:
        if name not in weight_map:
            raise KeyError(f"checkpoint tensor not found: {name}")
        by_file.setdefault(weight_map[name], []).append(name)
    tensors: Dict[str, torch.Tensor] = {}
    for filename, file_names in by_file.items():
        with safe_open(checkpoint / filename, framework="pt", device="cpu") as handle:
            for name in file_names:
                tensors[name] = handle.get_tensor(name)
    return tensors


def load_selector_scorer(checkpoint: Path, device: torch.device, dtype: torch.dtype):
    from token_compression.selector_scorer import TransformerScorer

    names = [
        SCORER_PREFIX + "k_proj.weight",
        SCORER_PREFIX + "k_proj.bias",
        SCORER_PREFIX + "q_proj.weight",
        SCORER_PREFIX + "q_proj.bias",
    ]
    tensors = load_named_tensors(checkpoint, names)
    in_features = int(tensors[SCORER_PREFIX + "k_proj.weight"].shape[1])
    hidden_dim = int(tensors[SCORER_PREFIX + "k_proj.weight"].shape[0])
    scorer = TransformerScorer(in_features=in_features, hidden_dim=hidden_dim)
    state = {name.removeprefix(SCORER_PREFIX): tensor for name, tensor in tensors.items()}
    scorer.load_state_dict(state, strict=True)
    return scorer.to(device=device, dtype=dtype).eval()


def verify_frozen_visual(model_visual, selector_checkpoint: Path) -> Dict[str, object]:
    """Compare representative beginning/end visual tensors exactly."""

    tensors = load_named_tensors(selector_checkpoint, VISUAL_SENTINELS)
    state = model_visual.state_dict()
    details = {}
    all_exact = True
    max_error = 0.0
    for full_name, checkpoint_tensor in tensors.items():
        model_name = full_name.removeprefix("visual.")
        model_tensor = state[model_name].detach().cpu()
        exact = torch.equal(model_tensor, checkpoint_tensor)
        error = float((model_tensor.float() - checkpoint_tensor.float()).abs().max().item())
        details[full_name] = {"exact": exact, "max_abs_error": error, "shape": list(model_tensor.shape)}
        all_exact = all_exact and exact
        max_error = max(max_error, error)
    return {"all_exact": all_exact, "max_abs_error": max_error, "tensors": details}
