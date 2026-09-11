"""
Qwen2.5-VL Selector Model Inference Implementation
==================================================

专为推理优化的 VisionSelector 实现。
核心特性：
1. 硬剪枝 (Hard Pruning): 使用 Hard Top-K 物理删除不重要的 Token，实现显存节省和推理加速。
2. 安全的序列重组: 修复了变量作用域问题，确保多模态（图/文/视频）输入时的位置编码（RoPE）计算正确。

使用方法:
from selector_model import Qwen2_5_VLForConditionalGeneration_Selector
model = Qwen2_5_VLForConditionalGeneration_Selector.from_pretrained(...)
"""

from copy import deepcopy
from typing import List, Optional, Tuple, Union
import math
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import vmap
from torch.func import grad
from torch.autograd import Function
from transformers.utils import logging
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VLCausalLMOutputWithPast,
    Qwen2_5_VLModel,
    Qwen2_5_VLForConditionalGeneration
)
from qwen25vl.modeling_qwen2_5_vl import (
    Qwen2_5_VisionTransformerPretrainedModel,
    Qwen2_5_VisionPatchEmbed,
    Qwen2_5_VisionRotaryEmbedding,
    Qwen2_5_VLVisionBlock,
    Qwen2_5_VLPatchMerger,
)

from .selector_scorer import TransformerScorer

logger = logging.get_logger(__name__)

sigmoid = torch.sigmoid
sigmoid_grad = vmap(vmap(grad(sigmoid)))

class TopK(Function):
    @staticmethod
    def forward(ctx, xs, k):
        ts, ps = _find_ts(xs, k)
        ctx.save_for_backward(xs, ts)
        return ps

    @staticmethod
    def backward(ctx, grad_output):
        xs, ts = ctx.saved_tensors
        v = sigmoid_grad(xs + ts)
        s = v.sum(dim=1, keepdims=True)
        uv = grad_output * v
        t1 = - uv.sum(dim=1, keepdims=True) * v / s
        return t1 + uv, None

@torch.no_grad()
def _find_ts(xs, k):
    b, n = xs.shape
    assert 0 < k < n
    lo = -xs.max(dim=1, keepdims=True).values - 10
    hi = -xs.min(dim=1, keepdims=True).values + 10
    for _ in range(64):
        mid = (hi + lo) / 2
        mask = sigmoid(xs + mid).sum(dim=1) < k
        lo[mask] = mid[mask]
        hi[~mask] = mid[~mask]
    ts = (lo + hi) / 2
    return ts, sigmoid(xs + ts)

topk = TopK.apply


def _should_store_selector_visualization_attrs() -> bool:
    flag = os.environ.get("SELECTOR_STORE_VIS_ATTRS", "")
    return flag.lower() in {"1", "true", "yes", "on"}


def _normalize_device(device: Optional[Union[int, str, torch.device]]) -> Optional[torch.device]:
    if device is None:
        return None
    if isinstance(device, torch.device):
        return device
    if isinstance(device, int):
        return torch.device(f"cuda:{device}")
    return torch.device(device)


def _get_module_execution_device(module: Optional[nn.Module]) -> Optional[torch.device]:
    if module is None:
        return None

    hook = getattr(module, "_hf_hook", None)
    execution_device = _normalize_device(getattr(hook, "execution_device", None))
    if execution_device is not None and execution_device.type != "meta":
        return execution_device

    for tensor in module.parameters(recurse=False):
        if tensor.device.type != "meta":
            return tensor.device

    for tensor in module.buffers(recurse=False):
        if tensor.device.type != "meta":
            return tensor.device

    return None


def _move_optional_tensor(tensor: Optional[torch.Tensor], device: Optional[torch.device]) -> Optional[torch.Tensor]:
    if tensor is None or device is None or tensor.device == device:
        return tensor
    return tensor.to(device)


def _iter_cache_tensors(past_key_values):
    if past_key_values is None:
        return

    key_cache = getattr(past_key_values, "key_cache", None)
    value_cache = getattr(past_key_values, "value_cache", None)
    if isinstance(key_cache, list) and isinstance(value_cache, list):
        for cache_list in (key_cache, value_cache):
            for tensor in cache_list:
                if isinstance(tensor, torch.Tensor):
                    yield tensor
        return

    if isinstance(past_key_values, (list, tuple)):
        for layer_cache in past_key_values:
            if isinstance(layer_cache, torch.Tensor):
                yield layer_cache
            elif isinstance(layer_cache, (list, tuple)):
                for tensor in layer_cache:
                    if isinstance(tensor, torch.Tensor):
                        yield tensor


def _estimate_kv_cache_mb(past_key_values) -> float:
    total_bytes = 0
    for tensor in _iter_cache_tensors(past_key_values):
        total_bytes += tensor.numel() * tensor.element_size()
    return total_bytes / (1024 * 1024)


def _grid_thw_to_token_lengths(grid_thw: Optional[torch.Tensor], merge_size: int) -> List[int]:
    if grid_thw is None:
        return []
    if isinstance(grid_thw, torch.Tensor) and grid_thw.dim() == 1:
        grid_thw = grid_thw.unsqueeze(0)
    lengths: List[int] = []
    if not isinstance(grid_thw, torch.Tensor) or grid_thw.numel() == 0:
        return lengths
    for entry in grid_thw.to(dtype=torch.long):
        if entry.numel() < 3:
            continue
        t, h, w = entry[:3].tolist()
        h_m = max(int(h) // merge_size, 1)
        w_m = max(int(w) // merge_size, 1)
        lengths.append(int(max(int(t), 1) * h_m * w_m))
    return lengths


def _normalize_cosine_params(
    penalty_weight: float,
    similarity_threshold: float,
) -> Tuple[float, float]:
    weight = max(0.0, float(penalty_weight))
    threshold = max(-1.0, min(1.0, float(similarity_threshold)))
    return weight, threshold


def _robust_iqr(x: torch.Tensor) -> torch.Tensor:
    x = x.to(dtype=torch.float32)
    if x.numel() <= 1:
        return torch.zeros((), device=x.device, dtype=x.dtype)
    q1 = torch.quantile(x, 0.25)
    q3 = torch.quantile(x, 0.75)
    return torch.clamp(q3 - q1, min=0.0)


def _compute_scale_matched_lambda(
    scores: torch.Tensor,
    penalty_base: torch.Tensor,
    alpha: float,
    eps: float = 1e-6,
    max_lambda: float = 100.0,
) -> float:
    if alpha <= 0.0 or scores.numel() == 0 or penalty_base.numel() == 0:
        return 0.0

    score_scale = _robust_iqr(scores)
    if float(score_scale) <= eps:
        score_scale = torch.clamp(scores.max() - scores.min(), min=0.0)

    penalty_scale = _robust_iqr(penalty_base)
    if float(penalty_scale) <= eps:
        penalty_scale = torch.clamp(penalty_base.max() - penalty_base.min(), min=0.0)

    if float(penalty_scale) <= eps:
        return 0.0

    lam = alpha * (score_scale / (penalty_scale + eps))
    lam = torch.clamp(lam, min=0.0, max=max_lambda)
    return float(lam.item())


@torch.no_grad()
def _select_indices_with_cosine_penalty(
    scores: torch.Tensor,
    token_features: Optional[torch.Tensor],
    k: int,
    penalty_weight: float,
    similarity_threshold: float,
    reference_features: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    total = int(scores.numel())
    if total == 0 or k <= 0:
        return torch.empty((0,), dtype=torch.long, device=scores.device)

    k = min(int(k), total)
    if k >= total:
        return torch.arange(total, dtype=torch.long, device=scores.device)

    penalty_weight, similarity_threshold = _normalize_cosine_params(
        penalty_weight, similarity_threshold
    )

    if (
        penalty_weight <= 0.0
        or token_features is None
        or token_features.dim() != 2
        or token_features.shape[0] != total
    ):
        return torch.topk(scores, k=k, dim=0, largest=True).indices

    feat = F.normalize(
        token_features.to(dtype=torch.float32),
        p=2,
        dim=-1,
        eps=1e-6,
    )
    if feat is None:
        return torch.topk(scores, k=k, dim=0, largest=True).indices
    base_scores = scores.to(dtype=torch.float32)
    max_sim = torch.zeros(total, device=scores.device, dtype=torch.float32)

    if (
        reference_features is not None
        and isinstance(reference_features, torch.Tensor)
        and reference_features.dim() == 2
        and reference_features.numel() > 0
        and reference_features.shape[1] == feat.shape[1]
    ):
        ref_feat = F.normalize(
            reference_features.to(dtype=torch.float32),
            p=2,
            dim=-1,
            eps=1e-6,
        )
        max_sim = torch.matmul(feat, ref_feat.transpose(0, 1)).max(dim=1).values

    selected_mask = torch.zeros(total, dtype=torch.bool, device=scores.device)
    selected: List[int] = []
    has_reference = (
        reference_features is not None
        and isinstance(reference_features, torch.Tensor)
        and reference_features.dim() == 2
        and reference_features.numel() > 0
        and reference_features.shape[1] == feat.shape[1]
    )
    lambda_eff = 0.0

    # Single-shot adaptive lambda: calibrate once per sample, then keep it fixed
    # while the greedy loop updates only the similarity penalty.
    if has_reference:
        penalty_base = torch.relu(max_sim - similarity_threshold)
        lambda_eff = _compute_scale_matched_lambda(
            base_scores,
            penalty_base,
            alpha=penalty_weight,
        )
    else:
        seed_idx = int(torch.argmax(base_scores).item())
        selected_mask[seed_idx] = True
        selected.append(seed_idx)
        if len(selected) < k:
            sim_to_new = torch.matmul(feat, feat[seed_idx])
            max_sim = torch.maximum(max_sim, sim_to_new)
            penalty_base = torch.relu(max_sim - similarity_threshold)
            candidate_mask = ~selected_mask
            lambda_eff = _compute_scale_matched_lambda(
                base_scores[candidate_mask],
                penalty_base[candidate_mask],
                alpha=penalty_weight,
            )

    while len(selected) < k:
        penalty_base = torch.relu(max_sim - similarity_threshold)
        penalized = base_scores - lambda_eff * penalty_base
        penalized = penalized.masked_fill(selected_mask, float("-inf"))

        next_idx = int(torch.argmax(penalized).item())
        if selected_mask[next_idx]:
            break
        selected_mask[next_idx] = True
        selected.append(next_idx)
        if len(selected) >= k:
            break

        sim_to_new = torch.matmul(feat, feat[next_idx])
        max_sim = torch.maximum(max_sim, sim_to_new)

    if len(selected) < k:
        remaining = torch.where(~selected_mask)[0]
        if remaining.numel() > 0:
            need = min(k - len(selected), int(remaining.numel()))
            top_rel = torch.topk(base_scores[remaining], k=need, largest=True).indices
            selected.extend(remaining[top_rel].tolist())

    if not selected:
        return torch.empty((0,), dtype=torch.long, device=scores.device)
    return torch.tensor(selected, dtype=torch.long, device=scores.device)


class Qwen2_5_VisionTransformerPretrainedModel_Selector(Qwen2_5_VisionTransformerPretrainedModel):
    """
    推理专用：带有硬剪枝功能的视觉 Transformer
    """
    
    def __init__(self, config, *inputs, **kwargs) -> None:
        super().__init__(config, *inputs, **kwargs)
        
        # === 初始化标准组件 ===
        self.spatial_merge_size = config.spatial_merge_size
        self.patch_size = config.patch_size
        self.fullatt_block_indexes = config.fullatt_block_indexes
        self.window_size = config.window_size
        self.spatial_merge_unit = self.spatial_merge_size * self.spatial_merge_size

        self.patch_embed = Qwen2_5_VisionPatchEmbed(
            patch_size=config.patch_size,
            temporal_patch_size=config.temporal_patch_size,
            in_channels=config.in_channels,
            embed_dim=config.hidden_size,
        )

        head_dim = config.hidden_size // config.num_heads
        self.rotary_pos_emb = Qwen2_5_VisionRotaryEmbedding(head_dim // 2)

        self.blocks = nn.ModuleList(
            [Qwen2_5_VLVisionBlock(config, config._attn_implementation, layer_idx=i) for i in range(config.depth)]
        )
        
        self.merger = Qwen2_5_VLPatchMerger(
            dim=config.out_hidden_size,
            context_dim=config.hidden_size,
            spatial_merge_size=config.spatial_merge_size,
        )
        
        # === 初始化评分器 ===
        # 加载训练好的权重
        self.importance_scorer = TransformerScorer(
            in_features=config.out_hidden_size,
            hidden_dim=config.out_hidden_size//2
        )
        # 读取保留比例 (Budget)
        self.budgets = config.budgets if hasattr(config, 'budgets') else 0.2

        # Focal/spatial split (ratio among kept tokens).
        spatial_ratio = getattr(config, "spatial_ratio", None)
        if spatial_ratio is None:
            spatial_ratio = getattr(config, "contextual_ratio", None)
        if spatial_ratio is None:
            spatial_ratio = os.getenv("SELECTOR_SPATIAL_RATIO", os.getenv("SELECTOR_CONTEXTUAL_RATIO", "0.1"))
        self.spatial_ratio = float(spatial_ratio)
        self.contextual_ratio = self.spatial_ratio
        self.cosine_penalty_weight = max(
            0.0,
            float(
                getattr(
                    config,
                    "cosine_penalty_weight",
                    float(os.getenv("SELECTOR_COSINE_PENALTY_WEIGHT", "0.15")),
                )
            ),
        )
        self.cosine_penalty_threshold = max(
            -1.0,
            min(
                1.0,
                float(
                    getattr(
                        config,
                        "cosine_penalty_threshold",
                        float(os.getenv("SELECTOR_COSINE_PENALTY_THRESHOLD", "0.85")),
                    )
                ),
            ),
        )
        self.selector_focal_redundancy_limit = max(
            0,
            int(
                float(
                    getattr(
                        config,
                        "selector_focal_redundancy_limit",
                        float(os.getenv("SELECTOR_FOCAL_REDUNDANCY_LIMIT", "0")),
                    )
                )
            ),
        )
        # === 可视化属性初始化 ===
        self.last_combined_scores = None
        self.last_grid_thw = None
        self.last_selected_indices = None
        self.last_focal_indices = None
        self.last_spatial_indices = None
        self.last_dominant_indices = None
        self.last_contextual_indices = None
        self.last_token_features = None
        self.last_selected_features = None
        self.last_selected_scores = None
        self.last_selected_labels = None
        self.last_focal_features = None
        self.last_spatial_features = None
        self.last_vision_encoder_time_ms = None
        self.last_selector_score_select_time_ms = None
        self._last_vision_encoder_start_event = None
        self._last_vision_encoder_end_event = None
        self._last_selector_start_event = None
        self._last_selector_end_event = None

    def forward(
        self, 
        hidden_states: torch.Tensor, 
        grid_thw: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        timing_enabled = os.environ.get("EVAL_TIME", "").lower() == "true"
        use_cuda_timing = bool(timing_enabled and hidden_states.is_cuda)
        vision_encoder_start_time = None
        vision_encoder_end_time = None
        selector_start_time = None

        self.last_vision_encoder_time_ms = None
        self.last_selector_score_select_time_ms = None
        self._last_vision_encoder_start_event = None
        self._last_vision_encoder_end_event = None
        self._last_selector_start_event = None
        self._last_selector_end_event = None

        if use_cuda_timing:
            self._last_vision_encoder_start_event = torch.cuda.Event(enable_timing=True)
            self._last_vision_encoder_start_event.record()
        elif timing_enabled:
            vision_encoder_start_time = time.perf_counter()
        
        # === 1. 标准 Vision Encoder 流程 (直到 Merger) ===
        hidden_states = self.patch_embed(hidden_states)
        if grid_thw.device != hidden_states.device:
            grid_thw = grid_thw.to(hidden_states.device)
        rotary_pos_emb = self.rot_pos_emb(grid_thw)
        
        window_index, cu_window_seqlens = self.get_window_index(grid_thw)
        cu_window_seqlens = torch.tensor(
            cu_window_seqlens,
            device=hidden_states.device,
            dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
        )
        cu_window_seqlens = torch.unique_consecutive(cu_window_seqlens)

        seq_len, _ = hidden_states.size()
        hidden_states = hidden_states.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
        hidden_states = hidden_states[window_index, :, :]
        hidden_states = hidden_states.reshape(seq_len, -1)
        
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
        rotary_pos_emb = rotary_pos_emb[window_index, :, :]
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())

        cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
            dim=0,
            dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
        )
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)
        
        for layer_num, blk in enumerate(self.blocks):
            if layer_num in self.fullatt_block_indexes:
                cu_seqlens_now = cu_seqlens
            else:
                cu_seqlens_now = cu_window_seqlens
            
            hidden_states = blk(hidden_states, cu_seqlens=cu_seqlens_now, position_embeddings=position_embeddings)

        hidden_states = self.merger(hidden_states)
        
        reverse_indices = torch.argsort(window_index)
        hidden_states = hidden_states[reverse_indices, :]

        if use_cuda_timing:
            self._last_vision_encoder_end_event = torch.cuda.Event(enable_timing=True)
            self._last_vision_encoder_end_event.record()
            self._last_selector_start_event = torch.cuda.Event(enable_timing=True)
            self._last_selector_start_event.record()
        elif timing_enabled:
            vision_encoder_end_time = time.perf_counter()
            selector_start_time = vision_encoder_end_time
        
        # === 2. 推理阶段的核心修改：硬剪枝 ===
        
        total_token_num = hidden_states.shape[0]
        focal_indices = None
        spatial_indices = None
        
        # hidden_states: [N, D] -> unsqueeze -> [1, N, D]
        hidden_states_unsqueezed = hidden_states.unsqueeze(0) 
        
        learned_scores = self.importance_scorer(hidden_states_unsqueezed).squeeze(0) # [N]
        
        # 根据预算计算保留数量 k
        k = max(1, int(total_token_num * self.budgets))

        if k >= total_token_num or total_token_num <= 1:
            all_indices = torch.arange(total_token_num, device=hidden_states.device)
            hidden_states_new = hidden_states
            focal_indices = all_indices
            spatial_indices = torch.empty((0,), dtype=torch.long, device=hidden_states.device)
        else:
            r = max(0.0, min(float(getattr(self, "spatial_ratio", getattr(self, "contextual_ratio", 0.1))), 1.0))
            spa_budget = int(round(k * r))
            spa_budget = max(0, min(spa_budget, k))
            k_foc = k - spa_budget
            if k_foc <= 0:
                k_foc = 1
                spa_budget = k - k_foc

            cosine_penalty_weight, cosine_penalty_threshold = _normalize_cosine_params(
                getattr(self, "cosine_penalty_weight", 0.0),
                getattr(self, "cosine_penalty_threshold", 0.85),
            )
            focal_redundancy_limit = max(
                0, int(getattr(self, "selector_focal_redundancy_limit", 0))
            )
            use_limited_focal_redundancy = (
                cosine_penalty_weight > 0.0
                and focal_redundancy_limit > 0
                and focal_redundancy_limit < k_foc
            )

            if use_limited_focal_redundancy:
                focal_seed_k = min(k_foc, focal_redundancy_limit)
                focal_indices = _select_indices_with_cosine_penalty(
                    learned_scores,
                    hidden_states,
                    focal_seed_k,
                    cosine_penalty_weight,
                    cosine_penalty_threshold,
                )
                if focal_indices.numel() != focal_seed_k:
                    focal_indices = torch.topk(learned_scores, focal_seed_k, dim=0).indices

                if focal_indices.numel() < k_foc:
                    selected_mask = torch.zeros(total_token_num, dtype=torch.bool, device=hidden_states.device)
                    selected_mask.scatter_(0, focal_indices, True)
                    remaining = torch.where(~selected_mask)[0]
                    need = min(k_foc - int(focal_indices.numel()), int(remaining.numel()))
                    if need > 0:
                        top_rel = torch.topk(
                            learned_scores[remaining],
                            k=need,
                            dim=0,
                            largest=True,
                        ).indices
                        focal_indices = torch.cat([focal_indices, remaining[top_rel]], dim=0)
            else:
                focal_indices = _select_indices_with_cosine_penalty(
                    learned_scores,
                    hidden_states,
                    k_foc,
                    cosine_penalty_weight,
                    cosine_penalty_threshold,
                )
            if focal_indices.numel() != k_foc:
                focal_indices = torch.topk(learned_scores, k_foc, dim=0).indices
            focal_indices, _ = torch.sort(focal_indices)

            if spa_budget == 0:
                all_indices = focal_indices
                hidden_states_new = hidden_states[all_indices, :]
                spatial_indices = torch.empty((0,), dtype=torch.long, device=hidden_states.device)
            else:
                device = hidden_states.device
                merge_size = max(1, int(getattr(self, "spatial_merge_size", 2)))
                is_focal = torch.zeros(total_token_num, dtype=torch.bool, device=device)
                is_focal.scatter_(0, focal_indices, True)

                grid = grid_thw
                if isinstance(grid, torch.Tensor) and grid.dim() == 1:
                    grid = grid.unsqueeze(0)

                token_lens = _grid_thw_to_token_lengths(grid, merge_size)
                grid_meta: List[Tuple[int, int, int, int]] = []
                if isinstance(grid, torch.Tensor) and grid.numel() > 0 and grid.size(-1) >= 3:
                    for entry in grid.to(dtype=torch.long):
                        t, h, w = entry[:3].tolist()
                        h_m = max(int(h) // merge_size, 1)
                        w_m = max(int(w) // merge_size, 1)
                        token_len = int(max(int(t), 1) * h_m * w_m)
                        grid_meta.append((int(t), h_m, w_m, token_len))

                use_grid = (len(token_lens) > 0 and sum(token_lens) == total_token_num)
                spatial_indices_list: List[torch.Tensor] = []

                if use_grid:
                    offsets: List[int] = []
                    off = 0
                    for tl in token_lens:
                        offsets.append(off)
                        off += tl

                    capacities = []
                    frac_parts = []
                    base_counts = []
                    for off, tl in zip(offsets, token_lens):
                        foc_count = int(is_focal[off : off + tl].sum().item())
                        cap = max(tl - foc_count, 0)
                        capacities.append(cap)
                        target = (spa_budget * float(tl) / float(total_token_num)) if total_token_num > 0 else 0.0
                        base = int(math.floor(target))
                        base = min(base, cap)
                        base_counts.append(base)
                        frac_parts.append(target - float(base))

                    spa_per_img = list(base_counts)
                    remain = spa_budget - sum(spa_per_img)
                    order = sorted(range(len(frac_parts)), key=lambda i: frac_parts[i], reverse=True)
                    for img_idx in order:
                        if remain <= 0:
                            break
                        if spa_per_img[img_idx] < capacities[img_idx]:
                            spa_per_img[img_idx] += 1
                            remain -= 1
                    if remain > 0:
                        for img_idx in range(len(token_lens)):
                            if remain <= 0:
                                break
                            addable = max(capacities[img_idx] - spa_per_img[img_idx], 0)
                            if addable <= 0:
                                continue
                            add = min(addable, remain)
                            spa_per_img[img_idx] += add
                            remain -= add

                    for img_idx, (off, tl, meta) in enumerate(zip(offsets, token_lens, grid_meta)):
                        spa_i = int(spa_per_img[img_idx])
                        if spa_i <= 0 or tl <= 0:
                            continue

                        t_i, h_m, w_m, _ = meta
                        h_tot = max(int(t_i) * int(h_m), 1)
                        w_tot = max(int(w_m), 1)

                        local_features = hidden_states[off : off + tl]
                        non_foc_mask = ~is_focal[off : off + tl]
                        non_foc_within = torch.where(non_foc_mask)[0]
                        if non_foc_within.numel() == 0:
                            continue

                        scores_non_foc = learned_scores[off : off + tl][non_foc_within]
                        features_non_foc = local_features[non_foc_within]
                        foc_features_img = local_features[is_focal[off : off + tl]]

                        grid_cols = max(1, int(math.ceil(math.sqrt(spa_i * (float(w_tot) / float(h_tot))))))
                        grid_rows = max(1, int(math.ceil(float(spa_i) / float(grid_cols))))
                        row_step = max(1, int(math.ceil(float(h_tot) / float(grid_rows))))
                        col_step = max(1, int(math.ceil(float(w_tot) / float(grid_cols))))

                        row = (non_foc_within // w_tot).clamp(min=0, max=h_tot - 1)
                        col = (non_foc_within % w_tot).clamp(min=0, max=w_tot - 1)
                        cell_r = (row // row_step).clamp(min=0, max=grid_rows - 1)
                        cell_c = (col // col_step).clamp(min=0, max=grid_cols - 1)
                        cell_id = cell_r * grid_cols + cell_c

                        keep_rel = torch.zeros(non_foc_within.shape[0], dtype=torch.bool, device=device)
                        num_cells = grid_rows * grid_cols

                        candidate_rel_indices = []
                        for cell in range(num_cells):
                            rel_idxs = torch.where(cell_id == cell)[0]
                            if rel_idxs.numel() == 0:
                                continue
                            best_rel = rel_idxs[scores_non_foc[rel_idxs].argmax()]
                            candidate_rel_indices.append(best_rel)

                        if candidate_rel_indices:
                            candidate_tensor = torch.stack(candidate_rel_indices)
                            if candidate_tensor.numel() > spa_i:
                                candidate_scores = scores_non_foc[candidate_tensor]
                                candidate_features = features_non_foc[candidate_tensor]
                                keep_rel_local = _select_indices_with_cosine_penalty(
                                    candidate_scores,
                                    candidate_features,
                                    spa_i,
                                    cosine_penalty_weight,
                                    cosine_penalty_threshold,
                                    reference_features=foc_features_img,
                                )
                                if keep_rel_local.numel() != spa_i:
                                    keep_rel_local = torch.topk(candidate_scores, spa_i, largest=True).indices
                                final_candidates = candidate_tensor[keep_rel_local]
                            else:
                                final_candidates = candidate_tensor

                            keep_rel[final_candidates] = True

                        selected_within = non_foc_within[keep_rel]
                        if selected_within.numel() < spa_i:
                            remaining_rel = torch.where(~keep_rel)[0]
                            need = spa_i - int(selected_within.numel())
                            if remaining_rel.numel() > 0 and need > 0:
                                need = min(need, int(remaining_rel.numel()))
                                remaining_scores = scores_non_foc[remaining_rel]
                                remaining_features = features_non_foc[remaining_rel]

                                ref_features = foc_features_img
                                kept_rel = torch.where(keep_rel)[0]
                                if kept_rel.numel() > 0:
                                    kept_features = features_non_foc[kept_rel]
                                    if ref_features is None or ref_features.numel() == 0:
                                        ref_features = kept_features
                                    else:
                                        ref_features = torch.cat([ref_features, kept_features], dim=0)

                                topk_rel = _select_indices_with_cosine_penalty(
                                    remaining_scores,
                                    remaining_features,
                                    need,
                                    cosine_penalty_weight,
                                    cosine_penalty_threshold,
                                    reference_features=ref_features,
                                )
                                if topk_rel.numel() != need:
                                    topk_rel = torch.topk(remaining_scores, k=need, largest=True).indices
                                selected_within = torch.cat(
                                    [selected_within, non_foc_within[remaining_rel[topk_rel]]], dim=0
                                )

                        if selected_within.numel() > spa_i:
                            selected_scores = learned_scores[off : off + tl][selected_within]
                            selected_features = hidden_states[off : off + tl][selected_within]
                            topk_rel = _select_indices_with_cosine_penalty(
                                selected_scores,
                                selected_features,
                                spa_i,
                                cosine_penalty_weight,
                                cosine_penalty_threshold,
                                reference_features=foc_features_img,
                            )
                            if topk_rel.numel() != spa_i:
                                topk_rel = torch.topk(selected_scores, k=spa_i, largest=True).indices
                            selected_within = selected_within[topk_rel]

                        spatial_indices_list.append(off + selected_within[:spa_i])

                if spatial_indices_list:
                    spatial_indices = torch.cat(spatial_indices_list, dim=0)
                else:
                    spatial_indices = torch.empty((0,), dtype=torch.long, device=device)

                if spatial_indices.numel() < spa_budget:
                    is_spatial = torch.zeros(total_token_num, dtype=torch.bool, device=device)
                    if spatial_indices.numel() > 0:
                        is_spatial.scatter_(0, spatial_indices, True)
                    cand = torch.where((~is_focal) & (~is_spatial))[0]
                    need = spa_budget - int(spatial_indices.numel())
                    if cand.numel() > 0 and need > 0:
                        need = min(need, int(cand.numel()))
                        ref_features = hidden_states[focal_indices]
                        if spatial_indices.numel() > 0:
                            ref_features = torch.cat([ref_features, hidden_states[spatial_indices]], dim=0)
                        extra = _select_indices_with_cosine_penalty(
                            learned_scores[cand],
                            hidden_states[cand],
                            need,
                            cosine_penalty_weight,
                            cosine_penalty_threshold,
                            reference_features=ref_features,
                        )
                        if extra.numel() != need:
                            extra = torch.topk(
                                learned_scores[cand],
                                k=need,
                                largest=True,
                            ).indices
                        spatial_indices = torch.cat([spatial_indices, cand[extra]], dim=0)

                if spatial_indices.numel() > spa_budget:
                    topm = _select_indices_with_cosine_penalty(
                        learned_scores[spatial_indices],
                        hidden_states[spatial_indices],
                        spa_budget,
                        cosine_penalty_weight,
                        cosine_penalty_threshold,
                        reference_features=hidden_states[focal_indices] if focal_indices.numel() > 0 else None,
                    )
                    if topm.numel() != spa_budget:
                        topm = torch.topk(learned_scores[spatial_indices], k=spa_budget, largest=True).indices
                    spatial_indices = spatial_indices[topm]
                spatial_indices, _ = torch.sort(spatial_indices)

                all_indices = torch.cat([focal_indices, spatial_indices], dim=0)
                all_indices, _ = torch.sort(all_indices)
                hidden_states_new = hidden_states[all_indices, :]
        
        # === 保存可视化属性 ===
        # These tensors are only needed for explicit visualization. Avoid forcing
        # extra TopK work and GPU->CPU sync on every evaluation sample.
        if _should_store_selector_visualization_attrs():
            k_vis = int(all_indices.numel()) if isinstance(all_indices, torch.Tensor) else int(k)
            if k_vis <= 0 or k_vis >= int(total_token_num):
                soft_scores = torch.ones_like(learned_scores)
            else:
                soft_scores = topk(learned_scores.unsqueeze(0), k_vis).squeeze(0)
            self.last_combined_scores = soft_scores.detach()
            self.last_grid_thw = grid_thw.detach() if isinstance(grid_thw, torch.Tensor) else grid_thw
            self.last_selected_indices = all_indices.detach()
            if focal_indices is None:
                focal_indices = torch.empty((0,), dtype=torch.long, device=hidden_states.device)
            if spatial_indices is None:
                spatial_indices = torch.empty((0,), dtype=torch.long, device=hidden_states.device)
            self.last_focal_indices = focal_indices.detach()
            self.last_spatial_indices = spatial_indices.detach()
            self.last_dominant_indices = self.last_focal_indices
            self.last_contextual_indices = self.last_spatial_indices
            self.last_token_features = hidden_states.detach()
            self.last_selected_features = hidden_states[all_indices, :].detach()
            self.last_selected_scores = learned_scores[all_indices].detach()
            selected_labels = torch.full(
                (all_indices.numel(),),
                1,
                dtype=torch.long,
                device=hidden_states.device,
            )
            if focal_indices.numel() > 0:
                selected_labels[torch.isin(all_indices, focal_indices)] = 0
            self.last_selected_labels = selected_labels.detach()
            self.last_focal_features = hidden_states[focal_indices, :].detach()
            self.last_spatial_features = hidden_states[spatial_indices, :].detach()
        else:
            self.last_combined_scores = None
            self.last_grid_thw = None
            self.last_selected_indices = None
            self.last_focal_indices = None
            self.last_spatial_indices = None
            self.last_dominant_indices = None
            self.last_contextual_indices = None
            self.last_token_features = None
            self.last_selected_features = None
            self.last_selected_scores = None
            self.last_selected_labels = None
            self.last_focal_features = None
            self.last_spatial_features = None
        # Debug: 打印网格与分数信息，便于检查是否偏向边缘
        if os.environ.get("SELECTOR_DEBUG"):
            grid_cpu = grid_thw.detach().cpu()
            scores_cpu = learned_scores.detach().cpu()

            # 计算每张图的 patch 数
            patch_counts = []
            for g in grid_cpu:
                if g.numel() >= 3:
                    t, h, w = g.tolist()[:3]
                    patch_counts.append(int(t * h * w))
                else:
                    patch_counts.append(int(torch.prod(g).item()))
            patch_sum = sum(patch_counts)

            def idx_to_coord(idx: int):
                acc = 0
                for img_i, pc in enumerate(patch_counts):
                    if idx < acc + pc:
                        rel = idx - acc
                        if grid_cpu[img_i].numel() >= 3:
                            h_dim = int(grid_cpu[img_i][1])
                            w_dim = int(grid_cpu[img_i][2])
                            row = rel // w_dim
                            col = rel % w_dim
                            return (img_i, row, col)
                        return (img_i, rel, 0)
                    acc += pc
                return (-1, -1, -1)

            top_k = min(10, scores_cpu.numel())
            top_idx = torch.topk(scores_cpu, top_k).indices.tolist()
            bot_idx = torch.topk(-scores_cpu, top_k).indices.tolist()

            print(f"[SELECTOR_DEBUG] grid_thw={grid_cpu.tolist()}, patch_tokens={total_token_num}, "
                  f"grid_patch_sum={patch_sum}, selected={len(all_indices)}, "
                  f"cos_w={float(getattr(self, 'cosine_penalty_weight', 0.0)):.3f}, "
                  f"cos_thr={float(getattr(self, 'cosine_penalty_threshold', 0.85)):.2f}")
            print("[SELECTOR_DEBUG] top scores idx->(img,row,col):",
                  [(i, idx_to_coord(i)) for i in top_idx])
            print("[SELECTOR_DEBUG] bottom scores idx->(img,row,col):",
                  [(i, idx_to_coord(i)) for i in bot_idx])

        if use_cuda_timing:
            self._last_selector_end_event = torch.cuda.Event(enable_timing=True)
            self._last_selector_end_event.record()
        elif timing_enabled:
            selector_end_time = time.perf_counter()
            self.last_vision_encoder_time_ms = (vision_encoder_end_time - vision_encoder_start_time) * 1000.0
            self.last_selector_score_select_time_ms = (selector_end_time - selector_start_time) * 1000.0
        
        return hidden_states_new, all_indices, total_token_num


class Qwen2_5_VLForConditionalGeneration_Selector(Qwen2_5_VLForConditionalGeneration):
    """
    推理专用生成模型，集成序列重组逻辑
    """
    
    def __init__(self, config):
        super().__init__(config)
        # 使用推理专用的 Vision Tower
        self.visual = Qwen2_5_VisionTransformerPretrainedModel_Selector._from_config(config.vision_config)
        
        self.model = Qwen2_5_VLModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.rope_deltas = None
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
    ) -> Union[Tuple, Qwen2_5_VLCausalLMOutputWithPast]:
        try:
            has_eval_time = os.environ["EVAL_TIME"]
        except KeyError:
            has_eval_time = None
        timing_enabled = bool(has_eval_time and os.environ["EVAL_TIME"].lower() == "true")
        use_cuda_timing = bool(timing_enabled and torch.cuda.is_available())
        if use_cuda_timing:
            start = torch.cuda.Event(enable_timing=True)
            start.record()
        elif timing_enabled:
            prefill_start_time = time.perf_counter()

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        visual_token_num = 0
        retained_visual_tokens = 0
        vision_encoder_time_ms = 0.0
        selector_score_select_time_ms = 0.0
        llm_prefill_time_ms = 0.0
        selector_apply_event_pairs = []
        selector_apply_time_ms = 0.0
        llm_prefill_start = None
        llm_prefill_end = None
        llm_prefill_start_time = None

        # With Accelerate dispatch, module weights can stay on `meta` until the hook
        # materializes them on the execution device. Use the hook target instead of the
        # parameter device to avoid accidentally moving inputs onto `meta`.
        model_device = inputs_embeds.device if inputs_embeds is not None else _get_module_execution_device(self.model.embed_tokens)
        input_ids = _move_optional_tensor(input_ids, model_device)
        attention_mask = _move_optional_tensor(attention_mask, model_device)
        position_ids = _move_optional_tensor(position_ids, model_device)
        cache_position = _move_optional_tensor(cache_position, model_device)
        labels = _move_optional_tensor(labels, model_device)
        rope_deltas = _move_optional_tensor(rope_deltas, model_device)
        image_grid_thw = _move_optional_tensor(image_grid_thw, model_device)
        video_grid_thw = _move_optional_tensor(video_grid_thw, model_device)
        pixel_values = _move_optional_tensor(pixel_values, model_device)
        pixel_values_videos = _move_optional_tensor(pixel_values_videos, model_device)
        if isinstance(second_per_grid_ts, torch.Tensor):
            second_per_grid_ts = _move_optional_tensor(second_per_grid_ts, model_device)

        if inputs_embeds is None:
            # 1. 获取所有 Token 的 Embedding
            inputs_embeds = self.model.embed_tokens(input_ids)
            
            # [Safe Init]: 初始化 origin_input_ids、origin_attention_mask 和 selected_indices
            # 无论是否发生剪枝，这些变量在后续 RoPE 计算中都是必须的
            # 使用 input_ids 和 attention_mask 的初始状态作为基准
            origin_input_ids = input_ids 
            origin_attention_mask = deepcopy(attention_mask) if attention_mask is not None else None
            selected_indices = None # None 表示未发生剪枝

            # ==================== 图像处理 ====================
            if pixel_values is not None:
                pixel_values = pixel_values.type(self.visual.dtype)
                
                # 显式保存副本，因为 input_ids 马上要被切片了
                # 这里的 origin_input_ids 将被用于后续 RoPE 计算
                origin_input_ids = deepcopy(input_ids)

                # 对齐检查：<image> 占位数量（patch级别）与 grid_thw 图像数可能不等，这是正常现象
                origin_image_indices = torch.where(input_ids == self.config.image_token_id)[1]
                # 完全禁用警告输出，避免日志噪音
                # if image_grid_thw is not None and image_grid_thw.shape[0] != origin_image_indices.shape[0]:
                #     logger.warning(
                #         "image count mismatch: patch_tokens=%d, grid_imgs=%d; proceeding with patch count",
                #         origin_image_indices.shape[0], image_grid_thw.shape[0]
                #     )
                
                # [调用]: Vision Tower
                image_embeds, all_indices, visual_token_num = self.visual(
                    pixel_values, 
                    grid_thw=image_grid_thw,
                )
                if timing_enabled:
                    if use_cuda_timing and image_embeds.is_cuda:
                        selector_apply_start = torch.cuda.Event(enable_timing=True)
                        selector_apply_start.record()
                    else:
                        selector_apply_start_time = time.perf_counter()
                
                # [序列重组]: 根据选中的 indices 物理缩短 input_ids
                retain_image_indices = origin_image_indices[all_indices] # 只保留选中的图片 Token 位置
                origin_text_indices = torch.where(input_ids != self.config.image_token_id)[1] # 所有文本位置保留
                
                # 合并索引并排序
                combined_indices = torch.cat((retain_image_indices, origin_text_indices))
                selected_indices, _ = torch.sort(combined_indices)
                
                # [剪枝]: 物理切片 input_ids 和 inputs_embeds
                input_ids = input_ids[:, selected_indices]
                inputs_embeds = inputs_embeds[:, selected_indices, :]

                # 将计算好的 visual features 填入对应的占位符
                # 此时 input_ids 里的 image_token_id 数量已经等于 image_embeds 的长度
                mask = input_ids == self.config.image_token_id
                mask_expanded = mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
                image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                retained_visual_tokens += int(image_embeds.shape[0])
                
                inputs_embeds = inputs_embeds.masked_scatter(mask_expanded, image_embeds)
                if timing_enabled:
                    if use_cuda_timing and image_embeds.is_cuda:
                        selector_apply_end = torch.cuda.Event(enable_timing=True)
                        selector_apply_end.record()
                        selector_apply_event_pairs.append((selector_apply_start, selector_apply_end))
                    else:
                        selector_apply_time_ms += (time.perf_counter() - selector_apply_start_time) * 1000.0

            # ==================== 视频处理 ====================
            if pixel_values_videos is not None:
                pixel_values_videos = pixel_values_videos.type(self.visual.dtype)
                
                # 显式保存副本 (如果之前已经处理过图像，这里保存的是处理后的 input_ids)
                # 注意：如果同时存在图和视频，RoPE 计算会基于最后一次保存的 origin_input_ids
                # 在标准 Qwen2.5-VL 处理流中，通常图/视频是互斥或顺序处理的
                origin_input_ids = deepcopy(input_ids)
                
                # [调用]: Vision Tower
                video_embeds, all_indices, visual_token_num = self.visual(
                    pixel_values_videos, 
                    grid_thw=video_grid_thw,
                )
                n_video_tokens = video_embeds.shape[0]
                if timing_enabled:
                    if use_cuda_timing and video_embeds.is_cuda:
                        selector_apply_start = torch.cuda.Event(enable_timing=True)
                        selector_apply_start.record()
                    else:
                        selector_apply_start_time = time.perf_counter()
                
                # Selector 仅支持 batch_size=1 (简化逻辑)
                assert input_ids.shape[0] == 1, 'Inference Selector currently supports single batch only'
                
                # 视频 Token 的定位与重组逻辑
                VIDEO_START_ID = 151652
                VIDEO_END_ID = 151653
                
                position_video_begin_token = (input_ids[0] == VIDEO_START_ID).nonzero(as_tuple=True)[0]
                before_idx = position_video_begin_token[0].item() + 1
                before_vid_idx = input_ids[:, :before_idx]
                
                position_video_end_token = (input_ids[0] == VIDEO_END_ID).nonzero(as_tuple=True)[0]
                post_idx = position_video_end_token[-1].item()
                post_vid_idx = input_ids[:, post_idx:]
                
                # 构建新的 Video Token 占位符
                vid_tensor = torch.full(
                    (input_ids.shape[0], n_video_tokens), 
                    self.config.video_token_id, 
                    dtype=input_ids.dtype, 
                    device=input_ids.device
                )
                
                # 重组 input_ids
                input_ids = torch.cat((before_vid_idx, vid_tensor, post_vid_idx), dim=1)
                
                # 调整 all_indices 的偏移量
                all_indices = all_indices + before_idx
                
                # 构建 selected_indices
                total_len = origin_input_ids.shape[-1]
                combined_indices = torch.cat((
                    torch.arange(0, before_idx, device=all_indices.device),
                    all_indices, 
                    torch.arange(post_idx, total_len, device=all_indices.device)
                ))
                selected_indices, _ = torch.sort(combined_indices)
                
                # 切片 inputs_embeds
                inputs_embeds = inputs_embeds[:, selected_indices, :]
                
                # 填入视频特征
                video_mask = (input_ids == self.config.video_token_id).unsqueeze(-1).expand_as(inputs_embeds)
                video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                retained_visual_tokens += int(video_embeds.shape[0])
                inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)
                if timing_enabled:
                    if use_cuda_timing and video_embeds.is_cuda:
                        selector_apply_end = torch.cuda.Event(enable_timing=True)
                        selector_apply_end.record()
                        selector_apply_event_pairs.append((selector_apply_start, selector_apply_end))
                    else:
                        selector_apply_time_ms += (time.perf_counter() - selector_apply_start_time) * 1000.0

                # 设置特殊 mask
                text_image_mask = (input_ids != self.config.video_token_id)
                self.base_model.text_image_mask = text_image_mask
                for layer in self.base_model.layers:
                    layer.self_attn.text_image_mask = text_image_mask

            # 更新 attention_mask (通用逻辑)
            # 如果发生了剪枝 (selected_indices 被赋值)，则对 mask 进行同样的切片
            # 注意：这里更新的是用于后续计算的 attention_mask，但 origin_attention_mask 保持不变
            if selected_indices is not None and attention_mask is not None:
                attention_mask = attention_mask[:, selected_indices]

        # =========================================================================
        # 位置编码处理 (RoPE)
        # =========================================================================
        if timing_enabled:
            if use_cuda_timing and inputs_embeds.is_cuda:
                llm_prefill_start = torch.cuda.Event(enable_timing=True)
                llm_prefill_start.record()
            else:
                llm_prefill_start_time = time.perf_counter()

        if position_ids is None and (attention_mask is None or attention_mask.ndim == 2):
            if (
                (cache_position is not None and cache_position[0] == 0)
                or self.rope_deltas is None
                or (past_key_values is None or past_key_values.get_seq_length() == 0)
            ):
                # 计算 RoPE: 必须使用 origin_input_ids (未剪枝的序列) 和 origin_attention_mask
                # 这样计算出的 spatial grid 相对位置才是正确的
                position_ids, rope_deltas = self.get_rope_index(
                    origin_input_ids,  # <--- 使用原始未剪枝的 input_ids
                    image_grid_thw,
                    video_grid_thw,
                    second_per_grid_ts,
                    origin_attention_mask,  # <--- 使用原始未剪枝的 attention_mask
                )
                
                # [关键]: 如果发生了剪枝，必须对位置编码也进行同样的物理切片
                if selected_indices is not None:
                    position_ids = position_ids[:, :, selected_indices]
                
                self.rope_deltas = rope_deltas

            else:
                # 增量生成模式 (KV Cache) - 推理通常是 prefill 后接 decode
                batch_size, seq_length, _ = inputs_embeds.shape
                delta = (
                    (cache_position[0] + self.rope_deltas).to(inputs_embeds.device)
                    if cache_position is not None
                    else 0
                )
                position_ids = torch.arange(seq_length, device=inputs_embeds.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                if cache_position is not None:
                    delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        # =========================================================================
        # LLM 前向传播
        # =========================================================================
        outputs = self.model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        if timing_enabled:
            if use_cuda_timing and hidden_states.is_cuda:
                llm_prefill_end = torch.cuda.Event(enable_timing=True)
                llm_prefill_end.record()
            else:
                llm_prefill_time_ms = (time.perf_counter() - llm_prefill_start_time) * 1000.0

        if timing_enabled and hidden_states.shape[1] != 1:
            if use_cuda_timing:
                end = torch.cuda.Event(enable_timing=True)
                end.record()
                torch.cuda.synchronize()
                generation_prefill_time = start.elapsed_time(end)
            else:
                generation_prefill_time = (time.perf_counter() - prefill_start_time) * 1000.0

            if getattr(self.visual, "_last_vision_encoder_start_event", None) is not None and getattr(self.visual, "_last_vision_encoder_end_event", None) is not None:
                vision_encoder_time_ms = self.visual._last_vision_encoder_start_event.elapsed_time(
                    self.visual._last_vision_encoder_end_event
                )
            elif self.visual.last_vision_encoder_time_ms is not None:
                vision_encoder_time_ms = self.visual.last_vision_encoder_time_ms

            if getattr(self.visual, "_last_selector_start_event", None) is not None and getattr(self.visual, "_last_selector_end_event", None) is not None:
                selector_score_select_time_ms = self.visual._last_selector_start_event.elapsed_time(
                    self.visual._last_selector_end_event
                )
            elif self.visual.last_selector_score_select_time_ms is not None:
                selector_score_select_time_ms = self.visual.last_selector_score_select_time_ms

            if selector_apply_event_pairs:
                selector_score_select_time_ms += sum(start_evt.elapsed_time(end_evt) for start_evt, end_evt in selector_apply_event_pairs)
            else:
                selector_score_select_time_ms += selector_apply_time_ms

            if llm_prefill_start is not None and llm_prefill_end is not None:
                llm_prefill_time_ms = llm_prefill_start.elapsed_time(llm_prefill_end)

            self.visual.last_vision_encoder_time_ms = vision_encoder_time_ms
            self.visual.last_selector_score_select_time_ms = selector_score_select_time_ms
            kv_cache_mb = _estimate_kv_cache_mb(outputs.past_key_values)
            print(f"VisionEncoderTime (ms): {vision_encoder_time_ms}")
            print(f"SelectorScoreSelectTime (ms): {selector_score_select_time_ms}")
            print(f"LLMPrefillTime (ms): {llm_prefill_time_ms}")
            print(f"PrefillTime (ms): {generation_prefill_time}")
            print(f"KVCache (MB): {kv_cache_mb}")
            print(f"Retained Visual Tokens: {retained_visual_tokens}")

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (None,) + output

        return Qwen2_5_VLCausalLMOutputWithPast(
            loss=None,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=self.rope_deltas,
        )
