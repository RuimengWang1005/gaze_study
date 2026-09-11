import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
import time

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache, SlidingWindowCache, StaticCache
from transformers.generation import GenerationMixin
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.modeling_outputs import BaseModelOutputWithPast, ModelOutput
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    is_torchdynamo_compiling,
    logging,
    replace_return_docstrings,
)
from qwen25vl.configuration_qwen2_5_vl import Qwen2_5_VLConfig, Qwen2_5_VLVisionConfig
import os

if is_flash_attn_2_available():
    from flash_attn import flash_attn_varlen_func
    from flash_attn.layers.rotary import apply_rotary_emb

else:
    flash_attn_varlen_func = None
    apply_rotary_emb = None


if is_flash_attn_2_available():
    from transformers.modeling_flash_attention_utils import _flash_attention_forward
else:
    flash_attn_varlen_func = None
logger = logging.get_logger(__name__)
from qwen25vl.modeling_qwen2_5_vl import Qwen2_5_VLCausalLMOutputWithPast, _estimate_kv_cache_mb, apply_rotary_pos_emb_flashatt
from copy import deepcopy
import numpy as np

DEFAULT_PRUMERGE_PLUS_ATTN_CHUNK_SIZE = 128


def _resolve_prumerge_plus_attn_chunk_size(module: nn.Module) -> int:
    chunk_size = getattr(module, "prumerge_plus_attn_chunk_size", None)
    if chunk_size is None:
        chunk_size = os.environ.get("PRUMERGE_PLUS_ATTN_CHUNK_SIZE", DEFAULT_PRUMERGE_PLUS_ATTN_CHUNK_SIZE)
    try:
        chunk_size = int(chunk_size)
    except (TypeError, ValueError):
        chunk_size = DEFAULT_PRUMERGE_PLUS_ATTN_CHUNK_SIZE
    return max(chunk_size, 1)


def _compute_prumerge_plus_attention_summary(
    q: torch.Tensor,
    k: torch.Tensor,
    cu_seqlens: torch.Tensor,
    chunk_size: int,
) -> torch.Tensor:
    seq_length = q.shape[0]
    num_heads = q.shape[1]
    scale = 1.0 / math.sqrt(q.shape[-1])
    attn_sum = torch.zeros(seq_length, dtype=torch.float32, device=q.device)

    q_by_head = q.transpose(0, 1)
    k_by_head = k.transpose(0, 1)

    for idx in range(1, len(cu_seqlens)):
        start = int(cu_seqlens[idx - 1].item())
        end = int(cu_seqlens[idx].item())
        if end <= start:
            continue

        k_segment_t = k_by_head[:, start:end, :].transpose(1, 2).float().contiguous()
        for q_start in range(start, end, chunk_size):
            q_end = min(q_start + chunk_size, end)
            q_chunk = q_by_head[:, q_start:q_end, :].float()
            attn_logits = torch.matmul(q_chunk, k_segment_t) * scale
            attn_probs = F.softmax(attn_logits, dim=-1, dtype=torch.float32)
            attn_sum[start:end] += attn_probs.sum(dim=(0, 1))

    return attn_sum / float(num_heads * seq_length)

def outlier_dectection_prumerge_plus(attn):

    attn_shape = attn.shape
    image_num = attn_shape[0]
    ratios = []
    
    # for each image (multi_images) or for base image (single_image)
    for i in range(image_num):
        cur_attn = attn[i].to(dtype=torch.float32).cpu().numpy().flatten()
        
        Q1 = np.percentile(cur_attn, 25)
        Q3 = np.percentile(cur_attn, 75)
        IQR = Q3 - Q1
        
        upper_bound = Q3 + 1.5 * IQR
        
        outlier_indices = np.where((cur_attn > upper_bound))[0]
        ratio = len(outlier_indices) / len(cur_attn)
        ratios.append(ratio)
    
    return sum(ratios) / len(ratios)

def complement_idx_prumerge_plus(idx, dim):
    a = torch.arange(dim, device=idx.device)
    ndim = idx.ndim
    dims = idx.shape
    n_idx = dims[-1]
    dims = dims[:-1] + (-1, )
    for i in range(1, ndim):
        a = a.unsqueeze(0)
    a = a.expand(*dims)
    masked = torch.scatter(a, -1, idx, 0)
    compl, _ = torch.sort(masked, dim=-1, descending=False)
    compl = compl.permute(-1, *tuple(range(ndim - 1)))
    compl = compl[n_idx:].permute(*(tuple(range(1, ndim)) + (0,)))
    return compl


def qwen25vl_vision_flash_attention2_forward_prumerge_plus(
    self,
    hidden_states: torch.Tensor,
    cu_seqlens: torch.Tensor,
    rotary_pos_emb: Optional[torch.Tensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
) -> torch.Tensor:
    if self.layer_idx == 31:
        self.metric = None
        self.attention_summary_prumerge_plus = None

    seq_length = hidden_states.shape[0]
    q, k, v = self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
    if position_embeddings is None:
        logger.warning_once(
            "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
            "through `rotary_pos_emb` (2D tensor of RoPE theta values), to using externally computed "
            "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.54 `rotary_pos_emb` will be "
            "removed and `position_embeddings` will be mandatory."
        )
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        cos = emb.cos().float()
        sin = emb.sin().float()
    else:
        cos, sin = position_embeddings
    q, k = apply_rotary_pos_emb_flashatt(q.unsqueeze(0), k.unsqueeze(0), cos, sin)
    q = q.squeeze(0)
    k = k.squeeze(0)

    if self.layer_idx == 31:
        self.metric = k.reshape(seq_length, -1)
        attention_summary = _compute_prumerge_plus_attention_summary(
            q=q,
            k=k,
            cu_seqlens=cu_seqlens,
            chunk_size=_resolve_prumerge_plus_attn_chunk_size(self),
        )
        self.attention_summary_prumerge_plus = attention_summary.view(-1, 4).mean(dim=-1)

    max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
    attn_output = flash_attn_varlen_func(q, k, v, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen).reshape(
        seq_length, -1
    )
    attn_output = self.proj(attn_output)
    return attn_output

def qwen25vl_vision_tower_forward_prumerge_plus(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
    """
    Args:
        hidden_states (`torch.Tensor` of shape `(seq_len, hidden_size)`):
            The final hidden states of the model.
        grid_thw (`torch.Tensor` of shape `(num_images_or_videos, 3)`):
            The temporal, height and width of feature shape of each image in LLM.

    Returns:
        `torch.Tensor`: hidden_states.
    """
    hidden_states = self.patch_embed(hidden_states)
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
        # Select dtype based on the following factors:
        #  - FA2 requires that cu_seqlens_q must have dtype int32
        #  - torch.onnx.export requires that cu_seqlens_q must have same dtype as grid_thw
        # See https://github.com/huggingface/transformers/pull/34852 for more information
        dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
    )
    cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

    for layer_num, blk in enumerate(self.blocks):
        if layer_num in self.fullatt_block_indexes:
            cu_seqlens_now = cu_seqlens
        else:
            cu_seqlens_now = cu_window_seqlens
        if self.gradient_checkpointing and self.training:
            hidden_states = self._gradient_checkpointing_func(
                blk.__call__, hidden_states, cu_seqlens_now, None, position_embeddings
            )
        else:
            hidden_states = blk(hidden_states, cu_seqlens=cu_seqlens_now, position_embeddings=position_embeddings)

    cls_attn = self.blocks[-1].attn.attention_summary_prumerge_plus
    if cls_attn is None:
        raise RuntimeError("prumerge+ attention summary was not computed for the final vision block")

    hidden_states = self.merger(hidden_states)
    reverse_indices = torch.argsort(window_index)
    image_features = hidden_states[reverse_indices, :]
    image_features = image_features.unsqueeze(0)
    B, N, C = image_features.shape

    desired_layer_k = self.blocks[-1].attn.metric  # shape:torch.Size([2320, 1280])
    desired_layer_k = desired_layer_k.view(desired_layer_k.shape[0] // 4, 4, -1)
    desired_layer_k = desired_layer_k.mean(dim=1).unsqueeze(0)  # shape: torch.Size([1, 580, 1280])
    k_C = desired_layer_k.shape[-1]
    cls_attn = cls_attn.unsqueeze(0) # shape: torch.Size([1, 580])
    #####新增#########
    cls_attn = cls_attn[:,reverse_indices]
    desired_layer_k = desired_layer_k[:,reverse_indices,:]
    ###################
    assert cls_attn.ndim == 2

    reduction_ratio = outlier_dectection_prumerge_plus(cls_attn)#*3.5
    # Maintaining the preset budget
    budgets_token = max(int(self.budgets * N), 1)
    iqr_token = max(int(N*reduction_ratio), 1)
    image_num = cls_attn.shape[0]

    if budgets_token > iqr_token:
        _, iqr_idx = torch.topk(cls_attn, iqr_token, dim=1, largest=True)  # [B, left_tokens]
        idx = torch.zeros((image_num, budgets_token), dtype=iqr_idx.dtype, device=self.device)
        
        for i in range(image_num):

            remaining_tokens = budgets_token - iqr_token
            
            # Sampling by arithmetic progression
            step_length = max(1, int(N / budgets_token))
            arithmetic_sequence = torch.arange(0, N, step_length).to(device=self.device)
            
            original_tensor_1d = iqr_idx[i].flatten()
            filtered_sequence = torch.tensor([x for x in arithmetic_sequence if x not in original_tensor_1d]).to(device=self.device)
            
            # If the filtered sequence is too long, truncate it
            if len(filtered_sequence) > remaining_tokens:
                filtered_sequence = filtered_sequence[:remaining_tokens]
            # If the filtered sequence is too short, randomly select additional indices
            elif len(filtered_sequence) < remaining_tokens:
                # code will not reach here
                available_indices = torch.tensor([x for x in range(N) if x not in original_tensor_1d and x not in filtered_sequence], 
                                            device=self.device)
                if len(available_indices) > 0:
                    extra_indices = available_indices[torch.randperm(len(available_indices))[:remaining_tokens - len(filtered_sequence)]]
                    filtered_sequence = torch.cat([filtered_sequence, extra_indices])
            
            # make sure the length of idx is budgets_token
            concatenated_tensor = torch.cat([iqr_idx[i], filtered_sequence])[:budgets_token]
            idx[i] = concatenated_tensor    
    else:
        _, idx = torch.topk(cls_attn, budgets_token, dim=1, largest=True)  # [B, left_tokens] , sorted=True
    
    index_features = idx.unsqueeze(-1).expand(-1, -1, C)  # [B, left_tokens, C]
    index_k = idx.unsqueeze(-1).expand(-1, -1, k_C)  # [B, left_tokens, C]


    x_others = torch.gather(image_features, dim=1, index=index_features)  # [B, left_tokens, C]
    Key_others = torch.gather(desired_layer_k, dim=1, index=index_k)  # [B, left_tokens, C]
    x_others_attn = torch.gather(cls_attn, dim=1, index=idx)  
    compl = complement_idx_prumerge_plus(idx, N)  # [B, N-1-left_tokens]
    non_topk = torch.gather(image_features, dim=1, index=compl.unsqueeze(-1).expand(-1, -1, C))  # [B, N-1-left_tokens, C]
    non_topk_Key = torch.gather(desired_layer_k, dim=1, index=compl.unsqueeze(-1).expand(-1, -1, k_C))
    non_topk_attn = torch.gather(cls_attn, dim=1, index=compl)  # [B, N-1-left_tokens]

    Key_others_norm = nn.functional.normalize(Key_others, p=2, dim=-1)
    non_topk_Key_norm = nn.functional.normalize(non_topk_Key, p=2, dim=-1)

    B, left_tokens, C = x_others.size()
    updated_x_others = torch.zeros_like(x_others)

    for b in range(B):
        for i in range(left_tokens):
            key_others_norm = Key_others_norm[b,i,:].unsqueeze(0).unsqueeze(0)

            before_i_Key = Key_others_norm[b, :i, :].unsqueeze(0)  
            after_i_Key = Key_others_norm[b, i+1:, :].unsqueeze(0) 

            before_i_x_others = x_others[b, :i, :].unsqueeze(0)  
            after_i_x_others = x_others[b, i+1:, :].unsqueeze(0)   
            rest_x_others = torch.cat([before_i_x_others, after_i_x_others, non_topk[b,:,:].unsqueeze(0)], dim=1)   
            before_i_x_others_attn = x_others_attn[b, :i].unsqueeze(0)  
            after_i_x_others_attn = x_others_attn[b, i+1:].unsqueeze(0)  
            rest_x_others_attn = torch.cat([before_i_x_others_attn, after_i_x_others_attn, non_topk_attn[b,:].unsqueeze(0)], dim=1)  

            rest_Keys = torch.cat([before_i_Key, after_i_Key, non_topk_Key_norm[b,:,:].unsqueeze(0)], dim=1)
            cos_sim_matrix = torch.bmm(key_others_norm, rest_Keys.transpose(1, 2))

            cos_sim_num = max(min(int(32), cos_sim_matrix.shape[2]), 1)
            _, cluster_indices = torch.topk(cos_sim_matrix, k=cos_sim_num, dim=2, largest=True)

            cluster_tokens = rest_x_others[:,cluster_indices.squeeze(),:]
            weights = rest_x_others_attn[:,cluster_indices.squeeze()].unsqueeze(-1)

            # update cluster centers
            weighted_avg = torch.sum(cluster_tokens * weights, dim=1) #/ torch.sum(weights)
            updated_center = x_others[b, i, :]  + weighted_avg 
            updated_x_others[b, i, :] = updated_center 
        
    image_features = updated_x_others.squeeze(0)
    all_keep_indices = idx.squeeze(0)
    all_keep_indices = all_keep_indices.sort().values
    image_features = image_features.to(dtype=self.dtype)
    self.blocks[-1].attn.attention_summary_prumerge_plus = None
    self.blocks[-1].attn.metric = None

    return image_features, all_keep_indices, N


def qwen25vl_vision_block_forward_prumerge_plus(
    self,
    hidden_states: torch.Tensor,
    cu_seqlens: torch.Tensor,
    rotary_pos_emb: Optional[torch.Tensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
) -> torch.Tensor:
    if self.layer_idx == 31:
        self.hidden_states = None
    
    hidden_states = hidden_states + self.attn(
        self.norm1(hidden_states),
        cu_seqlens=cu_seqlens,
        rotary_pos_emb=rotary_pos_emb,
        position_embeddings=position_embeddings,
    )
    hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))

    if self.layer_idx == 31:
        self.hidden_states = hidden_states

    return hidden_states


def qwen25vl_generation_forward_prumerge_plus(
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
    r"""
    Args:
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

    Returns:

    Example:

    ```python
    >>> from PIL import Image
    >>> import requests
    >>> from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    >>> model = Qwen2_5_VLForConditionalGeneration.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
    >>> processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

    >>> messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "What is shown in this image?"},
            ],
        },
    ]
    >>> url = "https://www.ilankelman.org/stopsigns/australia.jpg"
    >>> image = Image.open(requests.get(url, stream=True).raw)

    >>> text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    >>> inputs = processor(text=[text], images=[image], vision_infos=[vision_infos])

    >>> # Generate
    >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
    >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    "The image shows a street scene with a red stop sign in the foreground. In the background, there is a large red gate with Chinese characters ..."
    ```"""
    has_eval_time = os.environ.get("EVAL_TIME")
    timing_enabled = bool(has_eval_time and os.environ["EVAL_TIME"].lower() == "true")
    use_cuda_timing = bool(timing_enabled and torch.cuda.is_available())
    vision_event_pairs = []
    selector_score_select_time_ms = 0.0
    vision_encoder_time_ms = 0.0
    llm_prefill_time_ms = 0.0
    retained_visual_tokens = 0
    llm_prefill_start = None
    llm_prefill_end = None
    llm_prefill_start_time = None
    prefill_start_time = None

    if timing_enabled:
        if use_cuda_timing:
            start = torch.cuda.Event(enable_timing=True)
            start.record()
        else:
            prefill_start_time = time.perf_counter()

    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if inputs_embeds is None:
        inputs_embeds = self.model.embed_tokens(input_ids)
        if pixel_values is not None:
            if timing_enabled:
                if use_cuda_timing:
                    vision_start = torch.cuda.Event(enable_timing=True)
                    vision_end = torch.cuda.Event(enable_timing=True)
                    vision_start.record()
                else:
                    vision_start_time = time.perf_counter()
            pixel_values = pixel_values.type(self.visual.dtype)
            image_embeds, all_indices, visual_token_num = self.visual(pixel_values, grid_thw=image_grid_thw)  
            if timing_enabled:
                if use_cuda_timing:
                    vision_end.record()
                    vision_event_pairs.append((vision_start, vision_end))
                else:
                    vision_encoder_time_ms += (time.perf_counter() - vision_start_time) * 1000.0
            n_image_tokens = image_embeds.shape[0]   
            retained_visual_tokens = n_image_tokens
            total_len = input_ids.shape[-1] 

            assert input_ids.shape[0] == 1, 'prumerge only support single batch, assert is in qwen2vl_generation_forward_prumerge_plus function'
            position_image_begin_token = (input_ids[0] == 151652).nonzero(as_tuple=True)[0]
            before_idx = position_image_begin_token[0].item() + 1   # before image length
            before_img = input_ids[:, :before_idx]
            position_image_end_token = (input_ids[0] == 151653).nonzero(as_tuple=True)[0]
            post_idx = position_image_end_token[-1].item() 
            post_img = input_ids[:, post_idx:]
            img_tensor = torch.full(
                (input_ids.shape[0], n_image_tokens), 
                self.config.image_token_id, 
                dtype=input_ids.dtype, 
                device=input_ids.device
            )

            origin_input_ids = deepcopy(input_ids)  
            input_ids = torch.cat((before_img, img_tensor, post_img), dim=1)   
            all_indices = all_indices + before_idx  
            all_indices = torch.cat((torch.arange(0, before_idx, device=all_indices.device), all_indices, torch.arange(post_idx, total_len, device=all_indices.device)))

            inputs_embeds = inputs_embeds[:, all_indices, :]

            
            image_mask = (
                (input_ids == self.config.image_token_id)
                .unsqueeze(-1)
                .expand_as(inputs_embeds)
                .to(inputs_embeds.device)
            )
            image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)   # 151655
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

            text_image_mask = (input_ids != 151655)   
            self.base_model.text_image_mask = text_image_mask
            for layer in self.base_model.layers:
                    layer.self_attn.text_image_mask = text_image_mask

        # if pixel_values_videos is not None:
        #     pixel_values_videos = pixel_values_videos.type(self.visual.dtype)
        #     video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw)
        #     n_video_tokens = (input_ids == self.config.video_token_id).sum().item()
        #     n_video_features = video_embeds.shape[0]
        #     if n_video_tokens != n_video_features:
        #         raise ValueError(
        #             f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {n_video_features}"
        #         )

        #     mask = input_ids == self.config.video_token_id
        #     mask_unsqueezed = mask.unsqueeze(-1)
        #     mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
        #     video_mask = mask_expanded.to(inputs_embeds.device)

        #     video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        #     inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        #     text_image_mask = (input_ids != self.config.video_token_id)
        #     self.base_model.text_image_mask = text_image_mask
        #     for layer in self.base_model.layers:
        #         layer.self_attn.text_image_mask = text_image_mask
        if pixel_values_videos is not None:
            # import pdb; pdb.set_trace()
            if timing_enabled:
                if use_cuda_timing:
                    vision_start = torch.cuda.Event(enable_timing=True)
                    vision_end = torch.cuda.Event(enable_timing=True)
                    vision_start.record()
                else:
                    vision_start_time = time.perf_counter()
            pixel_values_videos = pixel_values_videos.type(self.visual.dtype)
            video_embeds, all_indices, visual_token_num = self.visual(pixel_values_videos, grid_thw=video_grid_thw)
            if timing_enabled:
                if use_cuda_timing:
                    vision_end.record()
                    vision_event_pairs.append((vision_start, vision_end))
                else:
                    vision_encoder_time_ms += (time.perf_counter() - vision_start_time) * 1000.0
            n_video_tokens = video_embeds.shape[0]
            retained_visual_tokens = n_video_tokens

            total_len = input_ids.shape[-1]   
            assert input_ids.shape[0] == 1, 'divprune only support single batch, assert is in qwen2vl_generation_forward_divprune function'
            position_video_begin_token = (input_ids[0] == 151652).nonzero(as_tuple=True)[0]
            before_idx = position_video_begin_token[0].item() + 1   
            before_vid_idx = input_ids[:, :before_idx]
            position_video_end_token = (input_ids[0] == 151653).nonzero(as_tuple=True)[0]
            post_idx = position_video_end_token[-1].item() 
            post_vid_idx = input_ids[:, post_idx:]
            vid_tensor = torch.full(
                (input_ids.shape[0], n_video_tokens), 
                self.config.video_token_id, 
                dtype=input_ids.dtype, 
                device=input_ids.device
            )
            origin_input_ids = deepcopy(input_ids)   
            input_ids = torch.cat((before_vid_idx, vid_tensor, post_vid_idx), dim=1)  
            all_indices = all_indices + before_idx  
            all_indices = torch.cat((torch.arange(0, before_idx, device=all_indices.device), all_indices, torch.arange(post_idx, total_len, device=all_indices.device)))  
            inputs_embeds = inputs_embeds[:, all_indices, :]

            video_mask = (input_ids == self.config.video_token_id).unsqueeze(-1).expand_as(inputs_embeds)
            video_mask = video_mask.to(inputs_embeds.device)
            video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

            text_image_mask = (input_ids != self.config.video_token_id)
            self.base_model.text_image_mask = text_image_mask
            for layer in self.base_model.layers:
                layer.self_attn.text_image_mask = text_image_mask

        if attention_mask is not None:
            attention_mask = attention_mask.to(inputs_embeds.device)

    # if we get 4D attention mask we cannot calculate rope deltas anymore. TODO @raushan fixme
    if position_ids is None and (attention_mask is None or attention_mask.ndim == 2):
        # calculate RoPE index once per generation in the pre-fill stage only
        if (
            (cache_position is not None and cache_position[0] == 0)
            or self.rope_deltas is None
            or (past_key_values is None or past_key_values.get_seq_length() == 0)
        ):
            position_ids, rope_deltas = self.get_rope_index(
                origin_input_ids,
                image_grid_thw,
                video_grid_thw,
                second_per_grid_ts,
                attention_mask,
            )
            self.rope_deltas = rope_deltas
            
            ######### post handle #########
            position_ids = position_ids[:, :, all_indices]    # because we don't prune text token, so we don't need to change rope_deltas
            attention_mask = attention_mask[:, all_indices]

        # then use the prev pre-calculated rope-deltas to get the correct position ids
        else:
            batch_size, seq_length, _ = inputs_embeds.shape
            delta = (
                (cache_position[0] + self.rope_deltas).to(inputs_embeds.device)
                if cache_position is not None
                else 0
            )
            position_ids = torch.arange(seq_length, device=inputs_embeds.device)
            position_ids = position_ids.view(1, -1).expand(batch_size, -1)
            if cache_position is not None:  # otherwise `deltas` is an int `0`
                delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
            position_ids = position_ids.add(delta)
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

    if timing_enabled:
        if use_cuda_timing:
            llm_prefill_start = torch.cuda.Event(enable_timing=True)
            llm_prefill_start.record()
        else:
            llm_prefill_start_time = time.perf_counter()

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
        if use_cuda_timing:
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
            if vision_event_pairs:
                vision_encoder_time_ms = sum(start_evt.elapsed_time(end_evt) for start_evt, end_evt in vision_event_pairs)
            if llm_prefill_start is not None and llm_prefill_end is not None:
                llm_prefill_time_ms = llm_prefill_start.elapsed_time(llm_prefill_end)
        else:
            generation_prefill_time = (time.perf_counter() - prefill_start_time) * 1000.0

        kv_cache_mb = _estimate_kv_cache_mb(outputs.past_key_values)
        print(f"VisionEncoderTime (ms): {vision_encoder_time_ms}")
        print(f"SelectorScoreSelectTime (ms): {selector_score_select_time_ms}")
        print(f"LLMPrefillTime (ms): {llm_prefill_time_ms}")
        print(f"PrefillTime (ms): {generation_prefill_time}")
        print(f"KVCache (MB): {kv_cache_mb}")
        print(f"Retained Visual Tokens: {retained_visual_tokens}")
    

    loss = None
    if labels is not None:
        # Upcast to float if we need to compute the loss to avoid potential precision issues
        logits = logits.float()
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = CrossEntropyLoss()
        shift_logits = shift_logits.view(-1, self.config.vocab_size)
        shift_labels = shift_labels.view(-1)
        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        loss = loss_fct(shift_logits, shift_labels)


    if not return_dict:
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output

    return Qwen2_5_VLCausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        rope_deltas=self.rope_deltas,
    )
