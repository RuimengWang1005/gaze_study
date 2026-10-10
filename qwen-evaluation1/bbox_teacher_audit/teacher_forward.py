"""Frozen Qwen2.5-VL feature extraction and token-subset LLM forwarding."""

from __future__ import annotations

import copy
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence

import torch
import torch.nn.functional as F
from PIL import Image

from .metrics import normalized_exact_match
from .token_mapping import foreground_indices, pixel_bboxs_to_merged_mask, validate_mapping
from .utility import deletion_count, deletion_sets, grad_input_utility, retained_indices


CHAT_TEMPLATE = """{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"""
SYSTEM_MESSAGE = "You are a helpful assistant."
SHORT_ANSWER_SUFFIX = "Answer the question using a single word or phrase."


@dataclass
class TokenizedExample:
    prompt_ids: torch.Tensor
    teacher_ids: torch.Tensor
    labels: torch.Tensor
    answer_token_count: int


@dataclass
class PreparedSample:
    record: Mapping[str, object]
    grid_thw: torch.Tensor
    pixel_values: torch.Tensor
    visual_features: torch.Tensor
    foreground_indices: torch.Tensor
    tokens: TokenizedExample
    vision_ms: float


@dataclass
class PrunedInputs:
    input_ids: torch.Tensor
    inputs_embeds: torch.Tensor
    attention_mask: torch.Tensor
    position_ids: torch.Tensor
    labels: Optional[torch.Tensor]
    kept_sequence_indices: torch.Tensor


def _cuda_sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def freeze_model(model) -> None:
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)


def parameters_are_frozen(model) -> bool:
    return all(not parameter.requires_grad for parameter in model.parameters())


def parameter_grads_absent(model) -> bool:
    return all(parameter.grad is None for parameter in model.parameters())


def model_sentinels(model) -> Dict[str, list[float]]:
    names = (
        "visual.patch_embed.proj.weight",
        "model.layers.0.self_attn.q_proj.weight",
        "lm_head.weight",
    )
    state = model.state_dict()
    return {name: state[name].detach().float().reshape(-1)[:8].cpu().tolist() for name in names}


class TeacherRuntime:
    def __init__(self, model, processor, device: torch.device, dtype: torch.dtype, max_new_tokens: int = 32):
        self.model = model
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.device = device
        self.dtype = dtype
        self.max_new_tokens = max_new_tokens
        self.merge_size = int(model.config.vision_config.spatial_merge_size)
        self.image_token_id = int(model.config.image_token_id)
        stop_ids = {int(self.tokenizer.eos_token_id)}
        im_end = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        if isinstance(im_end, int) and im_end >= 0:
            stop_ids.add(im_end)
        self.stop_ids = stop_ids
        freeze_model(model)
        if not parameters_are_frozen(model):
            raise RuntimeError("model freeze verification failed")

    @classmethod
    def load(
        cls,
        model_path: str,
        device: str = "cuda:0",
        dtype: str = "bfloat16",
        min_pixels: int = 3136,
        max_pixels: int = 401408,
        max_new_tokens: int = 32,
    ) -> "TeacherRuntime":
        from transformers import AutoProcessor
        from qwen25vl import Qwen2_5_VLForConditionalGeneration

        torch_dtype = getattr(torch, dtype)
        target = torch.device(device)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            attn_implementation="flash_attention_2",
            local_files_only=True,
        ).to(target)
        processor = AutoProcessor.from_pretrained(
            model_path,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            local_files_only=True,
        )
        return cls(model, processor, target, torch_dtype, max_new_tokens=max_new_tokens)

    def _visual_text(self, visual_tokens: int, question: str) -> str:
        vision = (
            "<|vision_start|>"
            + "<|image_pad|>" * visual_tokens
            + "<|vision_end|>"
        )
        return f"{vision}\n{question.strip()}\n{SHORT_ANSWER_SUFFIX}"

    def tokenize(self, question: str, answer: str, visual_tokens: int) -> TokenizedExample:
        user_content = self._visual_text(visual_tokens, question)
        prefix_messages = [
            {"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": user_content},
        ]
        prompt = self.tokenizer.apply_chat_template(
            prefix_messages,
            chat_template=CHAT_TEMPLATE,
            tokenize=True,
            add_generation_prompt=True,
        )
        answer_ids = self.tokenizer.encode(str(answer), add_special_tokens=False)
        full = self.tokenizer.apply_chat_template(
            prefix_messages + [{"role": "assistant", "content": str(answer)}],
            chat_template=CHAT_TEMPLATE,
            tokenize=True,
            add_generation_prompt=False,
        )
        if not answer_ids:
            raise ValueError("ground-truth answer has zero tokens")
        if full[len(prompt) : len(prompt) + len(answer_ids)] != answer_ids:
            raise ValueError("answer token boundary does not match the GAZE chat template")
        prompt_tensor = torch.tensor(prompt, dtype=torch.long, device=self.device).unsqueeze(0)
        teacher_tensor = torch.tensor(prompt + answer_ids, dtype=torch.long, device=self.device).unsqueeze(0)
        labels = torch.full_like(teacher_tensor, -100)
        labels[:, len(prompt) :] = teacher_tensor[:, len(prompt) :]
        if int((labels != -100).sum()) != len(answer_ids):
            raise RuntimeError("teacher labels are not restricted to answer tokens")
        image_count = int((teacher_tensor == self.image_token_id).sum())
        if image_count != visual_tokens:
            raise ValueError(f"prompt has {image_count} image tokens; expected {visual_tokens}")
        return TokenizedExample(prompt_tensor, teacher_tensor, labels, len(answer_ids))

    def prepare(self, record: Mapping[str, object]) -> PreparedSample:
        with Image.open(str(record["image_path"])) as opened:
            image = opened.convert("RGB")
        if image.size != (int(record["width"]), int(record["height"])):
            raise ValueError("metadata image dimensions do not match the file")
        processed = self.processor.image_processor(images=[image], return_tensors="pt")
        grid = processed["image_grid_thw"].to(self.device)
        pixels = processed["pixel_values"].to(device=self.device, dtype=self.dtype)
        _cuda_sync(self.device)
        start = time.perf_counter()
        with torch.inference_mode():
            visual = self.model.visual(pixels, grid_thw=grid).detach()
        _cuda_sync(self.device)
        vision_ms = (time.perf_counter() - start) * 1000.0
        mask = pixel_bboxs_to_merged_mask(
            record.get("bboxs"), record.get("width"), record.get("height"), grid[0].cpu(), self.merge_size
        )
        if mask is None:
            raise ValueError("bbox mapping produced an empty mask")
        validate_mapping(mask, grid[0].cpu(), self.merge_size)
        foreground = foreground_indices(mask).to(self.device)
        if visual.shape[0] != mask.numel():
            raise ValueError(f"visual feature count {visual.shape[0]} != mapped tokens {mask.numel()}")
        expected = record.get("bbox_token_indices")
        if expected is not None and list(map(int, expected)) != foreground.cpu().tolist():
            raise ValueError("runtime bbox mapping differs from prevalidated manifest")
        tokens = self.tokenize(str(record["question"]), str(record["answer"]), int(visual.shape[0]))
        return PreparedSample(record, grid, pixels, visual, foreground, tokens, vision_ms)

    def _pruned_inputs(
        self,
        input_ids: torch.Tensor,
        grid_thw: torch.Tensor,
        visual_features: torch.Tensor,
        visual_indices: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> PrunedInputs:
        visual_indices = visual_indices.to(device=self.device, dtype=torch.long).sort().values
        if visual_indices.numel() == 0:
            raise ValueError("at least one visual token must be retained")
        if int(visual_indices.min()) < 0 or int(visual_indices.max()) >= visual_features.shape[0]:
            raise IndexError("visual token index out of range")
        image_positions = (input_ids[0] == self.image_token_id).nonzero(as_tuple=False).flatten()
        if image_positions.numel() != visual_features.shape[0]:
            raise ValueError("image placeholder count does not match visual features")
        retained_image_positions = image_positions[visual_indices]
        text_positions = (input_ids[0] != self.image_token_id).nonzero(as_tuple=False).flatten()
        keep = torch.cat((text_positions, retained_image_positions)).sort().values
        pruned_ids = input_ids[:, keep]
        text_embeds = self.model.model.embed_tokens(pruned_ids)
        selected_visual = visual_features[visual_indices].to(text_embeds.dtype)
        image_mask = (pruned_ids == self.image_token_id).unsqueeze(-1).expand_as(text_embeds)
        inputs_embeds = text_embeds.masked_scatter(image_mask, selected_visual)
        full_attention = torch.ones_like(input_ids)
        full_positions, _ = self.model.get_rope_index(
            input_ids=input_ids,
            image_grid_thw=grid_thw,
            attention_mask=full_attention,
        )
        position_ids = full_positions[:, :, keep]
        pruned_labels = labels[:, keep] if labels is not None else None
        return PrunedInputs(
            input_ids=pruned_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=torch.ones_like(pruned_ids),
            position_ids=position_ids,
            labels=pruned_labels,
            kept_sequence_indices=keep,
        )

    def _answer_nll_from_hidden(self, hidden: torch.Tensor, labels: torch.Tensor) -> Dict[str, object]:
        target_positions = (labels[0] != -100).nonzero(as_tuple=False).flatten()
        prediction_positions = target_positions - 1
        if bool((prediction_positions < 0).any()):
            raise ValueError("answer label cannot occur at sequence position zero")
        logits = self.model.lm_head(hidden[:, prediction_positions, :]).float()
        targets = labels[:, target_positions]
        losses = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction="none"
        )
        return {
            "mean_nll_tensor": losses.mean(),
            "total_nll": float(losses.detach().sum().item()),
            "mean_nll": float(losses.detach().mean().item()),
            "answer_tokens": int(losses.numel()),
        }

    def nll(
        self,
        prepared: PreparedSample,
        visual_features: torch.Tensor,
        visual_indices: torch.Tensor,
        enable_grad: bool = False,
    ) -> Dict[str, object]:
        inputs = self._pruned_inputs(
            prepared.tokens.teacher_ids,
            prepared.grid_thw,
            visual_features,
            visual_indices,
            prepared.tokens.labels,
        )
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        _cuda_sync(self.device)
        start = time.perf_counter()
        context = torch.enable_grad() if enable_grad else torch.inference_mode()
        with context:
            outputs = self.model.model(
                input_ids=None,
                inputs_embeds=inputs.inputs_embeds,
                attention_mask=inputs.attention_mask,
                position_ids=inputs.position_ids,
                use_cache=False,
                return_dict=True,
            )
            metrics = self._answer_nll_from_hidden(outputs.last_hidden_state, inputs.labels)
        _cuda_sync(self.device)
        metrics.update(
            {
                "forward_ms": (time.perf_counter() - start) * 1000.0,
                "peak_memory_mb": (
                    float(torch.cuda.max_memory_allocated(self.device) / (1024**2))
                    if self.device.type == "cuda"
                    else float("nan")
                ),
                "sequence_length": int(inputs.input_ids.shape[1]),
                "position_ids": inputs.position_ids,
            }
        )
        return metrics

    def generate(
        self,
        prepared: PreparedSample,
        visual_features: torch.Tensor,
        visual_indices: torch.Tensor,
    ) -> Dict[str, object]:
        inputs = self._pruned_inputs(
            prepared.tokens.prompt_ids,
            prepared.grid_thw,
            visual_features,
            visual_indices,
        )
        _cuda_sync(self.device)
        start = time.perf_counter()
        generated: list[int] = []
        with torch.inference_mode():
            outputs = self.model.model(
                input_ids=None,
                inputs_embeds=inputs.inputs_embeds,
                attention_mask=inputs.attention_mask,
                position_ids=inputs.position_ids,
                use_cache=True,
                return_dict=True,
            )
            logits = self.model.lm_head(outputs.last_hidden_state[:, -1:, :])
            cache = outputs.past_key_values
            semantic_position = inputs.position_ids[:, :, -1:] + 1
            attention = inputs.attention_mask
            for _ in range(self.max_new_tokens):
                token = int(logits[0, -1].argmax().item())
                if token in self.stop_ids:
                    break
                generated.append(token)
                token_tensor = torch.tensor([[token]], device=self.device, dtype=torch.long)
                attention = torch.cat((attention, torch.ones_like(token_tensor)), dim=1)
                outputs = self.model.model(
                    input_ids=token_tensor,
                    attention_mask=attention,
                    position_ids=semantic_position,
                    past_key_values=cache,
                    use_cache=True,
                    return_dict=True,
                )
                cache = outputs.past_key_values
                logits = self.model.lm_head(outputs.last_hidden_state[:, -1:, :])
                semantic_position = semantic_position + 1
        _cuda_sync(self.device)
        text = self.tokenizer.decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=False).strip()
        return {
            "answer": text,
            "correct": normalized_exact_match(text, prepared.record["answer"]),
            "generated_tokens": len(generated),
            "generation_ms": (time.perf_counter() - start) * 1000.0,
        }

    def original_full_nll(self, prepared: PreparedSample) -> float:
        """Reference path used only by smoke tests."""

        with torch.inference_mode():
            output = self.model(
                input_ids=prepared.tokens.teacher_ids,
                attention_mask=torch.ones_like(prepared.tokens.teacher_ids),
                labels=prepared.tokens.labels,
                pixel_values=prepared.pixel_values,
                image_grid_thw=prepared.grid_thw,
                use_cache=False,
                return_dict=True,
            )
        return float(output.loss.item())

    def evaluate_sufficiency(self, prepared: PreparedSample, smoke_reference: bool = False) -> Dict[str, object]:
        full_indices = torch.arange(prepared.visual_features.shape[0], device=self.device)
        full_nll = self.nll(prepared, prepared.visual_features, full_indices)
        full_generation = self.generate(prepared, prepared.visual_features, full_indices)
        bbox_nll = self.nll(prepared, prepared.visual_features, prepared.foreground_indices)
        bbox_generation = self.generate(prepared, prepared.visual_features, prepared.foreground_indices)
        result = {
            "sample_id": prepared.record["sample_id"],
            "source": prepared.record["source"],
            "question": prepared.record["question"],
            "ground_truth": prepared.record["answer"],
            "full_mean_nll": full_nll["mean_nll"],
            "full_total_nll": full_nll["total_nll"],
            "full_answer_tokens": full_nll["answer_tokens"],
            "full_generated_answer": full_generation["answer"],
            "full_correct": full_generation["correct"],
            "bbox_mean_nll": bbox_nll["mean_nll"],
            "bbox_total_nll": bbox_nll["total_nll"],
            "bbox_answer_tokens": bbox_nll["answer_tokens"],
            "bbox_generated_answer": bbox_generation["answer"],
            "bbox_correct": bbox_generation["correct"],
            "delta_nll": float(bbox_nll["mean_nll"] - full_nll["mean_nll"]),
            "full_visual_tokens": int(prepared.visual_features.shape[0]),
            "bbox_visual_tokens": int(prepared.foreground_indices.numel()),
            "visual_token_retention": float(prepared.foreground_indices.numel() / prepared.visual_features.shape[0]),
            "full_sequence_length": full_nll["sequence_length"],
            "bbox_sequence_length": bbox_nll["sequence_length"],
            "vision_ms": prepared.vision_ms,
            "full_forward_ms": full_nll["forward_ms"],
            "bbox_forward_ms": bbox_nll["forward_ms"],
            "full_generation_ms": full_generation["generation_ms"],
            "bbox_generation_ms": bbox_generation["generation_ms"],
            "latency_ratio": float(bbox_nll["forward_ms"] / full_nll["forward_ms"]),
            "full_peak_memory_mb": full_nll["peak_memory_mb"],
            "bbox_peak_memory_mb": bbox_nll["peak_memory_mb"],
        }
        if smoke_reference:
            reference = self.original_full_nll(prepared)
            result["original_full_mean_nll"] = reference
            result["injected_full_nll_abs_error"] = abs(reference - float(full_nll["mean_nll"]))
        return result

    def evaluate_utility(
        self,
        prepared: PreparedSample,
        seed: int,
        bbox_base_correct: bool,
    ) -> Dict[str, object]:
        num_bbox = int(prepared.foreground_indices.numel())
        count = deletion_count(num_bbox)
        z_bbox = prepared.visual_features[prepared.foreground_indices].detach().requires_grad_(True)
        visual_for_grad = prepared.visual_features.detach().clone()
        visual_for_grad[prepared.foreground_indices] = z_bbox
        base = self.nll(
            prepared,
            visual_for_grad,
            prepared.foreground_indices,
            enable_grad=True,
        )
        objective = -base["mean_nll_tensor"]
        gradient = torch.autograd.grad(objective, z_bbox, retain_graph=False, create_graph=False)[0]
        utility = grad_input_utility(gradient, z_bbox)
        finite = bool(torch.isfinite(gradient).all() and torch.isfinite(utility).all())
        nonzero = int((utility > 0).sum().item())
        common = {
            "sample_id": prepared.record["sample_id"],
            "source": prepared.record["source"],
            "bbox_tokens": num_bbox,
            "deleted_tokens": count,
            "bbox_base_correct": bool(bbox_base_correct),
            "base_nll": base["mean_nll"],
            "gradient_finite": finite,
            "gradient_norm": float(gradient.detach().float().norm().item()),
            "utility_positive_tokens": nonzero,
            "utility_max": float(utility.max().item()) if finite else float("nan"),
            "utility_mean": float(utility.mean().item()) if finite else float("nan"),
            "parameter_grads_absent": parameter_grads_absent(self.model),
        }
        if not finite or nonzero == 0:
            return {
                **common,
                "degenerate": True,
                "degenerate_reason": "nonfinite_gradient_or_utility" if not finite else "all_utility_zero",
            }

        scores_cpu = utility.detach().cpu()
        sets = deletion_sets(scores_cpu, str(prepared.record["sample_id"]), seed, random_repeats=3)

        def evaluate_deleted(relative_deleted: torch.Tensor) -> float:
            relative_keep = retained_indices(num_bbox, relative_deleted.tolist()).to(self.device)
            absolute_keep = prepared.foreground_indices[relative_keep]
            result = self.nll(prepared, prepared.visual_features, absolute_keep)
            return float(result["mean_nll"])

        top_nll = evaluate_deleted(sets["top"])
        bottom_nll = evaluate_deleted(sets["bottom"])
        random_nlls = [evaluate_deleted(indices) for indices in sets["random"]]
        random_nll = float(sum(random_nlls) / len(random_nlls))
        base_nll = float(base["mean_nll"])
        delta_top = top_nll - base_nll
        delta_bottom = bottom_nll - base_nll
        delta_random = random_nll - base_nll
        return {
            **common,
            "degenerate": False,
            "degenerate_reason": None,
            "top_nll": top_nll,
            "bottom_nll": bottom_nll,
            "random_nll_mean": random_nll,
            "random_nlls": random_nlls,
            "delta_top": delta_top,
            "delta_bottom": delta_bottom,
            "delta_random": delta_random,
            "causal_excess": delta_top - delta_random,
            "top_vs_bottom": delta_top - delta_bottom,
            "utility_values": scores_cpu.tolist(),
            "top_deleted_relative": sets["top"].tolist(),
            "bottom_deleted_relative": sets["bottom"].tolist(),
            "random_deleted_relative": [indices.tolist() for indices in sets["random"]],
        }

    def release(self, prepared: PreparedSample) -> None:
        del prepared
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
