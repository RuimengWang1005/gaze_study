from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import torch


EVAL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EVAL_ROOT))

from bbox_teacher_audit.teacher_forward import (  # noqa: E402
    PreparedSample,
    TeacherRuntime,
    TokenizedExample,
    freeze_model,
    parameter_grads_absent,
    parameters_are_frozen,
)


class FakeTokenizer:
    eos_token_id = 0

    def convert_tokens_to_ids(self, token):
        return 99

    def encode(self, text, add_special_tokens=False):
        return [20 + index for index, _ in enumerate(text.split())]

    def apply_chat_template(self, messages, chat_template, tokenize, add_generation_prompt):
        image_count = sum(str(message["content"]).count("<|image_pad|>") for message in messages)
        prefix = [1, 2] + [7] * image_count + [3]
        if len(messages) == 2 and add_generation_prompt:
            return prefix
        answer = self.encode(messages[-1]["content"], add_special_tokens=False)
        return prefix + answer + [99]


class TinyDecoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(128, 4)

    def forward(self, inputs_embeds=None, **kwargs):
        return SimpleNamespace(last_hidden_state=inputs_embeds.cumsum(dim=1), past_key_values=None)


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = TinyDecoder()
        self.lm_head = torch.nn.Linear(4, 128, bias=False)
        self.config = SimpleNamespace(
            image_token_id=7,
            vision_config=SimpleNamespace(spatial_merge_size=2),
        )

    def get_rope_index(self, input_ids, **kwargs):
        positions = torch.arange(input_ids.shape[1]).view(1, 1, -1).expand(3, 1, -1)
        return positions, torch.zeros(1, 1, dtype=torch.long)


def tiny_runtime():
    runtime = TeacherRuntime.__new__(TeacherRuntime)
    runtime.model = TinyModel()
    runtime.processor = SimpleNamespace(tokenizer=FakeTokenizer())
    runtime.tokenizer = runtime.processor.tokenizer
    runtime.device = torch.device("cpu")
    runtime.dtype = torch.float32
    runtime.max_new_tokens = 2
    runtime.merge_size = 2
    runtime.image_token_id = 7
    runtime.stop_ids = {0, 99}
    freeze_model(runtime.model)
    return runtime


def test_teacher_labels_only_cover_answer_tokens():
    runtime = tiny_runtime()
    tokens = runtime.tokenize("where", "two words", visual_tokens=4)
    assert tokens.answer_token_count == 2
    assert int((tokens.labels != -100).sum()) == 2
    assert torch.all(tokens.labels[:, : tokens.prompt_ids.shape[1]] == -100)


def test_visual_leaf_gets_finite_nonzero_gradient_while_parameters_stay_frozen():
    runtime = tiny_runtime()
    teacher_ids = torch.tensor([[1, 7, 7, 7, 7, 3, 21]])
    labels = torch.full_like(teacher_ids, -100)
    labels[:, -1] = 21
    tokens = TokenizedExample(teacher_ids[:, :-1], teacher_ids, labels, 1)
    features = torch.randn(4, 4)
    prepared = PreparedSample(
        record={"sample_id": "x", "answer": "x"},
        grid_thw=torch.tensor([[1, 4, 4]]),
        pixel_values=torch.empty(0),
        visual_features=features,
        foreground_indices=torch.tensor([0, 1, 2]),
        tokens=tokens,
        vision_ms=0.0,
    )
    z = features.detach().requires_grad_(True)
    result = runtime.nll(prepared, z, torch.arange(4), enable_grad=True)
    gradient = torch.autograd.grad(-result["mean_nll_tensor"], z)[0]
    assert torch.isfinite(gradient).all()
    assert float(gradient.abs().sum()) > 0
    assert parameters_are_frozen(runtime.model)
    assert parameter_grads_absent(runtime.model)


def test_bbox_leaf_gradient_survives_scatter_into_full_visual_tensor():
    runtime = tiny_runtime()
    teacher_ids = torch.tensor([[1, 7, 7, 7, 7, 3, 21]])
    labels = torch.full_like(teacher_ids, -100)
    labels[:, -1] = 21
    tokens = TokenizedExample(teacher_ids[:, :-1], teacher_ids, labels, 1)
    features = torch.randn(4, 4)
    foreground = torch.tensor([0, 1, 2])
    prepared = PreparedSample(
        record={"sample_id": "x", "answer": "x"},
        grid_thw=torch.tensor([[1, 4, 4]]),
        pixel_values=torch.empty(0),
        visual_features=features,
        foreground_indices=foreground,
        tokens=tokens,
        vision_ms=0.0,
    )
    z_bbox = features[foreground].detach().requires_grad_(True)
    visual_for_grad = features.detach().clone()
    visual_for_grad[foreground] = z_bbox
    result = runtime.nll(prepared, visual_for_grad, foreground, enable_grad=True)
    gradient = torch.autograd.grad(-result["mean_nll_tensor"], z_bbox)[0]
    assert torch.isfinite(gradient).all()
    assert float(gradient.abs().sum()) > 0


def test_pruning_keeps_original_visual_positions():
    runtime = tiny_runtime()
    ids = torch.tensor([[1, 7, 7, 7, 7, 3]])
    features = torch.randn(4, 4)
    pruned = runtime._pruned_inputs(ids, torch.tensor([[1, 4, 4]]), features, torch.tensor([1, 3]))
    assert pruned.input_ids.tolist() == [[1, 7, 7, 3]]
    assert pruned.position_ids[0, 0].tolist() == [0, 2, 4, 5]
