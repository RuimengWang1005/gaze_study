import argparse
import json
import os
import random
import re
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from qwen25vl.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
from qwen25vl.processing_qwen2_5_vl import Qwen2_5_VLProcessor
from qwen_vl_utils import process_vision_info
from token_compression.dynamic_model import Qwen2_5_VLForConditionalGeneration_Dynamic
from token_compression.selector_model import Qwen2_5_VLForConditionalGeneration_Selector
from token_compression.visionzip_official import (
    Qwen2_5_VLForConditionalGeneration_VisionZip,
)


DEFAULT_OPTIONS_ROOT = (
    Path(__file__).resolve().parents[2] / "FinePruner_TIP2026" / "data"
)
DEFAULT_RESULTS_ROOT = Path("./resultrw/fgvc_eval")

DATASET_CONFIGS = {
    "bird-200": {
        "options_file": "bird-200-ViT-L-14-laion2b_s32b_b82k-4.json",
        "prompt_query": "Which of these birds is shown in the image?",
    },
    "fgvc_aircraft": {
        "options_file": "fgvc_aircraft-ViT-L-14-laion2b_s32b_b82k-4.json",
        "prompt_query": "Which of these aircrafts is shown in the image?",
    },
    "stanford_cars": {
        "options_file": "stanford_cars-ViT-L-14-laion2b_s32b_b82k-4.json",
        "prompt_query": "Which of these cars is shown in the image?",
    },
    "flowers102": {
        "options_file": "flowers102-ViT-L-14-laion2b_s32b_b82k-4.json",
        "prompt_query": "Which of these flowers is shown in the image?",
    },
    "oxford_pet": {
        "options_file": "oxford_pet-ViT-L-14-laion2b_s32b_b82k-4.json",
        "prompt_query": "Which of these pets is shown in the image?",
    },
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def resolve_spatial_ratio(args: argparse.Namespace, default: float = 0.1) -> float:
    spatial_ratio = getattr(args, "spatial_ratio", None)
    if spatial_ratio is None:
        spatial_ratio = getattr(args, "contextual_ratio", None)
    return float(default if spatial_ratio is None else spatial_ratio)


def parse_bool(value):
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean, got {value!r}")


def configure_processor_pixel_limits(
    processor: Qwen2_5_VLProcessor,
    min_pixels: int,
    max_pixels: int,
) -> Qwen2_5_VLProcessor:
    if hasattr(processor, "min_pixels"):
        processor.min_pixels = int(min_pixels)
    if hasattr(processor, "max_pixels"):
        processor.max_pixels = int(max_pixels)

    image_processor = getattr(processor, "image_processor", None)
    if image_processor is not None:
        if hasattr(image_processor, "min_pixels"):
            image_processor.min_pixels = int(min_pixels)
        if hasattr(image_processor, "max_pixels"):
            image_processor.max_pixels = int(max_pixels)
        size = getattr(image_processor, "size", None)
        if isinstance(size, dict):
            if "shortest_edge" in size:
                size["shortest_edge"] = int(min_pixels)
            if "longest_edge" in size:
                size["longest_edge"] = int(max_pixels)
    return processor


def load_model(args: argparse.Namespace):
    model_args = {
        "attn_implementation": args.attn_implementation,
        "device_map": args.device_map,
        "torch_dtype": args.torch_dtype,
    }

    method = args.method.lower()
    if method == "visionzip_official":
        model = Qwen2_5_VLForConditionalGeneration_VisionZip.from_pretrained(
            args.model_path, **model_args
        ).eval()
        model.budget = args.budgets
    elif method == "selector":
        model = Qwen2_5_VLForConditionalGeneration_Selector.from_pretrained(
            args.model_path, **model_args
        ).eval()
        model.visual.budgets = args.budgets
        model.visual.config.budgets = args.budgets
        model.visual.spatial_ratio = args.spatial_ratio
        model.visual.contextual_ratio = args.spatial_ratio
        model.visual.config.spatial_ratio = args.spatial_ratio
        model.visual.config.contextual_ratio = args.spatial_ratio
        model.visual.cosine_penalty_weight = max(
            0.0, float(args.cosine_penalty_weight)
        )
        model.visual.config.cosine_penalty_weight = (
            model.visual.cosine_penalty_weight
        )
        model.visual.cosine_penalty_threshold = max(
            -1.0, min(1.0, float(args.cosine_penalty_threshold))
        )
        model.visual.config.cosine_penalty_threshold = (
            model.visual.cosine_penalty_threshold
        )
        model.visual.distance_aware_redundancy = bool(args.distance_aware_redundancy)
        model.visual.distance_sigma = float(args.distance_sigma)
        model.visual.distance_floor = max(0.0, min(1.0, float(args.distance_floor)))
        model.visual.distance_apply_to = str(args.distance_apply_to)
        if model.visual.distance_sigma <= 0:
            raise ValueError("distance_sigma must be positive")
        if model.visual.distance_apply_to != "focal":
            raise ValueError("Only distance_apply_to='focal' is supported")
        for name in ("distance_aware_redundancy", "distance_sigma", "distance_floor", "distance_apply_to"):
            setattr(model.visual.config, name, getattr(model.visual, name))
    elif method == "dynamic":
        model = Qwen2_5_VLForConditionalGeneration_Dynamic.from_pretrained(
            args.model_path, **model_args
        ).eval()
        model.model.budgets = args.budgets
    else:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.model_path, **model_args
        ).eval()

    processor = Qwen2_5_VLProcessor.from_pretrained(args.model_path)
    processor = configure_processor_pixel_limits(
        processor,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    return model, processor, tokenizer


def replace_layers(args: argparse.Namespace, model) -> None:
    from token_compression.monkeypatch import replace_qwen25vl

    replace_qwen25vl(args, model, args.method.lower())


def load_examples(args: argparse.Namespace):
    dataset_cfg = DATASET_CONFIGS[args.dataset]
    options_path = Path(args.options_root) / dataset_cfg["options_file"]
    if not options_path.exists():
        raise FileNotFoundError(
            f"Options file not found: {options_path}. "
            "Please point --options-root to FinePruner_TIP2026/data."
        )

    with open(options_path, "r", encoding="utf-8") as f:
        label_options = json.load(f)

    examples = []
    for _, label_examples in label_options.items():
        for example in label_examples[: args.max_examples_per_class]:
            examples.append(example)

    if args.limit is not None:
        examples = examples[: args.limit]
    return examples, str(options_path)


def prepare_prompt_mc(options, prompt_query, choice_enumeration="ABCD"):
    if len(options) > len(choice_enumeration):
        raise ValueError(
            f"Expected at most {len(choice_enumeration)} options, got {len(options)}."
        )

    shuffle_idx = np.random.permutation(len(options)).tolist()
    gold_idx = shuffle_idx.index(0)
    gold_label = choice_enumeration[gold_idx]
    formatted_options = "\n".join(
        f"{choice_enumeration[i]}. {options[j]}" for i, j in enumerate(shuffle_idx)
    )
    label_to_option = {
        choice_enumeration[i]: options[j] for i, j in enumerate(shuffle_idx)
    }
    prompt = (
        f"{prompt_query}\nChoices:\n{formatted_options}\n"
        "Answer with the letter from the given choices directly."
    )
    return prompt, gold_label, label_to_option


def parse_prediction(text: str, choice_enumeration="ABCD"):
    stripped = text.strip()
    for letter in choice_enumeration:
        if stripped.lower().startswith(letter.lower()):
            return letter

    match = re.search(r"\b([" + re.escape(choice_enumeration) + r"])\b", stripped, re.I)
    if match:
        return match.group(1).upper()
    return None


def build_messages(image_path: str, prompt: str, args: argparse.Namespace):
    return [
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image_path,
                        "min_pixels": args.min_pixels,
                        "max_pixels": args.max_pixels,
                    },
                    {"type": "text", "text": prompt},
                ],
            },
        ]
    ]


def resolve_image_path(dataset: str, image_folder: str, image_rel_path: str) -> str:
    candidates = [os.path.join(image_folder, image_rel_path)]

    if dataset == "fgvc_aircraft" and image_rel_path.startswith("data/images/"):
        candidates.append(
            os.path.join(image_folder, image_rel_path.replace("data/images/", "images/", 1))
        )
    elif dataset == "flowers102":
        if image_rel_path.startswith("jpg/"):
            candidates.append(
                os.path.join(image_folder, image_rel_path.replace("jpg/", "images/", 1))
            )
        elif image_rel_path.startswith("images/"):
            candidates.append(
                os.path.join(image_folder, image_rel_path.replace("images/", "jpg/", 1))
            )

    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate

    raise FileNotFoundError(
        f"Image not found for dataset={dataset}: tried {candidates}"
    )


def generate_answer(
    model,
    processor: Qwen2_5_VLProcessor,
    tokenizer,
    image_path: str,
    prompt: str,
    args: argparse.Namespace,
) -> str:
    messages = build_messages(image_path, prompt, args)
    texts = [
        processor.apply_chat_template(
            msg, tokenize=False, add_generation_prompt=True
        )
        for msg in messages
    ]
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=texts,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    if args.device_map == "auto":
        inputs = inputs.to("cuda")
    else:
        inputs = inputs.to(args.device)

    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id

    output_ids = model.generate(
        **inputs,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=pad_token_id,
        do_sample=bool(args.temperature > 0),
        temperature=args.temperature,
        top_p=args.top_p,
        num_beams=args.num_beams,
        max_new_tokens=args.max_new_tokens,
        use_cache=True,
    )

    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, output_ids)
    ]
    answers = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return answers[0].strip()


def sanitize_model_name(model_path: str) -> str:
    return model_path.strip("/").replace("/", "__")


def resolve_output_dir(args: argparse.Namespace) -> Path:
    requested = Path(args.results_output_folder)
    if requested.resolve() == DEFAULT_RESULTS_ROOT.resolve():
        return (
            requested
            / args.dataset
            / f"{args.method}_{args.budgets}"
            / sanitize_model_name(args.model_path)
        )
    return requested


def eval_model(args: argparse.Namespace) -> Path:
    if args.dataset not in DATASET_CONFIGS:
        raise ValueError(f"Unsupported dataset: {args.dataset}")
    if not os.path.isdir(args.image_folder):
        raise FileNotFoundError(f"Image folder not found: {args.image_folder}")

    examples, options_path = load_examples(args)
    if not examples:
        raise ValueError("No examples found to evaluate.")

    model, processor, tokenizer = load_model(args)
    replace_layers(args, model)

    dataset_cfg = DATASET_CONFIGS[args.dataset]
    prompt_query = args.prompt_query or dataset_cfg["prompt_query"]

    results = []
    correct = 0
    generation_failures = 0

    start_time = time.time()

    for example in tqdm(examples, desc=f"Evaluating {args.dataset}"):
        image_path = resolve_image_path(
            dataset=args.dataset,
            image_folder=args.image_folder,
            image_rel_path=example["image"],
        )

        prompt, gold_label, label_to_option = prepare_prompt_mc(
            example["options"], prompt_query, choice_enumeration=args.choice_enumeration
        )
        answer = generate_answer(
            model=model,
            processor=processor,
            tokenizer=tokenizer,
            image_path=image_path,
            prompt=prompt,
            args=args,
        )
        parsed = parse_prediction(answer, choice_enumeration=args.choice_enumeration)

        if parsed is None:
            generation_failures += 1
        elif parsed == gold_label:
            correct += 1

        results.append(
            {
                "image": example["image"],
                "prediction_raw": answer,
                "prediction_parsed": parsed,
                "prediction_label_text": label_to_option.get(parsed),
                "gold_label": gold_label,
                "gold_text": label_to_option[gold_label],
                "options": label_to_option,
            }
        )

    elapsed = time.time() - start_time
    metrics = {
        "num_examples": len(results),
        "accuracy": correct / len(results),
        "generation_failures": generation_failures / len(results),
        "elapsed_seconds": elapsed,
    }

    output_dir = resolve_output_dir(args)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"{timestamp}_results.json"
    payload = {
        "dataset": args.dataset,
        "metrics": metrics,
        "config": vars(args),
        "options_path": options_path,
        "predictions": results,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(json.dumps({"dataset": args.dataset, "metrics": metrics}, indent=2))
    print(f"Saved results to {output_path}")
    return output_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate Qwen2.5-VL token-compression checkpoints on the 5 FGVC FOCI datasets."
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=sorted(DATASET_CONFIGS.keys()),
    )
    parser.add_argument("--image-folder", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument(
        "--options-root",
        default=str(DEFAULT_OPTIONS_ROOT),
        help="Directory containing FinePruner FGVC options json files.",
    )
    parser.add_argument(
        "--results-output-folder",
        default=str(DEFAULT_RESULTS_ROOT),
    )
    parser.add_argument("--prompt-query", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-examples-per-class", type=int, default=999999)
    parser.add_argument("--choice-enumeration", default="ABCD")

    parser.add_argument("--method", default="selector")
    parser.add_argument("--budgets", type=float, default=0.1)
    parser.add_argument("--spatial_ratio", type=float, default=None)
    parser.add_argument("--contextual_ratio", type=float, default=None)
    parser.add_argument("--cosine_penalty_weight", type=float, default=1.0)
    parser.add_argument("--cosine_penalty_threshold", type=float, default=0.7)
    parser.add_argument("--distance_aware_redundancy", type=parse_bool, default=False)
    parser.add_argument("--distance_sigma", type=float, default=0.1)
    parser.add_argument("--distance_floor", type=float, default=0.0)
    parser.add_argument("--distance_apply_to", choices=("focal",), default="focal")

    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=32)

    parser.add_argument("--attn_implementation", default="flash_attention_2")
    parser.add_argument("--torch_dtype", default="bfloat16")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--device_map", default="cuda:0")
    parser.add_argument("--min_pixels", type=int, default=3686400)
    parser.add_argument("--max_pixels", type=int, default=3686400)
    args = parser.parse_args()
    args.spatial_ratio = resolve_spatial_ratio(args, default=0.1)
    args.contextual_ratio = args.spatial_ratio
    return args


if __name__ == "__main__":
    parsed_args = parse_args()
    set_seed(parsed_args.seed)
    eval_model(parsed_args)
