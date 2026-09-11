# inference.py

import copy
import json
import base64
import logging
import argparse
import torch
import os
from io import BytesIO
from qwen25vl.processing_qwen2_5_vl import Qwen2_5_VLProcessor
from qwen25vl.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
from transformers import AutoModel, AutoTokenizer, AutoProcessor, AutoModelForCausalLM
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from PIL import Image, ImageDraw
import requests
import transformers
import numpy as np
import random

# Added for visualization
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

logger = logging.getLogger(__name__)
os.environ["WANDB_DISABLED"] = "true"


def _resolve_spatial_ratio(args, default: float = 0.1) -> float:
    spatial_ratio = getattr(args, "spatial_ratio", None)
    if spatial_ratio is None:
        spatial_ratio = getattr(args, "contextual_ratio", None)
    return float(default if spatial_ratio is None else spatial_ratio)

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)
    
def parse_args():
    parser = argparse.ArgumentParser(description="A simple inference script to test token prune and kv cache compression methods.")
    # settings for path/basic
    parser.add_argument("--seed", type=int, default=42, help="")
    parser.add_argument("--image_path", type=str, default="../docs/logo.png")
    parser.add_argument("--question", type=str, default="What is shown in this image?",help="the question to ask the model")
    parser.add_argument("--output_dir", type=str, default="vis_results", help="directory to save visualization results")

    # settings for model configuration
    parser.add_argument('--pretrained', type=str, default="Qwen/Qwen2.5-VL-7B-Instruct", help='Pretrained model path or identifier.')
    parser.add_argument('--model_name', type=str, default='qwen25-vl', help='Model name. such as Qwen2-VL-7B-Instruct.')
    parser.add_argument("--use_cache", type=bool, default=True, help="")
    parser.add_argument("--max_new_tokens", type=int, default=2048, help="")
    parser.add_argument("--temperature", type=float, default=0, help="")
    parser.add_argument("--top_p", type=float, default=None, help="")
    parser.add_argument("--num_beams", type=int, default=1, help="")
    parser.add_argument("--attn_implementation", type=str, default="flash_attention_2", help="")
    parser.add_argument("--torch_dtype", type=str, default="bfloat16", help="")
    parser.add_argument("--multimodal", type=bool, default=True, help="")
    parser.add_argument("--device", type=str, default="cuda:0", help="")
    parser.add_argument("--device_map", type=str, default="cuda:0", help="")
    # for qwen25-vl
    parser.add_argument("--max_pixels", type=int, default=2800, help="the max pixels of the image for qwen25-vl")
    parser.add_argument("--min_pixels", type=int, default=400, help="the min pixels of the image for qwen25-vl")
    
    # settings for token pruning method
    parser.add_argument('--method', type=str, choices=['orig', # original model
                                                       'fastv', # token prune method
                                                       'visionzip_official',
                                                       'prumerge+',
                                                       'dart',
                                                       'divprune',
                                                       'dynamic',
                                                       'selector'], help='Token pruning method to use.')
    parser.add_argument("--budgets", type=float, default=0.2, help="budgets of Token Pruning")
    parser.add_argument(
        "--spatial_ratio",
        type=float,
        default=None,
        help="spatial token ratio among kept tokens (selector)",
    )
    parser.add_argument(
        "--contextual_ratio",
        type=float,
        default=None,
        help="deprecated alias of --spatial_ratio",
    )
    parser.add_argument(
        "--cosine_penalty_weight",
        type=float,
        default=0.15,
        help="Scale-matching alpha for cosine penalty (lambda = alpha * IQR(score)/IQR(penalty))",
    )
    parser.add_argument(
        "--cosine_penalty_threshold",
        type=float,
        default=0.85,
        help="Cosine similarity threshold for selector penalty",
    )
    args = parser.parse_args()
    args.spatial_ratio = _resolve_spatial_ratio(args, default=0.1)
    args.contextual_ratio = args.spatial_ratio
    return args

def replace_layers(args,model):
    from token_compression.monkeypatch import replace_qwen25vl
    
    if "qwen25-vl" in args.model_name.lower():
        replace_qwen25vl(args,model,args.method.lower())
    else:
        raise ValueError(f"Model name {args.model_name} not supported")


def run_inference(args):

    if "qwen25-vl" in args.model_name.lower():
        run_inference_qwen25vl(args)
    else:
        raise ValueError(f"Model name {args.model_name} not supported")
    
def _configure_processor_pixel_limits(processor, min_pixels: int, max_pixels: int):
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


def load_model_qwen25vl(args, pretrained, model_name):
    from token_compression.visionzip_official import Qwen2_5_VLForConditionalGeneration_VisionZip
    from token_compression.selector_model import Qwen2_5_VLForConditionalGeneration_Selector
    from token_compression.dynamic_model import Qwen2_5_VLForConditionalGeneration_Dynamic

    model_args = {
        "attn_implementation": args.attn_implementation,
        "device_map": args.device_map, 
        "torch_dtype": args.torch_dtype,
    }

    if args.method.lower() == 'visionzip_official':
        print('using visionzip official')
        model = Qwen2_5_VLForConditionalGeneration_VisionZip.from_pretrained(pretrained, **model_args).eval() 
        model.budget = args.budgets
    elif args.method.lower() == 'selector':
        print('using selector')
        model = Qwen2_5_VLForConditionalGeneration_Selector.from_pretrained(pretrained, **model_args).eval()
        model.visual.budgets = args.budgets
        model.visual.config.budgets = args.budgets
        model.visual.spatial_ratio = float(args.spatial_ratio)
        model.visual.contextual_ratio = model.visual.spatial_ratio
        model.visual.config.spatial_ratio = model.visual.spatial_ratio
        model.visual.config.contextual_ratio = model.visual.spatial_ratio
        model.visual.cosine_penalty_weight = max(0.0, float(args.cosine_penalty_weight))
        model.visual.config.cosine_penalty_weight = model.visual.cosine_penalty_weight
        model.visual.cosine_penalty_threshold = max(
            -1.0, min(1.0, float(args.cosine_penalty_threshold))
        )
        model.visual.config.cosine_penalty_threshold = model.visual.cosine_penalty_threshold
    elif args.method.lower() == 'dynamic':
        print('using dynamic')
        model = Qwen2_5_VLForConditionalGeneration_Dynamic.from_pretrained(pretrained, **model_args).eval()
        model.model.budgets = args.budgets
    else:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(pretrained, **model_args).eval() 

    qwen25vl_processor = Qwen2_5_VLProcessor.from_pretrained(pretrained)
    qwen25vl_processor = _configure_processor_pixel_limits(
        qwen25vl_processor,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
    )
    qwen25vl_tokenizer = AutoTokenizer.from_pretrained(pretrained)

    return model, qwen25vl_processor, qwen25vl_tokenizer

def plot_selector_visualization(
    model,
    image_path,
    question,
    model_answer,
    output_dir="vis_results",
    requested_min_pixels=None,
    requested_max_pixels=None,
    processed_image_sizes=None,
):
    """
    Visualizes the token selection/pruning process by generating a mask overlay and saving textual info.
    """
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    
    # 1. Retrieve visual module and check attributes
    # The visual module might be directly in model.visual (for SelectorModel wrapper) or model.model.visual
    visual_module = None
    if hasattr(model, 'visual'):
        visual_module = model.visual
    elif hasattr(model, 'model') and hasattr(model.model, 'visual'):
        visual_module = model.model.visual
    
    if visual_module is None:
        print("Error: Could not locate visual module for visualization.")
        return

    # Normalize attribute names for different methods (VisionZip/DivPrune compatibility)
    if not hasattr(visual_module, 'last_combined_scores'):
        if hasattr(visual_module, 'last_attention_sum'): # VisionZip
             visual_module.last_combined_scores = visual_module.last_attention_sum
        elif hasattr(visual_module, 'last_divprune_scores'): # DivPrune
             visual_module.last_combined_scores = visual_module.last_divprune_scores

    required_attrs = ['last_combined_scores', 'last_grid_thw', 'last_selected_indices']
    if not all(hasattr(visual_module, attr) and getattr(visual_module, attr) is not None for attr in required_attrs):
        print(f"Warning: Missing visualization attributes ({required_attrs}) in model. Skipping visualization.")
        return

    # 2. Get data
    scores = visual_module.last_combined_scores.float()
    grid_thw = visual_module.last_grid_thw
    selected_indices = visual_module.last_selected_indices.cpu().numpy()
    dom_indices = getattr(visual_module, "last_dominant_indices", None)
    if dom_indices is None:
        dom_indices = getattr(visual_module, "last_focal_indices", None)
    if dom_indices is not None:
        dom_indices = dom_indices.cpu().numpy()
    ctx_indices = getattr(visual_module, "last_contextual_indices", None)
    if ctx_indices is None:
        ctx_indices = getattr(visual_module, "last_spatial_indices", None)
    if ctx_indices is not None:
        ctx_indices = ctx_indices.cpu().numpy()
    
    # 3. Calculate Grid Shape
    # grid_thw shape is [1, 3] -> (T, H, W). We take H and W.
    grid_h, grid_w = grid_thw[0, 1].item(), grid_thw[0, 2].item()
    
    # Qwen2.5-VL typically does a 2x2 patch merge effectively.
    # The number of scores should match the spatial grid size.
    num_patches = len(scores)
    
    # Attempt to reshape scores to 2D heatmap
    # Default assumption: scores map to grid_h//2, grid_w//2
    heatmap_h, heatmap_w = grid_h // 2, grid_w // 2
    
    if heatmap_h * heatmap_w != num_patches:
         print(f"Shape mismatch: Calculated ({heatmap_h}, {heatmap_w}) vs Actual {num_patches}. Attempting auto-reshape.")
         side = int(np.sqrt(num_patches))
         heatmap_h, heatmap_w = side, side
    
    heatmap_shape = (heatmap_h, heatmap_w)
    
    try:
        heatmap = scores.reshape(heatmap_shape).cpu().detach().numpy()
    except Exception as e:
        print(f"Error reshaping scores to {heatmap_shape}: {e}")
        return

    # 4. Prepare Images
    image = Image.open(image_path).convert("RGB")
    original_np = np.array(image)
    image_size = (original_np.shape[1], original_np.shape[0]) # W, H

    # 5. Generate Masks
    # Use a consistent pink tint for retained tokens and keep pruned regions only mildly dimmed.
    dim_factor = 0.58
    highlight_alpha = 0.62
    dom_highlight_alpha = 0.62
    ctx_highlight_alpha = 0.62
    keep_color = (255, 150, 190)
    dom_color = (230, 159, 0)
    ctx_color = (0, 199, 255)
    spatial_ratio = getattr(visual_module, "spatial_ratio", None)
    if spatial_ratio is None:
        spatial_ratio = getattr(visual_module, "contextual_ratio", None)
    spatial_ratio = float(spatial_ratio) if spatial_ratio is not None else None
    use_keep_style_for_dom = spatial_ratio is not None and np.isclose(spatial_ratio, 0.0)
    dimmed_background_np = np.clip(
        original_np.astype(np.float32) * dim_factor,
        0,
        255,
    ).astype(np.uint8)

    def build_selected_mask(indices_np, label):
        selected_mask_lowres = np.zeros(heatmap_shape, dtype=np.uint8)
        try:
            if indices_np is not None and len(indices_np) > 0:
                rows, cols = np.unravel_index(np.asarray(indices_np, dtype=np.int64), heatmap_shape)
                selected_mask_lowres[rows, cols] = 255
        except Exception as e:
            print(f"Error unravelling indices for {label}: {e}")
            return None

        mask_pil = Image.fromarray(selected_mask_lowres, mode="L").resize(image_size, Image.NEAREST)
        return np.array(mask_pil) > 0

    def build_tinted_foreground(highlight_color, alpha=highlight_alpha):
        highlight_color_np = np.asarray(highlight_color, dtype=np.float32).reshape(1, 1, 3)
        return (
            original_np.astype(np.float32) * (1.0 - alpha)
            + highlight_color_np * alpha
        ).clip(0, 255).astype(np.uint8)

    def build_highlighted_image(indices_np, label, highlight_color, alpha=highlight_alpha):
        selected_mask_hires = build_selected_mask(indices_np, label)
        if selected_mask_hires is None:
            return None

        highlighted_np = build_tinted_foreground(highlight_color, alpha=alpha)

        composed_np = dimmed_background_np.copy()
        composed_np[selected_mask_hires] = highlighted_np[selected_mask_hires]
        return composed_np

    def build_combined_highlighted_image(index_groups):
        composed_np = dimmed_background_np.copy()
        for indices_np, label, highlight_color, alpha in index_groups:
            selected_mask_hires = build_selected_mask(indices_np, label)
            if selected_mask_hires is None or not np.any(selected_mask_hires):
                continue
            highlighted_np = build_tinted_foreground(highlight_color, alpha=alpha)
            composed_np[selected_mask_hires] = highlighted_np[selected_mask_hires]
        return composed_np

    def save_rgb_image(image_np, output_path):
        Image.fromarray(image_np.astype(np.uint8)).save(output_path)

    def build_token_grid_preview(
        indices_np,
        label,
        selected_color=keep_color,
        pruned_color=(52, 58, 64),
        grid_color=(255, 255, 255),
    ):
        token_grid = np.zeros((heatmap_h, heatmap_w, 3), dtype=np.uint8)
        token_grid[:, :] = np.asarray(pruned_color, dtype=np.uint8)

        try:
            if indices_np is not None and len(indices_np) > 0:
                rows, cols = np.unravel_index(np.asarray(indices_np, dtype=np.int64), heatmap_shape)
                token_grid[rows, cols] = np.asarray(selected_color, dtype=np.uint8)
        except Exception as e:
            print(f"Error building token grid for {label}: {e}")
            return None

        preview = Image.fromarray(token_grid, mode="RGB").resize(image_size, Image.NEAREST)
        draw = ImageDraw.Draw(preview)

        # Draw cell boundaries so token density changes are visible in the saved image.
        max_grid_lines = 160
        if heatmap_w <= max_grid_lines:
            for col in range(1, heatmap_w):
                x = int(round(col * image_size[0] / float(heatmap_w)))
                draw.line([(x, 0), (x, image_size[1])], fill=grid_color, width=1)
        if heatmap_h <= max_grid_lines:
            for row in range(1, heatmap_h):
                y = int(round(row * image_size[1] / float(heatmap_h)))
                draw.line([(0, y), (image_size[0], y)], fill=grid_color, width=1)

        return np.array(preview)

    masked_image_np = build_highlighted_image(
        selected_indices,
        "all",
        highlight_color=keep_color,
    )
    if masked_image_np is None:
        return

    # --- Heatmap Overlay (Optional but helpful) ---
    logger.info(f"Heatmap stats: min={np.min(heatmap):.4f}, max={np.max(heatmap):.4f}, mean={np.mean(heatmap):.4f}, std={np.std(heatmap):.4f}")

    norm = Normalize(vmin=np.min(heatmap), vmax=np.max(heatmap))
    cmap = plt.get_cmap('jet')
    heat_norm = norm(heatmap)
    colored_heatmap = (cmap(heat_norm)[:, :, :3] * 255).astype(np.uint8)
    heat_pil = Image.fromarray(colored_heatmap).resize(image_size, Image.NEAREST)
    alpha = 0.6
    heatmap_overlay_np = (original_np * (1 - alpha) + np.array(heat_pil) * alpha).astype(np.uint8)

    # 6. Save Outputs
    
    output_img_path = os.path.join(output_dir, f"{base_name}_mask.png")
    save_rgb_image(masked_image_np, output_img_path)

    token_grid_preview_np = build_token_grid_preview(selected_indices, "all")
    if token_grid_preview_np is not None:
        output_grid_path = os.path.join(output_dir, f"{base_name}_token_grid.png")
        save_rgb_image(token_grid_preview_np, output_grid_path)

    # Save Dominant/Contextual Masks (Selector only)
    dom_masked_image_np = (
        build_highlighted_image(dom_indices, "dom", highlight_color=keep_color)
        if dom_indices is not None
        else None
    )
    if dom_masked_image_np is not None:
        output_dom_path = os.path.join(output_dir, f"{base_name}_dom_mask.png")
        save_rgb_image(dom_masked_image_np, output_dom_path)

    ctx_masked_image_np = (
        build_highlighted_image(ctx_indices, "ctx", highlight_color=keep_color)
        if ctx_indices is not None
        else None
    )
    if ctx_masked_image_np is not None:
        output_ctx_path = os.path.join(output_dir, f"{base_name}_ctx_mask.png")
        save_rgb_image(ctx_masked_image_np, output_ctx_path)

    if dom_indices is not None or ctx_indices is not None:
        output_dom_ctx_path = os.path.join(output_dir, f"{base_name}_dom_ctx_mask.png")
        dom_combined_color = keep_color if use_keep_style_for_dom else dom_color
        dom_combined_alpha = highlight_alpha if use_keep_style_for_dom else dom_highlight_alpha
        dom_ctx_masked_image_np = build_combined_highlighted_image(
            [
                (ctx_indices, "ctx", ctx_color, ctx_highlight_alpha),
                (dom_indices, "dom", dom_combined_color, dom_combined_alpha),
            ]
        )
        save_rgb_image(dom_ctx_masked_image_np, output_dom_ctx_path)

    # Save Heatmap Image
    output_heat_path = os.path.join(output_dir, f"{base_name}_heatmap.png")
    save_rgb_image(heatmap_overlay_np, output_heat_path)

    # Save Info Text File
    info_path = os.path.join(output_dir, f"{base_name}_info.txt")
    with open(info_path, "w") as f:
        f.write(f"Image Path: {image_path}\n")
        if requested_min_pixels is not None:
            f.write(f"Requested Min Pixels: {requested_min_pixels}\n")
        if requested_max_pixels is not None:
            f.write(f"Requested Max Pixels: {requested_max_pixels}\n")
        if processed_image_sizes:
            f.write(f"Processed Image Size(s): {processed_image_sizes}\n")
        f.write(
            f"Method: {visual_module.budgets if hasattr(visual_module, 'budgets') else 'Unknown'}\n"
        )
        if dom_indices is not None:
            f.write(f"Dominant Tokens: {len(dom_indices)}\n")
        if ctx_indices is not None:
            f.write(f"Contextual Tokens: {len(ctx_indices)}\n")
        f.write("-" * 20 + "\n")
        f.write(f"Question: {question}\n")
        f.write("-" * 20 + "\n")
        f.write(f"Model Answer: {model_answer}\n")
        f.write("-" * 20 + "\n")
        f.write(f"Image Grid THW: {grid_thw.tolist()}\n")
        f.write(f"Token Grid: {heatmap_h} x {heatmap_w}\n")
        f.write(f"Total Patches: {num_patches}\n")
        f.write(f"Selected Patches: {len(selected_indices)}\n")
        f.write(f"Retention Rate: {len(selected_indices)/num_patches:.2%}\n")

    print(f"Visualizations saved to {output_dir}")


def run_inference_qwen25vl(args):

    model, qwen25vl_processor, qwen25vl_tokenizer = load_model_qwen25vl(args, args.pretrained, args.model_name)
    replace_layers(args,model)
    if args.method == "selector":
        os.environ.setdefault("SELECTOR_STORE_VIS_ATTRS", "1")

    messages = []
    context = args.question
    message = [{"role": "system", "content": "You are a helpful assistant."}]
    # process image
    if not os.path.exists(args.image_path):
        print(f"Error: Image path {args.image_path} does not exist.")
        return

    visual = Image.open(args.image_path)
    base64_image = visual.convert("RGB")
    buffer = BytesIO()
    base64_image.save(buffer, format="JPEG")
    base64_bytes = base64.b64encode(buffer.getvalue())
    base64_string = base64_bytes.decode("utf-8")

    message.append(
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": f"data:image/jpeg;base64,{base64_string}",
                    "min_pixels": args.min_pixels,
                    "max_pixels": args.max_pixels,
                },
                {"type": "text", "text": context},
            ],
        }
    )
    messages.append(message)
    texts = [qwen25vl_processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in messages]
    image_inputs, video_inputs = process_vision_info(messages)
    processed_image_sizes = [img.size for img in image_inputs] if image_inputs is not None else []
    print(
        f"Requested pixel range: min={args.min_pixels}, max={args.max_pixels}; "
        f"processed image size(s): {processed_image_sizes}"
    )
    inputs = qwen25vl_processor(text=texts, images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")

    if args.device_map == "auto":
        inputs = inputs.to("cuda")
    else:
        inputs = inputs.to(args.device)
    
    # --- Inject Grid Info for Visualization ---
    # The visual encoder needs 'grid_thw' to know the image layout for pruning.
    # Normally this is handled inside the model forward, but for visualization we might need to access it manually later
    # or ensuring it's passed correctly. The Qwen2.5-VL processor output 'image_grid_thw' usually.
    
    if 'image_grid_thw' in inputs:
        # Determine where 'visual' is located and inject the grid info
        target_visual = None
        if hasattr(model, 'visual'):
            target_visual = model.visual
        elif hasattr(model, 'model') and hasattr(model.model, 'visual'):
            target_visual = model.model.visual
        
        if target_visual is not None:
            # We explicitly set this so the 'forward' pass of the visual encoder 
            # (which might have run during generate -> prepare_inputs) uses or caches it.
            # However, Qwen2.5 typically processes vision *inside* generate -> model call.
            # We just need to make sure our visualization function can find it later if the model code saves it.
            # The 'Selector' model code provided earlier saves `last_grid_thw` during its forward pass.
            # So we typically don't need to force inject it HERE, but we can as a failsafe if needed.
            target_visual.last_grid_thw = inputs['image_grid_thw']
    # ------------------------------------------

    pad_token_id = qwen25vl_tokenizer.pad_token_id

    # generate
    cont = model.generate(
        **inputs,
        eos_token_id= qwen25vl_tokenizer.eos_token_id,
        pad_token_id=pad_token_id,
        do_sample=True if args.temperature > 0 else False,
        temperature=args.temperature,
        top_p=args.top_p,
        num_beams=args.num_beams,
        max_new_tokens=args.max_new_tokens,
        use_cache=args.use_cache,
    )

    generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, cont)]
    answers = qwen25vl_processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    
    final_answer = answers[0]
    print(f"Answer: {final_answer}")

    # Call Visualization
    if args.method in ['selector', 'visionzip_official', 'divprune']:
        print(f"Generating visualization for method: {args.method}")
        plot_selector_visualization(
            model,
            args.image_path,
            args.question,
            final_answer,
            output_dir=args.output_dir,
            requested_min_pixels=args.min_pixels,
            requested_max_pixels=args.max_pixels,
            processed_image_sizes=processed_image_sizes,
        )


if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    run_inference(args)
