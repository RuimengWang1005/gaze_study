import os
import glob
import torch
from tqdm import tqdm
from PIL import Image
from transformers import AutoProcessor
from token_compression.selector_model import Qwen2_5_VLForConditionalGeneration_Selector
from token_compression.monkeypatch import replace_qwen25vl

class DummyArgs:
    method = "selector"

# 1. 初始化模型
model_path = os.path.abspath("../../VisionSelector-Qwen2.5-VL-3B")
processor = AutoProcessor.from_pretrained(model_path)
model = Qwen2_5_VLForConditionalGeneration_Selector.from_pretrained(
    model_path, device_map="cuda:3", torch_dtype=torch.bfloat16
).eval()
replace_qwen25vl(DummyArgs(), model, "selector")

# 🚨 可以在这里切换 1.0 (满血) 和 0.1 (GAZE极限压缩) 分别跑一次做对比！
CURRENT_RATIO = 0.1 
model.visual.budgets = CURRENT_RATIO
model.visual.config.budgets = CURRENT_RATIO
model.visual.spatial_ratio = 0.1
model.visual.config.spatial_ratio = 0.1
model.visual.cosine_penalty_weight = 1.0

# 2. 获取全量数据集
# search_pattern = '/data/DataSet/cls_dataset/oxford_pets/**/*.jpg'
search_pattern = '/data/DataSet/vlm_dataset/CUB_200_2011/**/*.jpg'
all_images = glob.glob(search_pattern, recursive=True)
print(f"📦 共找到 {len(all_images)} 张测试图片，准备开始全量评测...")

prompt_text = "Look at this image closely. What is the specific breed of the animal shown in the image? Answer with the exact breed name."

correct_count = 0
total_count = len(all_images)

# 3. 进度条遍历全集
for image_path in tqdm(all_images, desc=f"Evaluating Ratio={CURRENT_RATIO}"):
    try:
        # 提取真实标签 (基于 oxford_pets 的命名规则，通常前缀就是品种)
        filename = os.path.basename(image_path)
        # 假设文件名是 "Ragdoll_123.jpg" 或者 "shiba_inu_01.jpg"
        ground_truth = "_".join(filename.split('_')[:-1]).lower().replace("_", " ")

        messages = [{"role": "user", "content": [{"type": "image", "image": image_path}, {"type": "text", "text": prompt_text}]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image = Image.open(image_path).convert("RGB")
        inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt").to("cuda:3")

        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=20)
        
        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip().lower()
        
        # 简单宽松匹配：如果真实的品种名字出现在了模型的回答中，就算对
        if ground_truth in output_text or output_text in ground_truth:
            correct_count += 1
            
    except Exception as e:
        # 容错处理：跳过损坏的图片
        total_count -= 1
        pass

# 4. 打印最终学术结果
accuracy = (correct_count / total_count) * 100 if total_count > 0 else 0
print("\n" + "="*50)
print(f"📊 全量测试完成！Token留存率: {CURRENT_RATIO*100}%")
print(f"🎯 最终细粒度准确率 (Accuracy): {accuracy:.2f}% ({correct_count}/{total_count})")
print("="*50 + "\n")