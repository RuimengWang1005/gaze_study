# import os
# import torch
# from PIL import Image
# from transformers import AutoProcessor

# # 🚨 关键真相1：必须导入师姐魔改后的 Selector 模型！
# # 确保 test_cat.py 放在 qwen-evaluation1 文件夹下运行，这样才能找到 token_compression 包
# from token_compression.selector_model import Qwen2_5_VLForConditionalGeneration_Selector 

# # 1. 绝对路径配置
# model_path = os.path.abspath("../../VisionSelector-Qwen2.5-VL-3B")
# # 🚨 请换成你挑的那张布偶猫的绝对路径
# image_path = '/data/DataSet/cls_dataset/oxford_pets/images/scottish_terrier_110.jpg'

# print("⏳ 正在加载模型和 Processor...")
# processor = AutoProcessor.from_pretrained(model_path)

# # 2. 加载师姐的魔改模型 (注意去掉了惹祸的 retention_ratio)
# model = Qwen2_5_VLForConditionalGeneration_Selector.from_pretrained(
#     model_path, 
#     device_map="cuda:3", 
#     torch_dtype=torch.bfloat16
# ).eval()
# print("✅ 模型加载完毕！")

# # ==========================================
# # 🚨 你要切换的开关：测完 1.0 后，把它改成 0.1 再跑一次！
# CURRENT_RATIO = 1.0  
# # ==========================================

# # 3. 🚨 关键真相2：像师姐源码里那样，把压缩率强行塞进模型底层！
# model.visual.budgets = CURRENT_RATIO
# model.visual.config.budgets = CURRENT_RATIO
# # 顺便把空间网格比例也配上 (默认 0.1)
# model.visual.spatial_ratio = 0.1
# model.visual.config.spatial_ratio = 0.1
# model.visual.cosine_penalty_weight = 1.0

# # 4. 构建直接发问的 Prompt (不搞选择题，直接考验模型的真本事)
# prompt_text = "Look at this image closely. What is the specific breed of the animal shown in the image? Answer with the exact breed name."

# messages = [
#     {"role": "user", "content": [
#         {"type": "image", "image": image_path},
#         {"type": "text", "text": prompt_text}
#     ]}
# ]
# text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
# image = Image.open(image_path).convert("RGB")

# inputs = processor(
#     text=[text],
#     images=[image],
#     padding=True,
#     return_tensors="pt"
# ).to("cuda:3")

# print(f"🧠 正在以 {CURRENT_RATIO*100}% 的留存率进行思考...")

# # 5. 开始生成回答
# with torch.no_grad():
#     generated_ids = model.generate(**inputs, max_new_tokens=50)

# generated_ids_trimmed = [
#     out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
# ]
# output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

# print("\n" + "🔥"*25)
# print(f"📸 测试图片: {image_path}")
# print(f"🔪 Token留存率: {CURRENT_RATIO}")
# print(f"🤖 模型的回答是: {output_text.strip()}")
# print("🔥"*25 + "\n")

import os
import glob
import torch
from PIL import Image
from transformers import AutoProcessor
from token_compression.selector_model import Qwen2_5_VLForConditionalGeneration_Selector
# 🚨 引入决定生死的补丁函数！
from token_compression.monkeypatch import replace_qwen25vl

class DummyArgs:
    method = "selector"

model_path = os.path.abspath("../../VisionSelector-Qwen2.5-VL-3B")
print("⏳ 正在加载模型和 Processor...")
processor = AutoProcessor.from_pretrained(model_path)
model = Qwen2_5_VLForConditionalGeneration_Selector.from_pretrained(
    model_path, 
    device_map="cuda:3", 
    torch_dtype=torch.bfloat16
).eval()

# ==========================================
# 🚨 致命一步：打上猴子补丁，强制接管底层运算！
replace_qwen25vl(DummyArgs(), model, "selector")
print("✅ 模型加载并成功打上 GAZE 补丁！")
# ==========================================

# 我们直接测 0.1，看看是不是真的能防住！
CURRENT_RATIO = 0.1  
model.visual.budgets = CURRENT_RATIO
model.visual.config.budgets = CURRENT_RATIO
model.visual.spatial_ratio = 0.1
model.visual.config.spatial_ratio = 0.1
model.visual.cosine_penalty_weight = 1.0

# 找 10 张图批量处刑
search_pattern = '/data/DataSet/cls_dataset/oxford_pets/**/*.jpg'
found_images = glob.glob(search_pattern, recursive=True)[:10]

prompt_text = "Look at this image closely. What is the specific breed of the animal shown in the image? Answer with the exact breed name."

print(f"\n🚀 开始批量测试 10 张图片，留存率强制锁定在 {CURRENT_RATIO*100}% ...")

correct_guesses = 0
for i, image_path in enumerate(found_images):
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image_path},
        {"type": "text", "text": prompt_text}
    ]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image = Image.open(image_path).convert("RGB")
    inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt").to("cuda:3")

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=20)
    
    generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()
    
    # 提取图片真实的品种名字 (由文件夹或者文件名决定)
    ground_truth = os.path.basename(image_path).split('_')[0]
    
    print(f"[{i+1}/10] 真实: {ground_truth:15s} | 预测: {output_text}")

print("🔥 测试结束！赶紧看看它是不是开始胡言乱语了！")