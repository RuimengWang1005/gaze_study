import os
# 🚨 终极核武器：在导入任何包之前，强行打开师姐的“坐标保留开关”！
os.environ["SELECTOR_STORE_VIS_ATTRS"] = "1"

import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import glob
from transformers import AutoProcessor
from token_compression.selector_model import Qwen2_5_VLForConditionalGeneration_Selector
from token_compression.monkeypatch import replace_qwen25vl

class DummyArgs:
    method = "selector"

model_path = os.path.abspath("../../VisionSelector-Qwen2.5-VL-3B")
print("⏳ 正在加载模型...")
processor = AutoProcessor.from_pretrained(model_path)
model = Qwen2_5_VLForConditionalGeneration_Selector.from_pretrained(
    model_path, device_map="cuda:3", torch_dtype=torch.bfloat16
).eval()
replace_qwen25vl(DummyArgs(), model, "selector")

# ================= 设定 10% 极限压缩 =================
CURRENT_RATIO = 0.1  
model.visual.budgets = CURRENT_RATIO
model.visual.config.budgets = CURRENT_RATIO
model.visual.spatial_ratio = 0.1
model.visual.cosine_penalty_weight = 1.0

# 找一张真实的猫狗图片
search_pattern = '/data/DataSet/cls_dataset/oxford_pets/images/Bombay_27.jpg'
found_images = glob.glob(search_pattern, recursive=True)
if not found_images:
    print("❌ 找不到图片，请检查路径！")
    exit()
image_path = found_images[0] # 也可以手动指定一张你喜欢的布偶猫绝对路径

image = Image.open(image_path).convert("RGB")
prompt_text = "What is the exact breed of the animal?"

messages = [{"role": "user", "content": [{"type": "image", "image": image_path}, {"type": "text", "text": prompt_text}]}]
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt").to("cuda:3")

print("🧠 模型正在看图 (这次它不敢把坐标扔掉了!)...")
with torch.no_grad():
    model.generate(**inputs, max_new_tokens=10)

# ================= 🚨 再次暴力搜索 =================
found_masks = []

for attr_name in dir(model.visual):
    if "mask" in attr_name.lower() or "idx" in attr_name.lower() or "keep" in attr_name.lower() or "vis" in attr_name.lower():
        attr_val = getattr(model.visual, attr_name)
        if isinstance(attr_val, torch.Tensor):
            print(f"🎯 抓到内鬼了！model.visual.{attr_name}, 形状: {attr_val.shape}")
            found_masks.append((attr_name, attr_val.cpu().numpy()))

for name, module in model.visual.named_modules():
    for attr_name in dir(module):
        if "mask" in attr_name.lower() or "keep" in attr_name.lower() or "idx" in attr_name.lower():
            attr_val = getattr(module, attr_name)
            if isinstance(attr_val, torch.Tensor):
                print(f"🎯 抓到内鬼了！{name}.{attr_name}, 形状: {attr_val.shape}")
                found_masks.append((f"{name}.{attr_name}", attr_val.cpu().numpy()))

if not found_masks:
    print("\n😭 难道开关名字不对？你可能需要把刚才那张图发给导师说：师姐隐藏了可视化接口...")
else:
    print(f"\n🎉 成功截获！画图准备中...")
    # 假设我们拿到了名字类似 spatial_idx 或 mask 的张量
    # 这里只是为了确认抓到了什么，一旦你把终端打印出来的形状发给我，我就能写出完美的画红点代码！