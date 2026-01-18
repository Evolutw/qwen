# 保存为 check_weight_keys.py
import torch
from safetensors.torch import load_file

# 你的safetensors路径
SAFETENSORS_PATH = "/home/aoi/new/resume/Dev_container/hunyuan_model/qwen2.5-0.5b-instruct/model.safetensors"

# 加载权重并打印所有键名
state_dict = load_file(SAFETENSORS_PATH, device="cpu")
print(f"权重总数：{len(state_dict)}")
print("\n所有权重键名（前20个+包含embedding/lm_head/wte的键名）：")

# 打印前20个键名
for i, (key, _) in enumerate(state_dict.items()):
    if i < 20:
        print(f"  {key}")
    # 筛选包含关键词的键名（找到Embedding和LMHead）
    if any(k in key.lower() for k in ["embedding", "wte", "lm_head", "lmhead"]):
        print(f"🔍 关键权重：{key}")

# 若上述筛选未找到，直接打印所有包含"weight"的键名
print("\n所有包含'weight'的键名（筛选关键权重）：")
for key, _ in state_dict.items():
    if "weight" in key:
        print(f"  {key}")
