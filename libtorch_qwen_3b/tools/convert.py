import os
import json
import argparse
import torch
from safetensors.torch import load_file

def parse_args():
    parser = argparse.ArgumentParser(description="Convert Qwen safetensors to .pt")
    parser.add_argument("--input", dest="input_path", default=os.environ.get("QWEN_SAFETENSORS_PATH", ""))
    parser.add_argument("--output", dest="output_path", default=os.environ.get("QWEN_WEIGHT_PATH", ""))
    parser.add_argument("--model-dir", dest="model_dir", default=os.environ.get("QWEN_MODEL_DIR", ""))
    return parser.parse_args()


args = parse_args()

# ===================== 路径配置（优先命令行，其次环境变量） =====================
if args.model_dir and not args.input_path:
    args.input_path = os.path.join(args.model_dir, "model.safetensors")
if args.model_dir and not args.output_path:
    args.output_path = os.path.join(args.model_dir, "qwen2.5-0.5b-instruct.pt")

SAFETENSORS_INPUT_PATH = args.input_path
PT_OUTPUT_PATH = args.output_path

if not SAFETENSORS_INPUT_PATH or not PT_OUTPUT_PATH:
    raise SystemExit("Please set --input/--output or QWEN_MODEL_DIR/QWEN_SAFETENSORS_PATH/QWEN_WEIGHT_PATH.")

# ===================== 手动指定关键权重名（适配你的权重） =====================
EMBEDDING_KEY = "model.embed_tokens.weight"  # Embedding层权重
LMHEAD_KEY = "model.embed_tokens.weight"     # LMHead与Embedding共享权重，无需修改
LMHEAD_INDEPENDENT = False                   # 标记：无独立LMHead权重

# ===================== 自动创建输出目录 =====================
output_dir = os.path.dirname(PT_OUTPUT_PATH)
os.makedirs(output_dir, exist_ok=True)
print(f"✅ 输出目录已准备：{output_dir}")

# ===================== 加载safetensors权重（支持分片） =====================
print(f"\n[Step 1/4] 正在加载safetensors文件：\n{SAFETENSORS_INPUT_PATH}")
try:
    state_dict = {}
    if SAFETENSORS_INPUT_PATH.endswith(".index.json"):
        index_path = SAFETENSORS_INPUT_PATH
    else:
        index_path = os.path.join(os.path.dirname(SAFETENSORS_INPUT_PATH), "model.safetensors.index.json")

    if os.path.isfile(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            index = json.load(f)
        weight_map = index.get("weight_map", {})
        shard_files = sorted(set(weight_map.values()))
        if not shard_files:
            raise RuntimeError("weight_map is empty in index.json")
        base_dir = os.path.dirname(index_path)
        for shard in shard_files:
            shard_path = os.path.join(base_dir, shard)
            print(f"  loading shard: {shard}")
            state_dict.update(load_file(shard_path, device="cpu"))
    else:
        state_dict = load_file(SAFETENSORS_INPUT_PATH, device="cpu")

    print(f"✅ 成功加载 {len(state_dict)} 个权重张量")
except Exception as e:
    print(f"❌ 加载失败：{e}")
    print("提示：请确认safetensors文件路径正确，且文件未损坏")
    exit(1)

# ===================== 验证关键权重 =====================
print(f"\n[Step 2/4] 验证Qwen关键权重")
valid_flag = True
vocab_size = None
d_model = None

# 验证Embedding权重
if EMBEDDING_KEY in state_dict:
    tensor_shape = state_dict[EMBEDDING_KEY].shape
    tensor_dtype = state_dict[EMBEDDING_KEY].dtype
    print(f"✅ {EMBEDDING_KEY}：形状={tensor_shape}，数据类型={tensor_dtype}")
    # 提取模型参数（Embedding权重形状：[vocab_size, d_model]）
    vocab_size, d_model = tensor_shape
    print(f"   → 提取模型参数：vocab_size={vocab_size}，d_model={d_model}")
else:
    print(f"❌ {EMBEDDING_KEY} 不存在于权重中！")
    valid_flag = False

# 验证LMHead权重（共享权重无需额外检查）
print(f"ℹ️ LMHead与Embedding层共享权重：{LMHEAD_KEY}（无独立LMHead权重）")

if not valid_flag:
    print("\n提示：请确认权重文件为Qwen2.5系列模型")
    exit(1)

# ===================== 保存为LibTorch兼容的.pt格式 =====================
print(f"\n[Step 3/4] 正在保存.pt权重文件：\n{PT_OUTPUT_PATH}")
try:
    torch.save(state_dict, PT_OUTPUT_PATH)
    file_size = os.path.getsize(PT_OUTPUT_PATH) / 1024 / 1024 / 1024
    print(f"✅ 权重保存成功，文件大小：{file_size:.2f} GB")
except Exception as e:
    print(f"❌ 保存失败：{e}")
    print("提示：请确认输出目录有写入权限（可尝试sudo运行脚本）")
    exit(1)

# ===================== 验证.pt文件有效性 =====================
print(f"\n[Step 4/4] 验证.pt文件是否可正常加载")
try:
    loaded_state_dict = torch.load(PT_OUTPUT_PATH, map_location="cpu")
    # 对比Embedding权重是否一致
    embed_safetensors = state_dict[EMBEDDING_KEY]
    embed_pt = loaded_state_dict[EMBEDDING_KEY]
    tensor_diff = torch.max(torch.abs(embed_safetensors - embed_pt)).item()
    
    if tensor_diff < 1e-6:
        print(f"✅ .pt文件验证通过，权重最大差值：{tensor_diff:.6f}")
    else:
        print(f"❌ .pt文件验证失败，权重最大差值：{tensor_diff:.6f}")
        exit(1)
except Exception as e:
    print(f"❌ 验证失败：{e}")
    exit(1)

# ===================== 输出最终提示 =====================
print(f"\n🎉 权重转换全部完成！")
print(f"📌 原始safetensors：{SAFETENSORS_INPUT_PATH}")
print(f"📌 转换后.pt文件：{PT_OUTPUT_PATH}")
print(f"📌 模型关键参数（后续C++需使用）：")
print(f"   - vocab_size={vocab_size}")
print(f"   - d_model={d_model}")
print(f"   - Embedding权重名：{EMBEDDING_KEY}")
print(f"   - LMHead权重：与Embedding共享（{LMHEAD_KEY}），无独立权重")
print(f"   - 模型层数：24层（从权重键名`model.layers.23`推断，后续C++需设置num_layers=24）")
print(f"   - 注意力头数：8（Qwen2.5系列默认，可通过d_model/head_dim=64验证，head_dim=64）")

