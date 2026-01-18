#!/usr/bin/env python3
"""
直观对比：正确tokenizer vs 错误tokenizer的影响
"""
import torch
from transformers import AutoTokenizer

MODEL_PATH = "/home/aoi/new/resume/Dev_container/hunyuan_model/qwen2.5-0.5b-instruct"
WEIGHT_PATH = "/home/aoi/new/resume/Dev_container/hunyuan_model/qwen2.5-0.5b-instruct/qwen2.5-0.5b-instruct.pt"

def visualize_tokenization():
    print("=" * 70)
    print("         【直观对比：不同Tokenizer产生的Token IDs】")
    print("=" * 70)
    
    # 加载Qwen tokenizer和embedding
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    state_dict = torch.load(WEIGHT_PATH, map_location="cpu", weights_only=False)
    embed_weight = state_dict["model.embed_tokens.weight"]
    
    test_texts = [
        "你好",
        "人工智能",
        "深度学习",
        "Hello World"
    ]
    
    for text in test_texts:
        print(f"\n📝 原始文本: '{text}'")
        print("-" * 70)
        
        # Qwen tokenizer (正确的)
        qwen_ids = tokenizer.encode(text, add_special_tokens=False)
        print(f"✅ Qwen Tokenizer → Token IDs: {qwen_ids}")
        print(f"   Token数量: {len(qwen_ids)}")
        
        # 显示每个token对应的embedding向量范数（表示向量大小）
        qwen_norms = []
        for tid in qwen_ids:
            if tid < embed_weight.shape[0]:
                vec = embed_weight[tid]
                norm = torch.norm(vec).item()
                qwen_norms.append(f"{norm:.3f}")
        print(f"   对应Embedding向量范数: [{', '.join(qwen_norms)}]")
        
        # 尝试用假设的"错误"token IDs
        # 模拟如果用了其他模型的tokenizer会怎样
        wrong_ids = [i * 1000 % embed_weight.shape[0] for i in range(1, len(qwen_ids) + 1)]
        print(f"\n❌ 假设用了错误的Tokenizer → Token IDs: {wrong_ids}")
        print(f"   (这些ID在Qwen的embedding中指向完全不相关的词)")
        
        wrong_norms = []
        for tid in wrong_ids:
            vec = embed_weight[tid]
            norm = torch.norm(vec).item()
            wrong_norms.append(f"{norm:.3f}")
        print(f"   错误映射的Embedding向量范数: [{', '.join(wrong_norms)}]")
        
        print(f"\n💡 结论: 即使Token数量相同，但ID完全不同，指向的词向量也完全不同!")
    
    print("\n" + "=" * 70)
    print("【关键要点总结】")
    print("=" * 70)
    print("""
1️⃣  每个模型的Tokenizer都是唯一的
    - Qwen有自己的词汇表（151936个token）
    - GPT、LLaMA等都有各自不同的词汇表

2️⃣  Token ID是词汇表的索引
    - "你好" 在Qwen中可能是 [108386, 3837]
    - 在GPT中可能完全不同，如 [19526, 254, 25001, ...]

3️⃣  Embedding权重与Tokenizer严格对应
    - Embedding矩阵第i行 = 词汇表第i个token的向量表示
    - 用错tokenizer → 拿到错误的ID → 查到错误的向量 → 输出乱码

4️⃣  我们的实现是正确的
    ✅ qwen_tokenize.py 使用 AutoTokenizer.from_pretrained(Qwen模型)
    ✅ C++调用这个Python脚本生成token IDs
    ✅ Token IDs正确映射到Qwen的embedding权重
    ✅ 整个流程保证了一致性

⚠️  千万不要混用不同模型的tokenizer和权重!
    """)

if __name__ == "__main__":
    visualize_tokenization()
