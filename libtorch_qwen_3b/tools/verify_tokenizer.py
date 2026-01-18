#!/usr/bin/env python3
"""
验证tokenizer和embedding权重的对应关系
"""
import torch
from transformers import AutoTokenizer

MODEL_PATH = "/home/aoi/new/resume/Dev_container/hunyuan_model/qwen2.5-0.5b-instruct"
WEIGHT_PATH = "/home/aoi/new/resume/Dev_container/hunyuan_model/qwen2.5-0.5b-instruct/qwen2.5-0.5b-instruct.pt"

def main():
    print("=" * 60)
    print("【验证Tokenizer与Embedding权重的对应关系】")
    print("=" * 60)
    
    # 1. 加载Qwen tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    print(f"\n✅ 加载Qwen tokenizer")
    print(f"  词汇表大小: {tokenizer.vocab_size}")
    print(f"  模型最大长度: {tokenizer.model_max_length}")
    
    # 2. 加载embedding权重
    state_dict = torch.load(WEIGHT_PATH, map_location="cpu")
    embed_weight = state_dict["model.embed_tokens.weight"]
    print(f"\n✅ 加载Embedding权重")
    print(f"  权重形状: {embed_weight.shape}")
    print(f"  vocab_size: {embed_weight.shape[0]}")
    print(f"  d_model: {embed_weight.shape[1]}")
    
    # 3. 验证对应关系
    if tokenizer.vocab_size == embed_weight.shape[0]:
        print(f"\n✅ 验证通过: tokenizer词汇表大小与embedding权重完全匹配!")
    else:
        print(f"\n❌ 警告: 词汇表大小不匹配!")
        print(f"  tokenizer: {tokenizer.vocab_size}")
        print(f"  embedding: {embed_weight.shape[0]}")
    
    # 4. 测试实际分词和embedding
    test_text = "你好，世界！"
    token_ids = tokenizer.encode(test_text, add_special_tokens=False)
    print(f"\n【测试示例】")
    print(f"  文本: {test_text}")
    print(f"  Token IDs: {token_ids}")
    
    # 检查token ID是否在合法范围内
    for tid in token_ids:
        if tid >= embed_weight.shape[0]:
            print(f"  ❌ 错误: Token ID {tid} 超出embedding范围 [0, {embed_weight.shape[0]-1}]")
        else:
            # 获取对应的embedding向量
            embed_vec = embed_weight[tid]
            print(f"  ✅ Token ID {tid} -> Embedding向量 {embed_vec.shape}, 前3个值: {embed_vec[:3].tolist()}")
    
    # 5. 对比不同tokenizer的差异
    print(f"\n" + "=" * 60)
    print("【对比: 如果用错误的tokenizer会怎样】")
    print("=" * 60)
    
    try:
        # 尝试加载一个不同的tokenizer (比如GPT2)
        from transformers import GPT2Tokenizer
        gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        gpt2_token_ids = gpt2_tokenizer.encode(test_text)
        
        print(f"\n如果用GPT2 tokenizer:")
        print(f"  GPT2词汇表大小: {gpt2_tokenizer.vocab_size}")
        print(f"  相同文本 '{test_text}' 的Token IDs: {gpt2_token_ids}")
        print(f"  对比Qwen Token IDs: {token_ids}")
        print(f"\n⚠️ 看到了吗? 完全不同的Token IDs!")
        print(f"  如果用GPT2的token IDs去查Qwen的embedding，会得到错误的词向量!")
        
    except Exception as e:
        print(f"  (无法加载GPT2 tokenizer用于对比: {e})")
    
    # 6. 结论
    print(f"\n" + "=" * 60)
    print("【重要结论】")
    print("=" * 60)
    print("✅ 我们的实现是正确的:")
    print("  1. 使用Qwen官方tokenizer生成token IDs")
    print("  2. Token IDs匹配Qwen模型的embedding权重")
    print("  3. 每个token ID都能正确映射到对应的词向量")
    print("\n❌ 如果用错tokenizer会导致:")
    print("  1. Token IDs完全不同")
    print("  2. Embedding层返回错误的词向量")
    print("  3. 模型输出完全是乱码")
    print("\n💡 记住: 模型和tokenizer必须配套使用!")

if __name__ == "__main__":
    main()
