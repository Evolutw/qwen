#!/usr/bin/env python3
"""
诊断工具：比较PyTorch和LibTorch的Qwen模型输出
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys

def main():
    model_path = '/home/aoi/new/resume/Dev_container/hunyuan_model/qwen2.5-0.5b-instruct'
    
    # 加载模型和tokenizer
    print("加载PyTorch模型...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        trust_remote_code=True,
        torch_dtype=torch.bfloat16
    )
    model.eval()
    model = model.cpu()  # 使用CPU以便调试
    
    # 准备输入
    text = "你好"
    input_ids = tokenizer.encode(text, return_tensors='pt', add_special_tokens=False)
    print(f"\n输入文本: {text}")
    print(f"Token IDs: {input_ids.tolist()[0]}")
    print(f"Token数量: {len(input_ids[0])}")
    
    # 前向传播
    with torch.no_grad():
        # 获取embedding
        embeddings = model.model.embed_tokens(input_ids)
        print(f"\nEmbedding形状: {embeddings.shape}")
        print(f"Embedding dtype: {embeddings.dtype}")
        print(f"Embedding前10个值: {embeddings[0, 0, :10].tolist()}")
        print(f"Embedding最后10个值: {embeddings[0, 0, -10:].tolist()}")
        print(f"Embedding统计: min={embeddings.min().item():.6f}, max={embeddings.max().item():.6f}, mean={embeddings.mean().item():.6f}")
        
        # 保存embedding到文件供C++对比
        torch.save({
            'embedding': embeddings.cpu(),
            'input_ids': input_ids.cpu()
        }, '/tmp/pytorch_embedding.pt')
        print("✅ Embedding已保存到 /tmp/pytorch_embedding.pt")
        
        # 完整前向传播
        outputs = model(input_ids, return_dict=True)
        logits = outputs.logits
        print(f"\nLogits形状: {logits.shape}")
        print(f"Logits dtype: {logits.dtype}")
        print(f"最后一个位置logits前10个值: {logits[0, -1, :10].tolist()}")
        print(f"Logits统计: min={logits.min().item():.6f}, max={logits.max().item():.6f}, mean={logits.mean().item():.6f}")
        
        # 预测下一个token
        next_token_logits = logits[0, -1, :]
        next_token = torch.argmax(next_token_logits).item()
        next_token_text = tokenizer.decode([next_token])
        print(f"\n预测的下一个token ID: {next_token}")
        print(f"预测的下一个token文本: '{next_token_text}'")
        
        # Top 5候选
        top5_logits, top5_indices = torch.topk(next_token_logits, 5)
        print(f"\nTop 5候选:")
        for i, (logit, idx) in enumerate(zip(top5_logits, top5_indices)):
            token_text = tokenizer.decode([idx.item()])
            print(f"  {i+1}. Token {idx.item()}: '{token_text}' (logit={logit.item():.4f})")
        
        # 保存完整logits供C++对比
        torch.save({
            'logits': logits.cpu(),
            'top5_indices': top5_indices.cpu(),
            'top5_logits': top5_logits.cpu()
        }, '/tmp/pytorch_logits.pt')
        print("\n✅ Logits已保存到 /tmp/pytorch_logits.pt")

if __name__ == "__main__":
    main()
