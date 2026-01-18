#!/usr/bin/env python3
"""
Qwen分词器Python脚本
用于将中文文本转换为token IDs，供LibTorch C++代码调用
"""
import os
import sys
import json
from transformers import AutoTokenizer

# Qwen模型路径（优先环境变量）
MODEL_PATH = os.environ.get("QWEN_TOKENIZER_MODEL_DIR") or os.environ.get("QWEN_MODEL_DIR") or ""

def tokenize(text: str, use_chat_template: bool = False) -> list:
    """
    将文本转换为token IDs
    Args:
        text: 输入的中文文本
        use_chat_template: 是否使用聊天模板格式
    Returns:
        token IDs列表
    """
    if not MODEL_PATH:
        print("错误：未设置模型路径。请设置环境变量 QWEN_MODEL_DIR 或 QWEN_TOKENIZER_MODEL_DIR", file=sys.stderr)
        sys.exit(1)

    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
        
        if use_chat_template:
            # 使用ChatML格式
            messages = [{"role": "user", "content": text}]
            formatted_text = tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            token_ids = tokenizer.encode(formatted_text, add_special_tokens=False)
        else:
            # 直接编码文本
            token_ids = tokenizer.encode(text, add_special_tokens=False)
        
        return token_ids
    except Exception as e:
        print(f"分词失败: {e}", file=sys.stderr)
        sys.exit(1)

def main():
    if len(sys.argv) < 2:
        print("用法: python tokenize.py <文本> [--chat]", file=sys.stderr)
        print("或通过stdin输入文本", file=sys.stderr)
        print("--chat: 使用聊天模板格式", file=sys.stderr)
        sys.exit(1)
    
    # 检查是否使用聊天模板
    use_chat = "--chat" in sys.argv
    args = [arg for arg in sys.argv[1:] if arg != "--chat"]
    
    # 支持两种输入方式：命令行参数或stdin
    if len(args) > 0 and args[0] == "-":
        text = sys.stdin.read().strip()
    else:
        text = " ".join(args)
    
    if not text:
        print("错误：输入文本为空", file=sys.stderr)
        sys.exit(1)
    
    # 分词并输出JSON格式
    token_ids = tokenize(text, use_chat_template=use_chat)
    result = {
        "text": text,
        "token_ids": token_ids,
        "length": len(token_ids),
        "chat_template": use_chat
    }
    print(json.dumps(result, ensure_ascii=False))

if __name__ == "__main__":
    main()
