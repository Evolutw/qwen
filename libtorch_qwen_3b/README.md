# Qwen 2.5 LibTorch C++ 部署

基于LibTorch的Qwen 2.5模型（0.5B/3B）C++部署实现，统一一套推理代码。

## 项目结构

```
qwen_libtorch_deploy/
├── README.md                 # 项目说明
├── cpp/                      # C++实现
│   ├── include/              # 头文件
│   │   ├── qwen_model.h      # 完整Qwen模型
│   │   ├── qwen_model_config.h # 模型规格配置（0.5B/3B切换）
│   │   ├── qwen_attention.h  # Attention层（含RoPE）
│   │   ├── qwen_mlp.h        # MLP层
│   │   ├── qwen_embedding.h  # Embedding层
│   │   ├── qwen_layernorm.h  # RMSNorm
│   │   └── qwen_transformer_block.h
│   ├── src/
│   │   ├── tests/            # 单元测试
│   │   │   ├── test_embedding.cpp
│   │   │   ├── test_attention.cpp
│   │   │   ├── test_transformer.cpp
│   │   │   ├── test_qwen_model.cpp
│   │   │   ├── test_qwen_simple.cpp
│   │   │   ├── test_qwen_generate.cpp
│   │   │   ├── test_qwen_chat.cpp    # ChatML格式聊天测试
│   │   │   ├── quick_test.cpp
│   │   │   └── final_test.cpp
│   │   ├── examples/         # 示例代码
│   │   │   └── quick_chat_test.cpp   # 简单的聊天示例
│   │   └── diagnostics/      # 诊断工具
│   │       ├── diagnose_model.cpp
│   │       ├── diagnose_chat.cpp
│   │       ├── diagnose_hidden_states.cpp
│   │       ├── diagnose_second_token.cpp
│   │       └── debug_chat.cpp
│   ├── CMakeLists.txt        # CMake构建配置
│   └── build/                # 构建输出目录
├── scripts/                  # 脚本工具
│   └── qwen_tokenize.py      # Python分词脚本
├── tools/                    # 辅助工具
│   ├── check_weight_keys.py  # 检查权重键
│   ├── compare_tokenizers.py # 分词器对比
│   ├── convert.py            # 模型转换
│   ├── diagnose_model.py     # Python诊断工具
│   └── verify_tokenizer.py  # 分词器验证
└── docs/                     # 文档
    ├── README.md             # 详细文档
    ├── PROJECT_SUMMARY.md    # 项目总结
    └── TOKENIZER_EXPLAINED.md # 分词器说明

```

## 模型特性

- **模型**: Qwen 2.5-0.5B / 3B-Instruct
- **词汇表大小**: 151,936
- **隐藏层维度**: 0.5B=896 / 3B=2048
- **层数**: 0.5B=24层 / 3B=36层 Transformer
- **注意力头**: 0.5B=14Q+2KV / 3B=16Q+2KV
- **精度**: BFloat16
- **位置编码**: RoPE (Rotary Position Embedding)
- **支持特性**:
  - ✅ Causal Mask（自回归）
  - ✅ KV Cache加速推理
  - ✅ ChatML格式对话
  - ✅ Temperature采样 + Top-k采样
  - ✅ 中文分词支持

## 快速开始

### 1. 环境要求

- CMake >= 3.18
- LibTorch 2.x (CUDA 11.8)
- Python >= 3.8 (用于分词)
- transformers库

### 2. 构建项目

```bash
cd cpp/build
cmake ..
make
```

### 3. 运行示例

```bash
# 快速聊天测试（贪婪解码）
./bin/quick_chat_test

# 完整聊天测试（温度采样）
./bin/test_qwen_chat

# 单元测试
./bin/test_qwen_model
```

## 核心实现

### 关键修复

**Causal Mask** (2026-01-12修复)
- 问题: Attention层缺少causal mask导致模型能看到未来token
- 修复: 在multi-head attention中添加下三角mask
- 位置: [qwen_attention.h](cpp/include/qwen_attention.h#L150-L165)

**Attention Bias** (2026-01-10修复)
- 问题: Q/K/V投影层缺少bias参数
- 修复: 添加bias支持并正确加载权重
- 位置: [qwen_attention.h](cpp/include/qwen_attention.h#L78-L84)

### 架构亮点

1. **Grouped Query Attention (GQA)**
   - 14个Query头共享2个KV头
   - 7:1的头比例降低KV cache内存

2. **RoPE位置编码**
   - 旋转位置编码，支持外推
   - base=1,000,000

3. **权重共享**
   - LM Head与Embedding层共享权重

## 性能指标

- **推理速度**: 依赖模型规模与硬件
- **内存占用**: 依赖模型规模与KV cache
- **精度**: 与PyTorch输出logit差异 < 0.2

## 常见问题

### Q: 生成质量不佳？
A: 尝试调整temperature (0.1-1.0) 和 top_k (20-100)参数。

### Q: 如何切换0.5B和3B？
A: 修改 cpp/include/qwen_model_config.h 里的 QWEN_MODEL_VARIANT，然后重新编译。

### Q: 编译错误找不到LibTorch？
A: 检查CMakeLists.txt中的CMAKE_PREFIX_PATH是否正确。

### Q: Python分词失败？
A: 确保已安装transformers库并正确设置模型路径。

## 更多文档

- [详细文档](docs/README.md)
- [项目总结](docs/PROJECT_SUMMARY.md)
- [分词器说明](docs/TOKENIZER_EXPLAINED.md)

## 许可证

MIT License

## 致谢

- [Qwen](https://github.com/QwenLM/Qwen) - 阿里云通义千问
- [LibTorch](https://pytorch.org/cppdocs/) - PyTorch C++ API
