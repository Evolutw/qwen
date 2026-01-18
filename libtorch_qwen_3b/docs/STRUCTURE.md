# 项目目录结构说明

## 整理后的目录结构

```
qwen_libtorch_deploy/
│
├── 📄 README.md              项目说明文档
├── 📄 Makefile               便捷构建脚本
├── 📄 .gitignore             Git忽略配置
│
├── 📁 cpp/                   C++实现主目录
│   ├── 📄 CMakeLists.txt     CMake构建配置
│   ├── 📁 build/             构建输出目录（自动生成）
│   │   └── bin/              可执行文件输出
│   ├── 📁 include/           头文件目录
│   │   ├── qwen_model.h              完整Qwen模型定义
│   │   ├── qwen_attention.h          Attention层（含RoPE、GQA）
│   │   ├── qwen_mlp.h                MLP前馈网络
│   │   ├── qwen_embedding.h          词嵌入层
│   │   ├── qwen_layernorm.h          RMSNorm实现
│   │   ├── qwen_transformer_block.h  Transformer Block
│   │   └── qwen_tokenizer.h          分词器接口（已废弃）
│   └── 📁 src/               源文件目录
│       ├── 📁 tests/         单元测试
│       │   ├── test_embedding.cpp        Embedding层测试
│       │   ├── test_attention.cpp        Attention层测试
│       │   ├── test_transformer.cpp      Transformer Block测试
│       │   ├── test_qwen_model.cpp       完整模型测试
│       │   ├── test_qwen_simple.cpp      简化前向传播测试
│       │   ├── test_qwen_generate.cpp    文本生成测试
│       │   ├── test_qwen_chat.cpp        ChatML对话测试（主要）
│       │   ├── quick_test.cpp            快速验证测试
│       │   └── final_test.cpp            综合测试
│       ├── 📁 examples/      示例代码
│       │   └── quick_chat_test.cpp       快速聊天示例（推荐）
│       └── 📁 diagnostics/   诊断工具
│           ├── diagnose_model.cpp        模型输出诊断
│           ├── diagnose_chat.cpp         ChatML输入诊断
│           ├── diagnose_hidden_states.cpp Hidden States诊断
│           ├── diagnose_second_token.cpp 第二个token诊断
│           └── debug_chat.cpp            ChatML调试工具
│
├── 📁 scripts/               运行时脚本
│   └── qwen_tokenize.py      Python分词脚本（C++调用）
│
├── 📁 tools/                 开发工具
│   ├── check_weight_keys.py   检查权重文件的键
│   ├── compare_tokenizers.py  对比不同分词器
│   ├── convert.py             模型格式转换
│   ├── diagnose_model.py      Python诊断脚本
│   └── verify_tokenizer.py    验证分词器正确性
│
└── 📁 docs/                  文档目录
    ├── README.md              详细技术文档
    ├── PROJECT_SUMMARY.md     项目开发总结
    └── TOKENIZER_EXPLAINED.md 分词器使用说明

```

## 文件分类说明

### 核心实现文件（cpp/include/）
- **qwen_model.h**: 完整模型封装，包含forward、load_weights等接口
- **qwen_attention.h**: ⚠️ 关键文件 - 包含Causal Mask修复
- **qwen_mlp.h**: SwiGLU激活的MLP层
- **qwen_embedding.h**: Token嵌入层
- **qwen_layernorm.h**: RMSNorm归一化
- **qwen_transformer_block.h**: Transformer层组装

### 推荐使用的文件
1. **测试**: `cpp/src/tests/test_qwen_chat.cpp` - 完整的ChatML对话测试
2. **示例**: `cpp/src/examples/quick_chat_test.cpp` - 简单易懂的使用示例
3. **分词**: `scripts/qwen_tokenize.py` - Python分词脚本

### 诊断工具（仅开发时使用）
- 用于调试模型输出、对比PyTorch、检查中间状态
- 正常使用无需关注

## 构建和使用

### 快速开始
```bash
# 查看可用命令
make help

# 编译并运行快速聊天示例
make run-quick

# 编译并运行完整测试
make run-chat

# 编译所有程序
make build

# 清理构建文件
make clean
```

### 手动构建
```bash
cd cpp/build
cmake ..
make test_qwen_chat
./bin/test_qwen_chat
```

## 关键修复记录

### 2026-01-12: Causal Mask修复
- **文件**: `cpp/include/qwen_attention.h`
- **问题**: Attention层缺少causal mask导致能看到未来token
- **影响**: ChatML格式生成立即停止
- **状态**: ✅ 已修复

### 2026-01-10: Attention Bias修复
- **文件**: `cpp/include/qwen_attention.h`
- **问题**: Q/K/V投影层缺少bias参数
- **影响**: 输出乱码
- **状态**: ✅ 已修复

## 依赖说明

### C++编译依赖
- CMake >= 3.18
- LibTorch 2.x
- CUDA 11.8 (可选，用于GPU加速)

### Python运行依赖
- Python >= 3.8
- transformers
- torch

## 注意事项

1. **权重路径**: 所有测试文件中硬编码了权重路径，需根据实际情况修改
2. **分词器路径**: 已统一更新为 `scripts/qwen_tokenize.py`
3. **构建目录**: `cpp/build/` 目录会在构建时自动创建
4. **输出位置**: 所有可执行文件输出到 `cpp/build/bin/`

## 文件清理

以下文件/目录已从项目中移除：
- ❌ 根目录下的测试文件（已移到cpp/src/tests/）
- ❌ __pycache__/ 缓存目录
- ❌ 空的examples/和tests/目录

## 后续计划

- [ ] 添加Python binding
- [ ] 支持更多采样策略（Top-p, Beam Search）
- [ ] 优化推理速度
- [ ] 支持批处理推理
- [ ] 添加更多使用示例
