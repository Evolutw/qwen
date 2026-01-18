# 生成质量改进 - Repetition Penalty

## 问题描述

模型生成的文本存在重复问题：
```
之前: "你好！很高兴与您有什么问题吗？有什么我可以帮您需要帮助吗？"
```

## 解决方案

### 1. 实现Repetition Penalty

```cpp
torch::Tensor apply_repetition_penalty(torch::Tensor logits, 
                                       const std::vector<int64_t>& generated_tokens,
                                       float penalty = 1.1) {
    if (penalty == 1.0 || generated_tokens.empty()) {
        return logits;
    }
    
    logits = logits.clone();
    for (int64_t token : generated_tokens) {
        float current_logit = logits[token].item<float>();
        // 对已生成的token应用惩罚
        if (current_logit > 0) {
            logits[token] = current_logit / penalty;
        } else {
            logits[token] = current_logit * penalty;
        }
    }
    return logits;
}
```

### 2. 优化采样参数

| 参数 | 之前 | 现在 | 说明 |
|------|------|------|------|
| temperature | 0.7 | **0.8** | 提高随机性 |
| top_k | 50 | **40** | 减少候选集，增加多样性 |
| repetition_penalty | 无 | **1.2** | 惩罚重复token |
| max_tokens | 50 | **30** | 控制长度，避免啰嗦 |

### 3. 生成流程

```cpp
// Prefill阶段
torch::Tensor logits = model->forward(input_ids, true);
torch::Tensor next_logits = apply_repetition_penalty(logits[0][-1], {}, penalty);
int64_t next_token = sample_with_temperature(next_logits, temperature, top_k);

// 自回归生成
for (int i = 1; i < max_tokens; ++i) {
    // 应用repetition penalty（传入已生成的所有tokens）
    next_logits = apply_repetition_penalty(logits[0][0], generated, penalty);
    next_token = sample_with_temperature(next_logits, temperature, top_k);
    generated.push_back(next_token);
}
```

## 效果对比

### 之前（贪婪解码）
- 输出: "你好！很高兴与您有什么问题吗？有什么我可以帮您需要帮助吗？止]"
- 问题: 
  - ❌ 明显的重复："有什么...吗"出现两次
  - ❌ 语句不连贯
  - ❌ 过于啰嗦

### 现在（Temperature + Top-k + Repetition Penalty）
- 输出: "你好！有什么可以帮助你的吗？"
- 改进:
  - ✅ 无重复
  - ✅ 简洁清晰
  - ✅ 符合自然对话

## 参数调优建议

### 不同场景的推荐设置

**1. 创意写作**（高随机性）
```cpp
temperature = 0.9
top_k = 40
repetition_penalty = 1.3  // 强烈避免重复
```

**2. 事实问答**（低随机性）
```cpp
temperature = 0.3
top_k = 20
repetition_penalty = 1.1  // 轻度避免重复
```

**3. 日常对话**（平衡）
```cpp
temperature = 0.7-0.8
top_k = 40-50
repetition_penalty = 1.1-1.2
```

## 技术细节

### Repetition Penalty原理

对于已生成的token，降低其logit值：
- 如果 `logit > 0`: `new_logit = logit / penalty`
- 如果 `logit < 0`: `new_logit = logit * penalty`

这使得模型在后续生成时不太可能再次选择这些token。

### 性能考虑

- **时间复杂度**: O(generated_tokens_count) 每个生成步骤
- **空间复杂度**: O(1) 只需克隆一次logits
- **建议**: penalty值不要太高（1.1-1.5），否则可能导致生成质量下降

## 文件位置

- 实现: [cpp/src/examples/quick_chat_test.cpp](../cpp/src/examples/quick_chat_test.cpp)
- 测试: [cpp/src/tests/test_repetition.cpp](../cpp/src/tests/test_repetition.cpp)

## 运行测试

```bash
make quick       # 编译
make run-quick   # 运行快速测试
```

## 未来改进

1. **实现Top-p (Nucleus) Sampling** - 动态调整候选集大小
2. **添加Temperature Decay** - 随生成长度降低temperature
3. **实现Beam Search** - 更好的序列搜索策略
4. **添加Length Penalty** - 控制生成长度
