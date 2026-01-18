#include "../include/qwen_model.h"
#include "../include/qwen_env.h"
#include "../include/qwen_model_config.h"
#include <iostream>
#include <iomanip>

// 配置Qwen模型参数（统一入口）
const auto& QWEN_CFG = qwen::get_model_config();
const std::string WEIGHT_PATH = qwen::get_weight_path();

int main() {
    std::cout << "======== Qwen模型诊断工具 ========" << std::endl;

    try {
        qwen::ensure_required_paths(WEIGHT_PATH, qwen::get_tokenizer_script(), qwen::get_tokenizer_model_dir());
    } catch (const std::exception& e) {
        std::cerr << "❌ 路径配置错误: " << e.what() << std::endl;
        return 1;
    }
    
    // 初始化模型（CPU）
    std::cout << "\n初始化模型..." << std::endl;
    QwenModel model = QwenModel(
        QWEN_CFG.vocab_size, QWEN_CFG.hidden_size, QWEN_CFG.num_layers,
        QWEN_CFG.num_heads, QWEN_CFG.num_kv_heads, QWEN_CFG.intermediate_size,
        QWEN_CFG.max_position_embeddings, QWEN_CFG.rope_theta,
        QWEN_CFG.rms_norm_eps, QWEN_CFG.bos_token_id, QWEN_CFG.eos_token_id
    );
    model->eval();
    
    // 加载权重
    std::cout << "加载权重..." << std::endl;
    model->load_weights(WEIGHT_PATH);
    std::cout << "✅ 权重加载完成" << std::endl;
    
    // 保持CPU以便调试
    model->to(torch::kCPU, torch::kBFloat16);
    
    // 准备输入：token ID 108386 ("你好")
    std::cout << "\n输入测试..." << std::endl;
    std::cout << "输入文本: 你好" << std::endl;
    int64_t token_id = 108386;
    std::cout << "Token ID: " << token_id << std::endl;
    
    torch::Tensor input_ids = torch::tensor({{token_id}}, torch::kInt64);
    std::cout << "Input shape: [" << input_ids.size(0) << ", " << input_ids.size(1) << "]" << std::endl;
    
    // 先检查embedding层（通过模型内部调用）
    std::cout << "\n===== 第1步：检查Embedding层 =====" << std::endl;
    std::cout << "说明：无法直接访问embedding层，将通过完整前向传播检查" << std::endl;
    
    std::cout << "\n===== 第2步：执行完整前向传播 =====" << std::endl;
    // 前向传播
    std::cout << "\n===== 第2步：执行完整前向传播 =====" << std::endl;
    // 前向传播
    std::cout << "执行前向传播..." << std::endl;
    torch::NoGradGuard no_grad;
    model->clear_cache();
    
    torch::Tensor logits = model->forward(input_ids, true);
    
    std::cout << "Logits shape: [" << logits.size(0) << ", " << logits.size(1) << ", " << logits.size(2) << "]" << std::endl;
    std::cout << "Logits dtype: " << logits.dtype() << std::endl;
    
    // 获取最后一个位置的logits
    torch::Tensor last_logits = logits[0][0].to(torch::kFloat32);
    
    // 显示前10个logit值
    std::cout << "\n最后一个位置的前10个logit值: ";
    for (int i = 0; i < 10; ++i) {
        std::cout << last_logits[i].item<float>() << " ";
    }
    std::cout << std::endl;
    
    // 统计信息
    std::cout << "Logits统计: ";
    std::cout << "min=" << last_logits.min().item<float>() << ", ";
    std::cout << "max=" << last_logits.max().item<float>() << ", ";
    std::cout << "mean=" << last_logits.mean().item<float>() << std::endl;
    
    std::cout << "Logits统计: ";
    std::cout << "min=" << last_logits.min().item<float>() << ", ";
    std::cout << "max=" << last_logits.max().item<float>() << ", ";
    std::cout << "mean=" << last_logits.mean().item<float>() << std::endl;
    
    std::cout << "\n===== 第3步：预测结果 =====" << std::endl;
    // 预测下一个token
    int64_t next_token = torch::argmax(last_logits).item<int64_t>();
    std::cout << "预测的下一个token ID: " << next_token << std::endl;
    
    // Top 5候选
    auto topk_result = torch::topk(last_logits, 5);
    auto top_logits = std::get<0>(topk_result);
    auto top_indices = std::get<1>(topk_result);
    
    std::cout << "\nTop 5候选:" << std::endl;
    for (int i = 0; i < 5; ++i) {
        int64_t idx = top_indices[i].item<int64_t>();
        float logit = top_logits[i].item<float>();
        std::cout << "  " << (i+1) << ". Token " << idx << " (logit=" << std::fixed << std::setprecision(4) << logit << ")" << std::endl;
    }
    
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "===== 对比PyTorch输出 =====" << std::endl;
    std::cout << std::string(60, '=') << std::endl;
    std::cout << "PyTorch Embedding前10个值应该是:" << std::endl;
    std::cout << "  0.008545, -0.005432, 0.015869, 0.009155, -0.004028, ..." << std::endl;
    std::cout << "\nPyTorch预测: Token 3837 ('，') logit=13.0000" << std::endl;
    std::cout << "C++预测:     Token " << next_token << " logit=" << top_logits[0].item<float>() << std::endl;
    
    if (next_token == 3837) {
        std::cout << "\n✅ 预测匹配！模型工作正常！" << std::endl;
    } else {
        std::cout << "\n❌ 预测不匹配！需要检查：" << std::endl;
        std::cout << "   1. Embedding层输出是否匹配PyTorch" << std::endl;
        std::cout << "   2. 模型架构是否有误（特别是attention机制）" << std::endl;
        std::cout << "   3. 权重加载是否正确" << std::endl;
    }
    std::cout << std::string(60, '=') << std::endl;
    
    return 0;
}
