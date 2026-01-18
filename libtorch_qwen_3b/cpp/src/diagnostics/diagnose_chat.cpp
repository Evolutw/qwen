#include "../include/qwen_model.h"
#include "../include/qwen_env.h"
#include "../include/qwen_model_config.h"
#include <iostream>
#include <iomanip>
#include <vector>

// 配置Qwen模型参数（统一入口）
const auto& QWEN_CFG = qwen::get_model_config();
const int64_t QWEN_IM_END_ID = QWEN_CFG.im_end_id;
const std::string WEIGHT_PATH = qwen::get_weight_path();

int main() {
    std::cout << "======== Qwen ChatML输入诊断工具 ========" << std::endl;

    try {
        qwen::ensure_required_paths(WEIGHT_PATH, qwen::get_tokenizer_script(), qwen::get_tokenizer_model_dir());
    } catch (const std::exception& e) {
        std::cerr << "❌ 路径配置错误: " << e.what() << std::endl;
        return 1;
    }
    
    // ChatML格式输入: <|im_start|>system\n...<|im_end|>\n<|im_start|>user\n你好<|im_end|>\n<|im_start|>assistant\n
    std::vector<int64_t> chat_tokens = {
        151644, 8948, 198, 2610, 525, 1207, 16948, 11, 3465, 553, 54364, 14817, 13, 1446, 525, 264, 10950, 17847, 13, 
        151645, 198, 151644, 872, 198, 108386, 151645, 198, 151644, 77091, 198
    };
    
    std::cout << "\n输入: ChatML格式对话 '你好'" << std::endl;
    std::cout << "Token数量: " << chat_tokens.size() << std::endl;
    std::cout << "最后5个tokens: ";
    for (size_t i = chat_tokens.size() - 5; i < chat_tokens.size(); ++i) {
        std::cout << chat_tokens[i] << " ";
    }
    std::cout << "\n(应该是: 151645 198 151644 77091 198)" << std::endl;
    std::cout << "(含义: <|im_end|> \\n <|im_start|> assistant \\n)" << std::endl;
    
    // 初始化模型
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
    
    // 移动到GPU（使用bfloat16）
    if (torch::cuda::is_available()) {
        std::cout << "使用CUDA设备" << std::endl;
        model->to(torch::kCUDA, torch::kBFloat16);
    } else {
        std::cout << "使用CPU设备" << std::endl;
        model->to(torch::kCPU, torch::kBFloat16);
    }
    
    // 转换为tensor
    torch::Tensor input_ids = torch::zeros({1, static_cast<long>(chat_tokens.size())}, torch::kInt64);
    for (size_t i = 0; i < chat_tokens.size(); ++i) {
        input_ids[0][i] = chat_tokens[i];
    }
    
    if (torch::cuda::is_available()) {
        input_ids = input_ids.to(torch::kCUDA);
    }
    
    std::cout << "\nInput shape: [" << input_ids.size(0) << ", " << input_ids.size(1) << "]" << std::endl;
    
    // 前向传播
    std::cout << "\n执行前向传播..." << std::endl;
    torch::NoGradGuard no_grad;
    model->clear_cache();
    
    torch::Tensor logits = model->forward(input_ids, true);
    
    std::cout << "Logits shape: [" << logits.size(0) << ", " << logits.size(1) << ", " << logits.size(2) << "]" << std::endl;
    std::cout << "Logits dtype: " << logits.dtype() << std::endl;
    
    // 获取最后一个位置的logits（assistant标记后的第一个预测位置）
    torch::Tensor last_logits = logits[0][-1].to(torch::kFloat32).to(torch::kCPU);
    
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
    
    // 检查IM_END的logit值
    float im_end_logit = last_logits[QWEN_IM_END_ID].item<float>();
    std::cout << "\nIM_END token (151645) 的logit: " << im_end_logit << std::endl;
    
    // 预测下一个token
    int64_t next_token = torch::argmax(last_logits).item<int64_t>();
    std::cout << "预测的下一个token ID: " << next_token << std::endl;
    
    // Top 10候选
    auto topk_result = torch::topk(last_logits, 10);
    auto top_logits = std::get<0>(topk_result);
    auto top_indices = std::get<1>(topk_result);
    
    std::cout << "\nTop 10候选:" << std::endl;
    for (int i = 0; i < 10; ++i) {
        int64_t idx = top_indices[i].item<int64_t>();
        float logit = top_logits[i].item<float>();
        std::cout << "  " << (i+1) << ". Token " << idx 
                  << " (logit=" << std::fixed << std::setprecision(4) << logit << ")";
        if (idx == QWEN_IM_END_ID) {
            std::cout << " <- IM_END";
        }
        std::cout << std::endl;
    }
    
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "===== 对比PyTorch输出 =====" << std::endl;
    std::cout << std::string(60, '=') << std::endl;
    std::cout << "PyTorch预测:" << std::endl;
    std::cout << "  Top-1: Token 108386 (你好) logit=24.3750" << std::endl;
    std::cout << "  IM_END: Token 151645 logit=-3.4531" << std::endl;
    std::cout << "\nC++预测:" << std::endl;
    std::cout << "  Top-1: Token " << next_token << " logit=" << top_logits[0].item<float>() << std::endl;
    std::cout << "  IM_END: Token 151645 logit=" << im_end_logit << std::endl;
    
    if (next_token == 108386 && im_end_logit < 0) {
        std::cout << "\n✅ 预测匹配！模型工作正常！" << std::endl;
    } else {
        std::cout << "\n❌ 预测不匹配！问题分析：" << std::endl;
        if (next_token != 108386) {
            std::cout << "   1. Top-1预测不匹配 (期望108386, 实际" << next_token << ")" << std::endl;
        }
        if (im_end_logit > 0) {
            std::cout << "   2. IM_END的logit异常高 (期望负值, 实际" << im_end_logit << ")" << std::endl;
        }
        std::cout << "\n可能的原因:" << std::endl;
        std::cout << "   - 权重加载不完整或错误" << std::endl;
        std::cout << "   - 模型架构实现有误" << std::endl;
        std::cout << "   - KV cache或位置编码问题" << std::endl;
    }
    std::cout << std::string(60, '=') << std::endl;
    
    return 0;
}
