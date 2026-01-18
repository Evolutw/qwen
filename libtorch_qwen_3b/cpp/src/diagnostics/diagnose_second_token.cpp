#include "../include/qwen_model.h"
#include "../include/qwen_env.h"
#include "../include/qwen_model_config.h"
#include <iostream>

const auto& QWEN_CFG = qwen::get_model_config();
const int64_t QWEN_IM_END_ID = QWEN_CFG.im_end_id;
const std::string WEIGHT_PATH = qwen::get_weight_path();

int main() {
    std::cout << "======== 诊断第二个Token生成 ========\n" << std::endl;

    try {
        qwen::ensure_required_paths(WEIGHT_PATH, qwen::get_tokenizer_script(), qwen::get_tokenizer_model_dir());
    } catch (const std::exception& e) {
        std::cerr << "❌ 路径配置错误: " << e.what() << std::endl;
        return 1;
    }
    
    // ChatML输入
    std::vector<int64_t> chat_tokens = {
        151644, 8948, 198, 2610, 525, 1207, 16948, 11, 3465, 553, 54364, 14817, 13, 1446, 525, 264, 10950, 17847, 13, 
        151645, 198, 151644, 872, 198, 108386, 151645, 198, 151644, 77091, 198
    };
    
    QwenModel model = QwenModel(
        QWEN_CFG.vocab_size, QWEN_CFG.hidden_size, QWEN_CFG.num_layers,
        QWEN_CFG.num_heads, QWEN_CFG.num_kv_heads, QWEN_CFG.intermediate_size,
        QWEN_CFG.max_position_embeddings, QWEN_CFG.rope_theta,
        QWEN_CFG.rms_norm_eps, QWEN_CFG.bos_token_id, QWEN_CFG.eos_token_id
    );
    model->eval();
    model->load_weights(WEIGHT_PATH);
    model->to(torch::kCUDA, torch::kBFloat16);
    
    torch::Tensor input_ids = torch::zeros({1, static_cast<long>(chat_tokens.size())}, torch::kInt64);
    for (size_t i = 0; i < chat_tokens.size(); ++i) {
        input_ids[0][i] = chat_tokens[i];
    }
    input_ids = input_ids.to(torch::kCUDA);
    
    torch::NoGradGuard no_grad;
    model->clear_cache();
    
    // 第一步：Prefill（30 tokens）
    std::cout << "第1步：Prefill (30 tokens)..." << std::endl;
    torch::Tensor logits = model->forward(input_ids, true);
    torch::Tensor logits1 = logits[0][-1].to(torch::kFloat32).to(torch::kCPU);
    int64_t token1 = logits1.argmax().item<int64_t>();
    float token1_logit = logits1[token1].item<float>();
    float im_end_logit1 = logits1[QWEN_IM_END_ID].item<float>();
    
    std::cout << "  预测token: " << token1 << " (logit=" << token1_logit << ")" << std::endl;
    std::cout << "  IM_END logit: " << im_end_logit1 << std::endl;
    
    // 第二步：生成第2个token（seq_len=1，使用KV cache）
    std::cout << "\n第2步：生成第2个token (seq_len=1, 使用KV cache)..." << std::endl;
    torch::Tensor input2 = torch::tensor({{token1}}, torch::kInt64).to(torch::kCUDA);
    logits = model->forward(input2, true);
    torch::Tensor logits2 = logits[0][0].to(torch::kFloat32).to(torch::kCPU);
    int64_t token2 = logits2.argmax().item<int64_t>();
    float token2_logit = logits2[token2].item<float>();
    float im_end_logit2 = logits2[QWEN_IM_END_ID].item<float>();
    
    std::cout << "  预测token: " << token2;
    if (token2 == QWEN_IM_END_ID) std::cout << " [IM_END!]";
    std::cout << " (logit=" << token2_logit << ")" << std::endl;
    std::cout << "  IM_END logit: " << im_end_logit2 << std::endl;
    
    // Top-5
    auto topk = torch::topk(logits2, 5);
    std::cout << "\n  Top-5候选:" << std::endl;
    auto top_logits = std::get<0>(topk);
    auto top_indices = std::get<1>(topk);
    for (int i = 0; i < 5; ++i) {
        int64_t idx = top_indices[i].item<int64_t>();
        float logit = top_logits[i].item<float>();
        std::cout << "    " << idx << " (logit=" << logit << ")";
        if (idx == QWEN_IM_END_ID) std::cout << " <- IM_END";
        std::cout << std::endl;
    }
    
    // 对比PyTorch
    std::cout << "\n======== 与PyTorch对比 ========" << std::endl;
    std::cout << "如果第2个token是IM_END，说明KV cache或position encoding有问题" << std::endl;
    
    return 0;
}
