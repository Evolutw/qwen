#include "../../include/qwen_model.h"
#include "../../include/qwen_env.h"
#include "../../include/qwen_model_config.h"
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <ctime>

const auto& QWEN_CFG = qwen::get_model_config();
const std::string WEIGHT_PATH = qwen::get_weight_path();

// Temperature采样
int64_t sample_with_temperature(const torch::Tensor& logits, float temperature, int top_k) {
    if (temperature <= 0.0001) {
        return logits.argmax(-1).item<int64_t>();
    }
    
    torch::Tensor scaled_logits = logits / temperature;
    
    if (top_k > 0 && top_k < logits.size(-1)) {
        auto topk_result = torch::topk(scaled_logits, top_k, -1);
        auto topk_logits = std::get<0>(topk_result);
        auto topk_indices = std::get<1>(topk_result);
        
        torch::Tensor probs = torch::softmax(topk_logits, -1);
        torch::Tensor cumsum_probs = torch::cumsum(probs, -1);
        
        float random_val = static_cast<float>(rand()) / RAND_MAX;
        for (int i = 0; i < topk_indices.size(-1); ++i) {
            if (cumsum_probs[i].item<float>() >= random_val) {
                return topk_indices[i].item<int64_t>();
            }
        }
        return topk_indices[-1].item<int64_t>();
    }
    
    return logits.argmax(-1).item<int64_t>();
}

// Repetition penalty
torch::Tensor apply_repetition_penalty(torch::Tensor logits, 
                                       const std::vector<int64_t>& generated_tokens,
                                       float penalty = 1.1) {
    if (penalty == 1.0 || generated_tokens.empty()) {
        return logits;
    }
    
    logits = logits.clone();
    for (int64_t token : generated_tokens) {
        float current_logit = logits[token].item<float>();
        if (current_logit > 0) {
            logits[token] = current_logit / penalty;
        } else {
            logits[token] = current_logit * penalty;
        }
    }
    return logits;
}

int main() {
    std::cout << "======== 简短测试（验证repetition penalty）========" << std::endl;

    try {
        qwen::ensure_required_paths(WEIGHT_PATH, qwen::get_tokenizer_script(), qwen::get_tokenizer_model_dir());
    } catch (const std::exception& e) {
        std::cerr << "❌ 路径配置错误: " << e.what() << std::endl;
        return 1;
    }
    
    srand(time(nullptr));
    
    // 初始化模型
    QwenModel model = QwenModel(
        QWEN_CFG.vocab_size, QWEN_CFG.hidden_size, QWEN_CFG.num_layers,
        QWEN_CFG.num_heads, QWEN_CFG.num_kv_heads, QWEN_CFG.intermediate_size,
        QWEN_CFG.max_position_embeddings, QWEN_CFG.rope_theta,
        QWEN_CFG.rms_norm_eps, QWEN_CFG.bos_token_id, QWEN_CFG.eos_token_id
    );
    model->eval();
    model->load_weights(WEIGHT_PATH);
    
    if (torch::cuda::is_available()) {
        model->to(torch::kCUDA, torch::kBFloat16);
        std::cout << "✅ 使用CUDA" << std::endl;
    }
    
    // 手动构造ChatML输入 (简化版)
    std::vector<int64_t> input_tokens = {
        151644, 8948, 198,  // <|im_start|>system\n
        2610, 525, 1207, 16948, 11, 3465, 553, 54364, 14817, 13, 1446, 525, 264, 10950, 17847, 13,  // system prompt
        151645, 198,  // <|im_end|>\n
        151644, 872, 198,  // <|im_start|>user\n
        108386,  // 你好
        151645, 198,  // <|im_end|>\n
        151644, 77091, 198  // <|im_start|>assistant\n
    };
    
    torch::Tensor input_ids = torch::zeros({1, static_cast<long>(input_tokens.size())}, torch::kInt64);
    for (size_t i = 0; i < input_tokens.size(); ++i) {
        input_ids[0][i] = input_tokens[i];
    }
    if (torch::cuda::is_available()) {
        input_ids = input_ids.to(torch::kCUDA);
    }
    
    std::cout << "\n测试1: 不使用repetition penalty (贪婪)" << std::endl;
    std::cout << "----------------------------------------" << std::endl;
    model->clear_cache();
    torch::NoGradGuard no_grad;
    torch::Tensor logits = model->forward(input_ids, true);
    
    std::vector<int64_t> tokens1;
    int64_t token = logits[0][-1].argmax(-1).item<int64_t>();
    for (int i = 0; i < 20 && token != QWEN_CFG.im_end_id; ++i) {
        tokens1.push_back(token);
        std::cout << token << " ";
        torch::Tensor next_input = torch::tensor({{token}}, torch::kInt64).to(input_ids.device());
        logits = model->forward(next_input, true);
        token = logits[0][0].argmax(-1).item<int64_t>();
    }
    std::cout << "\n生成了" << tokens1.size() << "个tokens" << std::endl;
    
    std::cout << "\n测试2: 使用repetition penalty=1.2" << std::endl;
    std::cout << "----------------------------------------" << std::endl;
    model->clear_cache();
    logits = model->forward(input_ids, true);
    
    std::vector<int64_t> tokens2;
    torch::Tensor penalized_logits = apply_repetition_penalty(logits[0][-1], {}, 1.2);
    token = penalized_logits.argmax(-1).item<int64_t>();
    
    for (int i = 0; i < 20 && token != QWEN_CFG.im_end_id; ++i) {
        tokens2.push_back(token);
        std::cout << token << " ";
        torch::Tensor next_input = torch::tensor({{token}}, torch::kInt64).to(input_ids.device());
        logits = model->forward(next_input, true);
        penalized_logits = apply_repetition_penalty(logits[0][0], tokens2, 1.2);
        token = penalized_logits.argmax(-1).item<int64_t>();
    }
    std::cout << "\n生成了" << tokens2.size() << "个tokens" << std::endl;
    
    std::cout << "\n测试3: Temperature=0.7 + Top-k=50 + Repetition penalty=1.1" << std::endl;
    std::cout << "----------------------------------------" << std::endl;
    model->clear_cache();
    logits = model->forward(input_ids, true);
    
    std::vector<int64_t> tokens3;
    penalized_logits = apply_repetition_penalty(logits[0][-1], {}, 1.1);
    token = sample_with_temperature(penalized_logits, 0.7, 50);
    
    for (int i = 0; i < 20 && token != QWEN_CFG.im_end_id; ++i) {
        tokens3.push_back(token);
        std::cout << token << " ";
        torch::Tensor next_input = torch::tensor({{token}}, torch::kInt64).to(input_ids.device());
        logits = model->forward(next_input, true);
        penalized_logits = apply_repetition_penalty(logits[0][0], tokens3, 1.1);
        token = sample_with_temperature(penalized_logits, 0.7, 50);
    }
    std::cout << "\n生成了" << tokens3.size() << "个tokens" << std::endl;
    
    return 0;
}
