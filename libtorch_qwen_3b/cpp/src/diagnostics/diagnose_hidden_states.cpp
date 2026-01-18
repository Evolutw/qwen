#include "../include/qwen_model.h"
#include "../include/qwen_env.h"
#include "../include/qwen_model_config.h"
#include <iostream>
#include <iomanip>
#include <vector>

// 配置Qwen模型参数（统一入口）
const auto& QWEN_CFG = qwen::get_model_config();
const std::string WEIGHT_PATH = qwen::get_weight_path();

int main() {
    std::cout << "======== 诊断Hidden States (ChatML输入) ========" << std::endl;

    try {
        qwen::ensure_required_paths(WEIGHT_PATH, qwen::get_tokenizer_script(), qwen::get_tokenizer_model_dir());
    } catch (const std::exception& e) {
        std::cerr << "❌ 路径配置错误: " << e.what() << std::endl;
        return 1;
    }
    
    // ChatML格式输入: 30 tokens
    std::vector<int64_t> chat_tokens = {
        151644, 8948, 198, 2610, 525, 1207, 16948, 11, 3465, 553, 54364, 14817, 13, 1446, 525, 264, 10950, 17847, 13, 
        151645, 198, 151644, 872, 198, 108386, 151645, 198, 151644, 77091, 198
    };
    std::cout << "测试ChatML输入: 30 tokens" << std::endl;
    std::cout << "最后5个tokens: 151645 198 151644 77091 198" << std::endl;
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
    
    // 移到CUDA
    if (torch::cuda::is_available()) {
        model->to(torch::kCUDA, torch::kBFloat16);
    } else {
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
    
    torch::NoGradGuard no_grad;
    model->clear_cache();
    
    // 手动执行前向传播的每一步
    std::cout << "\n===== 逐步前向传播 =====" << std::endl;
    
    // 1. Embedding
    std::cout << "\n1. Embedding层（最后一个位置）:" << std::endl;
    torch::Tensor hidden_states = model->embed_tokens->forward(input_ids);
    std::cout << "  shape: " << hidden_states.sizes() << std::endl;
    torch::Tensor hs_float = hidden_states[0][-1].to(torch::kFloat32).to(torch::kCPU);
    std::cout << "  最后位置前5个值: ";
    for (int i = 0; i < 5; ++i) {
        std::cout << hs_float[i].item<float>() << " ";
    }
    std::cout << std::endl;
    
    // 2. 通过所有Transformer层
    std::cout << "\n2. 通过" << QWEN_CFG.num_layers << "层Transformer..." << std::endl;
    for (int i = 0; i < QWEN_CFG.num_layers; ++i) {
        hidden_states = model->layers[i]->forward(hidden_states, true);
        if (i == QWEN_CFG.num_layers - 1) {  // 只看最后一层
            hs_float = hidden_states[0][-1].to(torch::kFloat32).to(torch::kCPU);
            std::cout << "  Layer " << (QWEN_CFG.num_layers - 1) << " 最后位置前5个值: ";
            for (int j = 0; j < 5; ++j) {
                std::cout << hs_float[j].item<float>() << " ";
            }
            std::cout << std::endl;
        }
    }
    
    // 3. 最终RMSNorm之前
    std::cout << "\n3. RMSNorm之前（最后位置前5个值）:" << std::endl;
    hs_float = hidden_states[0][-1].to(torch::kFloat32).to(torch::kCPU);
    for (int i = 0; i < 5; ++i) {
        std::cout << hs_float[i].item<float>() << " ";
    }
    std::cout << std::endl;
    
    // 4. 最终RMSNorm
    std::cout << "\n4. 应用RMSNorm（最后位置前5个值）:" << std::endl;
    hidden_states = model->norm->forward(hidden_states);
    hs_float = hidden_states[0][-1].to(torch::kFloat32).to(torch::kCPU);
    for (int i = 0; i < 5; ++i) {
        std::cout << hs_float[i].item<float>() << " ";
    }
    std::cout << std::endl;
    
    // 5. LMHead
    std::cout << "\n5. LMHead（最后位置）:" << std::endl;
    torch::Tensor logits = model->lm_head(hidden_states);
    torch::Tensor logits_float = logits[0][-1].to(torch::kFloat32).to(torch::kCPU);
    std::cout << "  前10个logit值: ";
    for (int i = 0; i < 10; ++i) {
        std::cout << logits_float[i].item<float>() << " ";
    }
    std::cout << std::endl;
    
    int64_t pred_token = torch::argmax(logits_float).item<int64_t>();
    float max_logit = logits_float[pred_token].item<float>();
    float im_end_logit = logits_float[151645].item<float>();
    std::cout << "  预测token: " << pred_token << " (logit=" << max_logit << ")" << std::endl;
    std::cout << "  IM_END (151645): logit=" << im_end_logit << std::endl;
    
    std::cout << "\n===== 与PyTorch对比 =====" << std::endl;
    std::cout << "PyTorch Embedding最后位置前5个值: -0.007659912109375, 0.03125, -0.019287109375, -0.02099609375, 0.020263671875" << std::endl;
    std::cout << "PyTorch Layer 23最后位置前5个值: 0.361328125, 4.5625, 7.53125, 2.171875, -1.4921875" << std::endl;
    std::cout << "PyTorch RMSNorm后前5个值: 0.318359375, 3.65625, 6.34375, 1.765625, -1.1484375" << std::endl;
    std::cout << "PyTorch Logits前10个值: 4.96875, 10.125, 3.328125, 2.8125, 1.0, 2.78125, 5.46875, 13.125, 5.53125, 6.75" << std::endl;
    std::cout << "PyTorch预测: Token 108386 (logit=24.3750)" << std::endl;
    std::cout << "PyTorch IM_END: logit=-3.4531" << std::endl;
    
    return 0;
}
