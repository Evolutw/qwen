#include "../include/qwen_model.h"
#include "../include/qwen_env.h"
#include "../include/qwen_model_config.h"
#include <iostream>
#include <cstdio>
#include <string>
#include <vector>
#include <chrono>

const auto& QWEN_CFG = qwen::get_model_config();
const std::string WEIGHT_PATH = qwen::get_weight_path();
const std::string TOKENIZER_SCRIPT = qwen::get_tokenizer_script();
const std::string TOKENIZER_MODEL_DIR = qwen::get_tokenizer_model_dir();
const std::string PYTHON_CMD = qwen::get_python_cmd();

std::vector<int64_t> encode_chat(const std::string& user_message) {
    std::string cmd = PYTHON_CMD + " " + TOKENIZER_SCRIPT + " \"" + user_message + "\" --chat 2>/dev/null";
    FILE* pipe = popen(cmd.c_str(), "r");
    if (!pipe) throw std::runtime_error("分词失败");
    
    char buffer[4096];
    std::string result;
    while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
        result += buffer;
    }
    pclose(pipe);
    
    std::vector<int64_t> token_ids;
    size_t start = result.find("[");
    size_t end = result.find("]");
    if (start != std::string::npos && end != std::string::npos) {
        std::string ids_str = result.substr(start + 1, end - start - 1);
        std::stringstream ss(ids_str);
        std::string item;
        while (std::getline(ss, item, ',')) {
            token_ids.push_back(std::stoll(item));
        }
    }
    return token_ids;
}

std::string decode_tokens(const std::vector<int64_t>& token_ids) {
    if (token_ids.empty()) return "";
    std::string ids_str = "[";
    for (size_t i = 0; i < token_ids.size(); ++i) {
        ids_str += std::to_string(token_ids[i]);
        if (i < token_ids.size() - 1) ids_str += ",";
    }
    ids_str += "]";
    std::string cmd = PYTHON_CMD + " -c \"from transformers import AutoTokenizer; tokenizer = AutoTokenizer.from_pretrained('" + TOKENIZER_MODEL_DIR + "', trust_remote_code=True); print(tokenizer.decode(" + ids_str + "), end='')\" 2>/dev/null";
    FILE* pipe = popen(cmd.c_str(), "r");
    if (!pipe) return "[解码失败]";
    char buffer[4096];
    std::string result;
    while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
        result += buffer;
    }
    pclose(pipe);
    return result;
}

int main() {
    std::cout << "======================================================================" << std::endl;
    std::cout << "=== 测试Chat完整流程（贪婪采样）===" << std::endl;
    std::cout << "======================================================================\n" << std::endl;

    try {
        qwen::ensure_required_paths(WEIGHT_PATH, TOKENIZER_SCRIPT, TOKENIZER_MODEL_DIR);
    } catch (const std::exception& e) {
        std::cerr << "❌ 路径配置错误: " << e.what() << std::endl;
        return 1;
    }
    
    QwenModel model = QwenModel(
        QWEN_CFG.vocab_size, QWEN_CFG.hidden_size, QWEN_CFG.num_layers,
        QWEN_CFG.num_heads, QWEN_CFG.num_kv_heads, QWEN_CFG.intermediate_size,
        QWEN_CFG.max_position_embeddings, QWEN_CFG.rope_theta,
        QWEN_CFG.rms_norm_eps, QWEN_CFG.bos_token_id, QWEN_CFG.eos_token_id
    );
    model->eval();
    std::cout << "加载权重..." << std::endl;
    model->load_weights(WEIGHT_PATH);
    
    if (torch::cuda::is_available()) {
        model->to(torch::kCUDA, torch::kBFloat16);
        std::cout << "✅ 使用CUDA\n" << std::endl;
    }
    
    // 测试
    std::string user_msg = "你好";
    std::cout << "用户: " << user_msg << std::endl;
    std::cout << "助手: " << std::flush;
    
    // Encode
    std::vector<int64_t> input_tokens = encode_chat(user_msg);
    torch::Tensor input_ids = torch::zeros({1, static_cast<long>(input_tokens.size())}, torch::kInt64);
    for (size_t i = 0; i < input_tokens.size(); ++i) {
        input_ids[0][i] = input_tokens[i];
    }
    if (torch::cuda::is_available()) {
        input_ids = input_ids.to(torch::kCUDA);
    }
    
    // Generate
    model->clear_cache();
    torch::NoGradGuard no_grad;
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Prefill
    torch::Tensor logits = model->forward(input_ids, true);
    int64_t next_token = logits[0][-1].argmax(-1).item<int64_t>();
    
    std::vector<int64_t> generated;
    generated.push_back(next_token);
    
    // Autoregressive (最多20 tokens)
    for (int i = 0; i < 20; ++i) {
        if (next_token == QWEN_CFG.im_end_id || next_token == QWEN_CFG.eos_token_id) {
            break;
        }
        
        torch::Tensor next_input = torch::tensor({{next_token}}, torch::kInt64).to(input_ids.device());
        logits = model->forward(next_input, true);
        next_token = logits[0][0].argmax(-1).item<int64_t>();
        generated.push_back(next_token);
        
        // 每5个token显示一次
        if ((i + 1) % 5 == 0) {
            std::string partial = decode_tokens(generated);
            size_t pos = partial.find("<|im_end|>");
            if (pos != std::string::npos) {
                partial = partial.substr(0, pos);
            }
            std::cout << "\r助手: " << partial << std::flush;
        }
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
    
    // 最终解码
    std::string response = decode_tokens(generated);
    size_t pos = response.find("<|im_end|>");
    if (pos != std::string::npos) {
        response = response.substr(0, pos);
    }
    
    std::cout << "\r助手: " << response << std::endl;
    std::cout << "\n⏱️  耗时: " << duration << "ms | 生成tokens: " << generated.size() 
              << " | 速度: " << (generated.size() * 1000.0 / duration) << " tokens/s" << std::endl;
    
    // 对比PyTorch预期结果
    std::cout << "\n======== 预期结果（PyTorch贪婪采样）========" << std::endl;
    std::cout << "Token IDs: [108386, 6313, 112169, 102804, 47874, 1773, 104139, 109944, 100364, 101214, 101037, 11319, 151645]" << std::endl;
    std::cout << "文本: 你好！很高兴为您服务。有什么我可以帮助您的吗？" << std::endl;
    
    std::cout << "\n======== 实际生成 ========" << std::endl;
    std::cout << "Token IDs: [";
    for (size_t i = 0; i < generated.size(); ++i) {
        std::cout << generated[i];
        if (i < generated.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    
    // 检查是否匹配
    std::vector<int64_t> expected = {108386, 6313, 112169, 102804, 47874, 1773, 104139, 109944, 100364, 101214, 101037, 11319, 151645};
    bool match = true;
    for (size_t i = 0; i < std::min(generated.size(), expected.size()); ++i) {
        if (generated[i] != expected[i]) {
            std::cout << "\n❌ 第" << (i+1) << "个token不匹配: 实际" << generated[i] << " vs 预期" << expected[i] << std::endl;
            match = false;
            break;
        }
    }
    
    if (match && generated.size() == expected.size()) {
        std::cout << "\n✅ 完美匹配PyTorch输出！Causal mask修复成功！" << std::endl;
    }
    
    return 0;
}
