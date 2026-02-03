#include "../../include/qwen_model.h"
#include "../../include/qwen_env.h"
#include "../../include/qwen_model_config.h"
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <string>
#include <vector>
#include <sstream>
#include <c10/cuda/CUDAGuard.h>

// Qwen配置（统一入口）
const auto& QWEN_CFG = qwen::get_model_config();
const std::string MODEL_DIR = qwen::get_model_dir();
const std::string WEIGHT_PATH = qwen::get_weight_path();
const std::string TOKENIZER_SCRIPT = qwen::get_tokenizer_script();
const std::string TOKENIZER_MODEL_DIR = qwen::get_tokenizer_model_dir();
const std::string PYTHON_CMD = qwen::get_python_cmd();

std::vector<int64_t> encode_chat(const std::string& user_message) {
    std::string cmd = PYTHON_CMD + " " + TOKENIZER_SCRIPT + " \"" + user_message + "\" --chat";
    FILE* pipe = popen(cmd.c_str(), "r");
    if (!pipe) throw std::runtime_error("分词失败: 无法启动分词脚本");

    char buffer[4096];
    std::string result;
    while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
        result += buffer;
    }
    int status = pclose(pipe);

    std::vector<int64_t> token_ids;
    size_t start = result.find("[");
    size_t end = result.find("]");
    if (start != std::string::npos && end != std::string::npos && end > start + 1) {
        std::string ids_str = result.substr(start + 1, end - start - 1);
        std::stringstream ss(ids_str);
        std::string item;
        while (std::getline(ss, item, ',')) {
            token_ids.push_back(std::stoll(item));
        }
    }
    if (status != 0 || token_ids.empty()) {
        throw std::runtime_error("分词失败: 分词脚本无输出或返回非零状态");
    }
    return token_ids;
}

std::string decode_tokens(const std::vector<int64_t>& token_ids) {
    if (token_ids.empty()) return "[空]";
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
    return result.empty() ? "[空]" : result;
}

int64_t sample_greedy(const torch::Tensor& logits) {
    return logits.argmax(-1).item<int64_t>();
}

// Temperature + Top-k采样
int64_t sample_with_temperature(const torch::Tensor& logits, float temperature = 0.7, int top_k = 50) {
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
    
    torch::Tensor probs = torch::softmax(scaled_logits, -1);
    return torch::multinomial(probs, 1).item<int64_t>();
}

// 应用repetition penalty（惩罚已生成的token）
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
    std::cout << "======== 快速Chat测试 ========" << std::endl;

    const char* force_cpu_env = std::getenv("QWEN_FORCE_CPU");
    bool force_cpu = force_cpu_env && std::string(force_cpu_env) != "0";

    std::cout << "CUDA可用: " << (torch::cuda::is_available() ? "Yes" : "No") << std::endl;
    if (torch::cuda::is_available() && !force_cpu) {
        std::cout << "CUDA设备数量: " << torch::cuda::device_count() << std::endl;
        std::cout << "当前CUDA设备: " << c10::cuda::current_device() << std::endl;
    }
    if (force_cpu) {
        std::cout << "⚠️ 已强制使用CPU (QWEN_FORCE_CPU=1)" << std::endl;
    }

    try {
        qwen::ensure_required_paths(WEIGHT_PATH, TOKENIZER_SCRIPT, TOKENIZER_MODEL_DIR);
    } catch (const std::exception& e) {
        std::cerr << "❌ 路径配置错误: " << e.what() << std::endl;
        return 1;
    }

    if (!MODEL_DIR.empty()) {
        setenv("QWEN_MODEL_DIR", MODEL_DIR.c_str(), 1);
    }
    if (!TOKENIZER_MODEL_DIR.empty()) {
        setenv("QWEN_TOKENIZER_MODEL_DIR", TOKENIZER_MODEL_DIR.c_str(), 1);
    }
    
    // 设置随机种子
    srand(time(nullptr));
    
    // 采样参数（更激进的设置来避免重复）
    float temperature = 0.8;  // 提高温度增加随机性
    int top_k = 40;  // 减少top-k让选择更多样化
    float repetition_penalty = 1.2;  // 提高penalty惩罚重复
    int max_tokens = 30;  // 减少max tokens避免啰嗦
    
    // 初始化模型
    QwenModel model = QwenModel(
        QWEN_CFG.vocab_size, QWEN_CFG.hidden_size, QWEN_CFG.num_layers,
        QWEN_CFG.num_heads, QWEN_CFG.num_kv_heads, QWEN_CFG.intermediate_size,
        QWEN_CFG.max_position_embeddings, QWEN_CFG.rope_theta,
        QWEN_CFG.rms_norm_eps, QWEN_CFG.bos_token_id, QWEN_CFG.eos_token_id
    );
    model->eval();
    model->load_weights(WEIGHT_PATH);
    
    if (torch::cuda::is_available() && !force_cpu) {
        model->to(torch::kCUDA, torch::kBFloat16);
        std::cout << "✅ 使用CUDA" << std::endl;
    } else {
        model->to(torch::kCPU, torch::kBFloat16);
    }

    // 打印模型参数设备（取第一个参数）
    for (const auto& p : model->parameters()) {
        std::cout << "模型参数设备: " << p.device() << std::endl;
        break;
    }
    
    std::cout << "\n测试: 你好" << std::endl;
    std::cout << "-------------------------------" << std::endl;
    
    // Encode
    std::vector<int64_t> input_tokens;
    try {
        input_tokens = encode_chat("你好");
    } catch (const std::exception& e) {
        std::cerr << "❌ " << e.what() << std::endl;
        return 1;
    }
    torch::Tensor input_ids = torch::zeros({1, static_cast<long>(input_tokens.size())}, torch::kInt64);
    for (size_t i = 0; i < input_tokens.size(); ++i) {
        input_ids[0][i] = input_tokens[i];
    }
    if (torch::cuda::is_available() && !force_cpu) {
        input_ids = input_ids.to(torch::kCUDA);
    }
    std::cout << "input_ids设备: " << input_ids.device() << std::endl;
    
    std::cout << "用户: 你好" << std::endl;
    std::cout << "助手: " << std::flush;
    
    // Prefill
    model->clear_cache();
    torch::NoGradGuard no_grad;
    torch::Tensor logits = model->forward(input_ids, true);
    std::cout << "logits设备: " << logits.device() << std::endl;
    
    // 应用repetition penalty并采样
    torch::Tensor next_logits = apply_repetition_penalty(logits[0][-1], {}, repetition_penalty);
    int64_t next_token = sample_with_temperature(next_logits, temperature, top_k);
    
    std::vector<int64_t> generated;
    generated.push_back(next_token);
    
    // Autoregressive generation
    for (int i = 1; i < max_tokens; ++i) {
        if (next_token == QWEN_CFG.im_end_id || next_token == QWEN_CFG.eos_token_id) {
            break;
        }
        
        // 显示token ID (调试用)
        if (i <= 5) {
            std::cout << "[" << next_token << "]";
        }
        
        torch::Tensor next_input = torch::tensor({{next_token}}, torch::kInt64).to(input_ids.device());
        logits = model->forward(next_input, true);
        
        // 应用repetition penalty
        next_logits = apply_repetition_penalty(logits[0][0], generated, repetition_penalty);
        next_token = sample_with_temperature(next_logits, temperature, top_k);
        generated.push_back(next_token);
    }
    
    // Decode
    std::string response = decode_tokens(generated);
    size_t pos = response.find("<|im_end|>");
    if (pos != std::string::npos) {
        response = response.substr(0, pos);
    }
    
    std::cout << response << std::endl;
    std::cout << "\n✅ 生成tokens数: " << generated.size() << std::endl;
    std::cout << "参数: temperature=" << temperature << ", top_k=" << top_k 
              << ", repetition_penalty=" << repetition_penalty << std::endl;
    
    return 0;
}
