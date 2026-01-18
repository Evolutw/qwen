#include "../include/qwen_model.h"
#include "../include/qwen_env.h"
#include "../include/qwen_model_config.h"
#include <iostream>
#include <vector>
#include <chrono>
#include <algorithm>
#include <random>
#include <unordered_set>

// 配置Qwen模型参数（统一入口）
const auto& QWEN_CFG = qwen::get_model_config();
const int64_t QWEN_EOS_TOKEN_ID = QWEN_CFG.eos_token_id;
const int64_t QWEN_IM_START_ID = 151644;  // <|im_start|>
const int64_t QWEN_IM_END_ID = QWEN_CFG.im_end_id;     // <|im_end|> (same as EOS)
const int64_t QWEN_ASSISTANT_TOKEN_ID = 77091; // token for "assistant" in ChatML
const std::string WEIGHT_PATH = qwen::get_weight_path();
const std::string TOKENIZER_SCRIPT = qwen::get_tokenizer_script();
const std::string TOKENIZER_MODEL_DIR = qwen::get_tokenizer_model_dir();
const std::string PYTHON_CMD = qwen::get_python_cmd();

// 编码文本为token IDs（使用聊天模板）
std::vector<int64_t> encode_chat(const std::string& user_message) {
    std::string cmd = PYTHON_CMD + " " + TOKENIZER_SCRIPT + " \"" + user_message + "\" --chat 2>/dev/null";
    
    FILE* pipe = popen(cmd.c_str(), "r");
    if (!pipe) {
        throw std::runtime_error("无法执行分词命令");
    }
    
    char buffer[4096];
    std::string result;
    while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
        result += buffer;
    }
    pclose(pipe);
    
    // 解析JSON格式的token IDs
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

// 解码token IDs为文本
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

// 如果生成到新的assistant起始标记，及时停止
bool is_new_assistant_turn(const std::vector<int64_t>& tokens) {
    if (tokens.empty()) return false;
    if (tokens.back() == QWEN_IM_START_ID) return true;
    if (tokens.size() >= 2 && tokens[tokens.size() - 2] == QWEN_IM_START_ID && tokens.back() == QWEN_ASSISTANT_TOKEN_ID) {
        return true;
    }
    if (tokens.size() >= 3 && tokens[tokens.size() - 3] == QWEN_IM_START_ID && tokens[tokens.size() - 2] == QWEN_ASSISTANT_TOKEN_ID && tokens.back() == 198) {
        return true;
    }
    return false;
}

// 去除特殊标记后的生成内容
std::vector<int64_t> trim_at_special_tokens(const std::vector<int64_t>& tokens) {
    size_t cut = tokens.size();
    for (size_t i = 0; i < tokens.size(); ++i) {
        if (tokens[i] == QWEN_IM_END_ID || tokens[i] == QWEN_IM_START_ID) {
            cut = i;
            break;
        }
    }
    return std::vector<int64_t>(tokens.begin(), tokens.begin() + cut);
}

// Temperature采样：选择概率最高的token
int64_t sample_with_temperature(const torch::Tensor& logits, float temperature = 1.0, int top_k = 50) {
    torch::Tensor probs;
    
    if (temperature <= 0.0001) {
        // temperature接近0，使用贪婪采样
        return logits.argmax(-1).item<int64_t>();
    }
    
    // 应用temperature
    torch::Tensor scaled_logits = logits / temperature;
    
    // Top-k采样
    if (top_k > 0 && top_k < logits.size(-1)) {
        auto topk_result = torch::topk(scaled_logits, top_k, -1);
        auto topk_values = std::get<0>(topk_result);
        auto topk_indices = std::get<1>(topk_result);
        
        // 计算概率
        probs = torch::softmax(topk_values, -1);
        
        // 采样
        torch::Tensor cumsum = torch::cumsum(probs, -1);
        float random_val = static_cast<float>(rand()) / RAND_MAX;
        
        for (int i = 0; i < top_k; ++i) {
            if (cumsum[i].item<float>() >= random_val) {
                return topk_indices[i].item<int64_t>();
            }
        }
        return topk_indices[-1].item<int64_t>();
    } else {
        // 标准softmax采样
        probs = torch::softmax(scaled_logits, -1);
        torch::Tensor cumsum = torch::cumsum(probs, -1);
        float random_val = static_cast<float>(rand()) / RAND_MAX;
        
        for (int i = 0; i < probs.size(-1); ++i) {
            if (cumsum[i].item<float>() >= random_val) {
                return i;
            }
        }
        return probs.size(-1) - 1;
    }
}

// 应用repetition penalty（惩罚已生成的token）
torch::Tensor apply_repetition_penalty(torch::Tensor logits,
                                       const std::vector<int64_t>& generated_tokens,
                                       float penalty = 1.1f) {
    if (penalty <= 1.0f || generated_tokens.empty()) {
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

// 禁止重复n-gram（默认3-gram）
torch::Tensor apply_no_repeat_ngram(torch::Tensor logits,
                                    const std::vector<int64_t>& generated_tokens,
                                    int no_repeat_ngram_size = 3) {
    if (no_repeat_ngram_size <= 1) return logits;
    if (generated_tokens.size() < static_cast<size_t>(no_repeat_ngram_size - 1)) return logits;

    logits = logits.clone();

    const int n = no_repeat_ngram_size;
    const size_t prefix_start = generated_tokens.size() - (n - 1);
    std::vector<int64_t> prefix(generated_tokens.begin() + prefix_start, generated_tokens.end());

    // 找到所有与prefix匹配的n-gram，并禁止其下一token
    for (size_t i = 0; i + n <= generated_tokens.size(); ++i) {
        bool match = true;
        for (int j = 0; j < n - 1; ++j) {
            if (generated_tokens[i + j] != prefix[j]) {
                match = false;
                break;
            }
        }
        if (match) {
            int64_t banned_token = generated_tokens[i + (n - 1)];
            logits[banned_token] = -1e9;
        }
    }

    return logits;
}

// 简单重复检测：最近窗口内重复比例过高则提前停止
bool should_stop_on_repetition(const std::vector<int64_t>& generated_tokens,
                               size_t window = 30,
                               float min_unique_ratio = 0.35f) {
    if (generated_tokens.size() < window) return false;
    std::unordered_set<int64_t> uniq;
    for (size_t i = generated_tokens.size() - window; i < generated_tokens.size(); ++i) {
        uniq.insert(generated_tokens[i]);
    }
    float ratio = static_cast<float>(uniq.size()) / static_cast<float>(window);
    return ratio < min_unique_ratio;
}

// 聊天生成函数
std::string chat(
    QwenModel& model,
    const std::string& user_message,
    int max_new_tokens = 100,
    float temperature = 0.7,
    int top_k = 50,
    float repetition_penalty = 1.3f,
    int no_repeat_ngram_size = 4,
    bool verbose = true
) {
    // 编码用户消息（包含聊天模板）
    if (verbose) {
        std::cout << "\n用户: " << user_message << std::endl;
        std::cout << "助手: " << std::flush;
    }
    
    std::vector<int64_t> input_tokens = encode_chat(user_message);
    if (input_tokens.empty()) {
        return "[编码失败]";
    }
    
    // 转换为tensor
    torch::Tensor input_ids = torch::from_blob(
        input_tokens.data(),
        {1, static_cast<long>(input_tokens.size())},
        torch::kInt64
    ).clone().to(model->parameters()[0].device());
    
    // 清空KV缓存
    model->clear_cache();
    
    torch::NoGradGuard no_grad;
    
    // Prefill阶段
    auto start_time = std::chrono::high_resolution_clock::now();
    torch::Tensor logits = model->forward(input_ids, true);
    
    std::vector<int64_t> generated_tokens;
    // 采样第一个token
    torch::Tensor next_token_logits = logits[0][-1];
    next_token_logits = apply_repetition_penalty(next_token_logits, generated_tokens, repetition_penalty);
    next_token_logits = apply_no_repeat_ngram(next_token_logits, generated_tokens, no_repeat_ngram_size);

    int64_t next_token = sample_with_temperature(next_token_logits, temperature, top_k);

    generated_tokens.push_back(next_token);
    
    // 自回归生成
    for (int i = 1; i < max_new_tokens; ++i) {
        // 检查是否遇到结束标记
        if (next_token == QWEN_IM_END_ID || next_token == QWEN_EOS_TOKEN_ID) {
            if (verbose) std::cout << " [停止]" << std::flush;
            break;
        }
        
        if (should_stop_on_repetition(generated_tokens)) {
            if (verbose) std::cout << " [重复停止]" << std::flush;
            break;
        }
        
        // 生成下一个token
        torch::Tensor next_input = torch::tensor({{next_token}}, torch::kInt64).to(input_ids.device());
        logits = model->forward(next_input, true);
        next_token_logits = logits[0][0];
        next_token_logits = apply_repetition_penalty(next_token_logits, generated_tokens, repetition_penalty);
        next_token_logits = apply_no_repeat_ngram(next_token_logits, generated_tokens, no_repeat_ngram_size);

        next_token = sample_with_temperature(next_token_logits, temperature, top_k);
        generated_tokens.push_back(next_token);

        if (is_new_assistant_turn(generated_tokens)) {
            if (verbose) std::cout << " [对话结束]" << std::flush;
            break;
        }
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
    
    // 解码完整回复
    std::vector<int64_t> trimmed_tokens = trim_at_special_tokens(generated_tokens);
    std::string response = decode_tokens(trimmed_tokens);
    
    // 移除特殊标记
    size_t pos = response.find("<|im_end|>");
    if (pos != std::string::npos) {
        response = response.substr(0, pos);
    }
    
    if (verbose) {
        std::cout << response << std::endl;
        std::cout << "\n⏱️ 生成时间: " << duration << "ms";
        std::cout << " | 生成tokens: " << generated_tokens.size();
        if (duration > 0) {
            std::cout << " | 速度: " << (generated_tokens.size() * 1000.0 / duration) << " tokens/s";
        }
        std::cout << std::endl;
    }
    
    return response;
}

int main() {
    std::cout << std::string(70, '=') << std::endl;
    std::cout << "=== Qwen 2.5 聊天测试（使用ChatML格式）===" << std::endl;
    std::cout << std::string(70, '=') << std::endl;

    try {
        qwen::ensure_required_paths(WEIGHT_PATH, TOKENIZER_SCRIPT, TOKENIZER_MODEL_DIR);
    } catch (const std::exception& e) {
        std::cerr << "❌ 路径配置错误: " << e.what() << std::endl;
        return 1;
    }
    
    // 设置随机种子
    srand(time(nullptr));
    
    // 1. 初始化设备
    torch::Device device = torch::kCPU;
    if (torch::cuda::is_available()) {
        device = torch::Device(torch::kCUDA, 0);
        std::cout << "✅ 使用CUDA设备" << std::endl;
    } else {
        std::cout << "ℹ️ 使用CPU设备" << std::endl;
    }

    // 2. 初始化模型
    std::cout << "\n初始化Qwen模型..." << std::endl;
    QwenModel model = QwenModel(
        QWEN_CFG.vocab_size, QWEN_CFG.hidden_size, QWEN_CFG.num_layers,
        QWEN_CFG.num_heads, QWEN_CFG.num_kv_heads, QWEN_CFG.intermediate_size,
        QWEN_CFG.max_position_embeddings, QWEN_CFG.rope_theta,
        QWEN_CFG.rms_norm_eps, QWEN_CFG.bos_token_id, QWEN_CFG.eos_token_id
    );
    model->eval();
    std::cout << "✅ 模型初始化完成" << std::endl;

    // 3. 加载权重
    std::cout << "\n正在加载权重..." << std::endl;
    auto start_load = std::chrono::high_resolution_clock::now();
    model->load_weights(WEIGHT_PATH);
    auto end_load = std::chrono::high_resolution_clock::now();
    auto load_time = std::chrono::duration_cast<std::chrono::seconds>(end_load - start_load).count();
    std::cout << "✅ 权重加载完成 (耗时: " << load_time << "秒)" << std::endl;
    
    // 4. 移动到设备
    std::cout << "正在将模型移动到 " << device << "..." << std::endl;
    model->to(device, torch::kBFloat16);
    std::cout << "✅ 模型已就绪" << std::endl;

    std::cout << "\n" << std::string(70, '=') << std::endl;
    std::cout << "【对话测试 - 使用ChatML格式获得更好的生成质量】" << std::endl;
    std::cout << std::string(70, '=') << std::endl;

    // 测试1：简单问候
    std::cout << "\n[测试 1/5] 简单问候" << std::endl;
    std::cout << std::string(70, '-') << std::endl;
    chat(model, "你好", 50, 0.7, 50, true);

    // 测试2：知识问答
    std::cout << "\n\n[测试 2/5] 知识问答" << std::endl;
    std::cout << std::string(70, '-') << std::endl;
    chat(model, "什么是人工智能？", 100, 0.7, 50, true);

    // 测试3：创意写作
    std::cout << "\n\n[测试 3/5] 创意写作（高temperature）" << std::endl;
    std::cout << std::string(70, '-') << std::endl;
    chat(model, "写一首关于春天的诗", 80, 0.9, 40, true);

    // 测试4：事实回答（低temperature）
    std::cout << "\n\n[测试 4/5] 事实回答（低temperature）" << std::endl;
    std::cout << std::string(70, '-') << std::endl;
    chat(model, "北京是中国的首都吗？", 50, 0.3, 50, true);

    // 测试5：简短问答
    std::cout << "\n\n[测试 5/5] 简短问答" << std::endl;
    std::cout << std::string(70, '-') << std::endl;
    chat(model, "1+1=?", 20, 0.1, 50, true);

    std::cout << "\n" << std::string(70, '=') << std::endl;
    std::cout << "🎉 所有聊天测试完成！" << std::endl;
    std::cout << std::string(70, '=') << std::endl;
    
    return 0;
}
