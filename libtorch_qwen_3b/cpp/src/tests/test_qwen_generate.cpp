#include "../include/qwen_model.h"
#include "../include/qwen_env.h"
#include "../include/qwen_model_config.h"
#include <iostream>
#include <vector>
#include <chrono>
#include <algorithm>
#include <random>

// 配置Qwen模型参数（统一入口）
const auto& QWEN_CFG = qwen::get_model_config();
const std::string WEIGHT_PATH = qwen::get_weight_path();
const std::string TOKENIZER_SCRIPT = qwen::get_tokenizer_script();
const std::string TOKENIZER_MODEL_DIR = qwen::get_tokenizer_model_dir();
const std::string PYTHON_CMD = qwen::get_python_cmd();

// 编码文本为token IDs
std::vector<int64_t> encode_text(const std::string& text) {
    std::string cmd = PYTHON_CMD + " " + TOKENIZER_SCRIPT + " \"" + text + "\" 2>/dev/null";
    
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

// 改进的生成函数
std::vector<int64_t> generate_text(
    QwenModel& model,
    const torch::Tensor& input_ids,
    int max_new_tokens = 50,
    float temperature = 0.7,
    int top_k = 50,
    int64_t eos_token_id = -1,
    bool verbose = true
) {
    if (eos_token_id < 0) {
        eos_token_id = QWEN_CFG.eos_token_id;
    }
    auto device = input_ids.device();
    std::vector<int64_t> generated_tokens;
    
    // 复制输入tokens（先移到CPU再访问）
    auto input_cpu = input_ids.to(torch::kCPU);
    auto input_accessor = input_cpu.accessor<int64_t, 2>();
    for (int i = 0; i < input_cpu.size(1); ++i) {
        generated_tokens.push_back(input_accessor[0][i]);
    }
    
    // 清空KV缓存
    model->clear_cache();
    
    torch::NoGradGuard no_grad;
    
    // Prefill阶段：处理所有输入tokens
    if (verbose) std::cout << "Prefill... " << std::flush;
    auto start_prefill = std::chrono::high_resolution_clock::now();
    torch::Tensor logits = model->forward(input_ids, true);
    auto end_prefill = std::chrono::high_resolution_clock::now();
    if (verbose) {
        auto prefill_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_prefill - start_prefill).count();
        std::cout << "(" << prefill_time << "ms) " << std::flush;
    }
    
    // 获取最后一个位置的logits并采样
    torch::Tensor next_token_logits = logits[0][-1];
    int64_t next_token = sample_with_temperature(next_token_logits, temperature, top_k);
    generated_tokens.push_back(next_token);
    
    if (verbose) std::cout << "\n生成中: " << std::flush;
    
    // 自回归生成
    auto start_decode = std::chrono::high_resolution_clock::now();
    for (int i = 1; i < max_new_tokens; ++i) {
        // 检查是否遇到EOS
        if (next_token == eos_token_id) {
            if (verbose) std::cout << " [EOS]" << std::endl;
            break;
        }
        
        if (verbose && i % 5 == 0) std::cout << "." << std::flush;
        
        // 将新token作为输入（使用KV缓存）
        torch::Tensor next_input = torch::tensor({{next_token}}, torch::kInt64).to(device);
        logits = model->forward(next_input, true);
        
        // 采样下一个token
        next_token_logits = logits[0][0];
        next_token = sample_with_temperature(next_token_logits, temperature, top_k);
        generated_tokens.push_back(next_token);
    }
    
    auto end_decode = std::chrono::high_resolution_clock::now();
    
    if (verbose) {
        auto decode_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_decode - start_decode).count();
        int new_tokens = generated_tokens.size() - input_ids.size(1);
        std::cout << "\n生成完成！生成了 " << new_tokens << " 个新tokens";
        if (decode_time > 0) {
            std::cout << " (速度: " << (new_tokens * 1000.0 / decode_time) << " tokens/s)";
        }
        std::cout << std::endl;
    }
    
    return generated_tokens;
}

int main() {
    try {
        qwen::ensure_required_paths(WEIGHT_PATH, TOKENIZER_SCRIPT, TOKENIZER_MODEL_DIR);
    } catch (const std::exception& e) {
        std::cerr << "❌ 路径配置错误: " << e.what() << std::endl;
        return 1;
    }

    std::cout << std::string(70, '=') << std::endl;
    std::cout << "=== Qwen 2.5 文本生成测试 ===" << std::endl;
    std::cout << std::string(70, '=') << std::endl;
    
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
    std::cout << "【测试1：贪婪解码生成】" << std::endl;
    std::cout << std::string(70, '=') << std::endl;

    // 测试1：贪婪解码（temperature=0）
    {
        std::string prompt = "你好";
        std::cout << "\n输入: \"" << prompt << "\"" << std::endl;
        
        std::vector<int64_t> input_tokens = encode_text(prompt);
        torch::Tensor input_ids = torch::from_blob(
            input_tokens.data(),
            {1, static_cast<long>(input_tokens.size())},
            torch::kInt64
        ).clone().to(device);
        
        std::vector<int64_t> output_tokens = generate_text(
            model, input_ids, 20, 0.0, 0, QWEN_CFG.eos_token_id, true
        );
        
        std::string output_text = decode_tokens(output_tokens);
        std::cout << "\n完整输出:\n" << output_text << std::endl;
    }

    std::cout << "\n" << std::string(70, '=') << std::endl;
    std::cout << "【测试2：带Temperature的随机采样】" << std::endl;
    std::cout << std::string(70, '=') << std::endl;

    // 测试2：随机采样（temperature=0.7）
    {
        std::string prompt = "今天天气";
        std::cout << "\n输入: \"" << prompt << "\"" << std::endl;
        std::cout << "参数: temperature=0.7, top_k=50\n" << std::endl;
        
        std::vector<int64_t> input_tokens = encode_text(prompt);
        torch::Tensor input_ids = torch::from_blob(
            input_tokens.data(),
            {1, static_cast<long>(input_tokens.size())},
            torch::kInt64
        ).clone().to(device);
        
        std::vector<int64_t> output_tokens = generate_text(
            model, input_ids, 30, 0.7, 50, QWEN_CFG.eos_token_id, true
        );
        
        std::string output_text = decode_tokens(output_tokens);
        std::cout << "\n完整输出:\n" << output_text << std::endl;
    }

    std::cout << "\n" << std::string(70, '=') << std::endl;
    std::cout << "【测试3：更长的生成】" << std::endl;
    std::cout << std::string(70, '=') << std::endl;

    // 测试3：更长的生成
    {
        std::string prompt = "人工智能的未来发展方向是";
        std::cout << "\n输入: \"" << prompt << "\"" << std::endl;
        std::cout << "参数: temperature=0.8, top_k=40, max_tokens=50\n" << std::endl;
        
        std::vector<int64_t> input_tokens = encode_text(prompt);
        torch::Tensor input_ids = torch::from_blob(
            input_tokens.data(),
            {1, static_cast<long>(input_tokens.size())},
            torch::kInt64
        ).clone().to(device);
        
        std::vector<int64_t> output_tokens = generate_text(
            model, input_ids, 50, 0.8, 40, QWEN_CFG.eos_token_id, true
        );
        
        std::string output_text = decode_tokens(output_tokens);
        std::cout << "\n完整输出:\n" << output_text << std::endl;
    }

    std::cout << "\n" << std::string(70, '=') << std::endl;
    std::cout << "【测试4：对比不同temperature】" << std::endl;
    std::cout << std::string(70, '=') << std::endl;

    // 测试4：对比不同temperature
    {
        std::string prompt = "北京是";
        std::cout << "\n输入: \"" << prompt << "\"" << std::endl;
        
        std::vector<int64_t> input_tokens = encode_text(prompt);
        
        for (float temp : {0.1f, 0.5f, 1.0f}) {
            std::cout << "\n--- Temperature = " << temp << " ---" << std::endl;
            
            torch::Tensor input_ids = torch::from_blob(
                input_tokens.data(),
                {1, static_cast<long>(input_tokens.size())},
                torch::kInt64
            ).clone().to(device);
            
            std::vector<int64_t> output_tokens = generate_text(
                model, input_ids, 15, temp, 50, QWEN_CFG.eos_token_id, false
            );
            
            std::string output_text = decode_tokens(output_tokens);
            std::cout << "输出: " << output_text << std::endl;
        }
    }

    std::cout << "\n" << std::string(70, '=') << std::endl;
    std::cout << "🎉 所有生成测试完成！" << std::endl;
    std::cout << std::string(70, '=') << std::endl;
    
    return 0;
}
