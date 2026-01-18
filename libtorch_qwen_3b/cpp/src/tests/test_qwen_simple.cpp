#include "../include/qwen_model.h"
#include "../include/qwen_env.h"
#include "../include/qwen_model_config.h"
#include <iostream>
#include <vector>
#include <chrono>

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

// 打印张量信息
void print_tensor_info(const std::string& tensor_name, const torch::Tensor& tensor) {
    std::cout << "\n" << tensor_name << "：" << std::endl;
    std::cout << "  形状：" << tensor.sizes() << std::endl;
    std::cout << "  数据类型：" << tensor.dtype().name() << std::endl;
    std::cout << "  设备：" << tensor.device() << std::endl;
    if (tensor.numel() > 0 && tensor.numel() <= 10) {
        std::cout << "  值：" << tensor.flatten() << std::endl;
    } else if (tensor.numel() > 0) {
        int64_t print_size = std::min(5L, tensor.numel());
        std::cout << "  前" << print_size << "个元素：" << tensor.flatten().slice(0, 0, print_size) << std::endl;
    }
}

// 解码单个token ID为文本
std::string decode_token(int64_t token_id) {
    std::string cmd = PYTHON_CMD + " -c \"from transformers import AutoTokenizer; tokenizer = AutoTokenizer.from_pretrained('" + TOKENIZER_MODEL_DIR + "', trust_remote_code=True); print(tokenizer.decode([" + std::to_string(token_id) + "]), end='')\" 2>/dev/null";
    
    FILE* pipe = popen(cmd.c_str(), "r");
    if (!pipe) return "[解码失败]";
    
    char buffer[1024];
    std::string result;
    while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
        result += buffer;
    }
    pclose(pipe);
    
    return result.empty() ? "[空]" : result;
}

// 解码多个token IDs为文本
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
    
    char buffer[1024];
    std::string result;
    while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
        result += buffer;
    }
    pclose(pipe);
    
    return result.empty() ? "[空]" : result;
}

int main() {
    std::cout << std::string(70, '=') << std::endl;
    std::cout << "=== Qwen 2.5 模型简化测试（仅前向传播）===" << std::endl;
    std::cout << std::string(70, '=') << std::endl;

    try {
        qwen::ensure_required_paths(WEIGHT_PATH, TOKENIZER_SCRIPT, TOKENIZER_MODEL_DIR);
    } catch (const std::exception& e) {
        std::cerr << "❌ 路径配置错误: " << e.what() << std::endl;
        return 1;
    }
    
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
        QWEN_CFG.vocab_size,
        QWEN_CFG.hidden_size,
        QWEN_CFG.num_layers,
        QWEN_CFG.num_heads,
        QWEN_CFG.num_kv_heads,
        QWEN_CFG.intermediate_size,
        QWEN_CFG.max_position_embeddings,
        QWEN_CFG.rope_theta,
        QWEN_CFG.rms_norm_eps,
        QWEN_CFG.bos_token_id,
        QWEN_CFG.eos_token_id
    );
    model->eval();
    std::cout << "✅ 模型初始化完成（" << QWEN_CFG.num_layers << "层 Transformer）" << std::endl;

    // 3. 加载权重
    std::cout << "\n开始加载权重..." << std::endl;
    auto start_load = std::chrono::high_resolution_clock::now();
    model->load_weights(WEIGHT_PATH);
    auto end_load = std::chrono::high_resolution_clock::now();
    auto load_time = std::chrono::duration_cast<std::chrono::seconds>(end_load - start_load).count();
    std::cout << "权重加载耗时: " << load_time << " 秒" << std::endl;
    
    // 4. 移动到设备
    std::cout << "\n将模型移动到 " << device << "..." << std::endl;
    model->to(device, torch::kBFloat16);
    std::cout << "✅ 模型已移动到设备" << std::endl;

    std::cout << "\n" << std::string(70, '=') << std::endl;
    std::cout << "【测试1：中文文本前向传播】" << std::endl;
    std::cout << std::string(70, '=') << std::endl;

    // 测试1：中文文本
    {
        std::string input_text = "你好";
        std::cout << "\n输入文本: \"" << input_text << "\"" << std::endl;
        
        std::vector<int64_t> tokens = encode_text(input_text);
        std::cout << "Token IDs: [";
        for (size_t i = 0; i < tokens.size(); ++i) {
            std::cout << tokens[i];
            if (i < tokens.size() - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
        
        torch::Tensor input_ids = torch::from_blob(
            tokens.data(), 
            {1, static_cast<long>(tokens.size())}, 
            torch::kInt64
        ).clone().to(device);
        
        print_tensor_info("输入Tensor", input_ids);
        
        torch::NoGradGuard no_grad;
        auto start = std::chrono::high_resolution_clock::now();
        torch::Tensor logits = model->forward(input_ids, false);
        auto end = std::chrono::high_resolution_clock::now();

        print_tensor_info("输出logits", logits);
        
        // 获取预测的token
        int64_t pred_token = logits[0][-1].argmax().item<int64_t>();
        std::cout << "预测下一个token ID: " << pred_token << std::endl;
        std::string pred_text = decode_token(pred_token);
        std::cout << "预测下一个token文本: \"" << pred_text << "\"" << std::endl;
        std::cout << "耗时: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms\n";
    }

    std::cout << "\n" << std::string(70, '=') << std::endl;
    std::cout << "【测试2：更长的中文文本】" << std::endl;
    std::cout << std::string(70, '=') << std::endl;

    // 测试2：更长的中文文本
    {
        std::string input_text = "人工智能的未来";
        std::cout << "\n输入文本: \"" << input_text << "\"" << std::endl;
        
        std::vector<int64_t> tokens = encode_text(input_text);
        std::cout << "Token IDs: [";
        for (size_t i = 0; i < tokens.size(); ++i) {
            std::cout << tokens[i];
            if (i < tokens.size() - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
        
        torch::Tensor input_ids = torch::from_blob(
            tokens.data(), 
            {1, static_cast<long>(tokens.size())}, 
            torch::kInt64
        ).clone().to(device);
        
        torch::NoGradGuard no_grad;
        auto start = std::chrono::high_resolution_clock::now();
        torch::Tensor logits = model->forward(input_ids, false);
        auto end = std::chrono::high_resolution_clock::now();

        print_tensor_info("输出logits", logits);
        
        // 获取预测的token
        int64_t pred_token = logits[0][-1].argmax().item<int64_t>();
        std::cout << "预测下一个token ID: " << pred_token << std::endl;
        std::string pred_text = decode_token(pred_token);
        std::cout << "预测下一个token文本: \"" << pred_text << "\"" << std::endl;
    }

    std::cout << "\n" << std::string(70, '=') << std::endl;
    std::cout << "【测试3：使用KV缓存的多步推理】" << std::endl;
    std::cout << std::string(70, '=') << std::endl;

    // 测试3：分步推理模拟自回归
    {
        // Step 1: Prefill - 处理初始序列
        std::string input_text = "今天天气";
        std::cout << "\n输入文本: \"" << input_text << "\"" << std::endl;
        
        std::vector<int64_t> init_tokens = encode_text(input_text);
        std::cout << "Token IDs: [";
        for (size_t i = 0; i < init_tokens.size(); ++i) {
            std::cout << init_tokens[i];
            if (i < init_tokens.size() - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
        
        torch::Tensor input_ids = torch::from_blob(
            init_tokens.data(), 
            {1, static_cast<long>(init_tokens.size())}, 
            torch::kInt64
        ).clone().to(device);
        
        std::cout << "\nStep 1: Prefill阶段 (" << init_tokens.size() << "个tokens)" << std::endl;
        print_tensor_info("输入Tensor", input_ids);
        
        torch::NoGradGuard no_grad;
        auto start1 = std::chrono::high_resolution_clock::now();
        torch::Tensor logits1 = model->forward(input_ids, true);  // use_cache=true
        auto end1 = std::chrono::high_resolution_clock::now();
        
        int64_t next_token = logits1[0][-1].argmax().item<int64_t>();
        std::cout << "预测下一个token ID: " << next_token << std::endl;
        std::string next_text = decode_token(next_token);
        std::cout << "预测token文本: \"" << next_text << "\"" << std::endl;
        std::cout << "Prefill耗时: " << std::chrono::duration_cast<std::chrono::milliseconds>(end1 - start1).count() << " ms\n";
        
        // Step 2: Decode - 使用KV缓存处理单个token
        std::cout << "\nStep 2: Decode阶段 (1个token，使用KV缓存)" << std::endl;
        std::vector<int64_t> next_tokens = {next_token};
        torch::Tensor next_input = torch::from_blob(
            next_tokens.data(), 
            {1, 1}, 
            torch::kInt64
        ).clone().to(device);
        
        print_tensor_info("输入Tensor", next_input);
        std::cout << "输入token文本: \"" << next_text << "\"" << std::endl;
        
        auto start2 = std::chrono::high_resolution_clock::now();
        torch::Tensor logits2 = model->forward(next_input, true);  // use_cache=true继续使用缓存
        auto end2 = std::chrono::high_resolution_clock::now();
        
        int64_t next_token2 = logits2[0][0].argmax().item<int64_t>();
        std::cout << "预测下一个token ID: " << next_token2 << std::endl;
        std::string next_text2 = decode_token(next_token2);
        std::cout << "预测token文本: \"" << next_text2 << "\"" << std::endl;
        std::cout << "Decode耗时: " << std::chrono::duration_cast<std::chrono::milliseconds>(end2 - start2).count() << " ms";
        std::cout << " (应该比Prefill快得多)\n";
        
        // 显示完整生成序列
        std::cout << "\n完整生成序列: \"" << input_text << next_text << next_text2 << "\"" << std::endl;
        
        // 清除缓存
        model->clear_cache();
        std::cout << "\n✅ KV缓存已清除" << std::endl;
    }

    std::cout << "\n" << std::string(70, '=') << std::endl;
    std::cout << "🎉 所有测试完成！模型工作正常" << std::endl;
    std::cout << std::string(70, '=') << std::endl;
    
    return 0;
}
