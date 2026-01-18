#include "../include/qwen_model.h"
#include "../include/qwen_tokenizer.h"
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

// 解码token IDs为文本（通过Python tokenizer）
std::string decode_tokens(const std::vector<int64_t>& token_ids) {
    // 构建Python命令
    std::string ids_str = "[";
    for (size_t i = 0; i < token_ids.size(); ++i) {
        ids_str += std::to_string(token_ids[i]);
        if (i < token_ids.size() - 1) ids_str += ",";
    }
    ids_str += "]";
    
    std::string cmd = PYTHON_CMD + " -c \"from transformers import AutoTokenizer; tokenizer = AutoTokenizer.from_pretrained('" + TOKENIZER_MODEL_DIR + "', trust_remote_code=True); print(tokenizer.decode(" + ids_str + "))\" 2>/dev/null";
    
    FILE* pipe = popen(cmd.c_str(), "r");
    if (!pipe) return "[解码失败]";
    
    char buffer[4096];
    std::string result;
    while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
        result += buffer;
    }
    pclose(pipe);
    
    // 移除末尾换行符
    if (!result.empty() && result.back() == '\n') {
        result.pop_back();
    }
    
    return result;
}

int main() {
    std::cout << std::string(70, '=') << std::endl;
    std::cout << "=== Qwen 2.5 完整模型测试 ===" << std::endl;
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
    std::cout << "✅ 模型初始化完成" << std::endl;
    std::cout << "  模型配置:" << std::endl;
    std::cout << "    - 层数: " << QWEN_CFG.num_layers << std::endl;
    std::cout << "    - 隐藏维度: " << QWEN_CFG.hidden_size << std::endl;
    std::cout << "    - 中间维度: " << QWEN_CFG.intermediate_size << std::endl;
    std::cout << "    - 注意力头数: " << QWEN_CFG.num_heads << " (Q) / " << QWEN_CFG.num_kv_heads << " (KV)" << std::endl;
    std::cout << "    - 词汇表大小: " << QWEN_CFG.vocab_size << std::endl;

    // 3. 加载权重
    auto start_load = std::chrono::high_resolution_clock::now();
    model->load_weights(WEIGHT_PATH);
    auto end_load = std::chrono::high_resolution_clock::now();
    auto load_time = std::chrono::duration_cast<std::chrono::seconds>(end_load - start_load).count();
    std::cout << "权重加载耗时: " << load_time << " 秒" << std::endl;
    
    // 4. 移动到设备
    std::cout << "\n正在将模型移动到 " << device << "..." << std::endl;
    auto start_move = std::chrono::high_resolution_clock::now();
    model->to(device, torch::kBFloat16);
    auto end_move = std::chrono::high_resolution_clock::now();
    auto move_time = std::chrono::duration_cast<std::chrono::seconds>(end_move - start_move).count();
    std::cout << "✅ 模型已移动到设备（耗时: " << move_time << " 秒）" << std::endl;

    std::cout << "\n" << std::string(70, '=') << std::endl;
    std::cout << "【测试1：基本前向传播】" << std::endl;
    std::cout << std::string(70, '=') << std::endl;

    // 5. 测试前向传播
    std::vector<int64_t> test_tokens = {100, 200, 300};
    torch::Tensor input_ids = torch::from_blob(
        test_tokens.data(), 
        {1, static_cast<long>(test_tokens.size())}, 
        torch::kInt64
    ).clone().to(device);
    print_tensor_info("输入Token IDs", input_ids);
    
    torch::Tensor logits;
    {
        torch::NoGradGuard no_grad;
        auto start_fwd = std::chrono::high_resolution_clock::now();
        logits = model->forward(input_ids, false);
        auto end_fwd = std::chrono::high_resolution_clock::now();
        auto fwd_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_fwd - start_fwd).count();
        
        print_tensor_info("模型输出logits", logits);
        std::cout << "前向传播耗时: " << fwd_time << " ms" << std::endl;
    }
    
    // 验证输出形状
    if (logits.size(0) == 1 && logits.size(1) == 3 && logits.size(2) == QWEN_CFG.vocab_size) {
        std::cout << "✅ 输出形状验证通过: [batch_size=1, seq_len=3, vocab_size=" << QWEN_CFG.vocab_size << "]" << std::endl;
    } else {
        std::cerr << "❌ 输出形状验证失败" << std::endl;
        return 1;
    }

    std::cout << "\n" << std::string(70, '=') << std::endl;
    std::cout << "【测试2：文本生成（简单示例）】" << std::endl;
    std::cout << std::string(70, '=') << std::endl;

    // 6. 初始化tokenizer
    QwenTokenizer tokenizer(TOKENIZER_SCRIPT);
    
    // 7. 测试文本生成
    std::string prompt = "你好";
    std::cout << "\n输入提示: \"" << prompt << "\"" << std::endl;
    
    try {
        // 分词
        torch::Tensor prompt_ids = tokenizer.encode(prompt).to(device);
        std::cout << "输入Token IDs: " << prompt_ids << std::endl;
        
        // 生成
        std::cout << "\n开始生成（最多20个token）";
        auto start_gen = std::chrono::high_resolution_clock::now();
        std::vector<int64_t> generated_ids = model->generate(prompt_ids, 20);
        auto end_gen = std::chrono::high_resolution_clock::now();
        auto gen_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_gen - start_gen).count();
        
        std::cout << "\n\n生成完成！" << std::endl;
        std::cout << "生成Token数: " << generated_ids.size() << std::endl;
        std::cout << "生成耗时: " << gen_time << " ms" << std::endl;
        std::cout << "平均速度: " << (generated_ids.size() * 1000.0 / gen_time) << " tokens/s" << std::endl;
        
        // 解码
        std::cout << "\n生成的Token IDs: [";
        for (size_t i = 0; i < std::min(generated_ids.size(), size_t(20)); ++i) {
            std::cout << generated_ids[i];
            if (i < generated_ids.size() - 1) std::cout << ", ";
        }
        if (generated_ids.size() > 20) std::cout << "...";
        std::cout << "]" << std::endl;
        
        std::cout << "\n解码生成的文本..." << std::endl;
        std::string generated_text = decode_tokens(generated_ids);
        std::cout << "\n" << std::string(70, '-') << std::endl;
        std::cout << "完整输出:\n" << generated_text << std::endl;
        std::cout << std::string(70, '-') << std::endl;
        
        std::cout << "\n✅ 文本生成测试成功" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ 文本生成失败: " << e.what() << std::endl;
        return 1;
    }

    std::cout << "\n" << std::string(70, '=') << std::endl;
    std::cout << "【测试3：更长的提示词】" << std::endl;
    std::cout << std::string(70, '=') << std::endl;

    // 8. 测试更复杂的提示词
    std::string long_prompt = "人工智能的未来发展方向是";
    std::cout << "\n输入提示: \"" << long_prompt << "\"" << std::endl;
    
    try {
        torch::Tensor long_prompt_ids = tokenizer.encode(long_prompt).to(device);
        std::cout << "输入Token IDs数量: " << long_prompt_ids.size(0) << std::endl;
        
        std::cout << "\n开始生成（最多30个token）";
        auto start_gen2 = std::chrono::high_resolution_clock::now();
        std::vector<int64_t> generated_ids2 = model->generate(long_prompt_ids, 30);
        auto end_gen2 = std::chrono::high_resolution_clock::now();
        auto gen_time2 = std::chrono::duration_cast<std::chrono::milliseconds>(end_gen2 - start_gen2).count();
        
        std::cout << "\n\n生成完成！" << std::endl;
        std::cout << "生成耗时: " << gen_time2 << " ms" << std::endl;
        
        std::string generated_text2 = decode_tokens(generated_ids2);
        std::cout << "\n" << std::string(70, '-') << std::endl;
        std::cout << "完整输出:\n" << generated_text2 << std::endl;
        std::cout << std::string(70, '-') << std::endl;
        
        std::cout << "\n✅ 长提示词生成测试成功" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ 长提示词生成失败: " << e.what() << std::endl;
    }

    std::cout << "\n" << std::string(70, '=') << std::endl;
    std::cout << "🎉 Qwen 2.5完整模型测试完成" << std::endl;
    std::cout << std::string(70, '=') << std::endl;
    
    return 0;
}
