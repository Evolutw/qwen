#include "../include/qwen_embedding.h"
#include "../include/qwen_tokenizer.h"
#include "../include/qwen_env.h"
#include "../include/qwen_model_config.h"
#include <iostream>
#include <vector>

// 配置Qwen模型精准参数（统一入口）
const auto& QWEN_CFG = qwen::get_model_config();
const std::string WEIGHT_PATH = qwen::get_weight_path();
const std::string TOKENIZER_SCRIPT = qwen::get_tokenizer_script();

// 生成测试用Token ID
torch::Tensor get_test_input_ids() {
    // 测试输入：batch_size=2，seq_len=3
    std::vector<int64_t> token_ids = {
        100, 200, 300,
        400, 500, 600
    };
    return torch::from_blob(token_ids.data(), {2, 3}, torch::kInt64).clone();
}

// 打印张量信息
void print_tensor_info(const std::string& tensor_name, const torch::Tensor& tensor) {
    std::cout << "\n" << tensor_name << "：" << std::endl;
    std::cout << "  形状：" << tensor.sizes() << std::endl;
    std::cout << "  数据类型：" << tensor.dtype().name() << std::endl;
    std::cout << "  设备：" << tensor.device() << std::endl;
    std::cout << "  前5个元素：" << tensor.flatten().slice(0, 0, std::min(5L, tensor.numel())) << std::endl;
}

// 测试中文文本输入
void test_chinese_text(QwenEmbedding& embedding, torch::Device device) {
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "【测试2：中文文本输入】" << std::endl;
    std::cout << std::string(60, '=') << std::endl;
    
    // 初始化分词器（自动检测Python命令）
    QwenTokenizer tokenizer(TOKENIZER_SCRIPT);
    
    // 测试中文文本
    std::vector<std::string> test_texts = {
        "你好，世界！",
        "人工智能是未来的发展方向。"
    };
    
    for (const auto& text : test_texts) {
        std::cout << "\n原始文本：" << text << std::endl;
        
        try {
            // 分词
            torch::Tensor token_ids = tokenizer.encode(text);
            tokenizer.print_tokenize_info(text, token_ids);
            
            // 添加batch维度并移到设备
            token_ids = token_ids.unsqueeze(0).to(device);
            
            // Embedding前向传播
            torch::NoGradGuard no_grad;
            torch::Tensor embed_output = embedding->forward(token_ids);
            print_tensor_info("Embedding层输出", embed_output);
            
        } catch (const std::exception& e) {
            std::cerr << "❌ 处理失败：" << e.what() << std::endl;
        }
    }
}

int main(int argc, char* argv[]) {
    try {
        qwen::ensure_required_paths(WEIGHT_PATH, TOKENIZER_SCRIPT, qwen::get_tokenizer_model_dir());
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

    // 2. 初始化Embedding层（传入精准参数）
    QwenEmbedding embedding = QwenEmbedding(QWEN_CFG.vocab_size, QWEN_CFG.hidden_size);
    embedding->eval(); // 推理模式
    std::cout << "✅ Embedding层初始化完成（vocab_size=" << QWEN_CFG.vocab_size << ", d_model=" << QWEN_CFG.hidden_size << "）" << std::endl;

    // 3. 加载权重（先加载权重再移动到目标设备）
    embedding->load_weights(WEIGHT_PATH);
    
    // 4. 移动模型到目标设备并转换数据类型
    embedding->to(device, torch::kBFloat16); // 转换为bfloat16，匹配权重类型
    std::cout << "✅ 模型已移动到设备：" << device << std::endl;

    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "【测试1：直接Token ID输入】" << std::endl;
    std::cout << std::string(60, '=') << std::endl;

    // 5. 生成测试输入
    torch::Tensor input_ids = get_test_input_ids().to(device);
    print_tensor_info("测试输入Token ID", input_ids);

    // 6. 前向传播（关闭梯度计算）
    torch::NoGradGuard no_grad;
    torch::Tensor embed_output = embedding->forward(input_ids);
    print_tensor_info("Embedding层输出", embed_output);

    // 7. 验证输出形状
    std::vector<int64_t> expected_shape = {2, 3, QWEN_CFG.hidden_size};
    bool shape_match = true;
    for (int i = 0; i < expected_shape.size(); ++i) {
        if (embed_output.size(i) != expected_shape[i]) {
            shape_match = false;
            break;
        }
    }
    if (shape_match) {
        std::cout << "\n✅ Embedding层输出形状验证通过" << std::endl;
    } else {
        std::cerr << "\n❌ Embedding层输出形状验证失败" << std::endl;
        return 1;
    }

    // 8. 验证数据类型（匹配bfloat16）
    if (embed_output.dtype() == torch::kBFloat16) {
        std::cout << "✅ Embedding层输出数据类型验证通过（bfloat16）" << std::endl;
    } else {
        std::cerr << "❌ Embedding层输出数据类型验证失败，预期bfloat16，实际" << embed_output.dtype().name() << std::endl;
        return 1;
    }

    // 9. 测试中文文本输入（如果有命令行参数，使用自定义文本）
    if (argc > 1) {
        std::string custom_text;
        for (int i = 1; i < argc; ++i) {
            if (i > 1) custom_text += " ";
            custom_text += argv[i];
        }
        
        std::cout << "\n" << std::string(60, '=') << std::endl;
        std::cout << "【测试3：自定义中文文本】" << std::endl;
        std::cout << std::string(60, '=') << std::endl;
        
        try {
            QwenTokenizer tokenizer(TOKENIZER_SCRIPT);
            std::cout << "\n原始文本：" << custom_text << std::endl;
            
            torch::Tensor token_ids = tokenizer.encode(custom_text);
            tokenizer.print_tokenize_info(custom_text, token_ids);
            
            token_ids = token_ids.unsqueeze(0).to(device);
            torch::NoGradGuard no_grad;
            torch::Tensor embed_output = embedding->forward(token_ids);
            print_tensor_info("Embedding层输出", embed_output);
            
            std::cout << "✅ 自定义文本处理成功" << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "❌ 自定义文本处理失败：" << e.what() << std::endl;
        }
    } else {
        // 使用预设的中文文本测试
        test_chinese_text(embedding, device);
    }

    std::cout << "\n🎉 Qwen Embedding层测试完成" << std::endl;
    return 0;
}
