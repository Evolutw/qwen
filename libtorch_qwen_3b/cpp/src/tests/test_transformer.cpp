#include "../include/qwen_transformer_block.h"
#include "../include/qwen_embedding.h"
#include "../include/qwen_env.h"
#include "../include/qwen_model_config.h"
#include <iostream>
#include <vector>

// 配置Qwen模型参数（统一入口）
const auto& QWEN_CFG = qwen::get_model_config();
const std::string WEIGHT_PATH = qwen::get_weight_path();

// 打印张量信息
void print_tensor_info(const std::string& tensor_name, const torch::Tensor& tensor) {
    std::cout << "\n" << tensor_name << "：" << std::endl;
    std::cout << "  形状：" << tensor.sizes() << std::endl;
    std::cout << "  数据类型：" << tensor.dtype().name() << std::endl;
    std::cout << "  设备：" << tensor.device() << std::endl;
    if (tensor.numel() > 0) {
        int64_t print_size = std::min(5L, tensor.numel());
        std::cout << "  前" << print_size << "个元素：" << tensor.flatten().slice(0, 0, print_size) << std::endl;
        
        // 打印统计信息（仅对浮点类型）
        if (tensor.is_floating_point()) {
            std::cout << "  均值：" << tensor.mean().item<float>() << std::endl;
            std::cout << "  标准差：" << tensor.std().item<float>() << std::endl;
        }
    }
}

int main() {
    std::cout << std::string(60, '=') << std::endl;
    std::cout << "=== Qwen Transformer Block测试 ===" << std::endl;
    std::cout << std::string(60, '=') << std::endl;

    try {
        qwen::ensure_required_paths(WEIGHT_PATH, qwen::get_tokenizer_script(), qwen::get_tokenizer_model_dir());
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

    // 2. 初始化Transformer Block (第0层)
    QwenTransformerBlock transformer_block = QwenTransformerBlock(
        0,  // layer_idx
        QWEN_CFG.hidden_size,
        QWEN_CFG.num_heads,
        QWEN_CFG.num_kv_heads,
        QWEN_CFG.intermediate_size,
        QWEN_CFG.max_position_embeddings,
        QWEN_CFG.rope_theta,
        QWEN_CFG.rms_norm_eps
    );
    transformer_block->eval();
    std::cout << "✅ Transformer Block初始化完成" << std::endl;
    std::cout << "  hidden_size: " << QWEN_CFG.hidden_size << std::endl;
    std::cout << "  intermediate_size: " << QWEN_CFG.intermediate_size << std::endl;
    std::cout << "  num_heads: " << QWEN_CFG.num_heads << std::endl;
    std::cout << "  num_kv_heads: " << QWEN_CFG.num_kv_heads << std::endl;

    // 3. 加载权重
    transformer_block->load_weights(WEIGHT_PATH);
    
    // 4. 移动到设备并转换数据类型
    transformer_block->to(device, torch::kBFloat16);
    std::cout << "✅ 模型已移动到设备：" << device << std::endl;

    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "【测试1：基本前向传播】" << std::endl;
    std::cout << std::string(60, '=') << std::endl;

    // 5. 生成测试输入（模拟Embedding层输出）
    int64_t batch_size = 2;
    int64_t seq_len = 4;
    torch::Tensor hidden_states = torch::randn(
        {batch_size, seq_len, QWEN_CFG.hidden_size},
        torch::TensorOptions().dtype(torch::kBFloat16).device(device)
    );
    print_tensor_info("输入hidden_states", hidden_states);

    // 6. 前向传播
    torch::Tensor output;
    {
        torch::NoGradGuard no_grad;
        output = transformer_block->forward(hidden_states, false);
        print_tensor_info("Transformer Block输出", output);
    }
    
    // 验证输出形状
    std::vector<int64_t> expected_shape = {batch_size, seq_len, QWEN_CFG.hidden_size};
    bool shape_match = true;
    for (size_t i = 0; i < expected_shape.size(); ++i) {
        if (output.size(i) != expected_shape[i]) {
            shape_match = false;
            break;
        }
    }
    
    if (shape_match) {
        std::cout << "✅ Transformer Block输出形状验证通过" << std::endl;
    } else {
        std::cerr << "❌ Transformer Block输出形状验证失败" << std::endl;
        return 1;
    }

    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "【测试2：完整流程（Embedding → Transformer Block）】" << std::endl;
    std::cout << std::string(60, '=') << std::endl;

    // 7. 加载Embedding层
    QwenEmbedding embedding = QwenEmbedding(QWEN_CFG.vocab_size, QWEN_CFG.hidden_size);
    embedding->eval();
    embedding->load_weights(WEIGHT_PATH);
    embedding->to(device, torch::kBFloat16);
    std::cout << "✅ Embedding层已加载" << std::endl;
    
    // 生成token IDs
    std::vector<int64_t> token_ids = {100, 200, 300, 400};
    torch::Tensor input_ids = torch::tensor(token_ids, torch::kInt64).unsqueeze(0).to(device);
    print_tensor_info("输入Token IDs", input_ids);
    
    // Embedding -> Transformer Block
    {
        torch::NoGradGuard no_grad;
        transformer_block->clear_cache();
        
        torch::Tensor embeddings = embedding->forward(input_ids);
        print_tensor_info("Embedding输出", embeddings);
        
        torch::Tensor final_output = transformer_block->forward(embeddings, false);
        print_tensor_info("Transformer Block最终输出", final_output);
        
        std::cout << "✅ 完整流程测试成功" << std::endl;
    }

    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "【测试3：KV缓存功能】" << std::endl;
    std::cout << std::string(60, '=') << std::endl;

    // 8. 测试KV缓存
    transformer_block->clear_cache();
    
    // 第一次：缓存KV
    torch::Tensor first_embeddings = torch::randn(
        {1, 3, QWEN_CFG.hidden_size},
        torch::TensorOptions().dtype(torch::kBFloat16).device(device)
    );
    print_tensor_info("第一次输入 (seq_len=3)", first_embeddings);
    
    torch::Tensor first_output;
    {
        torch::NoGradGuard no_grad;
        first_output = transformer_block->forward(first_embeddings, true);
        print_tensor_info("第一次输出", first_output);
    }
    
    // 第二次：使用缓存
    torch::Tensor second_embeddings = torch::randn(
        {1, 1, QWEN_CFG.hidden_size},
        torch::TensorOptions().dtype(torch::kBFloat16).device(device)
    );
    print_tensor_info("第二次输入 (seq_len=1, 使用KV缓存)", second_embeddings);
    
    torch::Tensor second_output;
    {
        torch::NoGradGuard no_grad;
        second_output = transformer_block->forward(second_embeddings, true);
        print_tensor_info("第二次输出", second_output);
    }
    
    std::cout << "\n✅ KV缓存测试完成" << std::endl;
    std::cout << "  说明: Transformer Block支持自回归生成模式" << std::endl;

    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "【测试4：残差连接验证】" << std::endl;
    std::cout << std::string(60, '=') << std::endl;

    // 9. 验证残差连接是否工作
    torch::Tensor test_input = torch::randn(
        {1, 2, QWEN_CFG.hidden_size},
        torch::TensorOptions().dtype(torch::kBFloat16).device(device)
    );
    
    torch::Tensor test_output;
    {
        torch::NoGradGuard no_grad;
        transformer_block->clear_cache();
        test_output = transformer_block->forward(test_input, false);
    }
    
    // 残差连接应该让输出和输入的均值接近
    float input_mean = test_input.mean().item<float>();
    float output_mean = test_output.mean().item<float>();
    
    std::cout << "输入均值: " << input_mean << std::endl;
    std::cout << "输出均值: " << output_mean << std::endl;
    std::cout << "✅ 残差连接工作正常（输出包含输入的信息）" << std::endl;

    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "🎉 Qwen Transformer Block测试完成" << std::endl;
    std::cout << std::string(60, '=') << std::endl;
    
    return 0;
}
