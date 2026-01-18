#include "../include/qwen_attention.h"
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
    }
}

int main() {
    std::cout << "=== Qwen Attention层测试 ===" << std::endl;

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

    // 2. 初始化Attention层
    QwenAttention attention = QwenAttention(
        QWEN_CFG.hidden_size,
        QWEN_CFG.num_heads,
        QWEN_CFG.num_kv_heads,
        QWEN_CFG.max_position_embeddings,
        QWEN_CFG.rope_theta
    );
    attention->eval();
    std::cout << "✅ Attention层初始化完成" << std::endl;
    std::cout << "  hidden_size: " << QWEN_CFG.hidden_size << std::endl;
    std::cout << "  num_heads: " << QWEN_CFG.num_heads << std::endl;
    std::cout << "  num_kv_heads: " << QWEN_CFG.num_kv_heads << " (GQA)" << std::endl;
    std::cout << "  head_dim: " << (QWEN_CFG.hidden_size / QWEN_CFG.num_heads) << std::endl;

    // 3. 加载权重（使用第0层的attention权重）
    attention->load_weights(WEIGHT_PATH, 0);
    
    // 4. 移动到设备并转换数据类型
    attention->to(device, torch::kBFloat16);
    std::cout << "✅ 模型已移动到设备：" << device << std::endl;

    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "【测试1：基本前向传播】" << std::endl;
    std::cout << std::string(60, '=') << std::endl;

    // 5. 生成测试输入
    // 模拟Embedding层的输出：[batch_size=2, seq_len=4, hidden_size=896]
    int64_t batch_size = 2;
    int64_t seq_len = 4;
    torch::Tensor hidden_states = torch::randn(
        {batch_size, seq_len, QWEN_CFG.hidden_size},
        torch::TensorOptions().dtype(torch::kBFloat16).device(device)
    );
    print_tensor_info("输入hidden_states", hidden_states);

    // 6. 前向传播（不使用KV缓存）
    {
        torch::NoGradGuard no_grad;
        torch::Tensor output = attention->forward(hidden_states, false);
        print_tensor_info("Attention输出", output);
        
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
            std::cout << "✅ Attention输出形状验证通过" << std::endl;
        } else {
            std::cerr << "❌ Attention输出形状验证失败" << std::endl;
            return 1;
        }
    }

    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "【测试2：KV缓存功能】" << std::endl;
    std::cout << std::string(60, '=') << std::endl;

    // 7. 测试KV缓存
    attention->clear_cache();
    
    // 第一次前向传播：缓存KV
    torch::Tensor first_input = torch::randn(
        {1, 3, QWEN_CFG.hidden_size},
        torch::TensorOptions().dtype(torch::kBFloat16).device(device)
    );
    print_tensor_info("第一次输入 (seq_len=3)", first_input);
    
    torch::Tensor first_output;
    {
        torch::NoGradGuard no_grad;
        first_output = attention->forward(first_input, true);
        print_tensor_info("第一次输出", first_output);
    }
    
    // 第二次前向传播：使用缓存的KV，只处理新token
    torch::Tensor second_input = torch::randn(
        {1, 1, QWEN_CFG.hidden_size},
        torch::TensorOptions().dtype(torch::kBFloat16).device(device)
    );
    print_tensor_info("第二次输入 (seq_len=1, 使用KV缓存)", second_input);
    
    torch::Tensor second_output;
    {
        torch::NoGradGuard no_grad;
        second_output = attention->forward(second_input, true);
        print_tensor_info("第二次输出", second_output);
    }
    
    std::cout << "\n✅ KV缓存测试完成" << std::endl;
    std::cout << "  第一次输入长度: 3, 输出长度: " << first_output.size(1) << std::endl;
    std::cout << "  第二次输入长度: 1, 输出长度: " << second_output.size(1) << std::endl;
    std::cout << "  说明: 第二次使用了缓存的KV，只需处理1个新token" << std::endl;

    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "【测试3：完整流程（Embedding + Attention）】" << std::endl;
    std::cout << std::string(60, '=') << std::endl;

    // 8. 结合Embedding层测试完整流程
    QwenEmbedding embedding = QwenEmbedding(QWEN_CFG.vocab_size, QWEN_CFG.hidden_size);
    embedding->eval();
    embedding->load_weights(WEIGHT_PATH);
    embedding->to(device, torch::kBFloat16);
    std::cout << "✅ Embedding层已加载" << std::endl;
    
    // 生成token IDs
    std::vector<int64_t> token_ids = {100, 200, 300, 400};
    torch::Tensor input_ids = torch::tensor(token_ids, torch::kInt64).unsqueeze(0).to(device);
    print_tensor_info("输入Token IDs", input_ids);
    
    // Embedding -> Attention
    {
        torch::NoGradGuard no_grad;
        attention->clear_cache();
        
        torch::Tensor embeddings = embedding->forward(input_ids);
        print_tensor_info("Embedding输出", embeddings);
        
        torch::Tensor attn_output = attention->forward(embeddings, false);
        print_tensor_info("Attention最终输出", attn_output);
        
        std::cout << "✅ 完整流程测试成功" << std::endl;
    }

    std::cout << "\n🎉 Qwen Attention层测试完成" << std::endl;
    return 0;
}
