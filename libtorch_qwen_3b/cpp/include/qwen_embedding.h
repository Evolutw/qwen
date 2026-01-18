#pragma once
#include <torch/torch.h>
#include <string>
#include <fstream>
#include <vector>

// Qwen Embedding层（适配model.embed_tokens前缀+bfloat16数据类型）
struct QwenEmbeddingImpl : torch::nn::Module {
    // 词嵌入层（模块名必须为"embed_tokens"，匹配权重前缀model.embed_tokens）
    torch::nn::Embedding embed_tokens{nullptr};
    // 模型参数
    int vocab_size;
    int d_model;

    // 构造函数（传入精准模型参数）
    QwenEmbeddingImpl(int vocab_size_, int d_model_) 
        : vocab_size(vocab_size_), d_model(d_model_) {
        // 初始化词嵌入层（与权重参数一致）
        embed_tokens = register_module("embed_tokens", torch::nn::Embedding(vocab_size, d_model));
    }

    // 前向传播：输入Token ID，输出词嵌入向量（支持bfloat16）
    torch::Tensor forward(torch::Tensor input_ids) {
        // input_ids形状：[batch_size, seq_len]
        // 输出形状：[batch_size, seq_len, d_model]
        return embed_tokens(input_ids);
    }

    // 加载权重（适配model.前缀的state_dict）
    void load_weights(const std::string& weight_path) {
        try {
            // 加载完整state_dict（使用torch::jit::load加载pickle格式）
            std::cout << "正在加载权重文件：" << weight_path << std::endl;
            
            // 读取文件内容到内存
            std::ifstream file(weight_path, std::ios::binary | std::ios::ate);
            if (!file) {
                throw std::runtime_error("无法打开权重文件：" + weight_path);
            }
            
            std::streamsize size = file.tellg();
            file.seekg(0, std::ios::beg);
            std::vector<char> buffer(size);
            if (!file.read(buffer.data(), size)) {
                throw std::runtime_error("读取权重文件失败");
            }
            
            // 反序列化state_dict
            c10::IValue state_dict_ivalue = torch::pickle_load(buffer);
            c10::Dict<c10::IValue, c10::IValue> state_dict = state_dict_ivalue.toGenericDict();
            
            // 提取Embedding层权重（键名：model.embed_tokens.weight）
            std::string embed_key = "model.embed_tokens.weight";
            auto key_ivalue = c10::IValue(embed_key);
            
            if (state_dict.contains(key_ivalue)) {
                torch::Tensor embed_weight = state_dict.at(key_ivalue).toTensor();
                std::cout << "  找到权重：" << embed_key 
                         << "，形状：" << embed_weight.sizes() 
                         << "，数据类型：" << embed_weight.dtype() << std::endl;
                
                // 将权重加载到Embedding层（权重会自动移动到模块所在设备）
                embed_tokens->weight.set_data(embed_weight);
                std::cout << "✅ Embedding层权重加载成功（权重设备：" << embed_tokens->weight.device() << "）" << std::endl;
            } else {
                std::cerr << "❌ 权重文件中未找到键：" << embed_key << std::endl;
                std::cerr << "提示：请检查权重文件是否包含Qwen Embedding层权重" << std::endl;
                exit(1);
            }
        } catch (const std::exception& e) {
            std::cerr << "❌ Embedding层权重加载失败：" << e.what() << std::endl;
            exit(1);
        }
    }

    // 从已加载的state_dict中加载权重（避免重复读取文件）
    void load_weights(const c10::Dict<c10::IValue, c10::IValue>& state_dict) {
        try {
            std::string embed_key = "model.embed_tokens.weight";
            auto key_ivalue = c10::IValue(embed_key);

            if (state_dict.contains(key_ivalue)) {
                torch::Tensor embed_weight = state_dict.at(key_ivalue).toTensor();
                std::cout << "  找到权重：" << embed_key
                          << "，形状：" << embed_weight.sizes()
                          << "，数据类型：" << embed_weight.dtype() << std::endl;
                embed_tokens->weight.set_data(embed_weight);
                std::cout << "✅ Embedding层权重加载成功（权重设备：" << embed_tokens->weight.device() << "）" << std::endl;
            } else {
                std::cerr << "❌ 权重文件中未找到键：" << embed_key << std::endl;
                std::cerr << "提示：请检查权重文件是否包含Qwen Embedding层权重" << std::endl;
                exit(1);
            }
        } catch (const std::exception& e) {
            std::cerr << "❌ Embedding层权重加载失败：" << e.what() << std::endl;
            exit(1);
        }
    }
};

// 生成Module封装类（LibTorch要求）
TORCH_MODULE(QwenEmbedding);
