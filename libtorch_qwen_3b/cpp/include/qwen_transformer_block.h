#pragma once
#include "qwen_attention.h"
#include "qwen_mlp.h"
#include "qwen_layernorm.h"
#include <torch/torch.h>
#include <string>
#include <vector>
#include <fstream>
#include <memory>

/**
 * Qwen Transformer Block (单层Decoder)
 * 
 * 结构:
 *   x = x + Attention(RMSNorm(x))      # Pre-norm架构
 *   x = x + MLP(RMSNorm(x))            # Pre-norm架构
 */
struct QwenTransformerBlockImpl : torch::nn::Module {
    int64_t layer_idx;
    
    // 子模块
    QwenRMSNorm input_layernorm{nullptr};   // Attention前的LayerNorm
    QwenAttention self_attn{nullptr};        // Self-Attention层
    QwenRMSNorm post_attention_layernorm{nullptr};  // MLP前的LayerNorm
    QwenMLP mlp{nullptr};                    // MLP层
    
    QwenTransformerBlockImpl(
        int layer_idx_ = 0,
        int64_t hidden_size = 896,
        int64_t num_heads = 14,
        int64_t num_key_value_heads = 2,
        int64_t intermediate_size = 4864,
        int64_t max_position_embeddings = 32768,
        double rope_theta = 1000000.0,
        double rms_norm_eps = 1e-6)
        : layer_idx(layer_idx_) {
        
        // 初始化各个子模块
        input_layernorm = register_module("input_layernorm", 
            QwenRMSNorm(hidden_size, rms_norm_eps));
        
        self_attn = register_module("self_attn", 
            QwenAttention(hidden_size, num_heads, num_key_value_heads, 
                         max_position_embeddings, rope_theta));
        
        post_attention_layernorm = register_module("post_attention_layernorm", 
            QwenRMSNorm(hidden_size, rms_norm_eps));
        
        mlp = register_module("mlp", 
            QwenMLP(hidden_size, intermediate_size));
    }
    
    /**
     * 前向传播
     * @param hidden_states 输入张量 [batch_size, seq_len, hidden_size]
     * @param use_cache 是否使用KV缓存
     * @return 输出张量 [batch_size, seq_len, hidden_size]
     */
    torch::Tensor forward(torch::Tensor hidden_states, bool use_cache = false) {
        // 1. Self-Attention块（带残差连接）
        torch::Tensor residual = hidden_states;
        hidden_states = input_layernorm->forward(hidden_states);
        hidden_states = self_attn->forward(hidden_states, use_cache);
        hidden_states = residual + hidden_states;  // 残差连接
        
        // 2. MLP块（带残差连接）
        residual = hidden_states;
        hidden_states = post_attention_layernorm->forward(hidden_states);
        hidden_states = mlp->forward(hidden_states);
        hidden_states = residual + hidden_states;  // 残差连接
        
        return hidden_states;
    }
    
    /**
     * 清空KV缓存
     */
    void clear_cache() {
        self_attn->clear_cache();
    }
    
    /**
     * 加载权重（从state_dict）
     */
    void load_weights(const std::string& weight_path) {
        try {
            std::cout << "\n正在加载Transformer Block " << layer_idx << " 的权重..." << std::endl;
            
            // 读取权重文件
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
            
            // 反序列化state_dict并复用通用加载逻辑
            c10::IValue state_dict_ivalue = torch::pickle_load(buffer);
            c10::Dict<c10::IValue, c10::IValue> state_dict = state_dict_ivalue.toGenericDict();
            load_weights(state_dict);
            
            std::cout << "✅ Transformer Block " << layer_idx << " 权重加载完成" << std::endl;
            
        } catch (const std::exception& e) {
            std::cerr << "❌ Transformer Block权重加载失败：" << e.what() << std::endl;
            throw;
        }
    }

    // 从已加载的state_dict中加载权重（避免重复读取文件）
    void load_weights(const c10::Dict<c10::IValue, c10::IValue>& state_dict) {
        try {
            std::cout << "\n正在加载Transformer Block " << layer_idx << " 的权重..." << std::endl;

            std::string prefix = "model.layers." + std::to_string(layer_idx) + ".";

            // 加载LayerNorm权重
            auto load_layernorm = [&](const std::string& name, QwenRMSNorm& norm) {
                std::string key = prefix + name + ".weight";
                auto key_ivalue = c10::IValue(key);
                if (state_dict.contains(key_ivalue)) {
                    torch::Tensor weight = state_dict.at(key_ivalue).toTensor();
                    norm->load_weight(weight);
                    std::cout << "  加载: " << key << " " << weight.sizes() << std::endl;
                } else {
                    std::cerr << "  ⚠️ 未找到权重: " << key << std::endl;
                }
            };

            load_layernorm("input_layernorm", input_layernorm);
            load_layernorm("post_attention_layernorm", post_attention_layernorm);

            // 加载Attention和MLP权重
            self_attn->load_weights(state_dict, layer_idx);
            mlp->load_weights(state_dict, layer_idx);

            std::cout << "✅ Transformer Block " << layer_idx << " 权重加载完成" << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "❌ Transformer Block权重加载失败：" << e.what() << std::endl;
            throw;
        }
    }
};

TORCH_MODULE(QwenTransformerBlock);
