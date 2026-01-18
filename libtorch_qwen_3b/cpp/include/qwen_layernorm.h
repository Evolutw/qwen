#pragma once
#include <torch/torch.h>
#include <string>
#include <vector>
#include <fstream>
#include <memory>

/**
 * RMS Normalization (Root Mean Square Layer Normalization)
 * Qwen使用RMSNorm而不是标准LayerNorm
 * 
 * 公式: y = x * w / RMS(x), where RMS(x) = sqrt(mean(x^2) + eps)
 */
struct QwenRMSNormImpl : torch::nn::Module {
    torch::Tensor weight;  // 可学习的缩放参数
    double eps;            // 数值稳定性参数
    int64_t hidden_size;
    
    QwenRMSNormImpl(int64_t hidden_size_ = 896, double eps_ = 1e-6)
        : hidden_size(hidden_size_), eps(eps_) {
        // 初始化权重为全1
        weight = register_parameter("weight", torch::ones({hidden_size}));
    }
    
    /**
     * 前向传播
     * @param hidden_states 输入张量 [batch_size, seq_len, hidden_size]
     * @return 归一化后的张量
     */
    torch::Tensor forward(torch::Tensor hidden_states) {
        // 保存输入的数据类型
        auto input_dtype = hidden_states.dtype();
        
        // 转换到float32进行计算（提高数值稳定性）
        hidden_states = hidden_states.to(torch::kFloat32);
        
        // 计算方差: mean(x^2)
        torch::Tensor variance = hidden_states.pow(2).mean(-1, true);
        
        // RMS归一化: x / sqrt(mean(x^2) + eps)
        hidden_states = hidden_states * torch::rsqrt(variance + eps);
        
        // 转换回原始数据类型并应用缩放权重
        return (weight * hidden_states).to(input_dtype);
    }
    
    /**
     * 加载权重
     */
    void load_weight(torch::Tensor weight_data) {
        weight.set_data(weight_data);
    }
};

TORCH_MODULE(QwenRMSNorm);
