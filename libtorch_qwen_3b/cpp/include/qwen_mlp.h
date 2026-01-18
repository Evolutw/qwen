#pragma once
#include <torch/torch.h>
#include <string>
#include <vector>
#include <fstream>
#include <memory>

/**
 * Qwen MLP (Multi-Layer Perceptron)
 * 使用Gate机制的前馈神经网络
 * 
 * 结构: Gate-Up-Down with SiLU activation
 * - gate_proj: hidden_size -> intermediate_size
 * - up_proj:   hidden_size -> intermediate_size  
 * - down_proj: intermediate_size -> hidden_size
 * 
 * 输出 = down_proj(SiLU(gate_proj(x)) * up_proj(x))
 */
struct QwenMLPImpl : torch::nn::Module {
    int64_t hidden_size;        // 896
    int64_t intermediate_size;  // 4864
    
    // 三个线性投影层
    torch::nn::Linear gate_proj{nullptr};
    torch::nn::Linear up_proj{nullptr};
    torch::nn::Linear down_proj{nullptr};
    
    QwenMLPImpl(int64_t hidden_size_ = 896, int64_t intermediate_size_ = 4864)
        : hidden_size(hidden_size_), intermediate_size(intermediate_size_) {
        
        // 初始化线性层（无bias）
        gate_proj = register_module("gate_proj", 
            torch::nn::Linear(torch::nn::LinearOptions(hidden_size, intermediate_size).bias(false)));
        up_proj = register_module("up_proj", 
            torch::nn::Linear(torch::nn::LinearOptions(hidden_size, intermediate_size).bias(false)));
        down_proj = register_module("down_proj", 
            torch::nn::Linear(torch::nn::LinearOptions(intermediate_size, hidden_size).bias(false)));
    }
    
    /**
     * 前向传播
     * @param hidden_states 输入张量 [batch_size, seq_len, hidden_size]
     * @return 输出张量 [batch_size, seq_len, hidden_size]
     */
    torch::Tensor forward(torch::Tensor hidden_states) {
        // Gate分支: gate_proj(x) -> SiLU激活
        torch::Tensor gate_output = gate_proj(hidden_states);
        gate_output = torch::silu(gate_output);
        
        // Up分支: up_proj(x)
        torch::Tensor up_output = up_proj(hidden_states);
        
        // 门控机制: SiLU(gate) * up
        torch::Tensor gated_output = gate_output * up_output;
        
        // Down投影: down_proj(gated)
        torch::Tensor output = down_proj(gated_output);
        
        return output;
    }
    
    /**
     * 加载权重（从state_dict）
     */
    void load_weights(const std::string& weight_path, int layer_idx = 0) {
        try {
            std::cout << "正在加载MLP层权重（Layer " << layer_idx << "）..." << std::endl;
            
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
            
            // 反序列化state_dict
            c10::IValue state_dict_ivalue = torch::pickle_load(buffer);
            c10::Dict<c10::IValue, c10::IValue> state_dict = state_dict_ivalue.toGenericDict();
            
            // 权重键名前缀
            std::string prefix = "model.layers." + std::to_string(layer_idx) + ".mlp.";
            
            // 加载各个权重
            auto load_weight = [&](const std::string& name, torch::nn::Linear& linear) {
                std::string key = prefix + name + ".weight";
                auto key_ivalue = c10::IValue(key);
                if (state_dict.contains(key_ivalue)) {
                    torch::Tensor weight = state_dict.at(key_ivalue).toTensor();
                    linear->weight.set_data(weight);
                    std::cout << "  加载: " << key << " " << weight.sizes() << std::endl;
                } else {
                    std::cerr << "  ⚠️ 未找到权重: " << key << std::endl;
                }
            };
            
            load_weight("gate_proj", gate_proj);
            load_weight("up_proj", up_proj);
            load_weight("down_proj", down_proj);
            
            std::cout << "✅ MLP层权重加载成功" << std::endl;
            
        } catch (const std::exception& e) {
            std::cerr << "❌ MLP层权重加载失败：" << e.what() << std::endl;
            throw;
        }
    }

    // 从已加载的state_dict中加载权重（避免重复读取文件）
    void load_weights(const c10::Dict<c10::IValue, c10::IValue>& state_dict, int layer_idx = 0) {
        try {
            std::cout << "正在加载MLP层权重（Layer " << layer_idx << "）..." << std::endl;

            std::string prefix = "model.layers." + std::to_string(layer_idx) + ".mlp.";
            auto load_weight = [&](const std::string& name, torch::nn::Linear& linear) {
                std::string key = prefix + name + ".weight";
                auto key_ivalue = c10::IValue(key);
                if (state_dict.contains(key_ivalue)) {
                    torch::Tensor weight = state_dict.at(key_ivalue).toTensor();
                    linear->weight.set_data(weight);
                    std::cout << "  加载: " << key << " " << weight.sizes() << std::endl;
                } else {
                    std::cerr << "  ⚠️ 未找到权重: " << key << std::endl;
                }
            };

            load_weight("gate_proj", gate_proj);
            load_weight("up_proj", up_proj);
            load_weight("down_proj", down_proj);

            std::cout << "✅ MLP层权重加载成功" << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "❌ MLP层权重加载失败：" << e.what() << std::endl;
            throw;
        }
    }
};

TORCH_MODULE(QwenMLP);
