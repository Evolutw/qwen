#pragma once
#include "qwen_embedding.h"
#include "qwen_transformer_block.h"
#include "qwen_layernorm.h"
#include "qwen_env.h"
#include <torch/torch.h>
#include <string>
#include <vector>
#include <fstream>
#include <memory>
#include <iostream>
#include <sstream>
#include <iomanip>

/**
 * 完整的Qwen 2.5模型
 * 
 * 结构:
 *   1. Embedding层
 *   2. N层Transformer Block
 *   3. 最终RMSNorm
 *   4. LMHead输出层
 */
struct QwenModelImpl : torch::nn::Module {
    // 模型配置
    int64_t vocab_size;
    int64_t hidden_size;
    int64_t num_hidden_layers;
    int64_t num_heads;
    int64_t num_key_value_heads;
    int64_t intermediate_size;
    int64_t max_position_embeddings;
    double rope_theta;
    double rms_norm_eps;
    
    // 特殊token IDs
    int64_t bos_token_id;
    int64_t eos_token_id;
    
    // 模型组件
    QwenEmbedding embed_tokens{nullptr};           // Embedding层
    std::vector<QwenTransformerBlock> layers;      // Transformer层列表
    QwenRMSNorm norm{nullptr};                     // 最终的RMSNorm
    torch::nn::Linear lm_head{nullptr};            // 输出层（共享embedding权重）
    
    QwenModelImpl(
        int64_t vocab_size_ = 151936,
        int64_t hidden_size_ = 2048,
        int64_t num_hidden_layers_ = 36,
        int64_t num_heads_ = 16,
        int64_t num_key_value_heads_ = 2,
        int64_t intermediate_size_ = 11008,
        int64_t max_position_embeddings_ = 32768,
        double rope_theta_ = 1000000.0,
        double rms_norm_eps_ = 1e-6,
        int64_t bos_token_id_ = 151643,
        int64_t eos_token_id_ = 151645)
        : vocab_size(vocab_size_),
          hidden_size(hidden_size_),
          num_hidden_layers(num_hidden_layers_),
          num_heads(num_heads_),
          num_key_value_heads(num_key_value_heads_),
          intermediate_size(intermediate_size_),
          max_position_embeddings(max_position_embeddings_),
          rope_theta(rope_theta_),
          rms_norm_eps(rms_norm_eps_),
          bos_token_id(bos_token_id_),
          eos_token_id(eos_token_id_) {
        
        // 初始化Embedding层
        embed_tokens = register_module("embed_tokens", 
            QwenEmbedding(vocab_size, hidden_size));
        
        // 初始化N层Transformer Block
        for (int i = 0; i < num_hidden_layers; ++i) {
            auto layer = QwenTransformerBlock(
                i, hidden_size, num_heads, num_key_value_heads,
                intermediate_size, max_position_embeddings,
                rope_theta, rms_norm_eps
            );
            layers.push_back(register_module("layer" + std::to_string(i), layer));
        }
        
        // 初始化最终RMSNorm
        norm = register_module("norm", QwenRMSNorm(hidden_size, rms_norm_eps));
        
        // 初始化LMHead（注意：Qwen的LMHead与embedding共享权重）
        lm_head = register_module("lm_head", 
            torch::nn::Linear(torch::nn::LinearOptions(hidden_size, vocab_size).bias(false)));
    }
    
    /**
     * 前向传播
     * @param input_ids Token IDs [batch_size, seq_len]
     * @param use_cache 是否使用KV缓存
     * @return logits [batch_size, seq_len, vocab_size]
     */
    torch::Tensor forward(torch::Tensor input_ids, bool use_cache = false) {
        // 1. Embedding
        torch::Tensor hidden_states = embed_tokens->forward(input_ids);
        
        // 2. 通过所有Transformer层
        for (auto& layer : layers) {
            hidden_states = layer->forward(hidden_states, use_cache);
        }
        
        // 3. 最终RMSNorm
        hidden_states = norm->forward(hidden_states);
        
        // 4. LMHead输出logits
        torch::Tensor logits = lm_head(hidden_states);
        
        return logits;
    }
    
    /**
     * 清空所有层的KV缓存
     */
    void clear_cache() {
        for (auto& layer : layers) {
            layer->clear_cache();
        }
    }
    
    /**
     * 文本生成（贪婪解码）
     * @param input_ids 输入token IDs [1, input_len]
     * @param max_new_tokens 最大生成长度
     * @param temperature 采样温度（暂未实现）
     * @return 生成的完整token序列
     */
    std::vector<int64_t> generate(
        torch::Tensor input_ids,
        int max_new_tokens = 50,
        float temperature = 1.0) {
        
        // 确保输入是2D的
        if (input_ids.dim() == 1) {
            input_ids = input_ids.unsqueeze(0);
        }
        
        std::vector<int64_t> generated_tokens;
        
        // 将输入token转换为vector
        auto input_accessor = input_ids.accessor<int64_t, 2>();
        for (int i = 0; i < input_ids.size(1); ++i) {
            generated_tokens.push_back(input_accessor[0][i]);
        }
        
        // 清空KV缓存
        clear_cache();
        
        torch::NoGradGuard no_grad;
        
        // Prefill阶段：处理所有输入token
        torch::Tensor logits = forward(input_ids, true);
        
        // 获取最后一个位置的logits
        torch::Tensor next_token_logits = logits.index({0, -1, torch::indexing::Slice()});
        
        // 贪婪解码：选择概率最高的token
        int64_t next_token = next_token_logits.argmax(-1).item<int64_t>();
        generated_tokens.push_back(next_token);
        
        // 自回归生成
        for (int i = 1; i < max_new_tokens; ++i) {
            // 如果生成了EOS token，停止生成
            if (next_token == eos_token_id) {
                std::cout << "\n[生成完成] 遇到EOS token" << std::endl;
                break;
            }
            
            // 将新token作为输入（使用KV缓存，只需处理1个token）
            torch::Tensor next_input = torch::tensor({{next_token}}, torch::kInt64).to(input_ids.device());
            logits = forward(next_input, true);
            
            // 获取logits并选择下一个token
            next_token_logits = logits.index({0, 0, torch::indexing::Slice()});
            next_token = next_token_logits.argmax(-1).item<int64_t>();
            
            generated_tokens.push_back(next_token);
            
            // 打印进度
            if ((i + 1) % 10 == 0) {
                std::cout << "." << std::flush;
            }
        }
        
        return generated_tokens;
    }
    
    /**
     * 加载模型权重
     */
    void load_weights(const std::string& weight_path) {
        try {
            std::cout << "\n" << std::string(60, '=') << std::endl;
            std::cout << "正在加载Qwen完整模型权重..." << std::endl;
            std::cout << std::string(60, '=') << std::endl;

            auto load_state_dict_from_file = [](const std::string& path) {
                std::ifstream file(path, std::ios::binary | std::ios::ate);
                if (!file) {
                    throw std::runtime_error("无法打开权重文件：" + path);
                }

                std::streamsize size = file.tellg();
                file.seekg(0, std::ios::beg);
                std::vector<char> buffer(size);
                if (!file.read(buffer.data(), size)) {
                    throw std::runtime_error("读取权重文件失败：" + path);
                }

                c10::IValue state_dict_ivalue = torch::pickle_load(buffer);
                return state_dict_ivalue.toGenericDict();
            };

            auto format_layer_path = [](const std::string& dir, int layer_idx) {
                std::ostringstream oss;
                oss << dir << "/layer_" << std::setw(3) << std::setfill('0') << layer_idx << ".pt";
                return oss.str();
            };

            auto shards_dir = qwen::get_weight_shards_dir();
            if (!shards_dir.empty()) {
                std::cout << "\n[Split Weights] 使用分片权重目录: " << shards_dir << std::endl;

                // 1. Embedding
                std::cout << "\n[1/4] 加载Embedding层..." << std::endl;
                auto embed_dict = load_state_dict_from_file(shards_dir + "/embed_tokens.pt");
                embed_tokens->load_weights(embed_dict);

                // 2. Transformer layers
                std::cout << "\n[2/4] 加载" << num_hidden_layers << "层Transformer Block..." << std::endl;
                for (int i = 0; i < num_hidden_layers; ++i) {
                    if (i % 6 == 0) {
                        std::cout << "  进度: " << i << "/" << num_hidden_layers << std::endl;
                    }
                    auto layer_dict = load_state_dict_from_file(format_layer_path(shards_dir, i));
                    layers[i]->load_weights(layer_dict);
                }
                std::cout << "  完成: " << num_hidden_layers << "/" << num_hidden_layers << std::endl;

                // 3. Final norm
                std::cout << "\n[3/4] 加载最终RMSNorm..." << std::endl;
                auto norm_dict = load_state_dict_from_file(shards_dir + "/norm.pt");
                auto norm_key = c10::IValue("model.norm.weight");
                if (norm_dict.contains(norm_key)) {
                    torch::Tensor weight = norm_dict.at(norm_key).toTensor();
                    norm->load_weight(weight);
                    std::cout << "  加载: model.norm.weight " << weight.sizes() << std::endl;
                }

                // 4. LMHead (optional)
                std::cout << "\n[4/4] 加载LMHead（共享embedding权重）..." << std::endl;
                std::string lm_head_path = shards_dir + "/lm_head.pt";
                std::ifstream lm_file(lm_head_path);
                if (lm_file.good()) {
                    auto lm_dict = load_state_dict_from_file(lm_head_path);
                    auto lm_head_key = c10::IValue("lm_head.weight");
                    if (lm_dict.contains(lm_head_key)) {
                        torch::Tensor weight = lm_dict.at(lm_head_key).toTensor();
                        lm_head->weight.set_data(weight);
                        std::cout << "  加载: lm_head.weight " << weight.sizes() << std::endl;
                    }
                } else {
                    std::cout << "  使用embedding权重（权重共享）" << std::endl;
                    lm_head->weight.set_data(embed_tokens->embed_tokens->weight.data());
                }

                std::cout << "\n" << std::string(60, '=') << std::endl;
                std::cout << "✅ Qwen模型权重加载完成" << std::endl;
                std::cout << "  总层数: " << num_hidden_layers << std::endl;
                std::cout << "  总参数: ~" << (num_hidden_layers * 4 + 2) << "个模块" << std::endl;
                std::cout << std::string(60, '=') << std::endl;
                return;
            }
            
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

            std::cout << "权重文件大小: " << (size / 1024.0 / 1024.0) << " MB" << std::endl;

            // 反序列化state_dict
            c10::IValue state_dict_ivalue = torch::pickle_load(buffer);
            c10::Dict<c10::IValue, c10::IValue> state_dict = state_dict_ivalue.toGenericDict();
            
            // 1. 加载Embedding权重
            std::cout << "\n[1/4] 加载Embedding层..." << std::endl;
            embed_tokens->load_weights(state_dict);
            
            // 2. 加载所有Transformer层
            std::cout << "\n[2/4] 加载" << num_hidden_layers << "层Transformer Block..." << std::endl;
            for (int i = 0; i < num_hidden_layers; ++i) {
                if (i % 6 == 0) {  // 每6层打印一次进度
                    std::cout << "  进度: " << i << "/" << num_hidden_layers << std::endl;
                }
                layers[i]->load_weights(state_dict);
            }
            std::cout << "  完成: " << num_hidden_layers << "/" << num_hidden_layers << std::endl;
            
            // 3. 加载最终RMSNorm
            std::cout << "\n[3/4] 加载最终RMSNorm..." << std::endl;
            auto norm_key = c10::IValue("model.norm.weight");
            if (state_dict.contains(norm_key)) {
                torch::Tensor weight = state_dict.at(norm_key).toTensor();
                norm->load_weight(weight);
                std::cout << "  加载: model.norm.weight " << weight.sizes() << std::endl;
            }
            
            // 4. 加载LMHead（与embedding共享权重）
            std::cout << "\n[4/4] 加载LMHead（共享embedding权重）..." << std::endl;
            auto lm_head_key = c10::IValue("lm_head.weight");
            if (state_dict.contains(lm_head_key)) {
                torch::Tensor weight = state_dict.at(lm_head_key).toTensor();
                lm_head->weight.set_data(weight);
                std::cout << "  加载: lm_head.weight " << weight.sizes() << std::endl;
            } else {
                // 如果没有单独的lm_head权重，使用embedding权重
                std::cout << "  使用embedding权重（权重共享）" << std::endl;
                // 注意：权重共享必须在to()之前完成，否则引用可能失效
                // 不使用直接赋值，而是使用.set_data()确保权重共享
                lm_head->weight.set_data(embed_tokens->embed_tokens->weight.data());
            }
            
            std::cout << "\n" << std::string(60, '=') << std::endl;
            std::cout << "✅ Qwen模型权重加载完成" << std::endl;
            std::cout << "  总层数: " << num_hidden_layers << std::endl;
            std::cout << "  总参数: ~" << (num_hidden_layers * 4 + 2) << "个模块" << std::endl;
            std::cout << std::string(60, '=') << std::endl;
            
        } catch (const std::exception& e) {
            std::cerr << "❌ 模型权重加载失败：" << e.what() << std::endl;
            throw;
        }
    }
};

TORCH_MODULE(QwenModel);
