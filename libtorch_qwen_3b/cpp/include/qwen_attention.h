#pragma once
#include <torch/torch.h>
#include <cmath>
#include <string>
#include <vector>
#include <fstream>
#include <memory>

/**
 * Rotary Position Embedding (RoPE)
 * 用于为Query和Key添加位置信息
 */
class RotaryEmbedding {
private:
    int64_t dim;           // head_dim
    int64_t max_seq_len;   // 最大序列长度
    double theta;          // RoPE基础频率
    torch::Tensor cos_cached;  // 缓存的cos值
    torch::Tensor sin_cached;  // 缓存的sin值

public:
    RotaryEmbedding(int64_t dim_, int64_t max_seq_len_ = 32768, double theta_ = 1000000.0)
        : dim(dim_), max_seq_len(max_seq_len_), theta(theta_) {
        // 预计算RoPE的cos和sin值
        // inv_freq = 1.0 / (theta ^ (torch.arange(0, dim, 2).float() / dim))
        torch::Tensor t = torch::arange(0, dim, 2, torch::kFloat32);
        torch::Tensor inv_freq = 1.0 / torch::pow(theta, t / dim);
        
        // t = torch.arange(max_seq_len).float()
        torch::Tensor seq = torch::arange(max_seq_len, torch::kFloat32);
        
        // freqs = torch.outer(t, inv_freq)
        torch::Tensor freqs = torch::outer(seq, inv_freq);  // [max_seq_len, dim/2]
        
        // emb = torch.cat([freqs, freqs], dim=-1)  # [max_seq_len, dim]
        torch::Tensor emb = torch::cat({freqs, freqs}, -1);
        
        // 缓存cos和sin值
        cos_cached = emb.cos().unsqueeze(0).unsqueeze(0);  // [1, 1, max_seq_len, dim]
        sin_cached = emb.sin().unsqueeze(0).unsqueeze(0);  // [1, 1, max_seq_len, dim]
    }

    /**
     * 应用旋转位置编码
     * @param x 输入张量 [batch_size, num_heads, seq_len, head_dim]
     * @param seq_len 序列长度
     * @param position_offset 位置偏移（用于KV cache场景）
     * @return 应用RoPE后的张量
     */
    std::pair<torch::Tensor, torch::Tensor> apply_rotary_pos_emb(
        torch::Tensor q, torch::Tensor k, int64_t seq_len, int64_t position_offset = 0) {
        
        // 获取对应长度的cos和sin（考虑position offset）
        torch::Tensor cos = cos_cached.narrow(2, position_offset, seq_len).to(q.device()).to(q.dtype());
        torch::Tensor sin = sin_cached.narrow(2, position_offset, seq_len).to(k.device()).to(k.dtype());
        
        // 应用旋转
        auto rotate_half = [](torch::Tensor x) {
            // 将x分成两半，并交换位置添加负号
            int64_t half = x.size(-1) / 2;
            torch::Tensor x1 = x.narrow(-1, 0, half);
            torch::Tensor x2 = x.narrow(-1, half, half);
            return torch::cat({-x2, x1}, -1);
        };
        
        torch::Tensor q_embed = q * cos + rotate_half(q) * sin;
        torch::Tensor k_embed = k * cos + rotate_half(k) * sin;
        
        return {q_embed, k_embed};
    }
};

/**
 * Qwen Multi-Head Attention层
 * 支持Grouped Query Attention (GQA)和KV缓存
 */
struct QwenAttentionImpl : torch::nn::Module {
    int64_t hidden_size;           // 896
    int64_t num_heads;             // 14 (query heads)
    int64_t num_key_value_heads;   // 2 (key/value heads, GQA)
    int64_t head_dim;              // hidden_size / num_heads = 64
    int64_t num_key_value_groups;  // num_heads / num_key_value_heads = 7
    
    // 线性投影层
    torch::nn::Linear q_proj{nullptr};
    torch::nn::Linear k_proj{nullptr};
    torch::nn::Linear v_proj{nullptr};
    torch::nn::Linear o_proj{nullptr};
    
    // RoPE位置编码
    std::shared_ptr<RotaryEmbedding> rotary_emb;
    
    // KV缓存（用于自回归生成）
    torch::Tensor past_key;   // [batch_size, num_kv_heads, past_seq_len, head_dim]
    torch::Tensor past_value; // [batch_size, num_kv_heads, past_seq_len, head_dim]
    
    QwenAttentionImpl(
        int64_t hidden_size_ = 896,
        int64_t num_heads_ = 14,
        int64_t num_key_value_heads_ = 2,
        int64_t max_position_embeddings = 32768,
        double rope_theta = 1000000.0)
        : hidden_size(hidden_size_),
          num_heads(num_heads_),
          num_key_value_heads(num_key_value_heads_),
          head_dim(hidden_size_ / num_heads_),
          num_key_value_groups(num_heads_ / num_key_value_heads_) {
        
        // 初始化线性投影层（Qwen使用bias）
        q_proj = register_module("q_proj", torch::nn::Linear(torch::nn::LinearOptions(hidden_size, num_heads * head_dim).bias(true)));
        k_proj = register_module("k_proj", torch::nn::Linear(torch::nn::LinearOptions(hidden_size, num_key_value_heads * head_dim).bias(true)));
        v_proj = register_module("v_proj", torch::nn::Linear(torch::nn::LinearOptions(hidden_size, num_key_value_heads * head_dim).bias(true)));
        o_proj = register_module("o_proj", torch::nn::Linear(torch::nn::LinearOptions(num_heads * head_dim, hidden_size).bias(false)));
        
        // 初始化RoPE
        rotary_emb = std::make_shared<RotaryEmbedding>(head_dim, max_position_embeddings, rope_theta);
    }

    /**
     * 重复KV头以匹配Q头数量（用于GQA）
     */
    torch::Tensor repeat_kv(torch::Tensor x, int64_t n_rep) {
        if (n_rep == 1) {
            return x;
        }
        // x: [batch_size, num_kv_heads, seq_len, head_dim]
        // -> [batch_size, num_kv_heads, 1, seq_len, head_dim]
        // -> [batch_size, num_kv_heads, n_rep, seq_len, head_dim]
        // -> [batch_size, num_heads, seq_len, head_dim]
        auto sizes = x.sizes().vec();
        x = x.unsqueeze(2).expand({sizes[0], sizes[1], n_rep, sizes[2], sizes[3]});
        return x.reshape({sizes[0], sizes[1] * n_rep, sizes[2], sizes[3]});
    }

    /**
     * 前向传播
     * @param hidden_states 输入张量 [batch_size, seq_len, hidden_size]
     * @param use_cache 是否使用KV缓存
     * @return 输出张量 [batch_size, seq_len, hidden_size]
     */
    torch::Tensor forward(torch::Tensor hidden_states, bool use_cache = false) {
        int64_t batch_size = hidden_states.size(0);
        int64_t seq_len = hidden_states.size(1);
        
        // Q, K, V投影
        torch::Tensor query_states = q_proj(hidden_states);  // [bs, seq_len, num_heads * head_dim]
        torch::Tensor key_states = k_proj(hidden_states);    // [bs, seq_len, num_kv_heads * head_dim]
        torch::Tensor value_states = v_proj(hidden_states);  // [bs, seq_len, num_kv_heads * head_dim]
        
        // Reshape: [bs, seq_len, num_heads, head_dim] -> [bs, num_heads, seq_len, head_dim]
        query_states = query_states.view({batch_size, seq_len, num_heads, head_dim}).transpose(1, 2);
        key_states = key_states.view({batch_size, seq_len, num_key_value_heads, head_dim}).transpose(1, 2);
        value_states = value_states.view({batch_size, seq_len, num_key_value_heads, head_dim}).transpose(1, 2);
        
        // 计算position offset（KV cache场景下需要）
        int64_t position_offset = (use_cache && past_key.defined()) ? past_key.size(2) : 0;
        
        // 应用RoPE位置编码
        auto [query_rot, key_rot] = rotary_emb->apply_rotary_pos_emb(query_states, key_states, seq_len, position_offset);
        query_states = query_rot;
        key_states = key_rot;
        
        // 如果使用KV缓存，拼接过去的key/value
        if (use_cache && past_key.defined() && past_value.defined()) {
            key_states = torch::cat({past_key, key_states}, 2);
            value_states = torch::cat({past_value, value_states}, 2);
        }
        
        // 更新KV缓存
        if (use_cache) {
            past_key = key_states;
            past_value = value_states;
        }
        
        // GQA: 重复KV头以匹配Q头数量
        key_states = repeat_kv(key_states, num_key_value_groups);
        value_states = repeat_kv(value_states, num_key_value_groups);
        
        // 计算注意力分数
        // Q @ K^T / sqrt(head_dim)
        torch::Tensor attn_weights = torch::matmul(query_states, key_states.transpose(-2, -1));
        attn_weights = attn_weights / std::sqrt(static_cast<double>(head_dim));
        
        // 应用causal mask（仅在seq_len > 1时需要）
        // 对于autoregressive模型，需要屏蔽未来位置
        int64_t kv_seq_len = key_states.size(2);  // KV的序列长度（可能包含past）
        if (seq_len > 1) {
            // 创建causal mask: 下三角矩阵，上三角为-inf
            // 注意：如果有KV cache，mask的大小应该是 [seq_len, kv_seq_len]
            torch::Tensor causal_mask = torch::ones({seq_len, kv_seq_len}, 
                torch::TensorOptions().dtype(torch::kFloat32).device(attn_weights.device()));
            
            // 使用triu创建上三角mask（diagonal=1表示保留对角线及以下，上三角为0）
            causal_mask = torch::triu(causal_mask, kv_seq_len - seq_len + 1);
            
            // 将上三角的0替换为-inf（这样softmax后会变成0）
            causal_mask = causal_mask.masked_fill(causal_mask == 1.0, -std::numeric_limits<float>::infinity());
            causal_mask = causal_mask.masked_fill(causal_mask == 0.0, 0.0);
            
            // 应用mask
            attn_weights = attn_weights + causal_mask.to(attn_weights.dtype());
        }
        
        // Softmax
        attn_weights = torch::softmax(attn_weights, -1);
        
        // 应用注意力到value
        // attn_weights @ V
        torch::Tensor attn_output = torch::matmul(attn_weights, value_states);
        
        // Reshape: [bs, num_heads, seq_len, head_dim] -> [bs, seq_len, hidden_size]
        attn_output = attn_output.transpose(1, 2).contiguous();
        attn_output = attn_output.view({batch_size, seq_len, hidden_size});
        
        // 输出投影
        attn_output = o_proj(attn_output);
        
        return attn_output;
    }

    /**
     * 清空KV缓存
     */
    void clear_cache() {
        past_key = torch::Tensor();
        past_value = torch::Tensor();
    }

    /**
     * 加载权重（从state_dict）
     */
    void load_weights(const std::string& weight_path, int layer_idx = 0) {
        try {
            std::cout << "正在加载Attention层权重（Layer " << layer_idx << "）..." << std::endl;
            
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
            std::string prefix = "model.layers." + std::to_string(layer_idx) + ".self_attn.";
            
            // 加载各个权重和bias
            auto load_weight = [&](const std::string& name, torch::nn::Linear& linear, bool has_bias = false) {
                std::string key = prefix + name + ".weight";
                auto key_ivalue = c10::IValue(key);
                if (state_dict.contains(key_ivalue)) {
                    torch::Tensor weight = state_dict.at(key_ivalue).toTensor();
                    linear->weight.set_data(weight);
                    std::cout << "  加载: " << key << " " << weight.sizes() << std::endl;
                } else {
                    std::cerr << "  ⚠️ 未找到权重: " << key << std::endl;
                }
                
                // 加载bias（如果存在）
                if (has_bias) {
                    std::string bias_key = prefix + name + ".bias";
                    auto bias_key_ivalue = c10::IValue(bias_key);
                    if (state_dict.contains(bias_key_ivalue)) {
                        torch::Tensor bias = state_dict.at(bias_key_ivalue).toTensor();
                        linear->bias.set_data(bias);
                        std::cout << "  加载: " << bias_key << " " << bias.sizes() << std::endl;
                    }
                }
            };
            
            load_weight("q_proj", q_proj, true);
            load_weight("k_proj", k_proj, true);
            load_weight("v_proj", v_proj, true);
            load_weight("o_proj", o_proj, false);
            
            std::cout << "✅ Attention层权重加载成功" << std::endl;
            
        } catch (const std::exception& e) {
            std::cerr << "❌ Attention层权重加载失败：" << e.what() << std::endl;
            throw;
        }
    }

    // 从已加载的state_dict中加载权重（避免重复读取文件）
    void load_weights(const c10::Dict<c10::IValue, c10::IValue>& state_dict, int layer_idx = 0) {
        try {
            std::cout << "正在加载Attention层权重（Layer " << layer_idx << "）..." << std::endl;

            std::string prefix = "model.layers." + std::to_string(layer_idx) + ".self_attn.";

            auto load_linear = [&](const std::string& name, torch::nn::Linear& linear) {
                std::string key = prefix + name + ".weight";
                auto key_ivalue = c10::IValue(key);
                if (state_dict.contains(key_ivalue)) {
                    torch::Tensor weight = state_dict.at(key_ivalue).toTensor();
                    linear->weight.set_data(weight);
                    std::cout << "  加载: " << key << " " << weight.sizes() << std::endl;
                } else {
                    std::cerr << "  ⚠️ 未找到权重: " << key << std::endl;
                }

                std::string bias_key = prefix + name + ".bias";
                auto bias_key_ivalue = c10::IValue(bias_key);
                if (state_dict.contains(bias_key_ivalue)) {
                    torch::Tensor bias = state_dict.at(bias_key_ivalue).toTensor();
                    linear->bias.set_data(bias);
                    std::cout << "  加载: " << bias_key << " " << bias.sizes() << std::endl;
                }
            };

            load_linear("q_proj", q_proj);
            load_linear("k_proj", k_proj);
            load_linear("v_proj", v_proj);
            load_linear("o_proj", o_proj);

            std::cout << "✅ Attention层权重加载成功" << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "❌ Attention层权重加载失败：" << e.what() << std::endl;
            throw;
        }
    }
};

TORCH_MODULE(QwenAttention);
