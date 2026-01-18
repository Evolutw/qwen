#pragma once
#include <string>
#include <cstdint>

// 手动切换模型："0.5b" 或 "3b"
#define QWEN_MODEL_VARIANT "3b"

// 手动设置默认模型目录（若未设置环境变量，将使用这些路径）
#define QWEN_MODEL_DIR_05B "/home/wx/llm_learning/models/qwen2.5-0.5b-instruct"
#define QWEN_MODEL_DIR_3B  "/home/wx/llm_learning/models/qwen2.5-3b-instruct"
#define QWEN_SHARDS_DIR_05B "/home/wx/llm_learning/models/qwen2.5-0.5b-instruct/weights_shards"
#define QWEN_SHARDS_DIR_3B  "/home/wx/llm_learning/models/qwen2.5-3b-instruct/weights_shards"

namespace qwen {

struct QwenModelConfig {
    int64_t vocab_size;
    int64_t hidden_size;
    int64_t num_layers;
    int64_t intermediate_size;
    int64_t num_heads;
    int64_t num_kv_heads;
    int64_t max_position_embeddings;
    double rope_theta;
    double rms_norm_eps;
    int64_t bos_token_id;
    int64_t eos_token_id;
    int64_t im_end_id;
    const char* weight_filename;
    const char* default_model_dir;
    const char* default_shards_dir;
};

inline const QwenModelConfig& get_model_config() {
    static const QwenModelConfig cfg = []() {
        std::string variant = QWEN_MODEL_VARIANT;
        if (variant == "0.5b" || variant == "0.5" || variant == "0_5b") {
            return QwenModelConfig{
                151936,
                896,
                24,
                4864,
                14,
                2,
                32768,
                1000000.0,
                1e-6,
                151643,
                151645,
                151645,
                "qwen2.5-0.5b-instruct.pt",
                QWEN_MODEL_DIR_05B,
                QWEN_SHARDS_DIR_05B
            };
        }
        return QwenModelConfig{
            151936,
            2048,
            36,
            11008,
            16,
            2,
            32768,
            1000000.0,
            1e-6,
            151643,
            151645,
            151645,
            "qwen2.5-3b-instruct.pt",
            QWEN_MODEL_DIR_3B,
            QWEN_SHARDS_DIR_3B
        };
    }();
    return cfg;
}

inline std::string model_variant() {
    return std::string(QWEN_MODEL_VARIANT);
}

inline std::string default_weight_filename() {
    return std::string(get_model_config().weight_filename);
}

inline std::string default_model_dir() {
    return std::string(get_model_config().default_model_dir);
}

inline std::string default_shards_dir() {
    return std::string(get_model_config().default_shards_dir);
}

} // namespace qwen
