#pragma once
#include <string>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <optional>

// 手动切换模型："0.5b" / "3b" / "7b"
#define QWEN_MODEL_VARIANT "3b"

// 手动设置默认模型目录（若未设置环境变量，将使用这些路径）
#define QWEN_MODEL_DIR_05B "/home/wx/llm_learning/models/qwen2.5-0.5b-instruct"
#define QWEN_MODEL_DIR_3B  "/home/wx/llm_learning/models/qwen2.5-3b-instruct"
#define QWEN_MODEL_DIR_7B  "/home/wx/llm_learning/models/qwen2.5-7b-instruct"
#define QWEN_SHARDS_DIR_05B "/home/wx/llm_learning/models/qwen2.5-0.5b-instruct/weights_shards"
#define QWEN_SHARDS_DIR_3B  "/home/wx/llm_learning/models/qwen2.5-3b-instruct/weights_shards"
#define QWEN_SHARDS_DIR_7B  "/home/wx/llm_learning/models/qwen2.5-7b-instruct/weights_shards"

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
        auto read_file = [](const std::string& path) -> std::string {
            std::ifstream ifs(path);
            if (!ifs) {
                return {};
            }
            return std::string((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
        };

        auto trim = [](std::string s) -> std::string {
            const char* ws = " \t\r\n";
            size_t b = s.find_first_not_of(ws);
            size_t e = s.find_last_not_of(ws);
            if (b == std::string::npos || e == std::string::npos) {
                return {};
            }
            return s.substr(b, e - b + 1);
        };

        auto extract_value = [&](const std::string& text, const std::string& key) -> std::optional<std::string> {
            std::string needle = "\"" + key + "\"";
            size_t pos = text.find(needle);
            if (pos == std::string::npos) return std::nullopt;
            pos = text.find(':', pos);
            if (pos == std::string::npos) return std::nullopt;
            pos++;
            while (pos < text.size() && (text[pos] == ' ' || text[pos] == '\t')) pos++;
            size_t end = pos;
            while (end < text.size() && text[end] != ',' && text[end] != '\n' && text[end] != '\r' && text[end] != '}') {
                end++;
            }
            return trim(text.substr(pos, end - pos));
        };

        auto extract_int = [&](const std::string& text, const std::string& key, int64_t& out) {
            auto v = extract_value(text, key);
            if (!v || v->empty()) return;
            out = std::stoll(*v);
        };

        auto extract_double = [&](const std::string& text, const std::string& key, double& out) {
            auto v = extract_value(text, key);
            if (!v || v->empty()) return;
            out = std::stod(*v);
        };

        auto env = [](const char* name) -> std::string {
            const char* v = std::getenv(name);
            return v ? std::string(v) : std::string();
        };

        std::string variant = QWEN_MODEL_VARIANT;
        QwenModelConfig cfg{};

        if (variant == "0.5b" || variant == "0.5" || variant == "0_5b") {
            cfg = QwenModelConfig{
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
        } else if (variant == "7b" || variant == "7" || variant == "7_0b") {
            cfg = QwenModelConfig{
                152064,
                3584,
                28,
                18944,
                28,
                4,
                32768,
                1000000.0,
                1e-6,
                151643,
                151645,
                151645,
                "qwen2.5-7b-instruct.pt",
                QWEN_MODEL_DIR_7B,
                QWEN_SHARDS_DIR_7B
            };
        } else {
            cfg = QwenModelConfig{
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
        }

        std::string config_path = env("QWEN_CONFIG_JSON");
        if (config_path.empty()) {
            std::string model_dir = env("QWEN_MODEL_DIR");
            if (model_dir.empty()) {
                model_dir = cfg.default_model_dir ? std::string(cfg.default_model_dir) : std::string();
            }
            if (!model_dir.empty()) {
                config_path = model_dir + "/config.json";
            }
        }

        if (!config_path.empty()) {
            std::string json = read_file(config_path);
            if (!json.empty()) {
                extract_int(json, "vocab_size", cfg.vocab_size);
                extract_int(json, "hidden_size", cfg.hidden_size);
                extract_int(json, "num_hidden_layers", cfg.num_layers);
                extract_int(json, "intermediate_size", cfg.intermediate_size);
                extract_int(json, "num_attention_heads", cfg.num_heads);
                extract_int(json, "num_key_value_heads", cfg.num_kv_heads);
                extract_int(json, "max_position_embeddings", cfg.max_position_embeddings);
                extract_double(json, "rope_theta", cfg.rope_theta);
                extract_double(json, "rms_norm_eps", cfg.rms_norm_eps);
                extract_int(json, "bos_token_id", cfg.bos_token_id);
                extract_int(json, "eos_token_id", cfg.eos_token_id);
                cfg.im_end_id = cfg.eos_token_id;
            }
        }

        return cfg;
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
