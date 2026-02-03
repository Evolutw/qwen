#pragma once
#include <cstdlib>
#include <filesystem>
#include <stdexcept>
#include <string>
#include "qwen_model_config.h"

namespace qwen {

inline std::string get_env(const char* name) {
    const char* value = std::getenv(name);
    return value ? std::string(value) : std::string();
}

inline std::string project_root() {
#ifdef QWEN_PROJECT_ROOT
    return std::string(QWEN_PROJECT_ROOT);
#else
    return std::string();
#endif
}

inline std::string model_dir_default() {
#ifdef QWEN_MODEL_DIR
    return std::string(QWEN_MODEL_DIR);
#else
    return std::string();
#endif
}

inline std::string tokenizer_script_default() {
    auto root = project_root();
    if (!root.empty()) {
        std::string primary = root + "/scripts/qwen_tokenize.py";
        if (std::filesystem::exists(primary)) {
            return primary;
        }
        std::string fallback = root + "/../scripts/qwen_tokenize.py";
        if (std::filesystem::exists(fallback)) {
            return fallback;
        }
    }
    return std::string();
}

inline std::string get_model_dir() {
    auto env = get_env("QWEN_MODEL_DIR");
    if (!env.empty()) {
        return env;
    }
    auto compiled = model_dir_default();
    if (!compiled.empty()) {
        return compiled;
    }
    return default_model_dir();
}

inline std::string get_tokenizer_script() {
    auto env = get_env("QWEN_TOKENIZER_SCRIPT");
    if (!env.empty()) {
        return env;
    }
    return tokenizer_script_default();
}

inline std::string get_python_cmd() {
    auto env = get_env("QWEN_PYTHON");
    if (!env.empty()) {
        return env;
    }
    return "python";
}

inline std::string get_weight_path() {
    auto env = get_env("QWEN_WEIGHT_PATH");
    if (!env.empty()) {
        return env;
    }
    auto dir = get_model_dir();
    if (!dir.empty()) {
        return dir + "/" + default_weight_filename();
    }
    return std::string();
}

inline std::string get_tokenizer_model_dir() {
    auto env = get_env("QWEN_TOKENIZER_MODEL_DIR");
    if (!env.empty()) {
        return env;
    }
    return get_model_dir();
}

inline std::string get_weight_shards_dir() {
    auto env = get_env("QWEN_WEIGHT_SHARDS_DIR");
    if (!env.empty()) {
        return env;
    }
    return default_shards_dir();
}

inline void ensure_required_paths(const std::string& weight_path,
                                  const std::string& tokenizer_script,
                                  const std::string& tokenizer_model_dir) {
    if (weight_path.empty()) {
        throw std::runtime_error("Missing QWEN_WEIGHT_PATH (or QWEN_MODEL_DIR). Set env var before running.");
    }
    if (tokenizer_script.empty()) {
        throw std::runtime_error("Missing QWEN_TOKENIZER_SCRIPT. Set env var before running.");
    }
    if (tokenizer_model_dir.empty()) {
        throw std::runtime_error("Missing QWEN_TOKENIZER_MODEL_DIR (or QWEN_MODEL_DIR). Set env var before running.");
    }
    if (!std::filesystem::exists(tokenizer_script)) {
        throw std::runtime_error("Tokenizer script not found: " + tokenizer_script);
    }
    if (!std::filesystem::exists(tokenizer_model_dir)) {
        throw std::runtime_error("Tokenizer model dir not found: " + tokenizer_model_dir);
    }
}

} // namespace qwen
