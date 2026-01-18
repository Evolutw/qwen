#pragma once
#include <torch/torch.h>
#include <string>
#include <vector>
#include <iostream>
#include <sstream>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <array>
#include <regex>
#include "qwen_env.h"

/**
 * Qwen分词器C++包装类
 * 通过调用Python tokenizer脚本将中文文本转换为token IDs
 */
class QwenTokenizer {
private:
    std::string python_script_path;
    std::string python_command;

    // 自动检测可用的Python命令
    std::string detect_python_command() {
        // 优先使用环境变量指定的Python命令
        std::vector<std::string> candidates;
        auto env_python = qwen::get_env("QWEN_PYTHON");
        if (!env_python.empty()) {
            candidates.push_back(env_python);
        }

        // 尝试常见的Python命令
        candidates.insert(candidates.end(), {"python3", "python"});
        
        for (const auto& cmd : candidates) {
            std::string test_cmd = cmd + " --version 2>&1";
            FILE* pipe = popen(test_cmd.c_str(), "r");
            if (pipe) {
                char buffer[128];
                bool has_output = (fgets(buffer, sizeof(buffer), pipe) != nullptr);
                int status = pclose(pipe);
                if (status == 0 && has_output) {
                    return cmd;
                }
            }
        }
        
        throw std::runtime_error("未找到可用的Python命令");
    }

    // 执行shell命令并获取输出
    std::string exec_command(const std::string& cmd) {
        std::array<char, 128> buffer;
        std::string result;
        std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(cmd.c_str(), "r"), pclose);
        
        if (!pipe) {
            throw std::runtime_error("无法执行命令: " + cmd);
        }
        
        while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
            result += buffer.data();
        }
        
        return result;
    }

    // 从JSON输出解析token IDs
    std::vector<int64_t> parse_token_ids(const std::string& json_output) {
        std::vector<int64_t> token_ids;
        
        // 简单的JSON解析（查找 "token_ids": [数字列表]）
        std::regex pattern(R"("token_ids":\s*\[([\d,\s]+)\])");
        std::smatch match;
        
        if (std::regex_search(json_output, match, pattern)) {
            std::string ids_str = match[1].str();
            std::istringstream iss(ids_str);
            std::string token;
            
            while (std::getline(iss, token, ',')) {
                // 移除空格
                token.erase(std::remove_if(token.begin(), token.end(), ::isspace), token.end());
                if (!token.empty()) {
                    token_ids.push_back(std::stoll(token));
                }
            }
        } else {
            throw std::runtime_error("无法解析JSON输出: " + json_output);
        }
        
        return token_ids;
    }

public:
    /**
     * 构造函数
     * @param script_path Python tokenizer脚本路径
     * @param python_cmd Python命令（默认"auto"自动检测）
     */
    QwenTokenizer(const std::string& script_path, const std::string& python_cmd = "auto")
        : python_script_path(script_path) {
        if (python_cmd == "auto") {
            python_command = detect_python_command();
            std::cout << "自动检测到Python命令: " << python_command << std::endl;
        } else {
            python_command = python_cmd;
        }
    }

    /**
     * 将中文文本转换为token IDs
     * @param text 输入的中文文本
     * @return token IDs的Tensor（形状：[seq_len]）
     */
    torch::Tensor encode(const std::string& text) {
        // 构建命令（使用echo避免特殊字符问题）
        std::string escaped_text = text;
        // 转义单引号
        size_t pos = 0;
        while ((pos = escaped_text.find("'", pos)) != std::string::npos) {
            escaped_text.replace(pos, 1, "'\\''");
            pos += 4;
        }
        
        std::string cmd = python_command + " " + python_script_path + " '" + escaped_text + "' 2>&1";
        
        try {
            // 执行Python脚本
            std::string output = exec_command(cmd);
            
            // 解析输出
            std::vector<int64_t> token_ids = parse_token_ids(output);
            
            if (token_ids.empty()) {
                throw std::runtime_error("分词结果为空");
            }
            
            // 转换为Tensor
            return torch::tensor(token_ids, torch::kInt64);
            
        } catch (const std::exception& e) {
            std::cerr << "❌ 分词失败: " << e.what() << std::endl;
            throw;
        }
    }

    /**
     * 批量编码多个文本
     * @param texts 文本列表
     * @param padding 是否padding到相同长度
     * @return token IDs的Tensor（形状：[batch_size, max_seq_len]）
     */
    torch::Tensor encode_batch(const std::vector<std::string>& texts, bool padding = true) {
        std::vector<torch::Tensor> token_tensors;
        int64_t max_len = 0;
        
        // 分别编码每个文本
        for (const auto& text : texts) {
            torch::Tensor tokens = encode(text);
            token_tensors.push_back(tokens);
            max_len = std::max(max_len, tokens.size(0));
        }
        
        // 如果需要padding
        if (padding && texts.size() > 1) {
            std::vector<torch::Tensor> padded_tensors;
            for (auto& tokens : token_tensors) {
                int64_t pad_len = max_len - tokens.size(0);
                if (pad_len > 0) {
                    // Padding with 0 (或使用特定的pad_token_id)
                    torch::Tensor padding_tensor = torch::zeros({pad_len}, torch::kInt64);
                    tokens = torch::cat({tokens, padding_tensor}, 0);
                }
                padded_tensors.push_back(tokens);
            }
            return torch::stack(padded_tensors);
        } else {
            return torch::stack(token_tensors);
        }
    }

    /**
     * 打印分词结果信息
     */
    void print_tokenize_info(const std::string& text, const torch::Tensor& token_ids) {
        std::cout << "\n【分词结果】" << std::endl;
        std::cout << "  原始文本: " << text << std::endl;
        std::cout << "  Token数量: " << token_ids.size(0) << std::endl;
        std::cout << "  Token IDs: " << token_ids << std::endl;
    }
};
