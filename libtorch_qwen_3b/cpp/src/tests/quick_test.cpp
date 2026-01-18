#include "../include/qwen_model.h"
#include "../include/qwen_env.h"
#include "../include/qwen_model_config.h"
#include <iostream>

const auto& QWEN_CFG = qwen::get_model_config();
const std::string WEIGHT_PATH = qwen::get_weight_path();
const std::string TOKENIZER_SCRIPT = qwen::get_tokenizer_script();
const std::string TOKENIZER_MODEL_DIR = qwen::get_tokenizer_model_dir();
const std::string PYTHON_CMD = qwen::get_python_cmd();

int main() {
    std::cout << "=== 快速验证测试 ===" << std::endl;

    try {
        qwen::ensure_required_paths(WEIGHT_PATH, TOKENIZER_SCRIPT, TOKENIZER_MODEL_DIR);
    } catch (const std::exception& e) {
        std::cerr << "❌ 路径配置错误: " << e.what() << std::endl;
        return 1;
    }
    
    // 初始化模型
    QwenModel model = QwenModel(
        QWEN_CFG.vocab_size, QWEN_CFG.hidden_size, QWEN_CFG.num_layers,
        QWEN_CFG.num_heads, QWEN_CFG.num_kv_heads, QWEN_CFG.intermediate_size,
        QWEN_CFG.max_position_embeddings, QWEN_CFG.rope_theta,
        QWEN_CFG.rms_norm_eps, QWEN_CFG.bos_token_id, QWEN_CFG.eos_token_id
    );
    model->eval();
    
    std::cout << "正在加载权重..." << std::endl;
    model->load_weights(WEIGHT_PATH);
    
    torch::Device device = torch::cuda::is_available() ? torch::Device(torch::kCUDA, 0) : torch::kCPU;
    model->to(device, torch::kBFloat16);
    std::cout << "✅ 模型就绪 (" << device << ")" << std::endl;
    
    // 测试ChatML格式：编码"你好"
    std::cout << "\n=== 测试1：直接输入'你好' ===" << std::endl;
    torch::Tensor input_ids = torch::tensor({{108386}}, torch::kInt64).to(device);
    
    std::cout << "输入: 你好 (token 108386)" << std::endl;
    std::cout << "开始生成..." << std::endl;
    
    model->clear_cache();
    torch::NoGradGuard no_grad;
    
    // 生成20个token
    std::vector<int64_t> tokens = {108386};
    for (int i = 0; i < 20; ++i) {
        torch::Tensor current_input = torch::tensor({{tokens.back()}}, torch::kInt64).to(device);
        torch::Tensor logits = model->forward(current_input, true);
        int64_t next_token = logits[0][0].argmax(-1).item<int64_t>();
        
        tokens.push_back(next_token);
        std::cout << next_token << " ";
        
        if (next_token == QWEN_CFG.eos_token_id) {
            std::cout << "[EOS]";
            break;
        }
    }
    
    std::cout << "\n\n生成的token IDs: ";
    for (auto t : tokens) std::cout << t << " ";
    std::cout << std::endl;
    
    // 用Python解码
    std::string ids_str = "[";
    for (size_t i = 0; i < tokens.size(); ++i) {
        ids_str += std::to_string(tokens[i]);
        if (i < tokens.size() - 1) ids_str += ",";
    }
    ids_str += "]";
    
    std::string cmd = PYTHON_CMD + " -c \"from transformers import AutoTokenizer; tok = AutoTokenizer.from_pretrained('" + TOKENIZER_MODEL_DIR + "', trust_remote_code=True); print('解码结果:', tok.decode(" + ids_str + "))\" 2>/dev/null";
    system(cmd.c_str());
    
    // 测试2：ChatML格式
    std::cout << "\n\n=== 测试2：ChatML格式 ===" << std::endl;
    std::cout << "使用Python编码'你好'的ChatML格式..." << std::endl;
    
    std::string encode_cmd = PYTHON_CMD + " " + TOKENIZER_SCRIPT + " \"你好\" --chat 2>/dev/null";
    FILE* pipe = popen(encode_cmd.c_str(), "r");
    char buffer[8192];
    std::string result;
    while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
        result += buffer;
    }
    pclose(pipe);
    
    // 解析token IDs
    std::vector<int64_t> chat_tokens;
    size_t start_pos = result.find("[");
    size_t end_pos = result.find("]");
    if (start_pos != std::string::npos && end_pos != std::string::npos) {
        std::string ids_str_inner = result.substr(start_pos + 1, end_pos - start_pos - 1);
        std::stringstream ss(ids_str_inner);
        std::string item;
        while (std::getline(ss, item, ',')) {
            chat_tokens.push_back(std::stoll(item));
        }
    }
    
    std::cout << "ChatML token count: " << chat_tokens.size() << std::endl;
    std::cout << "最后5个tokens: ";
    for (size_t i = std::max(0, (int)chat_tokens.size() - 5); i < chat_tokens.size(); ++i) {
        std::cout << chat_tokens[i] << " ";
    }
    std::cout << std::endl;
    
    // 生成
    torch::Tensor chat_input_ids = torch::from_blob(
        chat_tokens.data(),
        {1, static_cast<long>(chat_tokens.size())},
        torch::kInt64
    ).clone().to(device);
    
    model->clear_cache();
    std::cout << "从ChatML格式生成20个tokens:" << std::endl;
    
    // Prefill整个ChatML序列
    torch::Tensor logits_prefill = model->forward(chat_input_ids, true);
    int64_t first_token = logits_prefill[0][-1].argmax(-1).item<int64_t>();
    
    std::vector<int64_t> chat_generated;
    chat_generated.push_back(first_token);
    std::cout << first_token << " ";
    
    // 继续生成
    for (int i = 1; i < 20; ++i) {
        if (first_token == QWEN_CFG.eos_token_id || first_token == QWEN_CFG.im_end_id) {
            std::cout << "[STOP]";
            break;
        }
        
        torch::Tensor current = torch::tensor({{chat_generated.back()}}, torch::kInt64).to(device);
        torch::Tensor logits_out = model->forward(current, true);
        int64_t next = logits_out[0][0].argmax(-1).item<int64_t>();
        
        chat_generated.push_back(next);
        std::cout << next << " ";
        
        first_token = next;  // 更新用于检查
    }
    std::cout << std::endl;
    
    // 解码完整输出
    std::string chat_ids_str = "[";
    for (size_t i = 0; i < chat_generated.size(); ++i) {
        chat_ids_str += std::to_string(chat_generated[i]);
        if (i < chat_generated.size() - 1) chat_ids_str += ",";
    }
    chat_ids_str += "]";
    
    std::string decode_cmd = PYTHON_CMD + " -c \"from transformers import AutoTokenizer; tok = AutoTokenizer.from_pretrained('" + TOKENIZER_MODEL_DIR + "', trust_remote_code=True); print('ChatML生成:', tok.decode(" + chat_ids_str + "))\" 2>/dev/null";
    system(decode_cmd.c_str());
    
    return 0;
}
