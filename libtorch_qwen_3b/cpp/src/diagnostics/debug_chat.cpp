#include "../include/qwen_model.h"
#include "../include/qwen_env.h"
#include "../include/qwen_model_config.h"
#include <iostream>
#include <iomanip>

const auto& QWEN_CFG = qwen::get_model_config();
const int64_t QWEN_IM_END_ID = QWEN_CFG.im_end_id;
const std::string WEIGHT_PATH = qwen::get_weight_path();
const std::string TOKENIZER_SCRIPT = qwen::get_tokenizer_script();
const std::string TOKENIZER_MODEL_DIR = qwen::get_tokenizer_model_dir();
const std::string PYTHON_CMD = qwen::get_python_cmd();

std::vector<int64_t> encode_chat(const std::string& user_message) {
    std::string cmd = PYTHON_CMD + " " + TOKENIZER_SCRIPT + " \"" + user_message + "\" --chat 2>/dev/null";
    
    FILE* pipe = popen(cmd.c_str(), "r");
    if (!pipe) {
        throw std::runtime_error("无法执行分词命令");
    }
    
    char buffer[4096];
    std::string result;
    while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
        result += buffer;
    }
    pclose(pipe);
    
    std::vector<int64_t> token_ids;
    size_t start = result.find("[");
    size_t end = result.find("]");
    if (start != std::string::npos && end != std::string::npos) {
        std::string ids_str = result.substr(start + 1, end - start - 1);
        std::stringstream ss(ids_str);
        std::string item;
        while (std::getline(ss, item, ',')) {
            token_ids.push_back(std::stoll(item));
        }
    }
    
    return token_ids;
}

int main() {
    std::cout << "=== ChatML格式生成调试 ===" << std::endl;

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
    std::cout << "✅ 模型就绪\n" << std::endl;
    
    // 编码"你好"
    std::vector<int64_t> input_tokens = encode_chat("你好");
    std::cout << "Input tokens count: " << input_tokens.size() << std::endl;
    std::cout << "最后5个input tokens: ";
    for (size_t i = std::max(0, (int)input_tokens.size() - 5); i < input_tokens.size(); ++i) {
        std::cout << input_tokens[i] << " ";
    }
    std::cout << "\n" << std::endl;
    
    torch::Tensor input_ids = torch::from_blob(
        input_tokens.data(),
        {1, static_cast<long>(input_tokens.size())},
        torch::kInt64
    ).clone().to(device);
    
    model->clear_cache();
    torch::NoGradGuard no_grad;
    
    // 前向传播
    torch::Tensor logits = model->forward(input_ids, true);
    torch::Tensor last_logits = logits[0][-1].to(torch::kFloat32);
    
    // 检查im_end token的logit
    float im_end_logit = last_logits[QWEN_IM_END_ID].item<float>();
    std::cout << "im_end token (151645) logit: " << im_end_logit << std::endl;
    
    // Top 10候选
    auto topk_result = torch::topk(last_logits, 10);
    auto top_logits = std::get<0>(topk_result);
    auto top_indices = std::get<1>(topk_result);
    
    std::cout << "\nTop 10候选tokens:" << std::endl;
    for (int i = 0; i < 10; ++i) {
        int64_t idx = top_indices[i].item<int64_t>();
        float logit = top_logits[i].item<float>();
        std::cout << "  " << (i+1) << ". Token " << idx << " (logit=" << std::fixed << std::setprecision(4) << logit << ")";
        if (idx == QWEN_IM_END_ID) {
            std::cout << " [IM_END!]";
        }
        std::cout << std::endl;
    }
    
    // 使用softmax查看概率
    torch::Tensor probs = torch::softmax(last_logits, -1);
    float im_end_prob = probs[QWEN_IM_END_ID].item<float>();
    std::cout << "\nim_end概率: " << im_end_prob << std::endl;
    
    // Temperature=0.7采样
    std::cout << "\n使用temperature=0.7采样:" << std::endl;
    torch::Tensor scaled_logits = last_logits / 0.7;
    auto topk_result2 = torch::topk(scaled_logits, 50);
    auto topk_values = std::get<0>(topk_result2);
    auto topk_indices = std::get<1>(topk_result2);
    torch::Tensor topk_probs = torch::softmax(topk_values, -1);
    
    std::cout << "Top-5采样候选:" << std::endl;
    for (int i = 0; i < 5; ++i) {
        int64_t idx = topk_indices[i].item<int64_t>();
        float prob = topk_probs[i].item<float>();
        std::cout << "  Token " << idx << " prob=" << prob;
        if (idx == QWEN_IM_END_ID) {
            std::cout << " [IM_END!]";
        }
        std::cout << std::endl;
    }
    
    return 0;
}
