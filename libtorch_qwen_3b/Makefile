# Qwen LibTorch C++ 部署 - 便捷构建脚本

# 默认构建目录
BUILD_DIR = cpp/build
BIN_DIR = $(BUILD_DIR)/bin

# 默认目标
.PHONY: all
all: build

# 初始化构建目录
.PHONY: init
init:
	@echo "🔧 初始化构建环境..."
	@mkdir -p $(BUILD_DIR)
	@cd $(BUILD_DIR) && cmake ..
	@echo "✅ 构建环境初始化完成"

# 清理构建
.PHONY: clean
clean:
	@echo "🧹 清理构建文件..."
	@rm -rf $(BUILD_DIR)/*
	@echo "✅ 清理完成"

# 构建所有目标
.PHONY: build
build: init
	@echo "🔨 开始编译..."
	@cd $(BUILD_DIR) && make -j$(shell nproc)
	@echo "✅ 编译完成"

# 构建特定目标
.PHONY: chat
chat: init
	@echo "🔨 编译聊天测试..."
	@cd $(BUILD_DIR) && make test_qwen_chat
	@echo "✅ test_qwen_chat 编译完成"

.PHONY: quick
quick: init
	@echo "🔨 编译快速测试..."
	@cd $(BUILD_DIR) && make quick_chat_test
	@echo "✅ quick_chat_test 编译完成"

.PHONY: tests
tests: init
	@echo "🔨 编译所有测试..."
	@cd $(BUILD_DIR) && make test_embedding test_attention test_transformer test_qwen_model
	@echo "✅ 测试程序编译完成"

.PHONY: diagnostics
diagnostics: init
	@echo "🔨 编译诊断工具..."
	@cd $(BUILD_DIR) && make diagnose_model diagnose_chat diagnose_hidden_states
	@echo "✅ 诊断工具编译完成"

# 运行测试
.PHONY: run-chat
run-chat: chat
	@echo "🚀 运行聊天测试..."
	@$(BIN_DIR)/test_qwen_chat

.PHONY: run-quick
run-quick: quick
	@echo "🚀 运行快速聊天..."
	@$(BIN_DIR)/quick_chat_test

# 帮助信息
.PHONY: help
help:
	@echo "Qwen LibTorch C++ 部署 - 构建脚本"
	@echo ""
	@echo "可用命令:"
	@echo "  make init          - 初始化构建环境"
	@echo "  make build         - 编译所有目标"
	@echo "  make clean         - 清理构建文件"
	@echo "  make chat          - 编译聊天测试"
	@echo "  make quick         - 编译快速聊天测试"
	@echo "  make tests         - 编译所有单元测试"
	@echo "  make diagnostics   - 编译诊断工具"
	@echo "  make run-chat      - 编译并运行聊天测试"
	@echo "  make run-quick     - 编译并运行快速聊天"
	@echo "  make help          - 显示此帮助信息"
