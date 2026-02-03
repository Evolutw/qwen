cmake_minimum_required(VERSION 3.18)
project(qwen_libtorch_deploy)

# 设置C++标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 默认使用Debug构建（可通过 -DCMAKE_BUILD_TYPE=Release 覆盖）
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type" FORCE)
endif()

# 指定LibTorch路径（可通过 -DCMAKE_PREFIX_PATH 覆盖）
if(NOT CMAKE_PREFIX_PATH)
    set(CMAKE_PREFIX_PATH "/home/wx/miniconda3/envs/llm/lib/python3.10/site-packages/torch/share/cmake" CACHE STRING "LibTorch CMake prefix" FORCE)
endif()

# 供代码读取的项目与模型路径
set(QWEN_MODEL_DIR "" CACHE PATH "Path to Qwen model directory (contains qwen2.5-0.5b-instruct.pt)")

# 传递项目根目录与模型目录到编译宏
add_compile_definitions(QWEN_PROJECT_ROOT="${CMAKE_SOURCE_DIR}")
if(QWEN_MODEL_DIR)
    add_compile_definitions(QWEN_MODEL_DIR="${QWEN_MODEL_DIR}")
endif()

# 运行时库路径（优先使用Conda环境）
if(DEFINED ENV{CONDA_PREFIX})
    list(APPEND CMAKE_BUILD_RPATH "$ENV{CONDA_PREFIX}/lib")
    set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
endif()

# 查找LibTorch
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# Require CUDA-enabled LibTorch; abort if the CPU-only package is picked up
if(NOT TARGET torch_cuda)
    message(FATAL_ERROR "LibTorch without CUDA detected. Install the CUDA build (e.g., conda install pytorch pytorch-cuda=12.1 -c pytorch -c nvidia) and set CMAKE_PREFIX_PATH accordingly.")
endif()

# 包含头文件目录
include_directories(include)

# ============================================================
# 示例程序 (Examples)
# ============================================================
add_executable(quick_chat_test src/examples/quick_chat_test.cpp)
target_link_libraries(quick_chat_test "${TORCH_LIBRARIES}")

# 兼容旧的构建目标名（避免CMake Tools仍构建 test_qwen_chat）
add_custom_target(test_qwen_chat DEPENDS quick_chat_test)

# ============================================================
# 输出目录设置
# ============================================================
set_target_properties(
    quick_chat_test
    PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# Windows需复制DLL，Linux无需
if (MSVC)
    file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
    add_custom_command(TARGET quick_chat_test
        POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${TORCH_DLLS}
        $<TARGET_FILE_DIR:quick_chat_test>)
endif (MSVC)
