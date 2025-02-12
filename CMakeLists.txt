cmake_minimum_required(VERSION 3.20)
project(LightGBM_Example)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 17)

# 设置 LightGBM 安装目录
set(LIGHTGBM_DIR ${CMAKE_CURRENT_SOURCE_DIR}/LightGBM)

# 添加 LightGBM 的头文件路径
include_directories(${LIGHTGBM_DIR}/include)

# 设置 LightGBM dll 文件路径
set(LIGHTGBM_LIB ${LIGHTGBM_DIR}/bin/lib_lightgbm.dll)

# 设置要编译的源文件
add_executable(LightGBM_Demo lightGBM_demo.cpp)

# 链接 LightGBM 库
target_link_libraries(LightGBM_Demo PRIVATE ${LIGHTGBM_LIB})

# 将 DLL 文件复制到可执行文件的输出目录
add_custom_command(TARGET LightGBM_Demo POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${LIGHTGBM_LIB}
        $<TARGET_FILE_DIR:LightGBM_Demo>)
