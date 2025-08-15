cmake_minimum_required(VERSION 3.26)
project(bitonic LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 23)
set(CMAKE_CXX_STANDARD 23)

add_executable(bitonic src/bitonic.cu src/main.cu)
target_include_directories(bitonic PRIVATE ${CMAKE_CURRENT_LIST_DIR}/includes)

set_target_properties(bitonic PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
