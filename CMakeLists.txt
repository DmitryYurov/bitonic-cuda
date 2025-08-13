cmake_minimum_required(VERSION 4.0)
project(bitonic LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 23)
set(CMAKE_CXX_STANDARD 23)

add_executable(bitonic main.cu)

set_target_properties(bitonic PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
