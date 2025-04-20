cmake_minimum_required(VERSION 3.18)
set(CMAKE_CUDA_ARCHITECTURES 75)

project(acotsp LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED) #cuda

add_executable(acotsp src/main.cu)

target_include_directories(acotsp PRIVATE ${CUDAToolkit_INCLUDE_DIRS})