cmake_minimum_required(VERSION 3.18)
project(joins LANGUAGES CXX CUDA)

# Optionally, set the C++ standard
set(CMAKE_CXX_STANDARD 11)
#add_compile_options(-G)

# Set CUDA architecture for all targets
set(CMAKE_CUDA_ARCHITECTURES "80")
#set(JPEG_INCLUDE_DIR /share/apps/rc/software/libjpeg-turbo/2.1.4-GCCcore-12.2.0/include)
#set(JPEG_LIBRARY_DIR /share/apps/rc/software/libjpeg-turbo/2.1.4-GCCcore-12.2.0/lib)

#include_directories(${JPEG_INCLUDE_DIR})
add_executable(cpu_join src/cpu_join.cpp
        src/sort_ints.cu
        src/cpu_hash_table.cpp
)
add_executable(hashtable src/cpu_hash_table.cpp)
#add_executable(gpu_join src/gpu_join.cu)
#add_subdirectory(gpu/lab1)
#add_subdirectory(gpu/text_snippets)add_executable(task3 task3.cu)
add_subdirectory(src/GPU)
#add_executable()