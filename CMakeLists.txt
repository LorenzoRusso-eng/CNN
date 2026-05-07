cmake_minimum_required(VERSION 3.20)

project(NN_multithread_time LANGUAGES CXX CUDA)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

option(NN_ENABLE_NATIVE_ARCH "Enable -march=native when supported" ON)
option(NN_ENABLE_LTO "Enable link-time optimization when supported" ON)
option(NN_ENABLE_WARNINGS "Enable useful compiler warnings" ON)
option(NN_ENABLE_MSVC_AVX2 "Enable /arch:AVX2 with MSVC" ON)
set(NN_BINARY_NAME "NN_op" CACHE STRING "Nome del binario compilato")

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND NN_ENABLE_LTO)
    get_filename_component(NN_COMPILER_DIR "${CMAKE_CXX_COMPILER}" DIRECTORY)
    find_program(NN_GCC_AR NAMES gcc-ar PATHS "${NN_COMPILER_DIR}" NO_DEFAULT_PATH)
    find_program(NN_GCC_RANLIB NAMES gcc-ranlib PATHS "${NN_COMPILER_DIR}" NO_DEFAULT_PATH)

    if(NN_GCC_AR)
        set(CMAKE_AR "${NN_GCC_AR}" CACHE FILEPATH "GNU archiver with LTO plugin support" FORCE)
    endif()
    if(NN_GCC_RANLIB)
        set(CMAKE_RANLIB "${NN_GCC_RANLIB}" CACHE FILEPATH "GNU ranlib with LTO plugin support" FORCE)
    endif()
endif()

add_executable(NN_multithread_batch_optimized
    app/main.cpp
    cli/architecture_builder.cpp
    cli/cli_utils.cpp
    cli/training_prompts.cpp
    core/cuda_backend.cpp
    math/loss_cuda.cu
    engine/backward.cu
    engine/forward.cu
    engine/gradients.cu
    engine/optimize.cu
    kernels/general.cu
    kernels/im2col.cu
    kernels/pooling.cu
    kernels/softmax.cu
    evaluation/evaluation.cpp
    evaluation/evaluation_metrics.cpp
    training/parameter_buffer.cpp
    training/shuffle_rng.cpp
    training/training.cpp
    training/training_metadata.cpp
    training/training_method.cpp
    training/training_progress.cpp
    training/training_step.cpp
    IO/activation_codec.cpp
    IO/dataset_io.cpp
    IO/image_io.cpp
    IO/layer_text_codec.cpp
    IO/model_io.cpp
    IO/report_io.cpp
    shared/network_globals.cpp
)

set_target_properties(NN_multithread_batch_optimized PROPERTIES OUTPUT_NAME "${NN_BINARY_NAME}")

target_include_directories(NN_multithread_batch_optimized PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    if(NN_ENABLE_WARNINGS)
        target_compile_options(NN_multithread_batch_optimized PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic>)
    endif()

    if(NN_ENABLE_NATIVE_ARCH)
        target_compile_options(NN_multithread_batch_optimized PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-march=native>)
    endif()

    if(NN_ENABLE_LTO)
        target_compile_options(NN_multithread_batch_optimized PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-flto=auto>)
        target_link_options(NN_multithread_batch_optimized PRIVATE -flto=auto)
    endif()
endif()

if(MSVC)
    target_compile_options(NN_multithread_batch_optimized PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:preprocessor>)

    if(NN_ENABLE_WARNINGS)
        target_compile_options(NN_multithread_batch_optimized PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/W4 /permissive->)
    endif()

    if(NN_ENABLE_MSVC_AVX2)
        target_compile_options(NN_multithread_batch_optimized PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/arch:AVX2>)
    endif()

    # Ensure MSVC uses the standard-conforming preprocessor so CUDA's CCCL headers compile
    target_compile_options(NN_multithread_batch_optimized PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/Zc:preprocessor>)
    # If host preprocessor flag cannot be propagated to nvcc, allow suppressing CCCL warning
    target_compile_definitions(NN_multithread_batch_optimized PRIVATE CCCL_IGNORE_MSVC_TRADITIONAL_PREPROCESSOR_WARNING)

    if(NN_ENABLE_LTO)
        target_compile_options(NN_multithread_batch_optimized PRIVATE $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Release>>:/GL>)
        target_link_options(NN_multithread_batch_optimized PRIVATE $<$<CONFIG:Release>:/LTCG>)
    endif()
endif()

find_package(CUDAToolkit REQUIRED)
target_link_libraries(NN_multithread_batch_optimized PRIVATE CUDA::cudart CUDA::cublas)
