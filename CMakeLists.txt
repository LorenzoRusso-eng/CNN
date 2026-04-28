cmake_minimum_required(VERSION 3.20)

project(NN_multithread_time LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(NN_ENABLE_NATIVE_ARCH "Enable -march=native when supported" ON)
option(NN_ENABLE_LTO "Enable link-time optimization when supported" ON)

add_executable(NN_multithread_batch_optimized
    app/main.cpp
    cli/architecture_builder.cpp
    cli/cli_utils.cpp
    cli/training_prompts.cpp
    engine/backward.cpp
    engine/backward_batch.cpp
    evaluation/evaluation.cpp
    evaluation/evaluation_metrics.cpp
    engine/forward.cpp
    engine/forward_batch.cpp
    engine/im2col.cpp
    training/optimizer.cpp
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

set_target_properties(NN_multithread_batch_optimized PROPERTIES OUTPUT_NAME NN_op)

target_include_directories(NN_multithread_batch_optimized PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    if(NN_ENABLE_NATIVE_ARCH)
        target_compile_options(NN_multithread_batch_optimized PRIVATE -march=native)
    endif()

    if(NN_ENABLE_LTO)
        target_compile_options(NN_multithread_batch_optimized PRIVATE -flto)
        target_link_options(NN_multithread_batch_optimized PRIVATE -flto)
    endif()
endif()

set(NN_OPENBLAS_ROOT "D:/Project/OpenBLAS-install-mt" CACHE PATH "Root dell'installazione OpenBLAS")

find_path(NN_OPENBLAS_INCLUDE_DIR
    NAMES openblas/cblas.h
    PATHS
        "${NN_OPENBLAS_ROOT}"
        "${NN_OPENBLAS_ROOT}/include"
    NO_DEFAULT_PATH
)

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(NN_multithread_batch_optimized PRIVATE OpenMP::OpenMP_CXX)
endif()

find_library(NN_OPENBLAS_LIBRARY
    NAMES openblas libopenblas
    PATHS
        "${NN_OPENBLAS_ROOT}"
        "${NN_OPENBLAS_ROOT}/lib"
        "${NN_OPENBLAS_ROOT}/lib64"
    NO_DEFAULT_PATH
)

if(NOT NN_OPENBLAS_INCLUDE_DIR)
    message(FATAL_ERROR "Header OpenBLAS non trovato in ${NN_OPENBLAS_ROOT}. Atteso: include/openblas/cblas.h")
endif()

if(NOT NN_OPENBLAS_LIBRARY)
    message(FATAL_ERROR "Libreria OpenBLAS non trovata in ${NN_OPENBLAS_ROOT}. Attesa in lib/ oppure lib64/")
endif()

target_include_directories(NN_multithread_batch_optimized PRIVATE "${NN_OPENBLAS_INCLUDE_DIR}")
target_link_libraries(NN_multithread_batch_optimized PRIVATE ${NN_OPENBLAS_LIBRARY})

message(STATUS "OpenBLAS trovato:")
message(STATUS "  include: ${NN_OPENBLAS_INCLUDE_DIR}")
message(STATUS "  library: ${NN_OPENBLAS_LIBRARY}")
