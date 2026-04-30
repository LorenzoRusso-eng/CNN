cmake_minimum_required(VERSION 3.20)

project(NN_multithread_time LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(NN_ENABLE_NATIVE_ARCH "Enable -march=native when supported" ON)
option(NN_ENABLE_LTO "Enable link-time optimization when supported" ON)
option(NN_ENABLE_WARNINGS "Enable useful compiler warnings" ON)
option(NN_ENABLE_MSVC_AVX2 "Enable /arch:AVX2 with MSVC" ON)
set(NN_OPENBLAS_NUM_THREADS "1" CACHE STRING "Numero di thread OpenBLAS da impostare a runtime")
set(NN_BINARY_NAME "NN_op" CACHE STRING "Nome del binario compilato")
set(NN_OPENBLAS_ROOT "D:/Project/OpenBLAS-install-mt" CACHE PATH "Root dell'installazione OpenBLAS")

if(NOT NN_OPENBLAS_NUM_THREADS MATCHES "^[0-9]+$")
    message(FATAL_ERROR "NN_OPENBLAS_NUM_THREADS deve essere un intero non negativo")
endif()

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

set_target_properties(NN_multithread_batch_optimized PROPERTIES OUTPUT_NAME "${NN_BINARY_NAME}")

target_include_directories(NN_multithread_batch_optimized PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_definitions(NN_multithread_batch_optimized PRIVATE NN_OPENBLAS_NUM_THREADS=${NN_OPENBLAS_NUM_THREADS})

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    if(NN_ENABLE_WARNINGS)
        target_compile_options(NN_multithread_batch_optimized PRIVATE -Wall -Wextra -Wpedantic)
    endif()

    if(NN_ENABLE_NATIVE_ARCH)
        target_compile_options(NN_multithread_batch_optimized PRIVATE -march=native)
    endif()

    if(NN_ENABLE_LTO)
        target_compile_options(NN_multithread_batch_optimized PRIVATE -flto=auto)
        target_link_options(NN_multithread_batch_optimized PRIVATE -flto=auto)
    endif()
endif()

if(MSVC)
    if(NN_ENABLE_WARNINGS)
        target_compile_options(NN_multithread_batch_optimized PRIVATE /W4 /permissive-)
    endif()

    if(NN_ENABLE_MSVC_AVX2)
        target_compile_options(NN_multithread_batch_optimized PRIVATE /arch:AVX2)
    endif()

    if(NN_ENABLE_LTO)
        target_compile_options(NN_multithread_batch_optimized PRIVATE $<$<CONFIG:Release>:/GL>)
        target_link_options(NN_multithread_batch_optimized PRIVATE $<$<CONFIG:Release>:/LTCG>)
    endif()
endif()

find_package(OpenMP REQUIRED)
target_link_libraries(NN_multithread_batch_optimized PRIVATE OpenMP::OpenMP_CXX)

find_package(OpenBLAS CONFIG REQUIRED COMPONENTS pthread PATHS "${NN_OPENBLAS_ROOT}" NO_DEFAULT_PATH)

target_include_directories(NN_multithread_batch_optimized PRIVATE "${NN_OPENBLAS_ROOT}/include")
target_link_libraries(NN_multithread_batch_optimized PRIVATE OpenBLAS::OpenBLAS)

message(STATUS "OpenBLAS trovato:")
message(STATUS "  include: ${NN_OPENBLAS_ROOT}/include")
message(STATUS "  target: OpenBLAS::OpenBLAS")
message(STATUS "  runtime threads: ${NN_OPENBLAS_NUM_THREADS}")
