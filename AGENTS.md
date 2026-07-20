# Repository Guidelines

## Project Structure & Module Organization

This is a C++20/CUDA image-classification CNN project built with CMake. The CLI entry point is in `app/main.cpp`.
Interactive prompts and architecture setup live in `cli/`; layer definitions and CUDA backend glue are in `core/`;
CUDA forward, backward, gradient, and optimization code is in `engine/`;
reusable kernels are in `kernels/`;
activation, loss, and decay helpers are in `math/`.
Training loops are under `training/`,
evaluation metrics under `evaluation/`,
and dataset/model/report I/O under `IO/`.
Shared global activation/loss/decay objects live in `shared/`.
Third-party single-header image libraries are vendored as `stb_image.h` and `stb_image_write.h`.

## Build, Test, and Development Commands

Use a Developer Command Prompt on Windows with MSVC, CUDA Toolkit, and cuBLAS available.

```bat
run_vs_and_cmake.bat
run_vs_and_build.bat
```

These helper scripts configure and build the Release executable in `build_cuda_verify/`.

```bat
cmake -S . -B build_cuda_verify -DNN_ENABLE_LTO=OFF
cmake --build build_cuda_verify --config Release --parallel
build_cuda_verify\Release\NN_op.exe
```

Use the explicit CMake commands when adjusting options or debugging configuration. `NN_ENABLE_LTO=OFF` is useful for faster local verification with MSVC/CUDA.

## Coding Style & Naming Conventions

Keep C++ and CUDA code in C++20 style. Use four-space indentation, braces consistent with the surrounding file, and descriptive `snake_case` names for functions, files, and variables. Keep headers paired with matching `.cpp` or `.cu` files where practical, for example `training/training.hpp` and `training/training.cpp`. Prefer small module-local helpers over cross-directory coupling. Do not commit generated build directories, executables, logs, reports, snapshots, or CMake cache files.

## Testing Guidelines

No automated test suite is currently committed. Validate changes by building successfully and running focused manual CLI scenarios: training with a small class-structured image dataset, inference from a saved snapshot, and any affected evaluation path such as hold-out or k-fold. For CUDA debugging, define `NN_CUDA_SYNC_DEBUG` to synchronize after kernels and surface launch failures earlier.

## Commit & Pull Request Guidelines

Recent history uses short, imperative or descriptive messages such as `cli bug fix` and `removed CPU-legacy activation and loss functions`. Keep commits focused and mention the affected subsystem when helpful, for example `training: fix validation split edge case`. Pull requests should describe the change, list the build/manual verification performed, note CUDA or dataset assumptions, and include sample output or generated report paths when behavior changes.

## Security & Configuration Tips

Keep datasets, trained snapshots, and performance reports out of version control unless intentionally adding a small fixture. Avoid hard-coded absolute paths in source; accept paths from the CLI so Windows and Unix-like builds remain usable.
