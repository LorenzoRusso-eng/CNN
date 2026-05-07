# CNN CUDA

C++20/CUDA implementation of a configurable convolutional neural network, driven from an interactive CLI and intended for image classification experiments.

The project builds with CMake, uses the CUDA Runtime and cuBLAS, and produces an executable named `NN_op` by default.

## Features

- Interactive network architecture builder.
- Supported layers:
  - Input
  - Convolution
  - Pooling
  - Flatten
  - Dense
  - optional final Softmax
- Pooling modes: Max, Average and L2.
- Training variants:
  - full-batch training
  - mini-batch SGD
  - online SGD
- Evaluation methods:
  - stratified hold-out
  - stratified k-fold cross validation
  - full training
- Momentum and Nesterov Accelerated Gradient.
- Learning-rate decay strategies:
  - constant
  - exponential
  - time-based
  - step
  - cosine annealing
- Activation functions:
  - Identity
  - Sigmoid
  - Tanh
  - ReLU
  - LeakyReLU
  - ELU
  - Softplus
  - Swish
  - Mish
  - GELU
- Loss functions:
  - Simple loss
  - L1
  - L2
  - Smooth L1
  - Huber
  - Cross entropy
  - log-likelihood loss for Softmax output
- Loss reductions: Sum and Mean.
- Model snapshots with architecture, weights, class names and activation metadata.
- Inference on a single image from a saved snapshot.
- Markdown performance reports with training metrics, test metrics, per-class metrics and confusion matrices.

## CUDA Backend

Training and inference run through the CUDA batch backend. Dense and convolutional layers use cuBLAS GEMM calls, while convolution uses an `im2col` layout for the forward pass and `col2im` for backpropagation.

The runtime keeps network parameters in dedicated CUDA buffers and synchronizes them back to CPU memory only when a snapshot must be written or the architecture must be materialized on the host. Recent changes also reduce unnecessary device allocations and copies by:

- storing Dense and Conv parameters in compact parameter buffers;
- copying best weights and optimizer velocity directly between CUDA buffers;
- aliasing compatible runtime tensors for Flatten and Softmax transitions;
- using forward-only CUDA buffers during evaluation and inference.

The CUDA error helpers check kernel launches by default. Defining `NN_CUDA_SYNC_DEBUG` also synchronizes after kernel execution, which is useful for debugging CUDA failures.

## Validation And Early Stopping

Hold-out and k-fold runs keep the final test split separate from training. The training portion is split again into an effective training set and an internal validation set using stratified class-aware sampling.

The validation set is evaluated at the end of each epoch with a lightweight batched forward pass. Validation accuracy controls:

- the best weights kept in memory;
- the optimizer velocity associated with those best weights;
- the counter of epochs without significant improvement;
- early stopping when the patience threshold is reached.

When validation accuracy improves over the best value seen so far, the current CUDA parameters and velocity are copied into `best` buffers. At the end of training, if validation was used, the best parameters and best velocity are restored before final testing or snapshot writing.

Default early-stopping settings:

```text
validation ratio: chosen from the CLI for hold-out and k-fold
patience: 5 epochs
relative min delta: 0.001
validation batch size: 200
```

Full training does not create an internal validation split and does not use early stopping.

## Requirements

- CMake 3.20 or newer.
- A C++20 compiler.
- NVIDIA CUDA Toolkit.
- A CUDA-capable NVIDIA GPU.
- cuBLAS, included with the CUDA Toolkit.

On Windows, Visual Studio Build Tools with MSVC and `nvcc` available from a Developer Command Prompt are recommended.

## Project Layout

```text
app/          CLI application entry point
cli/          Input prompts and interactive architecture builder
core/         Layer definitions, validation helpers and CUDA backend
engine/       CUDA forward pass, backward pass, gradients and optimization
evaluation/   Test evaluation and classification metrics
IO/           Dataset/image loading, model snapshots and reports
kernels/      CUDA kernels for CNN operations
math/         Activations, losses and learning-rate decay
shared/       Global reusable activation/loss/decay objects
training/     Training loops, batches, progress tracking and metadata
```

## Build

### Windows With Visual Studio Build Tools

The repository includes two helper scripts for configuring and building with MSVC/CUDA:

```bat
run_vs_and_cmake.bat
run_vs_and_build.bat
```

The scripts generate the build in `build_cuda_verify/` and compile the `Release` configuration.

From a Developer Command Prompt, the equivalent commands are:

```bat
cmake -S . -B build_cuda_verify -DNN_ENABLE_LTO=OFF
cmake --build build_cuda_verify --config Release --parallel
```

The executable is produced at:

```text
build_cuda_verify\Release\NN_op.exe
```

### Generic CMake Build

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
```

Useful CMake options:

```text
NN_BINARY_NAME         Output binary name, default: NN_op
NN_ENABLE_NATIVE_ARCH Enable native CPU optimizations, default: ON
NN_ENABLE_LTO         Enable link-time optimization, default: ON
NN_ENABLE_WARNINGS    Enable compiler warnings, default: ON
NN_ENABLE_MSVC_AVX2   Enable /arch:AVX2 with MSVC, default: ON
```

For MSVC/CUDA builds, disabling LTO with `-DNN_ENABLE_LTO=OFF` can make local verification simpler.

## Usage

Start the executable:

```bat
build_cuda_verify\Release\NN_op.exe
```

At startup, choose one mode:

```text
1 = Training
2 = Inference
```

### Training

Training mode asks for:

- model name;
- dataset path;
- network architecture;
- optional final Softmax layer;
- hidden and output activations;
- loss function and reduction;
- initial learning rate and decay strategy;
- target loss threshold;
- maximum epoch count;
- training variant: batch, mini-batch SGD or online SGD;
- training window: chunk size for batch training, mini-batch size for SGD, or loss averaging steps for online SGD;
- optional momentum and Nesterov;
- evaluation method: hold-out, k-fold or full training.

When a final Softmax layer is selected, the preceding Dense layer is treated as logits, the output activation is fixed to Identity and the compatible loss is fixed to log-likelihood loss.

### Inference

Inference mode asks for:

- the path to a saved model snapshot;
- the image to classify.

The snapshot provides the architecture, weights, class names and saved activation metadata. The image is loaded and scaled to `[0, 1]` using the input shape stored in the snapshot. The program prints the predicted class index, confidence value and, when available, the class name.

## Dataset Format

The dataset must be organized as one subdirectory per class:

```text
dataset/
  class_1/
    image_001.png
    image_002.jpg
  class_2/
    image_003.png
    image_004.jpg
```

Supported extensions:

```text
.png, .jpg, .jpeg, .bmp, .tga
```

Class names are read from the subdirectory names and sorted alphabetically. The dataset loader stores image paths lazily and loads image tensors only when a batch is fed to the GPU.

All images must have the same height and width. The number of channels is detected from the first image and must be compatible with the network input shape.

## Generated Files

Hold-out training creates:

```text
network_performance_report_<model_name>.md
trained_model_<model_name>_snapshot.txt
```

K-fold cross validation creates:

```text
network_performance_report_<model_name>.md
```

Full training creates:

```text
trained_model_<model_name>_snapshot.txt
```

Snapshots are written atomically through a temporary file and contain the layer configuration, Dense and Conv parameters, class names, hidden activation and output activation. They are the input format used by inference mode.

## Notes

Build directories and generated artifacts are ignored by `.gitignore`. Source files, CMake configuration, helper scripts and this README are the parts intended to be versioned.
