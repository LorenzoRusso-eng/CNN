#pragma once

#include "core/cuda_backend.hpp"
#include "math/activations.hpp"

#include <vector>

void feed_input_batch(
    const LazyDataset &dataset,
    const std::vector<int> &indices, int start, int end,
    const Layer &first, cuda_backend::CudaBatchLayerRuntime &first_runtime
);

void feed_input_tensor(
    const Tensor &input,
    const Layer &first,
    cuda_backend::CudaBatchLayerRuntime &first_runtime
);

void fill_target_batch(
    const LazyDataset &dataset,
    const std::vector<int> &indices, int start, int end,
    const Layer &last,
    cuda_backend::CudaBatchTensor<float> &target_batch
);

void forwardprop_batch(
    const LayerList &architecture,
    cuda_backend::CudaBatchRuntimeList &runtime,
    int num_layers,
    const cuda_backend::CudaParameterBuffer &cuda_params,
    const Activation &hidden_activation,
    const Activation &output_activation,
    const cuda_backend::CudaParameterBuffer *velocity = nullptr,
    float momentum = 0.0f
);
