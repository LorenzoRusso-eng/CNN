#pragma once

#include "core/cuda_backend.hpp"

void build_cost_from_next_layer_all_batch(
    cuda_backend::CudaBatchLayerRuntime &current_runtime,
    const Layer &next,
    cuda_backend::CudaBatchLayerRuntime &next_runtime,
    const cuda_backend::CudaParameterBuffer &cuda_params,
    int next_layer_index,
    const cuda_backend::CudaParameterBuffer *velocity,
    float momentum,
    cuda_backend::CudaBatchTensor<float> &out_cost
);
