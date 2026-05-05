#pragma once

#include "core/cuda_backend.hpp"

void optimizer_step(
    LayerList &architecture,
    int num_layers,
    const cuda_backend::CudaParameterBuffer &gradients,
    cuda_backend::CudaParameterBuffer &velocity,
    cuda_backend::CudaParameterBuffer &cuda_params,
    float learning_rate,
    float momentum
);
