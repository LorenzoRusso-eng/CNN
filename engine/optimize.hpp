#pragma once

#include "core/cuda_backend.hpp"

void optimizer_step(
    const LayerList &architecture,
    const cuda_backend::CudaParameterBuffer &gradients,
    cuda_backend::CudaParameterBuffer &velocity,
    cuda_backend::CudaParameterBuffer &cuda_params,
    float learning_rate,
    float momentum
);

void accumulate_scaled_gradients(
    const LayerList &architecture,
    cuda_backend::CudaParameterBuffer &accumulated,
    const cuda_backend::CudaParameterBuffer &chunk,
    float scale
);
