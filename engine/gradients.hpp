#pragma once

#include "core/cuda_backend.hpp"
#include "math/activations.hpp"
#include "math/losses.hpp"


void backprop_batch(
    const LayerList &architecture,
    cuda_backend::CudaBatchRuntimeList &runtime,
    int num_layers,
    cuda_backend::CudaParameterBuffer &gradients,
    const cuda_backend::CudaParameterBuffer &cuda_params,
    const Loss &loss,
    float &loss_value,
    const cuda_backend::CudaBatchTensor<float> &desired_output,
    const Activation &hidden_activation,
    const Activation &output_activation,
    const cuda_backend::CudaParameterBuffer *velocity = nullptr,
    float momentum = 0.0f,
    float learning_rate = 0.0f
);
