#pragma once

#include "core/layer.hpp"
#include "math/activations.hpp"
#include "math/losses.hpp"
#include "core/cuda_backend.hpp"

#include <vector>


namespace training_batches {

float train_batch_chunk(
    const LayerList &architecture, const cuda_backend::CudaParameterBuffer &cuda_params,
    cuda_backend::CudaBatchRuntimeList &runtime, int num_layers,
    cuda_backend::CudaParameterBuffer &batch_gradients,
    cuda_backend::CudaBatchTensor<float> &target_batch, const LazyDataset &dataset,
    const std::vector<int> &indices, int start, int end,
    const Loss &loss, const Activation &hidden_activation, const Activation &output_activation
);

float train_batch_chunk_nesterov(
    const LayerList &architecture, const cuda_backend::CudaParameterBuffer &cuda_params,
    cuda_backend::CudaBatchRuntimeList &runtime, int num_layers,
    cuda_backend::CudaParameterBuffer &batch_gradients,
    cuda_backend::CudaBatchTensor<float> &target_batch,
    const LazyDataset &dataset, const std::vector<int> &indices, int start, int end,
    const Loss &loss, const Activation &hidden_activation, const Activation &output_activation,
    const cuda_backend::CudaParameterBuffer *velocity, float momentum
);
} // namespace training_batches
