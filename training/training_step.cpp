#include "training/training_step.hpp"

#include "core/cuda_backend.hpp"
#include "engine/forward.hpp"
#include "engine/gradients.hpp"
#include "math/loss_cuda.hpp"

namespace training_batches {

float train_batch_chunk(
    const LayerList &architecture, const cuda_backend::CudaParameterBuffer &cuda_params,
    cuda_backend::CudaBatchRuntimeList &runtime, int num_layers,
    cuda_backend::CudaParameterBuffer &batch_gradients,
    cuda_backend::CudaBatchTensor<float> &target_batch, const LazyDataset &dataset,
    const std::vector<int> &indices, int start, int end,
    const Loss &loss, const Activation &hidden_activation, const Activation &output_activation
){
    const int batch_size = end - start;
    cuda_backend::init_cuda_batch_runtime_buffers(architecture, runtime, batch_size);
    cuda_backend::zero_cuda_parameter_buffer(architecture, batch_gradients);
    feed_input_batch(dataset, indices, start, end, architecture[0], runtime[0]);
    fill_target_batch(dataset, indices, start, end, architecture[num_layers - 1], target_batch);
    forwardprop_batch(architecture, runtime, num_layers, cuda_params, hidden_activation, output_activation);

    const float loss_value = Loss_fun(runtime[num_layers - 1], loss, target_batch);
    backprop_batch(architecture, runtime, num_layers, batch_gradients, cuda_params, loss, target_batch, hidden_activation, output_activation);
    return loss_value;
}

float train_batch_chunk_nesterov(
    const LayerList &architecture, const cuda_backend::CudaParameterBuffer &cuda_params,
    cuda_backend::CudaBatchRuntimeList &runtime, int num_layers,
    cuda_backend::CudaParameterBuffer &batch_gradients,
    cuda_backend::CudaBatchTensor<float> &target_batch,
    const LazyDataset &dataset, const std::vector<int> &indices, int start, int end,
    const Loss &loss, const Activation &hidden_activation, const Activation &output_activation,
    const cuda_backend::CudaParameterBuffer *velocity, float momentum
){

    const int batch_size = end - start;
    cuda_backend::init_cuda_batch_runtime_buffers(architecture, runtime, batch_size);
    cuda_backend::zero_cuda_parameter_buffer(architecture, batch_gradients);
    feed_input_batch(dataset, indices, start, end, architecture[0], runtime[0]);
    fill_target_batch(dataset, indices, start, end, architecture[num_layers - 1], target_batch);
    forwardprop_batch(architecture, runtime, num_layers, cuda_params, hidden_activation, output_activation);

    float architecture_loss_value = Loss_fun(runtime[num_layers - 1], loss, target_batch);
    forwardprop_batch(architecture, runtime, num_layers, cuda_params, hidden_activation, output_activation, velocity, momentum);
    backprop_batch(architecture, runtime, num_layers, batch_gradients, cuda_params, loss, target_batch, hidden_activation, output_activation, velocity, momentum);

    return architecture_loss_value;
}

} // namespace training_batches
