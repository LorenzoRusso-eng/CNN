#include "training/training_step.hpp"

#include "core/layer_validation.hpp"
#include "engine/backward.hpp"
#include "engine/forward.hpp"
#include "training/parameter_buffer.hpp"
#include "shared/openmp_utils.hpp"

#include <algorithm>

namespace training_examples {

float train_example(const LayerList &architecture, RuntimeList &runtime, int num_layers, ParameterBuffer &sample_gradients, const Dataset4D &input, const Dataset4D &output, int sample_index, const Loss &loss, const Activation &hidden_activation, const Activation &output_activation, ExecutionPolicy policy){
    float loss_value = 0.0f;
    training_buffers::zero_parameter_buffer(architecture, sample_gradients, policy);
    feed_input(input[sample_index], architecture[0], runtime[0]);
    forwardprop(architecture, runtime, num_layers, hidden_activation, output_activation, policy);
    backprop(architecture, runtime, num_layers, sample_gradients, loss, loss_value, output[sample_index], hidden_activation, output_activation, policy);
    return loss_value;
}

float train_example_nesterov(const LayerList &architecture, RuntimeList &runtime, int num_layers, ParameterBuffer &sample_gradients, const ParameterBuffer &velocity, const Dataset4D &input, const Dataset4D &output, int sample_index, const Loss &loss, const Activation &hidden_activation, const Activation &output_activation, float momentum, ExecutionPolicy policy){
    float architecture_loss_value = 0.0f;
    float lookahead_loss_value = 0.0f;
    training_buffers::zero_parameter_buffer(architecture, sample_gradients, policy);
    feed_input(input[sample_index], architecture[0], runtime[0]);
    forwardprop(architecture, runtime, num_layers, hidden_activation, output_activation, policy);
    architecture_loss_value = loss.fn(architecture[num_layers - 1], runtime[num_layers - 1], output[sample_index]);
    forwardprop(architecture, runtime, num_layers, hidden_activation, output_activation, policy, &velocity, momentum);
    backprop(
        architecture, runtime, num_layers,
        sample_gradients,
        loss, lookahead_loss_value, output[sample_index],
        hidden_activation, output_activation,
        policy, &velocity, momentum
    );
    return architecture_loss_value;
}

} // namespace training_examples

namespace training_batches {

void fill_target_batch(const Dataset4D &output, const std::vector<int> &indices, int start, int end, const Layer &last, BatchTensor &target_batch){
    require_condition(start >= 0 && end >= start && end <= static_cast<int>(indices.size()), "fill_target_batch: range batch non valido");

    const int batch_size = end - start;
    const int flat_size = last.flat_output_size();
    target_batch.resize(batch_size, flat_size, 0.0f);

    NN_OMP_PARALLEL_FOR_IF(batch_size * flat_size > 256)
    for(int batch_index = 0; batch_index < batch_size; batch_index++){
        const Tensor3D &sample = output[indices[start + batch_index]];
        validate_tensor3d_shape(sample, last.dim_layer, "fill_target_batch");
        const std::size_t base = static_cast<std::size_t>(batch_index) * static_cast<std::size_t>(flat_size);
        std::copy(sample.data.begin(), sample.data.end(), target_batch.data.begin() + static_cast<std::ptrdiff_t>(base));
    }
}

float train_batch_chunk(const LayerList &architecture, BatchRuntimeList &runtime, int num_layers, ParameterBuffer &batch_gradients, BatchTensor &target_batch, const Dataset4D &input, const Dataset4D &output, const std::vector<int> &indices, int start, int end, const Loss &loss, const Activation &hidden_activation, const Activation &output_activation, ExecutionPolicy policy){
    const int batch_size = end - start;
    init_batch_runtime_buffers(architecture, runtime, batch_size);
    training_buffers::zero_parameter_buffer(architecture, batch_gradients, policy);
    feed_input_batch(input, indices, start, end, architecture[0], runtime[0]);
    fill_target_batch(output, indices, start, end, architecture[num_layers - 1], target_batch);
    forwardprop_batch(architecture, runtime, num_layers, hidden_activation, output_activation, policy);

    float loss_value = 0.0f;
    backprop_batch(architecture, runtime, num_layers, batch_gradients, loss, loss_value, target_batch, hidden_activation, output_activation, policy);
    return loss_value;
}

float train_batch_chunk_nesterov(const LayerList &architecture, BatchRuntimeList &runtime, int num_layers, ParameterBuffer &batch_gradients, BatchTensor &target_batch, const ParameterBuffer &velocity, const Dataset4D &input, const Dataset4D &output, const std::vector<int> &indices, int start, int end, const Loss &loss, const Activation &hidden_activation, const Activation &output_activation, float momentum, ExecutionPolicy policy){
    const int batch_size = end - start;
    init_batch_runtime_buffers(architecture, runtime, batch_size);
    training_buffers::zero_parameter_buffer(architecture, batch_gradients, policy);
    feed_input_batch(input, indices, start, end, architecture[0], runtime[0]);
    fill_target_batch(output, indices, start, end, architecture[num_layers - 1], target_batch);

    forwardprop_batch(architecture, runtime, num_layers, hidden_activation, output_activation, policy);
    const float architecture_loss_value = loss.fn_batch(runtime[num_layers - 1], target_batch);

    forwardprop_batch(architecture, runtime, num_layers, hidden_activation, output_activation, policy, &velocity, momentum);
    float lookahead_loss_value = 0.0f;
    backprop_batch(
        architecture, runtime, num_layers,
        batch_gradients,
        loss, lookahead_loss_value, target_batch,
        hidden_activation, output_activation,
        policy, &velocity, momentum
    );
    return architecture_loss_value;
}

} // namespace training_batches
