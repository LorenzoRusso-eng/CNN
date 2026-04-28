#pragma once

#include "core/layer.hpp"
#include "math/activations.hpp"
#include "math/losses.hpp"
#include "shared/execution_policy.hpp"

#include <vector>

namespace training_examples {

float train_example(const LayerList &architecture, RuntimeList &runtime, int num_layers, ParameterBuffer &sample_gradients, const Dataset4D &input, const Dataset4D &output, int sample_index, const Loss &loss, const Activation &hidden_activation, const Activation &output_activation, ExecutionPolicy policy = execution_policy::intra_example());
float train_example_nesterov(const LayerList &architecture, RuntimeList &runtime, int num_layers, ParameterBuffer &sample_gradients, const ParameterBuffer &velocity, const Dataset4D &input, const Dataset4D &output, int sample_index, const Loss &loss, const Activation &hidden_activation, const Activation &output_activation, float momentum, ExecutionPolicy policy = execution_policy::intra_example());

} // namespace training_examples

namespace training_batches {

void fill_target_batch(const Dataset4D &output, const std::vector<int> &indices, int start, int end, const Layer &last, BatchTensor &target_batch);
float train_batch_chunk(const LayerList &architecture, BatchRuntimeList &runtime, int num_layers, ParameterBuffer &batch_gradients, BatchTensor &target_batch, const Dataset4D &input, const Dataset4D &output, const std::vector<int> &indices, int start, int end, const Loss &loss, const Activation &hidden_activation, const Activation &output_activation, ExecutionPolicy policy = execution_policy::batch_samples());
float train_batch_chunk_nesterov(const LayerList &architecture, BatchRuntimeList &runtime, int num_layers, ParameterBuffer &batch_gradients, BatchTensor &target_batch, const ParameterBuffer &velocity, const Dataset4D &input, const Dataset4D &output, const std::vector<int> &indices, int start, int end, const Loss &loss, const Activation &hidden_activation, const Activation &output_activation, float momentum, ExecutionPolicy policy = execution_policy::batch_samples());

} // namespace training_batches
