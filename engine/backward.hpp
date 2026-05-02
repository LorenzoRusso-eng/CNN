#pragma once

// Questo file contiene l'API pubblica del backward pass e del calcolo dei gradienti.

#include "core/layer.hpp"
#include "core/layer_validation.hpp"
#include "math/activations.hpp"
#include "math/losses.hpp"
#include "shared/execution_policy.hpp"

void backprop(const LayerList &architecture, RuntimeList &runtime, int num_layers, ParameterBuffer &gradients, const Loss &loss, float &loss_value, const Tensor &desired_output, const Activation &hidden_activation, const Activation &output_activation, ExecutionPolicy policy = execution_policy::intra_example(), const ParameterBuffer *velocity = nullptr, float momentum = 0.0f);
void backprop_batch(const LayerList &architecture, BatchRuntimeList &runtime, int num_layers, ParameterBuffer &gradients, const Loss &loss, float &loss_value, const BatchTensor &desired_output, const Activation &hidden_activation, const Activation &output_activation, ExecutionPolicy policy = execution_policy::batch_samples(), const ParameterBuffer *velocity = nullptr, float momentum = 0.0f);
