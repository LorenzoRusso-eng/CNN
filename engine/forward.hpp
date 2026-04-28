#pragma once

// Questo file contiene l'API pubblica del forward pass e dell'alimentazione dell'input.

#include "core/layer.hpp"
#include "core/layer_validation.hpp"
#include "math/activations.hpp"
#include "shared/execution_policy.hpp"

void feed_input(const Tensor3D &input, const Layer &first, LayerRuntime &first_runtime);
void feed_input_batch(const Dataset4D &input, const std::vector<int> &indices, int start, int end, const Layer &first, BatchLayerRuntime &first_runtime);

void forwardprop(const LayerList &architecture, RuntimeList &runtime, int num_layers, const Activation &hidden_activation, const Activation &output_activation, ExecutionPolicy policy = execution_policy::intra_example(), const ParameterBuffer *velocity = nullptr, float momentum = 0.0f);
void forwardprop_batch(const LayerList &architecture, BatchRuntimeList &runtime, int num_layers, const Activation &hidden_activation, const Activation &output_activation, ExecutionPolicy policy = execution_policy::batch_samples(), const ParameterBuffer *velocity = nullptr, float momentum = 0.0f);
