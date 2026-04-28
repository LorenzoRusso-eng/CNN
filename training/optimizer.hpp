#pragma once

#include "core/layer.hpp"
#include "shared/execution_policy.hpp"

namespace training_buffers {

void optimizer_step(LayerList &architecture, int num_layers, const ParameterBuffer &gradients, ParameterBuffer &velocity, float learning_rate, float momentum, float gradient_scale, ExecutionPolicy policy = execution_policy::intra_example());

} // namespace training_buffers
