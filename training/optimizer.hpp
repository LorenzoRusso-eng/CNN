#pragma once

#include "core/layer.hpp"

namespace training_buffers {

void optimizer_step(LayerList &architecture, int num_layers, const ParameterBuffer &gradients, ParameterBuffer &velocity, float learning_rate, float momentum, float gradient_scale);

} // namespace training_buffers
