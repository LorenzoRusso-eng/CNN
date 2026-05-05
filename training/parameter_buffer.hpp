#pragma once

#include "core/layer.hpp"

namespace training_buffers {

void init_parameter_buffer(const LayerList &architecture, ParameterBuffer &buffer);
void zero_parameter_buffer(const LayerList &architecture, ParameterBuffer &buffer);

} // namespace training_buffers
