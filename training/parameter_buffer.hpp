#pragma once

#include "core/layer.hpp"
#include "shared/execution_policy.hpp"

namespace training_buffers {

void init_parameter_buffer(const LayerList &architecture, ParameterBuffer &buffer);
void zero_parameter_buffer(const LayerList &architecture, ParameterBuffer &buffer, ExecutionPolicy policy = execution_policy::intra_example());

} // namespace training_buffers
