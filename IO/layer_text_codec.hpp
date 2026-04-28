#pragma once

#include "core/layer.hpp"

#include <string>

namespace layer_text {

Layer_type parse_layer_type(const std::string &type);
Pooling_type parse_pooling_type(const std::string &type);
std::string layer_type_to_string(Layer_type type);
std::string pooling_type_to_string(Pooling_type type);

} // namespace layer_text
