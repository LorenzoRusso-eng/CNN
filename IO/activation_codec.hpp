#pragma once

#include "math/activations.hpp"

#include <string>

std::string activation_to_snapshot_name(const Activation &activation);
Activation activation_from_snapshot_name(const std::string &activation_name);
