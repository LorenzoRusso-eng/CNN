#pragma once

// Questo file contiene l'API pubblica per inference finale e metriche di test.

#include "core/layer.hpp"
#include "math/activations.hpp"

int argmax_target(const Tensor3D &target);
int argmax_output(const Layer &output_layer, const LayerRuntime &output_runtime);
TestPerformance run_test(LayerList &architecture, int num_layers, const std::vector<int> &test_indices, const Dataset4D &input, const Dataset4D &output, const Activation &hidden_activation, const Activation &output_activation);
