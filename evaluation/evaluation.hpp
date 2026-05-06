#pragma once

// Questo file contiene l'API pubblica per inference finale e metriche di test.

#include "core/layer.hpp"
#include "math/activations.hpp"

TestPerformance run_test(LayerList &architecture, int num_layers, const std::vector<int> &test_indices, const LazyDataset &dataset, const Activation &hidden_activation, const Activation &output_activation);
