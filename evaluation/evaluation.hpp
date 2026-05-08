#pragma once

// Questo file contiene l'API pubblica per inference finale e metriche di test.

#include "core/cuda_backend.hpp"
#include "core/layer.hpp"
#include "math/activations.hpp"

TestPerformance run_test(LayerList &architecture, const cuda_backend::CudaParameterBuffer &parameters, const std::vector<int> &test_indices, const LazyDataset &dataset, const Activation &hidden_activation, const Activation &output_activation);
float run_validation_accuracy(LayerList &architecture, const cuda_backend::CudaParameterBuffer &parameters, cuda_backend::CudaBatchRuntimeList &runtime, const std::vector<int> &validation_indices, const LazyDataset &dataset, const Activation &hidden_activation, const Activation &output_activation, int batch_size = 200);
