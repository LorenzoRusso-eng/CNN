#pragma once

// Questo file contiene gli entrypoint pubblici delle varianti di training.

#include "core/cuda_backend.hpp"
#include "core/layer.hpp"
#include "math/activations.hpp"
#include "math/losses.hpp"
#include "math/decay.hpp"

#include <vector>

TrainingSummary train_batch(LayerList &architecture, int num_layers, int chunk_size, const Decay &learning_rate_decay, int num_epochs, float target_loss, std::vector<int> &train_indices, const LazyDataset &dataset, const Loss &loss, const Activation &hidden_activation, const Activation &output_activation, float momentum, cuda_backend::CudaParameterBuffer &parameters, const std::vector<int> *validation_indices = nullptr, const EarlyStoppingConfig *early_stopping = nullptr);
TrainingSummary train_sgd(LayerList &architecture, int num_layers, int batch_size, const Decay &learning_rate_decay, int num_epochs, float target_loss, std::vector<int> &train_indices, const LazyDataset &dataset, const Loss &loss, const Activation &hidden_activation, const Activation &output_activation, float momentum, cuda_backend::CudaParameterBuffer &parameters, const std::vector<int> *validation_indices = nullptr, const EarlyStoppingConfig *early_stopping = nullptr);
TrainingSummary train_sgd_online(LayerList &architecture, int num_layers, int steps, const Decay &learning_rate_decay, int num_epochs, float target_loss, std::vector<int> &train_indices, const LazyDataset &dataset, const Loss &loss, const Activation &hidden_activation, const Activation &output_activation, float momentum, cuda_backend::CudaParameterBuffer &parameters, const std::vector<int> *validation_indices = nullptr, const EarlyStoppingConfig *early_stopping = nullptr);
TrainingSummary train_batch_nesterov(LayerList &architecture, int num_layers, int chunk_size, const Decay &learning_rate_decay, int num_epochs, float target_loss, std::vector<int> &train_indices, const LazyDataset &dataset, const Loss &loss, const Activation &hidden_activation, const Activation &output_activation, float momentum, cuda_backend::CudaParameterBuffer &parameters, const std::vector<int> *validation_indices = nullptr, const EarlyStoppingConfig *early_stopping = nullptr);
TrainingSummary train_sgd_nesterov(LayerList &architecture, int num_layers, int batch_size, const Decay &learning_rate_decay, int num_epochs, float target_loss, std::vector<int> &train_indices, const LazyDataset &dataset, const Loss &loss, const Activation &hidden_activation, const Activation &output_activation, float momentum, cuda_backend::CudaParameterBuffer &parameters, const std::vector<int> *validation_indices = nullptr, const EarlyStoppingConfig *early_stopping = nullptr);
TrainingSummary train_sgd_online_nesterov(LayerList &architecture, int num_layers, int steps, const Decay &learning_rate_decay, int num_epochs, float target_loss, std::vector<int> &train_indices, const LazyDataset &dataset, const Loss &loss, const Activation &hidden_activation, const Activation &output_activation, float momentum, cuda_backend::CudaParameterBuffer &parameters, const std::vector<int> *validation_indices = nullptr, const EarlyStoppingConfig *early_stopping = nullptr);
