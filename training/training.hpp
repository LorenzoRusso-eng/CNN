#pragma once

// Questo file contiene gli entrypoint pubblici delle varianti di training.

#include "core/layer.hpp"
#include "math/activations.hpp"
#include "math/losses.hpp"
#include "math/decay.hpp"

TrainingSummary train_batch(LayerList &architecture, int num_layers, const Decay &learning_rate_decay, int num_epochs, float target_loss, std::vector<int> &train_indices, const Dataset4D &input, const Dataset4D &output, const Loss &loss, const Activation &hidden_activation, const Activation &output_activation, float momentum, TrainingRuntimeState *runtime_state = nullptr);
TrainingSummary train_sgd(LayerList &architecture, int num_layers, int batch_size, const Decay &learning_rate_decay, int num_epochs, float target_loss, std::vector<int> &train_indices, const Dataset4D &input, const Dataset4D &output, const Loss &loss, const Activation &hidden_activation, const Activation &output_activation, float momentum, TrainingRuntimeState *runtime_state = nullptr);
TrainingSummary train_sgd_online(LayerList &architecture, int num_layers, int steps, const Decay &learning_rate_decay, int num_epochs, float target_loss, std::vector<int> &train_indices, const Dataset4D &input, const Dataset4D &output, const Loss &loss, const Activation &hidden_activation, const Activation &output_activation, float momentum, TrainingRuntimeState *runtime_state = nullptr);
TrainingSummary train_batch_nesterov(LayerList &architecture, int num_layers, const Decay &learning_rate_decay, int num_epochs, float target_loss, std::vector<int> &train_indices, const Dataset4D &input, const Dataset4D &output, const Loss &loss, const Activation &hidden_activation, const Activation &output_activation, float momentum, TrainingRuntimeState *runtime_state = nullptr);
TrainingSummary train_sgd_nesterov(LayerList &architecture, int num_layers, int batch_size, const Decay &learning_rate_decay, int num_epochs, float target_loss, std::vector<int> &train_indices, const Dataset4D &input, const Dataset4D &output, const Loss &loss, const Activation &hidden_activation, const Activation &output_activation, float momentum, TrainingRuntimeState *runtime_state = nullptr);
TrainingSummary train_sgd_online_nesterov(LayerList &architecture, int num_layers, int steps, const Decay &learning_rate_decay, int num_epochs, float target_loss, std::vector<int> &train_indices, const Dataset4D &input, const Dataset4D &output, const Loss &loss, const Activation &hidden_activation, const Activation &output_activation, float momentum = 0.0f, TrainingRuntimeState *runtime_state = nullptr);
