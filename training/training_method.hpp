#pragma once

// Questo file contiene gli entrypoint  pubblici dei diversi metodi di training.

#include "core/layer.hpp"
#include "IO/model_io.hpp"
#include "math/activations.hpp"
#include "math/losses.hpp"
#include "math/decay.hpp"

#include <string>
#include <vector>

namespace training_split {

void split_stratified_subset(const LazyDataset &dataset, int num_classes, const std::vector<int> &source_indices, float validation_ratio, std::vector<int> &train_effective_indices, std::vector<int> &validation_indices);

} // namespace training_split

void k_fold(std::string model_name, int k_folds, int num_examples, int training_type, bool use_nesterov, LayerList &architecture, int num_layers, const Decay &learning_rate_decay, int num_epochs, float target_loss, const LazyDataset &dataset, const Loss &loss, const Activation &hidden_activation, const Activation &output_activation, float momentum, float validation_ratio, const std::vector<std::string> &class_names);
void hold_out(std::string model_name, int num_examples, int training_type, bool use_nesterov, LayerList &architecture, int num_layers, const Decay &learning_rate_decay, int num_epochs, float target_loss, const LazyDataset &dataset, const Loss &loss, const Activation &hidden_activation, const Activation &output_activation, float momentum, float train_ratio, float validation_ratio, const std::vector<std::string> &class_names);
void full_training(std::string model_name, int num_examples, int training_type, bool use_nesterov, LayerList &architecture, int num_layers, const Decay &learning_rate_decay, int num_epochs, float target_loss, const LazyDataset &dataset, const Loss &loss, const Activation &hidden_activation, const Activation &output_activation, float momentum, const std::vector<std::string> &class_names);
