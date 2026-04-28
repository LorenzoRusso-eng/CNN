#pragma once

// Questo file contiene gli entrypoint  pubblici dei diversi metodi di training.

#include "core/layer.hpp"
#include "IO/model_io.hpp"
#include "math/activations.hpp"
#include "math/losses.hpp"
#include "math/decay.hpp"

#include <string>
#include <vector>

void k_fold(std::string model_name, int k_folds, int num_examples, int training_type, bool use_nesterov, LayerList &architecture, int num_layers, const Decay &learning_rate_decay, int num_epochs, float target_loss, const Dataset4D &input, const Dataset4D &output, const Loss &loss, const Activation &hidden_activation, const Activation &output_activation, float momentum, const std::vector<std::string> &class_names);
void hold_out(std::string model_name, int num_examples, int training_type, bool use_nesterov, LayerList &architecture, int num_layers, const Decay &learning_rate_decay, int num_epochs, float target_loss, const Dataset4D &input, const Dataset4D &output, const Loss &loss, const Activation &hidden_activation, const Activation &output_activation, float momentum, float train_ratio, const std::vector<std::string> &class_names, const std::vector<std::string> &dataset_manifest_paths);
void full_training(std::string model_name, int num_examples, int training_type, bool use_nesterov, LayerList &architecture, int num_layers, const Decay &learning_rate_decay, int num_epochs, float target_loss, const Dataset4D &input, const Dataset4D &output, const Loss &loss, const Activation &hidden_activation, const Activation &output_activation, float momentum, const std::vector<std::string> &class_names, const std::vector<std::string> &dataset_manifest_paths);
void hold_out_resume(std::string model_name, int num_examples, LayerList &architecture, int num_layers, const Decay &learning_rate_decay, int target_total_epochs, const Dataset4D &input, const Dataset4D &output, const Loss &loss, const Activation &hidden_activation, const Activation &output_activation, const std::vector<std::string> &class_names, const TrainingSnapshotMetadata &resume_snapshot, const std::vector<std::string> &dataset_manifest_paths);
void full_training_resume(std::string model_name, int num_examples, LayerList &architecture, int num_layers, const Decay &learning_rate_decay, int target_total_epochs, const Dataset4D &input, const Dataset4D &output, const Loss &loss, const Activation &hidden_activation, const Activation &output_activation, const std::vector<std::string> &class_names, const TrainingSnapshotMetadata &resume_snapshot, const std::vector<std::string> &dataset_manifest_paths);
