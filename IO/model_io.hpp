#pragma once

// Questo file contiene il caricamento e salvataggio degli snapshot del modello.

#include <filesystem>
#include <string>
#include <vector>

#include "core/layer.hpp"
#include "IO/activation_codec.hpp"
#include "math/activations.hpp"
#include "math/decay.hpp"
#include "math/losses.hpp"

struct TrainingSnapshotMetadata{
    std::string model_name;
    int training_method = 0; // 1=Hold-out 2=K-fold 3=Full-training
    int training_type = 0;   // 1=Batch 2=Mini-batch SGD 3=Online SGD
    int training_window = 0;
    bool use_nesterov = false;
    float momentum = 0.0f;
    float target_loss = 0.0f;
    int requested_epochs = 0;
    int completed_epochs = 0;
    long long optimizer_steps = 0;
    bool training_finalized = false;

    DecayKind decay_kind = DecayKind::Constant;
    float initial_learning_rate = 0.0f;
    float decay_rate = 0.0f;
    int decay_step_size = 0;
    float cosine_final_lr = 0.0f;
    int cosine_max_epoch = 0;

    LossKind loss_kind = LossKind::Simple;
    Reduction loss_reduction = Reduction::Sum;
    float loss_beta = 1.0f;

    float hold_out_ratio = 0.0f;
    int k_folds = 0;
    std::vector<int> train_indices;
    std::vector<int> test_indices;
    std::vector<std::string> dataset_manifest_paths;
    std::string shuffle_rng_state;
    ParameterBuffer optimizer_velocity;
};

void load_model_snapshot(const std::filesystem::path &snapshot_path, LayerList &architecture, std::vector<std::string> &class_names, std::string &hidden_activation_name, std::string &output_activation_name);
void save_model_snapshot(const LayerList &architecture, int num_layers, const std::filesystem::path &file_path, const std::vector<std::string> &class_names, const Activation &hidden_activation, const Activation &output_activation);
void save_training_snapshot(const std::filesystem::path &file_path, const LayerList &architecture, int num_layers, const std::vector<std::string> &class_names, const Activation &hidden_activation, const Activation &output_activation, const TrainingSnapshotMetadata &metadata);
void load_training_snapshot(const std::filesystem::path &file_path, LayerList &architecture, std::vector<std::string> &class_names, std::string &hidden_activation_name, std::string &output_activation_name, TrainingSnapshotMetadata &metadata);
