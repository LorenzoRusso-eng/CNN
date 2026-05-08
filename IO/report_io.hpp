#pragma once

// Questo file contiene la scrittura del report finale di performance.

#include <filesystem>
#include <string>

#include "core/core_definitions.hpp"
#include "core/layer.hpp"

struct ReportMetadata{
    std::string model_name;
    std::string dataset_description = "N/A";
    std::string run_type;
    int requested_epochs = 0;
    int total_examples = 0;
    int num_classes = 0;
    std::string training_variant;
    std::string training_window_label = "N/A";
    int training_window = 0;
    bool use_nesterov = false;
    float momentum = 0.0f;
    float target_loss = 0.0f;
    std::string decay_type;
    float initial_learning_rate = 0.0f;
    std::string hidden_activation;
    std::string output_activation;
    std::string loss_name;
    std::string loss_reduction;
    float hold_out_ratio = 0.0f;
    float validation_ratio = 0.0f;
    int k_folds = 0;
};

void write_performance_report_hold_out(const std::filesystem::path &file_path, const ReportMetadata &meta, const TrainingSummary &training_summary, const TestPerformance &perf, const LayerList &architecture, int train_examples, int validation_examples, int test_examples);
void write_performance_report_k_fold(const std::filesystem::path &file_path, const ReportMetadata &meta, const std::vector<TrainingSummary> &training_summaries, const std::vector<TestPerformance> &performances, const std::vector<int> &train_examples_by_fold, const std::vector<int> &validation_examples_by_fold, const LayerList &architecture);
