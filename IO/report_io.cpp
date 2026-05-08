#include "IO/report_io.hpp"

// Questo file contiene la serializzazione del report performance in formato Markdown.

#include "IO/layer_text_codec.hpp"
#include "evaluation/evaluation_metrics.hpp"

#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace{
    std::string bool_to_yes_no(bool value){
        return value ? "Yes" : "No";
    }

    std::string format_dim(const Shape3D &dim){
        return std::to_string(dim[0]) + "x" + std::to_string(dim[1]) + "x" + std::to_string(dim[2]);
    }

    std::string layer_details(const Layer &layer){
        switch(layer.type){
            case Layer_type::Dense:
                return "in=" + std::to_string(layer.dense_input_size) +
                       ", out=" + std::to_string(layer.dense_output_size) +
                       ", params=" + std::to_string(layer.dense_params.weights.size() + layer.dense_params.bias.size());
            case Layer_type::Conv:
                return "kernel=" + format_dim(layer.kernel_shape) +
                       ", stride=" + std::to_string(layer.stride[0]) + "x" + std::to_string(layer.stride[1]) +
                       ", padding=" + std::to_string(layer.padding[0]) + "x" + std::to_string(layer.padding[1]) +
                       ", params=" + std::to_string(layer.conv_params.conv_weights.size() + layer.conv_params.bias.size());
            case Layer_type::Pooling:
                return "type=" + layer_text::pooling_type_to_string(layer.pooling_type) +
                       ", window=" + format_dim(layer.pool_window_shape) +
                       ", stride=" + std::to_string(layer.stride[0]) + "x" + std::to_string(layer.stride[1]) +
                       ", padding=" + std::to_string(layer.padding[0]) + "x" + std::to_string(layer.padding[1]);
            case Layer_type::Input:
            case Layer_type::Flatten:
            case Layer_type::Softmax:
                return "-";
        }
        return "-";
    }

    void write_report_title(std::ostream &out, const ReportMetadata &meta){
        out << "# Performance Report - " << meta.model_name << " (" << meta.run_type << ")" << std::endl << std::endl;
    }

    void write_general_info_hold_out(std::ostream &out, const ReportMetadata &meta, int train_examples, int validation_examples, int test_examples){
        out << "## General Information" << std::endl;
        out << "| Field | Value |" << std::endl;
        out << "|---|---|" << std::endl;
        out << "| Model | " << meta.model_name << " |" << std::endl;
        out << "| Run Type | " << meta.run_type << " |" << std::endl;
        out << "| Dataset | " << meta.dataset_description << " |" << std::endl;
        out << "| Total Examples | " << meta.total_examples << " |" << std::endl;
        out << "| Train Effective Examples | " << train_examples << " |" << std::endl;
        out << "| Validation Examples | " << validation_examples << " |" << std::endl;
        out << "| Test Examples | " << test_examples << " |" << std::endl;
        out << "| Num Classes | " << meta.num_classes << " |" << std::endl;
        out << "| Requested Epochs | " << meta.requested_epochs << " |" << std::endl;
        out << "| Hold-out Ratio | " << meta.hold_out_ratio << " |" << std::endl;
        out << "| Validation Ratio | " << meta.validation_ratio << " |" << std::endl;
        out << std::endl;
    }

    void write_general_info_k_fold(std::ostream &out, const ReportMetadata &meta){
        out << "## General Information" << std::endl;
        out << "| Field | Value |" << std::endl;
        out << "|---|---|" << std::endl;
        out << "| Model | " << meta.model_name << " |" << std::endl;
        out << "| Run Type | " << meta.run_type << " |" << std::endl;
        out << "| Dataset | " << meta.dataset_description << " |" << std::endl;
        out << "| Total Examples | " << meta.total_examples << " |" << std::endl;
        out << "| Num Classes | " << meta.num_classes << " |" << std::endl;
        out << "| Requested Epochs (per fold) | " << meta.requested_epochs << " |" << std::endl;
        out << "| k_folds | " << meta.k_folds << " |" << std::endl;
        out << "| Internal Validation | " << (meta.validation_ratio > 0.0f ? "Yes" : "No") << " |" << std::endl;
        out << "| Validation Ratio | " << meta.validation_ratio << " |" << std::endl;
        out << std::endl;
    }

    void write_network_config_table(std::ostream &out, const LayerList &architecture){
        const int num_layers = static_cast<int>(architecture.size());
        out << "## Network Configuration" << std::endl;
        out << "| Layer | Type | Input Dim | Output Dim | Details |" << std::endl;
        out << "|---:|---|---|---|---|" << std::endl;
        for(int l = 0; l < num_layers; l++){
            const Layer &layer = architecture[l];
            out << "| " << l
                << " | " << layer_text::layer_type_to_string(layer.type)
                << " | " << format_dim(layer.input_dim)
                << " | " << format_dim(layer.dim_layer)
                << " | " << layer_details(layer)
                << " |" << std::endl;
        }
        out << std::endl;
    }

    void write_training_config_table(std::ostream &out, const ReportMetadata &meta){
        out << "## Training Configuration" << std::endl;
        out << "| Parameter | Value |" << std::endl;
        out << "|---|---|" << std::endl;
        out << "| Training Variant | " << meta.training_variant << " |" << std::endl;
        out << "| " << meta.training_window_label << " | " << meta.training_window << " |" << std::endl;
        out << "| Nesterov | " << bool_to_yes_no(meta.use_nesterov) << " |" << std::endl;
        out << "| Momentum | " << meta.momentum << " |" << std::endl;
        out << "| Target Loss | " << meta.target_loss << " |" << std::endl;
        out << "| Learning Rate Decay | " << meta.decay_type << " |" << std::endl;
        out << "| Initial Learning Rate | " << meta.initial_learning_rate << " |" << std::endl;
        out << "| Hidden Activation | " << meta.hidden_activation << " |" << std::endl;
        out << "| Output Activation | " << meta.output_activation << " |" << std::endl;
        out << "| Loss | " << meta.loss_name << " |" << std::endl;
        out << "| Loss Reduction | " << meta.loss_reduction << " |" << std::endl;
        out << std::endl;
    }

    void write_training_results_table(std::ostream &out, const TrainingSummary &summary){
        double avg_epoch_time = 0.0;
        if(!summary.epoch_times_seconds.empty()){
            double total_epoch_times = 0.0;
            for(double epoch_time : summary.epoch_times_seconds){
                total_epoch_times += epoch_time;
            }
            avg_epoch_time = total_epoch_times / static_cast<double>(summary.epoch_times_seconds.size());
        }

        out << "## Training Results" << std::endl;
        out << "| Metric | Value |" << std::endl;
        out << "|---|---:|" << std::endl;
        out << "| Executed Epochs | " << summary.epochs_completed << " |" << std::endl;
        out << "| Final Loss | " << summary.final_loss << " |" << std::endl;
        out << "| Total Training Seconds | " << summary.total_training_seconds << " |" << std::endl;
        out << "| Avg Epoch Seconds | " << avg_epoch_time << " |" << std::endl;
        out << "| Stopped By Loss | " << bool_to_yes_no(summary.stopped_by_loss) << " |" << std::endl;
        out << "| Stopped By Validation Patience | " << bool_to_yes_no(summary.stopped_by_validation) << " |" << std::endl;
        out << "| Used Validation | " << bool_to_yes_no(summary.used_validation) << " |" << std::endl;
        if(summary.used_validation){
            out << "| Best Validation Accuracy | " << summary.best_validation_accuracy << " |" << std::endl;
            out << "| Best Validation Epoch | " << summary.best_epoch << " |" << std::endl;
            out << "| Epochs Without Significant Improvement | " << summary.epochs_without_significant_improvement << " |" << std::endl;
        }
        out << std::endl;

        out << "### Epoch Times" << std::endl;
        out << "| Epoch | Seconds |" << std::endl;
        out << "|---:|---:|" << std::endl;
        for(std::size_t epoch = 0; epoch < summary.epoch_times_seconds.size(); epoch++){
            out << "| " << (epoch + 1) << " | " << summary.epoch_times_seconds[epoch] << " |" << std::endl;
        }
        out << std::endl;
    }

    void write_test_results_table(std::ostream &out, const TestPerformance &perf){
        out << "## Test Results" << std::endl;
        out << "| Metric | Value |" << std::endl;
        out << "|---|---:|" << std::endl;
        out << "| Test Examples | " << perf.test_count << " |" << std::endl;
        out << "| Correct Predictions | " << perf.correct << " |" << std::endl;
        out << "| Accuracy | " << perf.accuracy << " |" << std::endl;
        out << std::endl;
    }

    void write_confusion_matrix_table(std::ostream &out, const TestPerformance &perf){
        out << "## Confusion Matrix" << std::endl;
        out << "Rows: true class, Columns: predicted class" << std::endl << std::endl;

        out << "| True \\ Pred |";
        for(int c = 0; c < perf.num_classes; c++){
            out << " " << c << " |";
        }
        out << std::endl;

        out << "|---|";
        for(int c = 0; c < perf.num_classes; c++){
            out << "---:|";
        }
        out << std::endl;

        for(int r = 0; r < perf.num_classes; r++){
            out << "| " << r << " |";
            for(int c = 0; c < perf.num_classes; c++){
                out << " " << perf.confusion[r][c] << " |";
            }
            out << std::endl;
        }
        out << std::endl;
    }

    void write_class_metrics_table(std::ostream &out, const TestPerformance &perf){
        out << "## Metrics by Class" << std::endl;
        out << "| Class | TP | FP | TN | FN | Precision | Recall | F1 |" << std::endl;
        out << "|---:|---:|---:|---:|---:|---:|---:|---:|" << std::endl;
        for(int c = 0; c < perf.num_classes; c++){
            const ClassPerformance &m = perf.per_class[c];
            out << "| " << c
                << " | " << m.tp
                << " | " << m.fp
                << " | " << m.tn
                << " | " << m.fn
                << " | " << m.precision
                << " | " << m.recall
                << " | " << m.f1
                << " |" << std::endl;
        }
        out << std::endl;
    }

    void write_aggregate_metrics_table(std::ostream &out, const TestPerformance &perf){
        out << "## Aggregate Metrics" << std::endl;
        out << "| Metric | Value |" << std::endl;
        out << "|---|---:|" << std::endl;
        out << "| Macro Precision | " << perf.macro_precision << " |" << std::endl;
        out << "| Macro Recall | " << perf.macro_recall << " |" << std::endl;
        out << "| Macro F1 | " << perf.macro_f1 << " |" << std::endl;
        out << "| Accuracy | " << perf.accuracy << " |" << std::endl;
        out << std::endl;
    }

    TestPerformance aggregate_k_fold_performance(const std::vector<TestPerformance> &performances, int num_classes){
        int total_test_count = 0;
        int total_correct = 0;
        std::vector<std::vector<int>> total_confusion(
            static_cast<std::size_t>(num_classes),
            std::vector<int>(static_cast<std::size_t>(num_classes), 0)
        );

        for(const TestPerformance &perf : performances){
            if(perf.test_count <= 0){
                continue;
            }

            if(perf.num_classes != num_classes){
                throw std::invalid_argument("aggregate_k_fold_performance: numero classi incoerente tra fold");
            }
            if(static_cast<int>(perf.confusion.size()) != num_classes){
                throw std::invalid_argument("aggregate_k_fold_performance: confusion matrix incoerente tra fold");
            }

            total_test_count += perf.test_count;
            total_correct += perf.correct;

            for(int r = 0; r < num_classes; r++){
                if(static_cast<int>(perf.confusion[static_cast<std::size_t>(r)].size()) != num_classes){
                    throw std::invalid_argument("aggregate_k_fold_performance: riga confusion matrix incoerente tra fold");
                }

                for(int c = 0; c < num_classes; c++){
                    total_confusion[static_cast<std::size_t>(r)][static_cast<std::size_t>(c)] +=
                        perf.confusion[static_cast<std::size_t>(r)][static_cast<std::size_t>(c)];
                }
            }
        }

        if(total_test_count <= 0){
            return TestPerformance{};
        }

        return evaluation_metrics::build_test_performance(
            num_classes,
            total_test_count,
            total_correct,
            std::move(total_confusion)
        );
    }
}

void write_performance_report_hold_out(const std::filesystem::path &file_path, const ReportMetadata &meta, const TrainingSummary &training_summary, const TestPerformance &perf, const LayerList &architecture, int train_examples, int validation_examples, int test_examples){
    std::ofstream out(file_path);
    if(!out.good()){
        throw std::runtime_error("Impossibile creare il file performance: " + file_path.string());
    }

    out << std::fixed << std::setprecision(7);

    write_report_title(out, meta);
    write_general_info_hold_out(out, meta, train_examples, validation_examples, test_examples);
    write_network_config_table(out, architecture);
    write_training_config_table(out, meta);
    write_training_results_table(out, training_summary);
    write_test_results_table(out, perf);
    write_confusion_matrix_table(out, perf);
    write_class_metrics_table(out, perf);
    write_aggregate_metrics_table(out, perf);
}

void write_performance_report_k_fold(const std::filesystem::path &file_path, const ReportMetadata &meta, const std::vector<TrainingSummary> &training_summaries, const std::vector<TestPerformance> &performances, const std::vector<int> &train_examples_by_fold, const std::vector<int> &validation_examples_by_fold, const LayerList &architecture){
    std::ofstream out(file_path);
    if(!out.good()){
        throw std::runtime_error("Impossibile creare il file performance: " + file_path.string());
    }

    out << std::fixed << std::setprecision(7);

    double outer_total_training_seconds = 0.0;

    write_report_title(out, meta);
    write_general_info_k_fold(out, meta);
    write_network_config_table(out, architecture);
    write_training_config_table(out, meta);

    for(int k = 0; k < meta.k_folds; k++){
        const TestPerformance &perf = performances[k];
        const TrainingSummary &train_sum = training_summaries[k];
        const int train_examples = train_examples_by_fold[k];
        const int validation_examples = validation_examples_by_fold[k];

        out << "## Fold " << (k + 1) << std::endl << std::endl;

        out << "| Field | Value |" << std::endl;
        out << "|---|---|" << std::endl;
        out << "| Train Effective Examples | " << train_examples << " |" << std::endl;
        out << "| Validation Examples | " << validation_examples << " |" << std::endl;
        out << "| Test Examples | " << perf.test_count << " |" << std::endl;
        out << std::endl;

        write_training_results_table(out, train_sum);
        write_test_results_table(out, perf);
        write_confusion_matrix_table(out, perf);
        write_class_metrics_table(out, perf);
        write_aggregate_metrics_table(out, perf);

        outer_total_training_seconds += train_sum.total_training_seconds;
    }

    const TestPerformance aggregate_perf = aggregate_k_fold_performance(performances, meta.num_classes);

    out << "## Aggregated Metrics Across Folds" << std::endl;
    out << "| Metric | Value |" << std::endl;
    out << "|---|---:|" << std::endl;
    out << "| Test Examples | " << aggregate_perf.test_count << " |" << std::endl;
    out << "| Correct Predictions | " << aggregate_perf.correct << " |" << std::endl;
    out << "| Macro Precision | " << aggregate_perf.macro_precision << " |" << std::endl;
    out << "| Macro Recall | " << aggregate_perf.macro_recall << " |" << std::endl;
    out << "| Macro F1 | " << aggregate_perf.macro_f1 << " |" << std::endl;
    out << "| Accuracy | " << aggregate_perf.accuracy << " |" << std::endl;
    out << "| Avg Total Training Seconds | " << (outer_total_training_seconds / meta.k_folds) << " |" << std::endl;
    out << std::endl;

    write_confusion_matrix_table(out, aggregate_perf);
    write_class_metrics_table(out, aggregate_perf);
    write_aggregate_metrics_table(out, aggregate_perf);

    std::cout << "Risultati aggregati sui fold:" << std::endl;
    std::cout << "test_examples " << aggregate_perf.test_count << std::endl;
    std::cout << "correct " << aggregate_perf.correct << std::endl;
    std::cout << "macro_precision " << aggregate_perf.macro_precision << std::endl;
    std::cout << "macro_recall " << aggregate_perf.macro_recall << std::endl;
    std::cout << "macro_f1 " << aggregate_perf.macro_f1 << std::endl;
    std::cout << "accuracy " << aggregate_perf.accuracy << std::endl;
    std::cout << "avg_total_training_seconds " << (outer_total_training_seconds / meta.k_folds) << std::endl;
}
