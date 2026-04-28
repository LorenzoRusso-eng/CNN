#include "IO/report_io.hpp"

// Questo file contiene la serializzazione del report performance in formato Markdown.

#include "IO/layer_text_codec.hpp"

#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

namespace{
    std::string bool_to_yes_no(bool value){
        return value ? "Yes" : "No";
    }

    std::string format_dim(const int dim[3]){
        return std::to_string(dim[0]) + "x" + std::to_string(dim[1]) + "x" + std::to_string(dim[2]);
    }

    std::string layer_details(const Layer &layer){
        switch(layer.type){
            case Layer_type::Dense:
                return "in=" + std::to_string(layer.dense_input_size) +
                       ", out=" + std::to_string(layer.dense_output_size) +
                       ", params=" + std::to_string(layer.dense_params.weights.size() + layer.dense_params.bias.size());
            case Layer_type::Conv:
                return "kernel=" + format_dim(layer.kernel_dim) +
                       ", stride=" + std::to_string(layer.stride[0]) + "x" + std::to_string(layer.stride[1]) +
                       ", padding=" + std::to_string(layer.padding[0]) + "x" + std::to_string(layer.padding[1]) +
                       ", params=" + std::to_string(layer.conv_params.filters.size() + layer.conv_params.bias.size());
            case Layer_type::Pooling:
                return "type=" + layer_text::pooling_type_to_string(layer.pooling_type) +
                       ", kernel=" + format_dim(layer.kernel_dim) +
                       ", stride=" + std::to_string(layer.stride[0]) + "x" + std::to_string(layer.stride[1]) +
                       ", padding=" + std::to_string(layer.padding[0]) + "x" + std::to_string(layer.padding[1]);
            case Layer_type::LRN:
                return "local_size=" + std::to_string(layer.lrn_local_size) +
                       ", alpha=" + std::to_string(layer.lrn_alpha) +
                       ", beta=" + std::to_string(layer.lrn_beta) +
                       ", k=" + std::to_string(layer.lrn_k);
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

    void write_general_info_hold_out(std::ostream &out, const ReportMetadata &meta, int train_examples, int test_examples){
        out << "## General Information" << std::endl;
        out << "| Field | Value |" << std::endl;
        out << "|---|---|" << std::endl;
        out << "| Model | " << meta.model_name << " |" << std::endl;
        out << "| Run Type | " << meta.run_type << " |" << std::endl;
        out << "| Dataset | " << meta.dataset_description << " |" << std::endl;
        out << "| Total Examples | " << meta.total_examples << " |" << std::endl;
        out << "| Train Examples | " << train_examples << " |" << std::endl;
        out << "| Test Examples | " << test_examples << " |" << std::endl;
        out << "| Num Classes | " << meta.num_classes << " |" << std::endl;
        out << "| Requested Epochs | " << meta.requested_epochs << " |" << std::endl;
        out << "| Hold-out Ratio | " << meta.hold_out_ratio << " |" << std::endl;
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
        out << std::endl;
    }

    void write_network_config_table(std::ostream &out, const LayerList &architecture, int num_layers){
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
}

void write_performance_report_hold_out(const std::filesystem::path &file_path, const ReportMetadata &meta, const TrainingSummary &training_summary, const TestPerformance &perf, const LayerList &architecture, int num_layers, int train_examples, int test_examples){
    std::ofstream out(file_path);
    if(!out.good()){
        throw std::runtime_error("Impossibile creare il file performance: " + file_path.string());
    }

    out << std::fixed << std::setprecision(7);

    write_report_title(out, meta);
    write_general_info_hold_out(out, meta, train_examples, test_examples);
    write_network_config_table(out, architecture, num_layers);
    write_training_config_table(out, meta);
    write_training_results_table(out, training_summary);
    write_test_results_table(out, perf);
    write_confusion_matrix_table(out, perf);
    write_class_metrics_table(out, perf);
    write_aggregate_metrics_table(out, perf);
}

void write_performance_report_k_fold(const std::filesystem::path &file_path, const ReportMetadata &meta, const std::vector<TrainingSummary> &training_summaries, const std::vector<TestPerformance> &performances, const LayerList &architecture, int num_layers){
    std::ofstream out(file_path);
    if(!out.good()){
        throw std::runtime_error("Impossibile creare il file performance: " + file_path.string());
    }

    out << std::fixed << std::setprecision(7);

    float outer_precision = 0.0f;
    float outer_recall = 0.0f;
    float outer_f1 = 0.0f;
    float outer_accuracy = 0.0f;
    double outer_total_training_seconds = 0.0;

    write_report_title(out, meta);
    write_general_info_k_fold(out, meta);
    write_network_config_table(out, architecture, num_layers);
    write_training_config_table(out, meta);

    for(int k = 0; k < meta.k_folds; k++){
        const TestPerformance &perf = performances[k];
        const TrainingSummary &train_sum = training_summaries[k];
        const int train_examples = meta.total_examples - perf.test_count;

        out << "## Fold " << (k + 1) << std::endl << std::endl;

        out << "| Field | Value |" << std::endl;
        out << "|---|---|" << std::endl;
        out << "| Train Examples | " << train_examples << " |" << std::endl;
        out << "| Test Examples | " << perf.test_count << " |" << std::endl;
        out << std::endl;

        write_training_results_table(out, train_sum);
        write_test_results_table(out, perf);
        write_confusion_matrix_table(out, perf);
        write_class_metrics_table(out, perf);
        write_aggregate_metrics_table(out, perf);

        outer_precision += perf.macro_precision;
        outer_recall += perf.macro_recall;
        outer_f1 += perf.macro_f1;
        outer_accuracy += perf.accuracy;
        outer_total_training_seconds += train_sum.total_training_seconds;
    }

    out << "## Aggregated Metrics Across Folds" << std::endl;
    out << "| Metric | Value |" << std::endl;
    out << "|---|---:|" << std::endl;
    out << "| Macro Precision | " << (outer_precision / meta.k_folds) << " |" << std::endl;
    out << "| Macro Recall | " << (outer_recall / meta.k_folds) << " |" << std::endl;
    out << "| Macro F1 | " << (outer_f1 / meta.k_folds) << " |" << std::endl;
    out << "| Accuracy | " << (outer_accuracy / meta.k_folds) << " |" << std::endl;
    out << "| Avg Total Training Seconds | " << (outer_total_training_seconds / meta.k_folds) << " |" << std::endl;
    out << std::endl;

    std::cout << "Risultati aggregati sui fold:" << std::endl;
    std::cout << "macro_precision " << (outer_precision / meta.k_folds) << std::endl;
    std::cout << "macro_recall " << (outer_recall / meta.k_folds) << std::endl;
    std::cout << "macro_f1 " << (outer_f1 / meta.k_folds) << std::endl;
    std::cout << "accuracy " << (outer_accuracy / meta.k_folds) << std::endl;
    std::cout << "avg_total_training_seconds " << (outer_total_training_seconds / meta.k_folds) << std::endl;
}
