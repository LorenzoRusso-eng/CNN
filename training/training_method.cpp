#include "training/training_method.hpp"

// Questo file definisce il funzionamento della logica k_fold.

#include "evaluation/evaluation.hpp"
#include "training/shuffle_rng.hpp"
#include "training/training.hpp"
#include "training/training_metadata.hpp"
#include "IO/model_io.hpp"
#include "IO/report_io.hpp"
#include "cli/training_prompts.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>

using namespace std;

namespace{
    namespace fs = std::filesystem;

    std::vector<std::vector<int>> build_indices_per_class(const Dataset4D &output, int num_classes){
        std::vector<std::vector<int>> indices_per_class(num_classes);

        for(int i = 0; i < static_cast<int>(output.size()); i++){
            const int class_index = argmax_target(output[i]);
            if(class_index < 0 || class_index >= num_classes) throw std::invalid_argument("Indice di classe non valido nel dataset");
            indices_per_class[class_index].push_back(i);
        }

        return indices_per_class;
    }

    TrainingSummary train(int training_type, bool use_nesterov, int window, std::vector<int> &indices, LayerList &architecture, int num_layers, const Decay &learning_rate_decay, int num_epochs, float target_loss, const Dataset4D &input, const Dataset4D &output, const Loss &loss, const Activation &hidden_activation, const Activation &output_activation, float momentum, TrainingRuntimeState *runtime_state = nullptr){
        switch(training_type){
            case 1:
                return use_nesterov
                    ? train_batch_nesterov(
                        architecture, num_layers,
                        learning_rate_decay, num_epochs, target_loss,
                        indices, input, output,
                        loss, hidden_activation, output_activation,
                        momentum,
                        execution_policy::batch_samples(),
                        runtime_state
                    )
                    : train_batch(
                        architecture, num_layers,
                        learning_rate_decay, num_epochs, target_loss,
                        indices, input, output,
                        loss, hidden_activation, output_activation,
                        momentum,
                        execution_policy::batch_samples(),
                        runtime_state
                    );
            case 2:
                return use_nesterov
                    ? train_sgd_nesterov(
                        architecture, num_layers, window,
                        learning_rate_decay, num_epochs, target_loss,
                        indices, input, output,
                        loss, hidden_activation, output_activation,
                        momentum,
                        execution_policy::batch_samples(),
                        runtime_state
                    )
                    : train_sgd(
                        architecture, num_layers, window,
                        learning_rate_decay, num_epochs, target_loss,
                        indices, input, output,
                        loss, hidden_activation, output_activation,
                        momentum,
                        execution_policy::batch_samples(),
                        runtime_state
                    );
            case 3:
                return use_nesterov
                    ? train_sgd_online_nesterov(
                        architecture, num_layers, window,
                        learning_rate_decay, num_epochs, target_loss,
                        indices, input, output,
                        loss, hidden_activation, output_activation,
                        momentum,
                        execution_policy::intra_example(),
                        runtime_state
                    )
                    : train_sgd_online(
                        architecture, num_layers, window,
                        learning_rate_decay, num_epochs, target_loss,
                        indices, input, output,
                        loss, hidden_activation, output_activation,
                        momentum,
                        execution_policy::intra_example(),
                        runtime_state
                    );
            default:
                throw std::invalid_argument("Tipo di training non valido");
        }
    }

}

namespace training_split {

void build_stratified_hold_out_indices(const Dataset4D &output, int num_classes, float train_ratio, std::vector<int> &train_indices, std::vector<int> &test_indices){
    train_indices.clear();
    test_indices.clear();

    std::vector<std::vector<int>> indices_per_class = build_indices_per_class(output, num_classes);

    for(int c = 0; c < num_classes; c++){
        std::vector<int> &class_indices = indices_per_class[c];
        training_shuffle::shuffle_vec(class_indices);

        const int class_size = static_cast<int>(class_indices.size());
        int train_count = static_cast<int>(std::round(train_ratio * class_size));

        if(class_size >= 2) train_count = std::clamp(train_count, 1, class_size - 1);

        for(int i = 0; i < class_size; i++){
            if(i < train_count) train_indices.push_back(class_indices[i]);
            else test_indices.push_back(class_indices[i]);
        }
    }

    training_shuffle::shuffle_vec(train_indices);
    training_shuffle::shuffle_vec(test_indices);
}

std::vector<std::vector<int>> build_stratified_folds(const Dataset4D &output, int num_classes, int k_folds){
    if(k_folds <= 0) throw std::invalid_argument("K deve essere positivo");

    std::vector<std::vector<int>> folds(k_folds);
    std::vector<std::vector<int>> indices_per_class = build_indices_per_class(output, num_classes);

    for(int c = 0; c < num_classes; c++){
        std::vector<int> &class_indices = indices_per_class[c];
        if(static_cast<int>(class_indices.size()) < k_folds) throw std::invalid_argument("Una classe ha meno esempi del numero di fold K");

        training_shuffle::shuffle_vec(class_indices);

        for(int i = 0; i < static_cast<int>(class_indices.size()); i++) folds[i % k_folds].push_back(class_indices[i]);
    }

    for(int k = 0; k < k_folds; k++) training_shuffle::shuffle_vec(folds[k]);

    return folds;
}

} // namespace training_split

void hold_out(string model_name, int num_examples, int training_type, bool use_nesterov, LayerList &architecture, int num_layers, const Decay &learning_rate_decay, int num_epochs, float target_loss, const Dataset4D &input, const Dataset4D &output, const Loss &loss, const Activation &hidden_activation, const Activation &output_activation, float momentum, float train_ratio, const std::vector<std::string> &class_names, const std::vector<std::string> &dataset_manifest_paths){
    (void)num_examples;
    vector<int> train_indices, test_indices;
    TrainingSummary training_summary{};
    TestPerformance performance{};
    TrainingRuntimeState runtime_state{};

    const int num_classes = output[0].height;
    training_split::build_stratified_hold_out_indices(output, num_classes, train_ratio, train_indices, test_indices);

    int window = choose_training_window(training_type, static_cast<int>(train_indices.size()));

    training_summary = train(
        training_type, use_nesterov, window,
        train_indices,
        architecture, num_layers,
        learning_rate_decay, num_epochs, target_loss,
        input, output,
        loss, hidden_activation, output_activation,
        momentum,
        &runtime_state
    );
    performance = run_test(architecture, num_layers, test_indices, input, output, hidden_activation, output_activation);

    ReportMetadata report_meta = training_metadata::build_report_metadata(
        model_name, "Hold-out",
        num_epochs, num_examples, num_classes,
        training_type, window, use_nesterov,
        momentum, target_loss,
        learning_rate_decay, loss,
        hidden_activation, output_activation
    );
    report_meta.hold_out_ratio = train_ratio;
    const fs::path performance_report_path = "network_performance_report_" + model_name + ".md";
    write_performance_report_hold_out(
        performance_report_path,
        report_meta, training_summary, performance,
        architecture, num_layers,
        static_cast<int>(train_indices.size()),
        static_cast<int>(test_indices.size())
    );
    cout << "Report performance salvato in: " << performance_report_path << endl;

    const fs::path model_snapshot_path = "trained_model_" + model_name + "_snapshot.txt";
    save_model_snapshot(architecture, num_layers, model_snapshot_path, class_names, hidden_activation, output_activation);
    std::cout << "Struttura e pesi salvati in: " << model_snapshot_path << std::endl;

    TrainingSnapshotMetadata training_snapshot = training_metadata::build_training_snapshot_metadata(
        model_name, 1,
        training_type, window, use_nesterov,
        momentum, target_loss, num_epochs,
        learning_rate_decay, loss,
        runtime_state,
        training_summary.stopped_early
    );
    training_snapshot.hold_out_ratio = train_ratio;
    training_snapshot.train_indices = train_indices;
    training_snapshot.test_indices = test_indices;
    training_snapshot.dataset_manifest_paths = dataset_manifest_paths;
    training_snapshot.shuffle_rng_state = training_shuffle::serialize_rng_state();
    const fs::path training_snapshot_path = "trained_model_" + model_name + "_training_snapshot.txt";
    save_training_snapshot(training_snapshot_path, architecture, num_layers, class_names, hidden_activation, output_activation, training_snapshot);
    std::cout << "Snapshot completo di training salvato in: " << training_snapshot_path << std::endl;
}

void k_fold(string model_name, int k_folds, int num_examples, int training_type, bool use_nesterov, LayerList &architecture, int num_layers, const Decay &learning_rate_decay, int num_epochs, float target_loss, const Dataset4D &input, const Dataset4D &output, const Loss &loss, const Activation &hidden_activation, const Activation &output_activation, float momentum, const std::vector<std::string> &class_names){
    vector<int> train_indices, test_indices;
    LayerList architecture_copy = architecture;
    vector<TrainingSummary> training_summary{};
    vector<TestPerformance> performance{};

    const int num_classes = output[0].height;
    const std::vector<std::vector<int>> folds = training_split::build_stratified_folds(output, num_classes, k_folds);

    int min_train_size = num_examples;
    for(const auto &fold : folds) min_train_size = std::min(min_train_size, num_examples - static_cast<int>(fold.size()));

    int window = choose_training_window(training_type, min_train_size);

    for(int k=0; k<k_folds; k++){
        architecture = architecture_copy;
        cout << "Inizio dell'apprendimento per il fold numero " << k+1 << endl;
        train_indices.clear();
        test_indices = folds[k];
        for(int j = 0; j < k_folds; j++){
            if(j == k) continue;
            train_indices.insert(train_indices.end(), folds[j].begin(), folds[j].end());
        }
        training_shuffle::shuffle_vec(train_indices);
        training_shuffle::shuffle_vec(test_indices);

        training_summary.push_back(train(
            training_type, use_nesterov, window,
            train_indices,
            architecture, num_layers,
            learning_rate_decay, num_epochs, target_loss,
            input, output,
            loss, hidden_activation, output_activation,
            momentum,
            nullptr
        ));
        cout << "Fine dell'apprendimento per il fold numero " << k+1 << ", inizio del test" << endl;
        performance.push_back(run_test(architecture, num_layers, test_indices, input, output, hidden_activation, output_activation));
    }

    ReportMetadata report_meta = training_metadata::build_report_metadata(
        model_name, "K-Fold",
        num_epochs, num_examples, num_classes,
        training_type, window, use_nesterov,
        momentum, target_loss,
        learning_rate_decay, loss,
        hidden_activation, output_activation
    );
    report_meta.k_folds = k_folds;
    const fs::path performance_report_path = "network_performance_report_" + model_name + ".md";
    write_performance_report_k_fold(performance_report_path, report_meta, training_summary, performance, architecture_copy, num_layers);
    cout << "Report performance salvato in: " << performance_report_path << endl;

    const fs::path model_snapshot_path = "trained_model_" + model_name + "_snapshot.txt";
    save_model_snapshot(architecture, num_layers, model_snapshot_path, class_names, hidden_activation, output_activation);
    std::cout << "Struttura e pesi salvati in: " << model_snapshot_path << std::endl;
    std::cout << "Snapshot di training (resume) disabilitato per K-fold in questa versione." << std::endl;

}

void full_training(std::string model_name, int num_examples, int training_type, bool use_nesterov, LayerList &architecture, int num_layers, const Decay &learning_rate_decay, int num_epochs, float target_loss, const Dataset4D &input, const Dataset4D &output, const Loss &loss, const Activation &hidden_activation, const Activation &output_activation, float momentum, const std::vector<std::string> &class_names, const std::vector<std::string> &dataset_manifest_paths){
    vector<int> train_indices(num_examples);
    iota(train_indices.begin(), train_indices.end(), 0);
    int window = choose_training_window(training_type, num_examples);
    TrainingRuntimeState runtime_state{};
    TrainingSummary training_summary = train(
        training_type, use_nesterov, window,
        train_indices,
        architecture, num_layers,
        learning_rate_decay, num_epochs, target_loss,
        input, output,
        loss, hidden_activation, output_activation,
        momentum,
        &runtime_state
    );

    const fs::path model_snapshot_path = "trained_model_" + model_name + "_snapshot.txt";
    save_model_snapshot(architecture, num_layers, model_snapshot_path, class_names, hidden_activation, output_activation);
    std::cout << "Struttura e pesi salvati in: " << model_snapshot_path << std::endl;

    TrainingSnapshotMetadata training_snapshot = training_metadata::build_training_snapshot_metadata(
        model_name, 3,
        training_type, window, use_nesterov,
        momentum, target_loss, num_epochs,
        learning_rate_decay, loss,
        runtime_state,
        training_summary.stopped_early
    );
    training_snapshot.train_indices = train_indices;
    training_snapshot.dataset_manifest_paths = dataset_manifest_paths;
    training_snapshot.shuffle_rng_state = training_shuffle::serialize_rng_state();
    const fs::path training_snapshot_path = "trained_model_" + model_name + "_training_snapshot.txt";
    save_training_snapshot(training_snapshot_path, architecture, num_layers, class_names, hidden_activation, output_activation, training_snapshot);
    std::cout << "Snapshot completo di training salvato in: " << training_snapshot_path << std::endl;

}

void hold_out_resume(std::string model_name, int num_examples, LayerList &architecture, int num_layers, const Decay &learning_rate_decay, int target_total_epochs, const Dataset4D &input, const Dataset4D &output, const Loss &loss, const Activation &hidden_activation, const Activation &output_activation, const std::vector<std::string> &class_names, const TrainingSnapshotMetadata &resume_snapshot, const std::vector<std::string> &dataset_manifest_paths){
    require_condition(!resume_snapshot.train_indices.empty(), "Resume hold-out non valido: train_indices assenti nello snapshot");
    require_condition(!resume_snapshot.test_indices.empty(), "Resume hold-out non valido: test_indices assenti nello snapshot");
    require_condition(
        target_total_epochs > resume_snapshot.completed_epochs,
        "Resume hold-out non valido: target_total_epochs deve essere maggiore delle epoche gia' completate"
    );

    std::vector<int> train_indices = resume_snapshot.train_indices;
    std::vector<int> test_indices = resume_snapshot.test_indices;
    if(!resume_snapshot.shuffle_rng_state.empty()){
        training_shuffle::restore_rng_state(resume_snapshot.shuffle_rng_state);
    }

    TrainingRuntimeState runtime_state{};
    runtime_state.completed_epochs = resume_snapshot.completed_epochs;
    runtime_state.optimizer_steps = resume_snapshot.optimizer_steps;
    runtime_state.velocity = resume_snapshot.optimizer_velocity;

    const int training_type = resume_snapshot.training_type;
    const int window = resume_snapshot.training_window;
    const bool use_nesterov = resume_snapshot.use_nesterov;
    const float momentum = resume_snapshot.momentum;
    const float target_loss = resume_snapshot.target_loss;
    const float train_ratio = resume_snapshot.hold_out_ratio;

    TrainingSummary training_summary = train(
        training_type, use_nesterov, window,
        train_indices,
        architecture, num_layers,
        learning_rate_decay, target_total_epochs, target_loss,
        input, output,
        loss, hidden_activation, output_activation,
        momentum,
        &runtime_state
    );
    TestPerformance performance = run_test(architecture, num_layers, test_indices, input, output, hidden_activation, output_activation);

    const int num_classes = output[0].height;
    ReportMetadata report_meta = training_metadata::build_report_metadata(
        model_name, "Hold-out (resume)",
        target_total_epochs, num_examples, num_classes,
        training_type, window, use_nesterov,
        momentum, target_loss,
        learning_rate_decay, loss,
        hidden_activation, output_activation
    );
    report_meta.hold_out_ratio = train_ratio;
    const fs::path performance_report_path = "network_performance_report_" + model_name + ".md";
    write_performance_report_hold_out(
        performance_report_path,
        report_meta, training_summary, performance,
        architecture, num_layers,
        static_cast<int>(train_indices.size()),
        static_cast<int>(test_indices.size())
    );
    cout << "Report performance salvato in: " << performance_report_path << endl;

    const fs::path model_snapshot_path = "trained_model_" + model_name + "_snapshot.txt";
    save_model_snapshot(architecture, num_layers, model_snapshot_path, class_names, hidden_activation, output_activation);
    std::cout << "Struttura e pesi salvati in: " << model_snapshot_path << std::endl;

    TrainingSnapshotMetadata training_snapshot = training_metadata::build_training_snapshot_metadata(
        model_name, 1,
        training_type, window, use_nesterov,
        momentum, target_loss, target_total_epochs,
        learning_rate_decay, loss,
        runtime_state,
        training_summary.stopped_early
    );
    training_snapshot.hold_out_ratio = train_ratio;
    training_snapshot.train_indices = train_indices;
    training_snapshot.test_indices = test_indices;
    training_snapshot.dataset_manifest_paths = dataset_manifest_paths;
    training_snapshot.shuffle_rng_state = training_shuffle::serialize_rng_state();
    const fs::path training_snapshot_path = "trained_model_" + model_name + "_training_snapshot.txt";
    save_training_snapshot(training_snapshot_path, architecture, num_layers, class_names, hidden_activation, output_activation, training_snapshot);
    std::cout << "Snapshot completo di training salvato in: " << training_snapshot_path << std::endl;
}

void full_training_resume(std::string model_name, int num_examples, LayerList &architecture, int num_layers, const Decay &learning_rate_decay, int target_total_epochs, const Dataset4D &input, const Dataset4D &output, const Loss &loss, const Activation &hidden_activation, const Activation &output_activation, const std::vector<std::string> &class_names, const TrainingSnapshotMetadata &resume_snapshot, const std::vector<std::string> &dataset_manifest_paths){
    std::vector<int> train_indices;
    if(resume_snapshot.train_indices.empty()){
        train_indices.resize(num_examples);
        iota(train_indices.begin(), train_indices.end(), 0);
    } else {
        train_indices = resume_snapshot.train_indices;
    }

    require_condition(
        target_total_epochs > resume_snapshot.completed_epochs,
        "Resume full-training non valido: target_total_epochs deve essere maggiore delle epoche gia' completate"
    );
    if(!resume_snapshot.shuffle_rng_state.empty()){
        training_shuffle::restore_rng_state(resume_snapshot.shuffle_rng_state);
    }

    TrainingRuntimeState runtime_state{};
    runtime_state.completed_epochs = resume_snapshot.completed_epochs;
    runtime_state.optimizer_steps = resume_snapshot.optimizer_steps;
    runtime_state.velocity = resume_snapshot.optimizer_velocity;

    const int training_type = resume_snapshot.training_type;
    const int window = resume_snapshot.training_window;
    const bool use_nesterov = resume_snapshot.use_nesterov;
    const float momentum = resume_snapshot.momentum;
    const float target_loss = resume_snapshot.target_loss;

    TrainingSummary training_summary = train(
        training_type, use_nesterov, window,
        train_indices,
        architecture, num_layers,
        learning_rate_decay, target_total_epochs, target_loss,
        input, output,
        loss, hidden_activation, output_activation,
        momentum,
        &runtime_state
    );

    const fs::path model_snapshot_path = "trained_model_" + model_name + "_snapshot.txt";
    save_model_snapshot(architecture, num_layers, model_snapshot_path, class_names, hidden_activation, output_activation);
    std::cout << "Struttura e pesi salvati in: " << model_snapshot_path << std::endl;

    TrainingSnapshotMetadata training_snapshot = training_metadata::build_training_snapshot_metadata(
        model_name, 3,
        training_type, window, use_nesterov,
        momentum, target_loss, target_total_epochs,
        learning_rate_decay, loss,
        runtime_state,
        training_summary.stopped_early
    );
    training_snapshot.train_indices = train_indices;
    training_snapshot.dataset_manifest_paths = dataset_manifest_paths;
    training_snapshot.shuffle_rng_state = training_shuffle::serialize_rng_state();
    const fs::path training_snapshot_path = "trained_model_" + model_name + "_training_snapshot.txt";
    save_training_snapshot(training_snapshot_path, architecture, num_layers, class_names, hidden_activation, output_activation, training_snapshot);
    std::cout << "Snapshot completo di training salvato in: " << training_snapshot_path << std::endl;
}
