#include "training/training.hpp"

// Questo file contiene le routine pubbliche di training batch, SGD e Nesterov.

#include "cli/cli_utils.hpp"
#include "training/shuffle_rng.hpp"
#include "training/training_progress.hpp"
#include "training/training_step.hpp"

#include "engine/optimize.hpp"
#include "core/cuda_backend.hpp"
#include "core/layer_validation.hpp"
#include "evaluation/evaluation.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>

using namespace std;

namespace {

double elapsed_seconds(const std::chrono::steady_clock::time_point &start){
    const auto elapsed = std::chrono::steady_clock::now() - start;
    return std::chrono::duration_cast<std::chrono::duration<double>>(elapsed).count();
}

void validate_loss_output_compatibility(const Layer &output_layer, const Loss &cost, const Activation &output_activation){
    if(output_layer.type == Layer_type::Softmax){
        require_condition(
            cost.supports_softmax_output(),
            "La loss selezionata richiede che il layer di output Softmax sia usato con una loss compatibile."
        );
        return;
    }

    require_condition(
        !cost.supports_softmax_output(),
        "La loss selezionata richiede che il layer di output finale sia Softmax."
    );

    if(cost.requires_unit_interval_output()){
        require_condition(
            output_activation.outputs_in_unit_interval(),
            "La loss selezionata richiede che il layer di output usi una funzione di attivazione con output nell'intervallo [0, 1]."
        );
    }
}

void validate_training_setup(const LayerList &architecture, int num_layers, const std::vector<int> &indices, const LazyDataset &dataset, const Loss &loss, const Activation &output_activation, const std::string &context){
    validate_architecture(architecture, num_layers, context);
    validate_dataset_indices_io_shapes(architecture, num_layers, dataset, indices, context);
    validate_loss_output_compatibility(architecture[num_layers - 1], loss, output_activation);
}

int resolve_start_epoch(const TrainingRuntimeState *runtime_state, int max_epochs){
    if(runtime_state == nullptr){
        return 0;
    }
    return std::clamp(runtime_state->completed_epochs, 0, std::max(0, max_epochs));
}

std::int64_t resolve_optimizer_steps(const TrainingRuntimeState *runtime_state){
    if(runtime_state == nullptr){
        return 0;
    }
    return std::max<std::int64_t>(0, runtime_state->optimizer_steps);
}


void store_runtime_state(TrainingRuntimeState *runtime_state, int completed_epochs, std::int64_t optimizer_steps, const ParameterBuffer &velocity, const training_progress::ValidationTracker &validation_tracker){
    if(runtime_state == nullptr){
        return;
    }
    runtime_state->completed_epochs = completed_epochs;
    runtime_state->optimizer_steps = optimizer_steps;
    runtime_state->velocity = velocity;
    runtime_state->validation_observed = validation_tracker.observed;
    runtime_state->best_validation_accuracy = validation_tracker.best_accuracy;
    runtime_state->best_validation_epoch = validation_tracker.best_epoch;
    runtime_state->epochs_without_significant_improvement = validation_tracker.epochs_without_significant_improvement;
}

void store_cuda_runtime_state(const LayerList &architecture, TrainingRuntimeState *runtime_state, int completed_epochs, std::int64_t optimizer_steps, const cuda_backend::CudaParameterBuffer &velocity, const training_progress::ValidationTracker &validation_tracker){
    if(runtime_state == nullptr){
        return;
    }
    ParameterBuffer host_velocity;
    cuda_backend::sync_cuda_parameters_to_cpu(architecture, velocity, host_velocity);
    store_runtime_state(runtime_state, completed_epochs, optimizer_steps, host_velocity, validation_tracker);
}

bool validation_enabled(const std::vector<int> *validation_indices, const EarlyStoppingConfig *early_stopping){
    return early_stopping != nullptr &&
           early_stopping->enabled &&
           validation_indices != nullptr &&
           !validation_indices->empty();
}

bool has_significant_relative_improvement(float previous_best_accuracy, float current_accuracy, float threshold){
    if(previous_best_accuracy <= 0.0f){
        return current_accuracy > previous_best_accuracy;
    }

    const float relative_improvement = (current_accuracy - previous_best_accuracy) / previous_best_accuracy;
    return relative_improvement > threshold;
}

bool update_validation_tracker(
    LayerList &architecture,
    int num_layers,
    const cuda_backend::CudaParameterBuffer &parameters,
    cuda_backend::CudaBatchRuntimeList &validation_runtime,
    const std::vector<int> *validation_indices,
    const LazyDataset &dataset,
    const Activation &hidden_activation,
    const Activation &output_activation,
    const EarlyStoppingConfig *early_stopping,
    int completed_epoch,
    training_progress::ValidationTracker &tracker
){
    if(!validation_enabled(validation_indices, early_stopping)){
        return false;
    }

    tracker.enabled = true;
    const float previous_best_accuracy = tracker.best_accuracy;
    const bool first_observation = !tracker.observed;
    const float validation_accuracy = run_validation_accuracy(
        architecture, num_layers,
        parameters, validation_runtime,
        *validation_indices, dataset,
        hidden_activation, output_activation,
        early_stopping->validation_batch_size
    );

    const bool improved = first_observation || validation_accuracy > previous_best_accuracy;
    const bool significant = first_observation ||
        has_significant_relative_improvement(
            previous_best_accuracy,
            validation_accuracy,
            early_stopping->relative_delta_threshold
        );

    if(improved){
        tracker.best_accuracy = validation_accuracy;
        tracker.best_epoch = completed_epoch;
    }
    tracker.observed = true;

    if(significant){
        tracker.epochs_without_significant_improvement = 0;
    } else {
        tracker.epochs_without_significant_improvement++;
    }

    std::cout << "Validation accuracy epoca " << completed_epoch << ": " << validation_accuracy
              << " | best=" << tracker.best_accuracy << std::endl;
    return improved;
}

training_progress::ValidationTracker init_validation_tracker(const TrainingRuntimeState *runtime_state, bool enabled){
    training_progress::ValidationTracker tracker{};
    tracker.enabled = enabled;
    if(runtime_state == nullptr || !runtime_state->validation_observed){
        return tracker;
    }

    tracker.observed = true;
    tracker.best_accuracy = runtime_state->best_validation_accuracy;
    tracker.best_epoch = runtime_state->best_validation_epoch;
    tracker.epochs_without_significant_improvement = runtime_state->epochs_without_significant_improvement;
    return tracker;
}

void store_best_parameters_if_improved(
    bool improved,
    const LayerList &architecture,
    const cuda_backend::CudaParameterBuffer &parameters,
    const cuda_backend::CudaParameterBuffer &velocity,
    cuda_backend::CudaParameterBuffer &best_parameters,
    cuda_backend::CudaParameterBuffer &best_velocity
){
    if(!improved){
        return;
    }

    cuda_backend::copy_cuda_parameter_buffer(architecture, parameters, best_parameters);
    cuda_backend::copy_cuda_parameter_buffer(architecture, velocity, best_velocity);
}

void init_best_parameters_from_current_if_observed(
    const LayerList &architecture,
    const training_progress::ValidationTracker &tracker,
    const cuda_backend::CudaParameterBuffer &parameters,
    const cuda_backend::CudaParameterBuffer &velocity,
    cuda_backend::CudaParameterBuffer &best_parameters,
    cuda_backend::CudaParameterBuffer &best_velocity
){
    if(!tracker.observed){
        return;
    }

    cuda_backend::copy_cuda_parameter_buffer(architecture, parameters, best_parameters);
    cuda_backend::copy_cuda_parameter_buffer(architecture, velocity, best_velocity);
}

void restore_best_parameters_if_available(
    LayerList &architecture,
    const training_progress::ValidationTracker &tracker,
    const cuda_backend::CudaParameterBuffer &best_parameters,
    const cuda_backend::CudaParameterBuffer &best_velocity,
    cuda_backend::CudaParameterBuffer &parameters,
    cuda_backend::CudaParameterBuffer &velocity,
    int current_completed_epoch,
    bool current_parameters_match_completed_epoch
){
    if(!tracker.enabled || !tracker.observed){
        return;
    }
    if(current_parameters_match_completed_epoch && tracker.best_epoch == current_completed_epoch){
        return;
    }

    cuda_backend::copy_cuda_parameter_buffer(architecture, best_parameters, parameters);
    cuda_backend::copy_cuda_parameter_buffer(architecture, best_velocity, velocity);
}

bool should_stop_for_patience(const EarlyStoppingConfig *early_stopping, const training_progress::ValidationTracker &tracker){
    if(early_stopping == nullptr || !early_stopping->enabled || !tracker.enabled){
        return false;
    }
    if(early_stopping->patience <= 0){
        return false;
    }
    return tracker.epochs_without_significant_improvement >= early_stopping->patience;
}

bool update_loss_and_check_stop_without_completing_epoch(float loss_value, float target_loss, float &last_loss){
    last_loss = loss_value;
    return training_progress::should_stop_training(target_loss, loss_value);
}

}

namespace training_iteration {

template <typename Fn>
void for_each_batch(int num_tr, int batch_size, Fn &&fn){
    const int batches = static_cast<int>(std::ceil(static_cast<float>(num_tr) / batch_size));
    for(int b = 0; b < batches; b++){
        const int start = b * batch_size;
        const int end = (b == batches - 1) ? num_tr : (b + 1) * batch_size;
        const int size = end - start;
        fn(start, end, size);
    }
}

} // namespace training_iteration

TrainingSummary train_batch(
    LayerList &architecture, int num_layers, int chunk_size,
    const Decay &learning_rate_decay,
    int num_epochs, float target_loss, vector<int> &train_indices, const LazyDataset &dataset, const Loss &loss, const Activation &hidden_activation, const Activation &output_activation, float momentum, cuda_backend::CudaParameterBuffer &parameters, const std::vector<int> *validation_indices, const EarlyStoppingConfig *early_stopping, TrainingRuntimeState *runtime_state){
    training_progress::print_training_banner();
    validate_training_setup(architecture, num_layers, train_indices, dataset, loss, output_activation, "train_batch");
    int num_tr = static_cast<int>(train_indices.size());
    float last_epoch_loss = std::numeric_limits<float>::quiet_NaN();
    int executed_epochs = 0;
    bool stopped_by_loss = false;
    bool stopped_by_validation = false;
    std::vector<double> epoch_times_seconds;
    epoch_times_seconds.reserve(std::max(0, num_epochs));
    const auto training_start = std::chrono::steady_clock::now();

    const int effective_chunk_size = std::clamp(chunk_size, 1, num_tr);
    cuda_backend::CudaParameterBuffer gradients;
    cuda_backend::CudaParameterBuffer chunk_gradients;
    cuda_backend::CudaParameterBuffer velocity;
    cuda_backend::CudaParameterBuffer best_parameters;
    cuda_backend::CudaParameterBuffer best_velocity;
    cuda_backend::CudaBatchRuntimeList runtime;
    cuda_backend::CudaBatchRuntimeList validation_runtime;
    cuda_backend::CudaBatchTensor<float> target_batch;
    training_progress::ValidationTracker validation_tracker = init_validation_tracker(runtime_state, validation_enabled(validation_indices, early_stopping));
    cuda_backend::init_cuda_batch_runtime_buffers(architecture, runtime, effective_chunk_size);
    cuda_backend::init_cuda_parameter_buffer(architecture, gradients);
    cuda_backend::init_cuda_parameter_buffer(architecture, chunk_gradients);
    cuda_backend::init_or_load_velocity(architecture, velocity, runtime_state);
    init_best_parameters_from_current_if_observed(architecture, validation_tracker, parameters, velocity, best_parameters, best_velocity);
    const int start_epoch = resolve_start_epoch(runtime_state, num_epochs);
    executed_epochs = start_epoch;
    std::int64_t optimizer_steps = resolve_optimizer_steps(runtime_state);

    for(int e=start_epoch; e<num_epochs; e++){
        const auto epoch_start = std::chrono::steady_clock::now();
        const float current_learning_rate = learning_rate_decay.fn(e);
        training_progress::print_epoch_start(e, num_epochs);
        training_shuffle::shuffle_vec(train_indices);

        float loss_value = 0.0f;
        cuda_backend::zero_cuda_parameter_buffer(architecture, gradients);
        training_iteration::for_each_batch(num_tr, effective_chunk_size, [&](int start, int end, int size){
            loss_value += training_batches::train_batch_chunk(
                architecture, parameters, runtime, num_layers,
                chunk_gradients, target_batch,
                dataset, train_indices,
                start, end,
                loss, hidden_activation, output_activation
            );

            const float gradient_scale = static_cast<float>(size) / static_cast<float>(num_tr);
            accumulate_scaled_gradients(
                architecture, num_layers,
                gradients, chunk_gradients,
                gradient_scale
            );
        });

        optimizer_step(
            architecture, num_layers,
            gradients, velocity, parameters,
            current_learning_rate, momentum
        );
        optimizer_steps++;
        loss_value /= num_tr;
        training_progress::print_epoch_progress(e, num_epochs, loss_value);
        const double epoch_seconds = elapsed_seconds(epoch_start);
        epoch_times_seconds.push_back(epoch_seconds);
        const bool validation_improved = update_validation_tracker(
            architecture, num_layers,
            parameters, validation_runtime,
            validation_indices, dataset,
            hidden_activation, output_activation,
            early_stopping, e + 1,
            validation_tracker
        );
        store_best_parameters_if_improved(validation_improved, architecture, parameters, velocity, best_parameters, best_velocity);

        stopped_by_loss = training_progress::update_progress_and_check_stop(loss_value, e, target_loss, last_epoch_loss, executed_epochs);
        stopped_by_validation = should_stop_for_patience(early_stopping, validation_tracker);
        if(stopped_by_loss || stopped_by_validation){
            break;
        }
    }

    const double total_training_seconds = elapsed_seconds(training_start);
    std::cout << "Tempo training totale: " << total_training_seconds << " s" << std::endl;
    cout << endl;
    restore_best_parameters_if_available(architecture, validation_tracker, best_parameters, best_velocity, parameters, velocity, executed_epochs, true);
    store_cuda_runtime_state(architecture, runtime_state, executed_epochs, optimizer_steps, velocity, validation_tracker);
    return training_progress::make_summary(executed_epochs, last_epoch_loss, epoch_times_seconds, total_training_seconds, stopped_by_loss, stopped_by_validation, validation_tracker);
}

TrainingSummary train_sgd(LayerList &architecture, int num_layers, int batch_size, const Decay &learning_rate_decay, int num_epochs, float target_loss, vector<int> &train_indices, const LazyDataset &dataset, const Loss &loss, const Activation &hidden_activation, const Activation &output_activation, float momentum, cuda_backend::CudaParameterBuffer &parameters, const std::vector<int> *validation_indices, const EarlyStoppingConfig *early_stopping, TrainingRuntimeState *runtime_state){
    training_progress::print_training_banner();
    validate_training_setup(architecture, num_layers, train_indices, dataset, loss, output_activation, "train_sgd");
    int num_tr = static_cast<int>(train_indices.size());
    float last_epoch_loss = std::numeric_limits<float>::quiet_NaN();
    int executed_epochs = 0;
    bool stopped_by_loss = false;
    bool stopped_by_validation = false;
    std::vector<double> epoch_times_seconds;
    epoch_times_seconds.reserve(std::max(0, num_epochs));
    const auto training_start = std::chrono::steady_clock::now();

    cuda_backend::CudaParameterBuffer gradients;
    cuda_backend::CudaParameterBuffer velocity;
    cuda_backend::CudaParameterBuffer best_parameters;
    cuda_backend::CudaParameterBuffer best_velocity;
    cuda_backend::CudaBatchRuntimeList runtime;
    cuda_backend::CudaBatchRuntimeList validation_runtime;
    cuda_backend::CudaBatchTensor<float> target_batch;
    training_progress::ValidationTracker validation_tracker = init_validation_tracker(runtime_state, validation_enabled(validation_indices, early_stopping));
    cuda_backend::init_cuda_batch_runtime_buffers(architecture, runtime, batch_size);
    cuda_backend::init_cuda_parameter_buffer(architecture, gradients);
    cuda_backend::init_or_load_velocity(architecture, velocity, runtime_state);
    init_best_parameters_from_current_if_observed(architecture, validation_tracker, parameters, velocity, best_parameters, best_velocity);
    const int start_epoch = resolve_start_epoch(runtime_state, num_epochs);
    executed_epochs = start_epoch;
    std::int64_t optimizer_steps = resolve_optimizer_steps(runtime_state);

    for(int e=start_epoch; e<num_epochs; e++){
        const auto epoch_start = std::chrono::steady_clock::now();
        const float current_learning_rate = learning_rate_decay.fn(e);
        training_progress::print_epoch_start(e, num_epochs);
        training_shuffle::shuffle_vec(train_indices);
        float loss_value = 0.0f;

        training_iteration::for_each_batch(num_tr, batch_size, [&](int start, int end, int){
            loss_value += training_batches::train_batch_chunk(
                architecture, parameters, runtime, num_layers,
                gradients, target_batch,
                dataset, train_indices,
                start, end,
                loss, hidden_activation, output_activation
            );

            optimizer_step(
                architecture, num_layers,
                gradients, velocity, parameters,
                current_learning_rate, momentum
            );
            optimizer_steps++;
        });

        loss_value /= num_tr;
        training_progress::print_epoch_progress(e, num_epochs, loss_value);
        const double epoch_seconds = elapsed_seconds(epoch_start);
        epoch_times_seconds.push_back(epoch_seconds);
        const bool validation_improved = update_validation_tracker(
            architecture, num_layers,
            parameters, validation_runtime,
            validation_indices, dataset,
            hidden_activation, output_activation,
            early_stopping, e + 1,
            validation_tracker
        );
        store_best_parameters_if_improved(validation_improved, architecture, parameters, velocity, best_parameters, best_velocity);
        stopped_by_loss = training_progress::update_progress_and_check_stop(loss_value, e, target_loss, last_epoch_loss, executed_epochs);
        stopped_by_validation = should_stop_for_patience(early_stopping, validation_tracker);
        if(stopped_by_loss || stopped_by_validation){
            break;
        }
    }

    const double total_training_seconds = elapsed_seconds(training_start);
    std::cout << "Tempo training totale: " << total_training_seconds << " s" << std::endl;
    cout << endl;
    restore_best_parameters_if_available(architecture, validation_tracker, best_parameters, best_velocity, parameters, velocity, executed_epochs, true);
    store_cuda_runtime_state(architecture, runtime_state, executed_epochs, optimizer_steps, velocity, validation_tracker);
    return training_progress::make_summary(executed_epochs, last_epoch_loss, epoch_times_seconds, total_training_seconds, stopped_by_loss, stopped_by_validation, validation_tracker);
}

TrainingSummary train_sgd_online(LayerList &architecture, int num_layers, int steps, const Decay &learning_rate_decay, int num_epochs, float target_loss, vector<int> &train_indices, const LazyDataset &dataset, const Loss &loss, const Activation &hidden_activation, const Activation &output_activation, float momentum, cuda_backend::CudaParameterBuffer &parameters, const std::vector<int> *validation_indices, const EarlyStoppingConfig *early_stopping, TrainingRuntimeState *runtime_state){
    training_progress::print_training_banner();
    validate_training_setup(architecture, num_layers, train_indices, dataset, loss, output_activation, "train_sgd_online");
    int num_tr = static_cast<int>(train_indices.size());
    const int windows = static_cast<int>(std::ceil(static_cast<float>(num_tr) / steps));
    float last_observed_loss = std::numeric_limits<float>::quiet_NaN();
    int executed_epochs = 0;
    bool stopped_by_loss = false;
    bool stopped_by_validation = false;
    std::vector<double> epoch_times_seconds;
    epoch_times_seconds.reserve(std::max(0, num_epochs));
    const auto training_start = std::chrono::steady_clock::now();

    cuda_backend::CudaParameterBuffer gradients;
    cuda_backend::CudaParameterBuffer velocity;
    cuda_backend::CudaParameterBuffer best_parameters;
    cuda_backend::CudaParameterBuffer best_velocity;
    cuda_backend::CudaBatchRuntimeList runtime;
    cuda_backend::CudaBatchRuntimeList validation_runtime;
    cuda_backend::CudaBatchTensor<float> target_batch;
    training_progress::ValidationTracker validation_tracker = init_validation_tracker(runtime_state, validation_enabled(validation_indices, early_stopping));
    cuda_backend::init_cuda_batch_runtime_buffers(architecture, runtime, 1);
    cuda_backend::init_cuda_parameter_buffer(architecture, gradients);
    cuda_backend::init_or_load_velocity(architecture, velocity, runtime_state);
    init_best_parameters_from_current_if_observed(architecture, validation_tracker, parameters, velocity, best_parameters, best_velocity);
    const int start_epoch = resolve_start_epoch(runtime_state, num_epochs);
    executed_epochs = start_epoch;
    std::int64_t optimizer_steps = resolve_optimizer_steps(runtime_state);

    for(int e=start_epoch; e<num_epochs; e++){
        const auto epoch_start = std::chrono::steady_clock::now();
        const float current_learning_rate = learning_rate_decay.fn(e);
        training_progress::print_epoch_start(e, num_epochs);
        training_shuffle::shuffle_vec(train_indices);
        for(int s=0; s<windows; s++){
            float loss_value = 0.0f;
            const int start = s * steps;
            const int end = (s == windows - 1) ? num_tr : (s + 1) * steps;

            for(int t=start; t<end; t++){
                loss_value += training_batches::train_batch_chunk(
                    architecture, parameters, runtime, num_layers,
                    gradients, target_batch,
                    dataset, train_indices,
                    t, t + 1,
                    loss, hidden_activation, output_activation
                );
                optimizer_step(
                    architecture, num_layers,
                    gradients, velocity, parameters,
                    current_learning_rate, momentum
                );
                optimizer_steps++;
            }

            loss_value /= (end - start);
            training_progress::print_epoch_progress(e, num_epochs, loss_value);
            if(update_loss_and_check_stop_without_completing_epoch(loss_value, target_loss, last_observed_loss)){
                stopped_by_loss = true;
                const double epoch_seconds = elapsed_seconds(epoch_start);
                epoch_times_seconds.push_back(epoch_seconds);
                const double total_training_seconds = elapsed_seconds(training_start);
                std::cout << "Tempo training totale: " << total_training_seconds << " s" << std::endl;
                restore_best_parameters_if_available(architecture, validation_tracker, best_parameters, best_velocity, parameters, velocity, executed_epochs, false);
                store_cuda_runtime_state(architecture, runtime_state, executed_epochs, optimizer_steps, velocity, validation_tracker);
                return training_progress::make_summary(executed_epochs, last_observed_loss, epoch_times_seconds, total_training_seconds, stopped_by_loss, stopped_by_validation, validation_tracker);
            }
        }

        const double epoch_seconds = elapsed_seconds(epoch_start);
        epoch_times_seconds.push_back(epoch_seconds);
        executed_epochs = e + 1;
        const bool validation_improved = update_validation_tracker(
            architecture, num_layers,
            parameters, validation_runtime,
            validation_indices, dataset,
            hidden_activation, output_activation,
            early_stopping, executed_epochs,
            validation_tracker
        );
        store_best_parameters_if_improved(validation_improved, architecture, parameters, velocity, best_parameters, best_velocity);
        stopped_by_validation = should_stop_for_patience(early_stopping, validation_tracker);
        if(stopped_by_validation){
            break;
        }
    }

    const double total_training_seconds = elapsed_seconds(training_start);
    std::cout << "Tempo training totale: " << total_training_seconds << " s" << std::endl;
    cout << endl;
    restore_best_parameters_if_available(architecture, validation_tracker, best_parameters, best_velocity, parameters, velocity, executed_epochs, true);
    store_cuda_runtime_state(architecture, runtime_state, executed_epochs, optimizer_steps, velocity, validation_tracker);
    return training_progress::make_summary(executed_epochs, last_observed_loss, epoch_times_seconds, total_training_seconds, stopped_by_loss, stopped_by_validation, validation_tracker);
}

TrainingSummary train_batch_nesterov(LayerList &architecture, int num_layers, int chunk_size, const Decay &learning_rate_decay, int num_epochs, float target_loss, vector<int> &train_indices, const LazyDataset &dataset, const Loss &loss, const Activation &hidden_activation, const Activation &output_activation, float momentum, cuda_backend::CudaParameterBuffer &parameters, const std::vector<int> *validation_indices, const EarlyStoppingConfig *early_stopping, TrainingRuntimeState *runtime_state){
    training_progress::print_training_banner();
    validate_training_setup(architecture, num_layers, train_indices, dataset, loss, output_activation, "train_batch_nesterov");
    int num_tr = static_cast<int>(train_indices.size());
    float last_epoch_loss = std::numeric_limits<float>::quiet_NaN();
    int executed_epochs = 0;
    bool stopped_by_loss = false;
    bool stopped_by_validation = false;
    std::vector<double> epoch_times_seconds;
    epoch_times_seconds.reserve(std::max(0, num_epochs));
    const auto training_start = std::chrono::steady_clock::now();

    const int effective_chunk_size = std::clamp(chunk_size, 1, num_tr);
    cuda_backend::CudaParameterBuffer gradients;
    cuda_backend::CudaParameterBuffer chunk_gradients;
    cuda_backend::CudaParameterBuffer velocity;
    cuda_backend::CudaParameterBuffer best_parameters;
    cuda_backend::CudaParameterBuffer best_velocity;
    cuda_backend::CudaBatchRuntimeList runtime;
    cuda_backend::CudaBatchRuntimeList validation_runtime;
    cuda_backend::CudaBatchTensor<float> target_batch;
    training_progress::ValidationTracker validation_tracker = init_validation_tracker(runtime_state, validation_enabled(validation_indices, early_stopping));
    cuda_backend::init_cuda_batch_runtime_buffers(architecture, runtime, effective_chunk_size);
    cuda_backend::init_cuda_parameter_buffer(architecture, gradients);
    cuda_backend::init_cuda_parameter_buffer(architecture, chunk_gradients);
    cuda_backend::init_or_load_velocity(architecture, velocity, runtime_state);
    init_best_parameters_from_current_if_observed(architecture, validation_tracker, parameters, velocity, best_parameters, best_velocity);
    const int start_epoch = resolve_start_epoch(runtime_state, num_epochs);
    executed_epochs = start_epoch;
    std::int64_t optimizer_steps = resolve_optimizer_steps(runtime_state);

    for(int e=start_epoch; e<num_epochs; e++){
        const auto epoch_start = std::chrono::steady_clock::now();
        const float current_learning_rate = learning_rate_decay.fn(e);
        training_progress::print_epoch_start(e, num_epochs);
        training_shuffle::shuffle_vec(train_indices);
        float loss_value = 0.0f;
        cuda_backend::zero_cuda_parameter_buffer(architecture, gradients);
        training_iteration::for_each_batch(num_tr, effective_chunk_size, [&](int start, int end, int size){
            loss_value += training_batches::train_batch_chunk_nesterov(
                architecture, parameters, runtime, num_layers,
                chunk_gradients, target_batch,
                dataset, train_indices,
                start, end,
                loss, hidden_activation, output_activation,
                &velocity, momentum
            );

            const float gradient_scale = static_cast<float>(size) / static_cast<float>(num_tr);
            accumulate_scaled_gradients(
                architecture, num_layers,
                gradients, chunk_gradients,
                gradient_scale
            );
        });

        optimizer_step(
            architecture, num_layers,
            gradients, velocity, parameters,
            current_learning_rate, momentum
        );
        optimizer_steps++;
        loss_value /= num_tr;
        training_progress::print_epoch_progress(e, num_epochs, loss_value);
        const double epoch_seconds = elapsed_seconds(epoch_start);
        epoch_times_seconds.push_back(epoch_seconds);
        const bool validation_improved = update_validation_tracker(
            architecture, num_layers,
            parameters, validation_runtime,
            validation_indices, dataset,
            hidden_activation, output_activation,
            early_stopping, e + 1,
            validation_tracker
        );
        store_best_parameters_if_improved(validation_improved, architecture, parameters, velocity, best_parameters, best_velocity);

        stopped_by_loss = training_progress::update_progress_and_check_stop(loss_value, e, target_loss, last_epoch_loss, executed_epochs);
        stopped_by_validation = should_stop_for_patience(early_stopping, validation_tracker);
        if(stopped_by_loss || stopped_by_validation){
            break;
        }
    }

    const double total_training_seconds = elapsed_seconds(training_start);
    std::cout << "Tempo training totale: " << total_training_seconds << " s" << std::endl;
    cout << endl;
    restore_best_parameters_if_available(architecture, validation_tracker, best_parameters, best_velocity, parameters, velocity, executed_epochs, true);
    store_cuda_runtime_state(architecture, runtime_state, executed_epochs, optimizer_steps, velocity, validation_tracker);
    return training_progress::make_summary(executed_epochs, last_epoch_loss, epoch_times_seconds, total_training_seconds, stopped_by_loss, stopped_by_validation, validation_tracker);
}

TrainingSummary train_sgd_nesterov(LayerList &architecture, int num_layers, int batch_size, const Decay &learning_rate_decay, int num_epochs, float target_loss, vector<int> &train_indices, const LazyDataset &dataset, const Loss &loss, const Activation &hidden_activation, const Activation &output_activation, float momentum, cuda_backend::CudaParameterBuffer &parameters, const std::vector<int> *validation_indices, const EarlyStoppingConfig *early_stopping, TrainingRuntimeState *runtime_state){
    training_progress::print_training_banner();
    validate_training_setup(architecture, num_layers, train_indices, dataset, loss, output_activation, "train_sgd_nesterov");
    int num_tr = static_cast<int>(train_indices.size());
    float last_epoch_loss = std::numeric_limits<float>::quiet_NaN();
    int executed_epochs = 0;
    bool stopped_by_loss = false;
    bool stopped_by_validation = false;
    std::vector<double> epoch_times_seconds;
    epoch_times_seconds.reserve(std::max(0, num_epochs));
    const auto training_start = std::chrono::steady_clock::now();

    cuda_backend::CudaParameterBuffer gradients;
    cuda_backend::CudaParameterBuffer velocity;
    cuda_backend::CudaParameterBuffer best_parameters;
    cuda_backend::CudaParameterBuffer best_velocity;
    cuda_backend::CudaBatchRuntimeList runtime;
    cuda_backend::CudaBatchRuntimeList validation_runtime;
    cuda_backend::CudaBatchTensor<float> target_batch;
    training_progress::ValidationTracker validation_tracker = init_validation_tracker(runtime_state, validation_enabled(validation_indices, early_stopping));
    cuda_backend::init_cuda_batch_runtime_buffers(architecture, runtime, batch_size);
    cuda_backend::init_cuda_parameter_buffer(architecture, gradients);
    cuda_backend::init_or_load_velocity(architecture, velocity, runtime_state);
    init_best_parameters_from_current_if_observed(architecture, validation_tracker, parameters, velocity, best_parameters, best_velocity);
    const int start_epoch = resolve_start_epoch(runtime_state, num_epochs);
    executed_epochs = start_epoch;
    std::int64_t optimizer_steps = resolve_optimizer_steps(runtime_state);

    for(int e=start_epoch; e<num_epochs; e++){
        const auto epoch_start = std::chrono::steady_clock::now();
        const float current_learning_rate = learning_rate_decay.fn(e);
        training_progress::print_epoch_start(e, num_epochs);
        training_shuffle::shuffle_vec(train_indices);
        float loss_value = 0.0f;

        training_iteration::for_each_batch(num_tr, batch_size, [&](int start, int end, int){
            loss_value += training_batches::train_batch_chunk_nesterov(
                architecture, parameters, runtime, num_layers,
                gradients, target_batch,
                dataset, train_indices,
                start, end,
                loss, hidden_activation, output_activation,
                &velocity, momentum
            );

            optimizer_step(
                architecture, num_layers,
                gradients, velocity, parameters,
                current_learning_rate, momentum
            );
            optimizer_steps++;
        });

        loss_value /= num_tr;
        training_progress::print_epoch_progress(e, num_epochs, loss_value);
        const double epoch_seconds = elapsed_seconds(epoch_start);
        epoch_times_seconds.push_back(epoch_seconds);
        const bool validation_improved = update_validation_tracker(
            architecture, num_layers,
            parameters, validation_runtime,
            validation_indices, dataset,
            hidden_activation, output_activation,
            early_stopping, e + 1,
            validation_tracker
        );
        store_best_parameters_if_improved(validation_improved, architecture, parameters, velocity, best_parameters, best_velocity);
        stopped_by_loss = training_progress::update_progress_and_check_stop(loss_value, e, target_loss, last_epoch_loss, executed_epochs);
        stopped_by_validation = should_stop_for_patience(early_stopping, validation_tracker);
        if(stopped_by_loss || stopped_by_validation){
            break;
        }
    }

    const double total_training_seconds = elapsed_seconds(training_start);
    std::cout << "Tempo training totale: " << total_training_seconds << " s" << std::endl;
    cout << endl;
    restore_best_parameters_if_available(architecture, validation_tracker, best_parameters, best_velocity, parameters, velocity, executed_epochs, true);
    store_cuda_runtime_state(architecture, runtime_state, executed_epochs, optimizer_steps, velocity, validation_tracker);
    return training_progress::make_summary(executed_epochs, last_epoch_loss, epoch_times_seconds, total_training_seconds, stopped_by_loss, stopped_by_validation, validation_tracker);
}

TrainingSummary train_sgd_online_nesterov(LayerList &architecture, int num_layers, int steps, const Decay &learning_rate_decay, int num_epochs, float target_loss, vector<int> &train_indices, const LazyDataset &dataset, const Loss &loss, const Activation &hidden_activation, const Activation &output_activation, float momentum, cuda_backend::CudaParameterBuffer &parameters, const std::vector<int> *validation_indices, const EarlyStoppingConfig *early_stopping, TrainingRuntimeState *runtime_state){
    training_progress::print_training_banner();
    validate_training_setup(architecture, num_layers, train_indices, dataset, loss, output_activation, "train_sgd_online_nesterov");
    int num_tr = static_cast<int>(train_indices.size());
    const int windows = static_cast<int>(std::ceil(static_cast<float>(num_tr) / steps));
    float last_observed_loss = std::numeric_limits<float>::quiet_NaN();
    int executed_epochs = 0;
    bool stopped_by_loss = false;
    bool stopped_by_validation = false;
    std::vector<double> epoch_times_seconds;
    epoch_times_seconds.reserve(std::max(0, num_epochs));
    const auto training_start = std::chrono::steady_clock::now();

    cuda_backend::CudaParameterBuffer gradients;
    cuda_backend::CudaParameterBuffer velocity;
    cuda_backend::CudaParameterBuffer best_parameters;
    cuda_backend::CudaParameterBuffer best_velocity;
    cuda_backend::CudaBatchRuntimeList runtime;
    cuda_backend::CudaBatchRuntimeList validation_runtime;
    cuda_backend::CudaBatchTensor<float> target_batch;
    training_progress::ValidationTracker validation_tracker = init_validation_tracker(runtime_state, validation_enabled(validation_indices, early_stopping));
    cuda_backend::init_cuda_batch_runtime_buffers(architecture, runtime, 1);
    cuda_backend::init_cuda_parameter_buffer(architecture, gradients);
    cuda_backend::init_or_load_velocity(architecture, velocity, runtime_state);
    init_best_parameters_from_current_if_observed(architecture, validation_tracker, parameters, velocity, best_parameters, best_velocity);
    const int start_epoch = resolve_start_epoch(runtime_state, num_epochs);
    executed_epochs = start_epoch;
    std::int64_t optimizer_steps = resolve_optimizer_steps(runtime_state);

    for(int e=start_epoch; e<num_epochs; e++){
        const auto epoch_start = std::chrono::steady_clock::now();
        const float current_learning_rate = learning_rate_decay.fn(e);
        training_progress::print_epoch_start(e, num_epochs);
        training_shuffle::shuffle_vec(train_indices);
        for(int s=0; s<windows; s++){
            float loss_value = 0.0f;
            const int start = s * steps;
            const int end = (s == windows - 1) ? num_tr : (s + 1) * steps;

            for(int t=start; t<end; t++){
                loss_value += training_batches::train_batch_chunk_nesterov(
                    architecture, parameters, runtime, num_layers,
                    gradients, target_batch,
                    dataset, train_indices,
                    t, t + 1,
                    loss, hidden_activation, output_activation,
                    &velocity, momentum
                );
                optimizer_step(
                    architecture, num_layers,
                    gradients, velocity, parameters,
                    current_learning_rate, momentum
                );
                optimizer_steps++;
            }
            
            loss_value /= (end - start);
            training_progress::print_epoch_progress(e, num_epochs, loss_value);
            if(update_loss_and_check_stop_without_completing_epoch(loss_value, target_loss, last_observed_loss)){
                stopped_by_loss = true;
                const double epoch_seconds = elapsed_seconds(epoch_start);
                epoch_times_seconds.push_back(epoch_seconds);
                const double total_training_seconds = elapsed_seconds(training_start);
                std::cout << "Tempo training totale: " << total_training_seconds << " s" << std::endl;
                restore_best_parameters_if_available(architecture, validation_tracker, best_parameters, best_velocity, parameters, velocity, executed_epochs, false);
                store_cuda_runtime_state(architecture, runtime_state, executed_epochs, optimizer_steps, velocity, validation_tracker);
                return training_progress::make_summary(executed_epochs, last_observed_loss, epoch_times_seconds, total_training_seconds, stopped_by_loss, stopped_by_validation, validation_tracker);
            }
        }

        const double epoch_seconds = elapsed_seconds(epoch_start);
        epoch_times_seconds.push_back(epoch_seconds);
        executed_epochs = e + 1;
        const bool validation_improved = update_validation_tracker(
            architecture, num_layers,
            parameters, validation_runtime,
            validation_indices, dataset,
            hidden_activation, output_activation,
            early_stopping, executed_epochs,
            validation_tracker
        );
        store_best_parameters_if_improved(validation_improved, architecture, parameters, velocity, best_parameters, best_velocity);
        stopped_by_validation = should_stop_for_patience(early_stopping, validation_tracker);
        if(stopped_by_validation){
            break;
        }
    }

    const double total_training_seconds = elapsed_seconds(training_start);
    std::cout << "Tempo training totale: " << total_training_seconds << " s" << std::endl;
    cout << endl;
    restore_best_parameters_if_available(architecture, validation_tracker, best_parameters, best_velocity, parameters, velocity, executed_epochs, true);
    store_cuda_runtime_state(architecture, runtime_state, executed_epochs, optimizer_steps, velocity, validation_tracker);
    return training_progress::make_summary(executed_epochs, last_observed_loss, epoch_times_seconds, total_training_seconds, stopped_by_loss, stopped_by_validation, validation_tracker);
}
