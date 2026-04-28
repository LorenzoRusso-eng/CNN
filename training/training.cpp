#include "training/training.hpp"

// Questo file contiene le routine pubbliche di training batch, SGD e Nesterov.

#include "cli/cli_utils.hpp"
#include "engine/backward.hpp"
#include "engine/forward.hpp"
#include "training/optimizer.hpp"
#include "training/parameter_buffer.hpp"
#include "training/shuffle_rng.hpp"
#include "training/training_progress.hpp"
#include "training/training_step.hpp"

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

void validate_training_setup(const LayerList &architecture, int num_layers, const std::vector<int> &indices, const Dataset4D &input, const Dataset4D &output, const Loss &loss, const Activation &output_activation, const std::string &context){
    validate_architecture(architecture, num_layers, context);
    validate_dataset_indices_io_shapes(architecture, num_layers, input, output, indices, context);
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

void init_or_load_velocity(const LayerList &architecture, ParameterBuffer &velocity, const TrainingRuntimeState *runtime_state){
    training_buffers::init_parameter_buffer(architecture, velocity);
    if(runtime_state == nullptr){
        return;
    }
    if(runtime_state->velocity.dense_weights.size() == architecture.size() &&
       runtime_state->velocity.dense_biases.size() == architecture.size() &&
       runtime_state->velocity.conv_weights.size() == architecture.size() &&
       runtime_state->velocity.conv_biases.size() == architecture.size()){
        velocity = runtime_state->velocity;
    }
}

void store_runtime_state(TrainingRuntimeState *runtime_state, int completed_epochs, std::int64_t optimizer_steps, const ParameterBuffer &velocity){
    if(runtime_state == nullptr){
        return;
    }
    runtime_state->completed_epochs = completed_epochs;
    runtime_state->optimizer_steps = optimizer_steps;
    runtime_state->velocity = velocity;
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

TrainingSummary train_batch(LayerList &architecture, int num_layers, const Decay &learning_rate_decay, int num_epochs, float target_loss, vector<int> &train_indices, const Dataset4D &input, const Dataset4D &output, const Loss &loss, const Activation &hidden_activation, const Activation &output_activation, float momentum, ExecutionPolicy policy, TrainingRuntimeState *runtime_state){
    training_progress::print_training_banner();
    validate_training_setup(architecture, num_layers, train_indices, input, output, loss, output_activation, "train_batch");
    int num_tr = static_cast<int>(train_indices.size());
    float last_epoch_loss = std::numeric_limits<float>::quiet_NaN();
    int executed_epochs = 0;
    bool stopped_early = false;
    std::vector<double> epoch_times_seconds;
    epoch_times_seconds.reserve(std::max(0, num_epochs));
    const auto training_start = std::chrono::steady_clock::now();

    ParameterBuffer batch_gradients;
    ParameterBuffer velocity;
    BatchRuntimeList runtime;
    BatchTensor target_batch;
    training_buffers::init_parameter_buffer(architecture, batch_gradients);
    init_or_load_velocity(architecture, velocity, runtime_state);
    const int start_epoch = resolve_start_epoch(runtime_state, num_epochs);
    std::int64_t optimizer_steps = resolve_optimizer_steps(runtime_state);

    for(int e=start_epoch; e<num_epochs; e++){
        const auto epoch_start = std::chrono::steady_clock::now();
        const float current_learning_rate = learning_rate_decay.fn(e);
        training_progress::print_epoch_start(e, num_epochs);
        training_shuffle::shuffle_vec(train_indices);

        float loss_value = 0.0f;
        loss_value = training_batches::train_batch_chunk(
            architecture, runtime, num_layers,
            batch_gradients, target_batch,
            input, output, train_indices,
            0, num_tr,
            loss, hidden_activation, output_activation,
            policy
        );

        const ExecutionPolicy batch_parameter_policy = policy.allows_batch_sample_parallelism() ? execution_policy::intra_example() : policy;
        training_buffers::optimizer_step(
            architecture, num_layers,
            batch_gradients, velocity,
            current_learning_rate, momentum,
            1.0f / static_cast<float>(std::max(1, num_tr)),
            batch_parameter_policy
        );
        optimizer_steps++;
        loss_value /= num_tr;
        training_progress::print_epoch_progress(e, num_epochs, loss_value);
        const double epoch_seconds = elapsed_seconds(epoch_start);
        epoch_times_seconds.push_back(epoch_seconds);

        if(training_progress::update_progress_and_check_stop(loss_value, e, target_loss, last_epoch_loss, executed_epochs)){
            break;
        }
    }

    const double total_training_seconds = elapsed_seconds(training_start);
    std::cout << "Tempo training totale: " << total_training_seconds << " s" << std::endl;
    cout << endl;
    store_runtime_state(runtime_state, executed_epochs, optimizer_steps, velocity);
    return training_progress::make_summary(executed_epochs, last_epoch_loss, epoch_times_seconds, total_training_seconds, stopped_early);
}

TrainingSummary train_sgd(LayerList &architecture, int num_layers, int batch_size, const Decay &learning_rate_decay, int num_epochs, float target_loss, vector<int> &train_indices, const Dataset4D &input, const Dataset4D &output, const Loss &loss, const Activation &hidden_activation, const Activation &output_activation, float momentum, ExecutionPolicy policy, TrainingRuntimeState *runtime_state){
    training_progress::print_training_banner();
    validate_training_setup(architecture, num_layers, train_indices, input, output, loss, output_activation, "train_sgd");
    int num_tr = static_cast<int>(train_indices.size());
    float last_epoch_loss = std::numeric_limits<float>::quiet_NaN();
    int executed_epochs = 0;
    bool stopped_early = false;
    std::vector<double> epoch_times_seconds;
    epoch_times_seconds.reserve(std::max(0, num_epochs));
    const auto training_start = std::chrono::steady_clock::now();

    ParameterBuffer batch_gradients;
    ParameterBuffer velocity;
    BatchRuntimeList runtime;
    BatchTensor target_batch;
    training_buffers::init_parameter_buffer(architecture, batch_gradients);
    init_or_load_velocity(architecture, velocity, runtime_state);
    const int start_epoch = resolve_start_epoch(runtime_state, num_epochs);
    std::int64_t optimizer_steps = resolve_optimizer_steps(runtime_state);

    for(int e=start_epoch; e<num_epochs; e++){
        const auto epoch_start = std::chrono::steady_clock::now();
        const float current_learning_rate = learning_rate_decay.fn(e);
        training_progress::print_epoch_start(e, num_epochs);
        training_shuffle::shuffle_vec(train_indices);
        float loss_value = 0.0f;

        training_iteration::for_each_batch(num_tr, batch_size, [&](int start, int end, int size){
            loss_value += training_batches::train_batch_chunk(
                architecture, runtime, num_layers,
                batch_gradients, target_batch,
                input, output, train_indices,
                start, end,
                loss, hidden_activation, output_activation,
                policy
            );

            const ExecutionPolicy batch_parameter_policy = policy.allows_batch_sample_parallelism() ? execution_policy::intra_example() : policy;
            training_buffers::optimizer_step(
                architecture, num_layers,
                batch_gradients, velocity,
                current_learning_rate, momentum,
                1.0f / static_cast<float>(std::max(1, size)),
                batch_parameter_policy
            );
            optimizer_steps++;
        });

        loss_value /= num_tr;
        training_progress::print_epoch_progress(e, num_epochs, loss_value);
        const double epoch_seconds = elapsed_seconds(epoch_start);
        epoch_times_seconds.push_back(epoch_seconds);
        if(training_progress::update_progress_and_check_stop(loss_value, e, target_loss, last_epoch_loss, executed_epochs)){
            break;
        }
    }

    const double total_training_seconds = elapsed_seconds(training_start);
    std::cout << "Tempo training totale: " << total_training_seconds << " s" << std::endl;
    cout << endl;
    store_runtime_state(runtime_state, executed_epochs, optimizer_steps, velocity);
    return training_progress::make_summary(executed_epochs, last_epoch_loss, epoch_times_seconds, total_training_seconds, stopped_early);
}

TrainingSummary train_sgd_online(LayerList &architecture, int num_layers, int steps, const Decay &learning_rate_decay, int num_epochs, float target_loss, vector<int> &train_indices, const Dataset4D &input, const Dataset4D &output, const Loss &loss, const Activation &hidden_activation, const Activation &output_activation, float momentum, ExecutionPolicy policy, TrainingRuntimeState *runtime_state){
    training_progress::print_training_banner();
    validate_training_setup(architecture, num_layers, train_indices, input, output, loss, output_activation, "train_sgd_online");
    int num_tr = static_cast<int>(train_indices.size());
    const int windows = ceil(static_cast<float>(num_tr) / steps);
    float last_observed_loss = std::numeric_limits<float>::quiet_NaN();
    int executed_epochs = 0;
    bool stopped_early = false;
    std::vector<double> epoch_times_seconds;
    epoch_times_seconds.reserve(std::max(0, num_epochs));
    const auto training_start = std::chrono::steady_clock::now();

    ParameterBuffer sample_gradients;
    ParameterBuffer velocity;
    RuntimeList runtime;
    training_buffers::init_parameter_buffer(architecture, sample_gradients);
    init_or_load_velocity(architecture, velocity, runtime_state);
    init_runtime_buffers(architecture, runtime);
    const int start_epoch = resolve_start_epoch(runtime_state, num_epochs);
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

            const ExecutionPolicy online_policy = policy.allows_batch_sample_parallelism() ? execution_policy::intra_example() : policy;
            for(int t=start; t<end; t++){
                loss_value += training_examples::train_example(
                    architecture, runtime, num_layers,
                    sample_gradients,
                    input, output, train_indices[t],
                    loss, hidden_activation, output_activation,
                    online_policy
                );
                training_buffers::optimizer_step(architecture, num_layers, sample_gradients, velocity, current_learning_rate, momentum, 1.0f, online_policy);
                optimizer_steps++;
            }

            loss_value /= (end - start);
            training_progress::print_epoch_progress(e, num_epochs, loss_value);
            if(training_progress::update_progress_and_check_stop(loss_value, e, target_loss, last_observed_loss, executed_epochs)){
                stopped_early = true;
                const double epoch_seconds = elapsed_seconds(epoch_start);
                epoch_times_seconds.push_back(epoch_seconds);
                const double total_training_seconds = elapsed_seconds(training_start);
                std::cout << "Tempo training totale: " << total_training_seconds << " s" << std::endl;
                store_runtime_state(runtime_state, executed_epochs, optimizer_steps, velocity);
                return training_progress::make_summary(executed_epochs, last_observed_loss, epoch_times_seconds, total_training_seconds, stopped_early);
            }
        }

        const double epoch_seconds = elapsed_seconds(epoch_start);
        epoch_times_seconds.push_back(epoch_seconds);
    }

    const double total_training_seconds = elapsed_seconds(training_start);
    std::cout << "Tempo training totale: " << total_training_seconds << " s" << std::endl;
    cout << endl;
    store_runtime_state(runtime_state, executed_epochs, optimizer_steps, velocity);
    return training_progress::make_summary(executed_epochs, last_observed_loss, epoch_times_seconds, total_training_seconds, stopped_early);
}

TrainingSummary train_batch_nesterov(LayerList &architecture, int num_layers, const Decay &learning_rate_decay, int num_epochs, float target_loss, vector<int> &train_indices, const Dataset4D &input, const Dataset4D &output, const Loss &loss, const Activation &hidden_activation, const Activation &output_activation, float momentum, ExecutionPolicy policy, TrainingRuntimeState *runtime_state){
    training_progress::print_training_banner();
    validate_training_setup(architecture, num_layers, train_indices, input, output, loss, output_activation, "train_batch_nesterov");
    int num_tr = static_cast<int>(train_indices.size());
    float last_epoch_loss = std::numeric_limits<float>::quiet_NaN();
    int executed_epochs = 0;
    bool stopped_early = false;
    std::vector<double> epoch_times_seconds;
    epoch_times_seconds.reserve(std::max(0, num_epochs));
    const auto training_start = std::chrono::steady_clock::now();

    ParameterBuffer batch_gradients;
    ParameterBuffer velocity;
    BatchRuntimeList runtime;
    BatchTensor target_batch;
    training_buffers::init_parameter_buffer(architecture, batch_gradients);
    init_or_load_velocity(architecture, velocity, runtime_state);
    const int start_epoch = resolve_start_epoch(runtime_state, num_epochs);
    std::int64_t optimizer_steps = resolve_optimizer_steps(runtime_state);

    for(int e=start_epoch; e<num_epochs; e++){
        const auto epoch_start = std::chrono::steady_clock::now();
        const float current_learning_rate = learning_rate_decay.fn(e);
        training_progress::print_epoch_start(e, num_epochs);
        training_shuffle::shuffle_vec(train_indices);
        float loss_value = 0.0f;
        loss_value = training_batches::train_batch_chunk_nesterov(
            architecture, runtime, num_layers,
            batch_gradients, target_batch, velocity,
            input, output, train_indices,
            0, num_tr,
            loss, hidden_activation, output_activation,
            momentum,
            policy
        );

        const ExecutionPolicy batch_parameter_policy = policy.allows_batch_sample_parallelism() ? execution_policy::intra_example() : policy;
        training_buffers::optimizer_step(
            architecture, num_layers,
            batch_gradients, velocity,
            current_learning_rate, momentum,
            1.0f / static_cast<float>(std::max(1, num_tr)),
            batch_parameter_policy
        );
        optimizer_steps++;
        loss_value /= num_tr;
        training_progress::print_epoch_progress(e, num_epochs, loss_value);
        const double epoch_seconds = elapsed_seconds(epoch_start);
        epoch_times_seconds.push_back(epoch_seconds);

        if(training_progress::update_progress_and_check_stop(loss_value, e, target_loss, last_epoch_loss, executed_epochs)){
            break;
        }
    }

    const double total_training_seconds = elapsed_seconds(training_start);
    std::cout << "Tempo training totale: " << total_training_seconds << " s" << std::endl;
    cout << endl;
    store_runtime_state(runtime_state, executed_epochs, optimizer_steps, velocity);
    return training_progress::make_summary(executed_epochs, last_epoch_loss, epoch_times_seconds, total_training_seconds, stopped_early);
}

TrainingSummary train_sgd_nesterov(LayerList &architecture, int num_layers, int batch_size, const Decay &learning_rate_decay, int num_epochs, float target_loss, vector<int> &train_indices, const Dataset4D &input, const Dataset4D &output, const Loss &loss, const Activation &hidden_activation, const Activation &output_activation, float momentum, ExecutionPolicy policy, TrainingRuntimeState *runtime_state){
    training_progress::print_training_banner();
    validate_training_setup(architecture, num_layers, train_indices, input, output, loss, output_activation, "train_sgd_nesterov");
    int num_tr = static_cast<int>(train_indices.size());
    float last_epoch_loss = std::numeric_limits<float>::quiet_NaN();
    int executed_epochs = 0;
    bool stopped_early = false;
    std::vector<double> epoch_times_seconds;
    epoch_times_seconds.reserve(std::max(0, num_epochs));
    const auto training_start = std::chrono::steady_clock::now();

    ParameterBuffer batch_gradients;
    ParameterBuffer velocity;
    BatchRuntimeList runtime;
    BatchTensor target_batch;
    training_buffers::init_parameter_buffer(architecture, batch_gradients);
    init_or_load_velocity(architecture, velocity, runtime_state);
    const int start_epoch = resolve_start_epoch(runtime_state, num_epochs);
    std::int64_t optimizer_steps = resolve_optimizer_steps(runtime_state);

    for(int e=start_epoch; e<num_epochs; e++){
        const auto epoch_start = std::chrono::steady_clock::now();
        const float current_learning_rate = learning_rate_decay.fn(e);
        training_progress::print_epoch_start(e, num_epochs);
        training_shuffle::shuffle_vec(train_indices);
        float loss_value = 0.0f;

        training_iteration::for_each_batch(num_tr, batch_size, [&](int start, int end, int size){
            loss_value += training_batches::train_batch_chunk_nesterov(
                architecture, runtime, num_layers,
                batch_gradients, target_batch, velocity,
                input, output, train_indices,
                start, end,
                loss, hidden_activation, output_activation,
                momentum,
                policy
            );

            const ExecutionPolicy batch_parameter_policy = policy.allows_batch_sample_parallelism() ? execution_policy::intra_example() : policy;
            training_buffers::optimizer_step(
                architecture, num_layers,
                batch_gradients, velocity,
                current_learning_rate, momentum,
                1.0f / static_cast<float>(std::max(1, size)),
                batch_parameter_policy
            );
            optimizer_steps++;
        });

        loss_value /= num_tr;
        training_progress::print_epoch_progress(e, num_epochs, loss_value);
        const double epoch_seconds = elapsed_seconds(epoch_start);
        epoch_times_seconds.push_back(epoch_seconds);
        if(training_progress::update_progress_and_check_stop(loss_value, e, target_loss, last_epoch_loss, executed_epochs)){
            break;
        }
    }

    const double total_training_seconds = elapsed_seconds(training_start);
    std::cout << "Tempo training totale: " << total_training_seconds << " s" << std::endl;
    cout << endl;
    store_runtime_state(runtime_state, executed_epochs, optimizer_steps, velocity);
    return training_progress::make_summary(executed_epochs, last_epoch_loss, epoch_times_seconds, total_training_seconds, stopped_early);
}

TrainingSummary train_sgd_online_nesterov(LayerList &architecture, int num_layers, int steps, const Decay &learning_rate_decay, int num_epochs, float target_loss, vector<int> &train_indices, const Dataset4D &input, const Dataset4D &output, const Loss &loss, const Activation &hidden_activation, const Activation &output_activation, float momentum, ExecutionPolicy policy, TrainingRuntimeState *runtime_state){
    training_progress::print_training_banner();
    validate_training_setup(architecture, num_layers, train_indices, input, output, loss, output_activation, "train_sgd_online_nesterov");
    int num_tr = static_cast<int>(train_indices.size());
    const int windows = ceil(static_cast<float>(num_tr) / steps);
    float last_observed_loss = std::numeric_limits<float>::quiet_NaN();
    int executed_epochs = 0;
    bool stopped_early = false;
    std::vector<double> epoch_times_seconds;
    epoch_times_seconds.reserve(std::max(0, num_epochs));
    const auto training_start = std::chrono::steady_clock::now();

    ParameterBuffer sample_gradients;
    ParameterBuffer velocity;
    RuntimeList runtime;
    training_buffers::init_parameter_buffer(architecture, sample_gradients);
    init_or_load_velocity(architecture, velocity, runtime_state);
    init_runtime_buffers(architecture, runtime);
    const int start_epoch = resolve_start_epoch(runtime_state, num_epochs);
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

            const ExecutionPolicy online_policy = policy.allows_batch_sample_parallelism() ? execution_policy::intra_example() : policy;
            for(int t=start; t<end; t++){
                loss_value += training_examples::train_example_nesterov(
                    architecture, runtime, num_layers,
                    sample_gradients, velocity,
                    input, output, train_indices[t],
                    loss, hidden_activation, output_activation,
                    momentum,
                    online_policy
                );
                training_buffers::optimizer_step(architecture, num_layers, sample_gradients, velocity, current_learning_rate, momentum, 1.0f, online_policy);
                optimizer_steps++;
            }
            
            loss_value /= (end - start);
            training_progress::print_epoch_progress(e, num_epochs, loss_value);
            if(training_progress::update_progress_and_check_stop(loss_value, e, target_loss, last_observed_loss, executed_epochs)){
                stopped_early = true;
                const double epoch_seconds = elapsed_seconds(epoch_start);
                epoch_times_seconds.push_back(epoch_seconds);
                const double total_training_seconds = elapsed_seconds(training_start);
                std::cout << "Tempo training totale: " << total_training_seconds << " s" << std::endl;
                store_runtime_state(runtime_state, executed_epochs, optimizer_steps, velocity);
                return training_progress::make_summary(executed_epochs, last_observed_loss, epoch_times_seconds, total_training_seconds, stopped_early);
            }
        }

        const double epoch_seconds = elapsed_seconds(epoch_start);
        epoch_times_seconds.push_back(epoch_seconds);


    }

    const double total_training_seconds = elapsed_seconds(training_start);
    std::cout << "Tempo training totale: " << total_training_seconds << " s" << std::endl;
    cout << endl;
    store_runtime_state(runtime_state, executed_epochs, optimizer_steps, velocity);
    return training_progress::make_summary(executed_epochs, last_observed_loss, epoch_times_seconds, total_training_seconds, stopped_early);
}
