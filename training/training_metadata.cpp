#include "training/training_metadata.hpp"

namespace {

std::string training_variant_to_string(int training_type){
    switch(training_type){
        case 1:
            return "Batch";
        case 2:
            return "Mini-batch SGD";
        case 3:
            return "Online SGD";
        default:
            return "Unknown";
    }
}

std::string training_window_label_to_string(int training_type){
    switch(training_type){
        case 1:
            return "Batch Size";
        case 2:
            return "Batch Size";
        case 3:
            return "Loss Averaging Steps";
        default:
            return "Window";
    }
}

std::string decay_to_string(const Decay &decay){
    if(dynamic_cast<const Constant_decay*>(&decay) != nullptr){
        return "Constant";
    }
    if(dynamic_cast<const Exponential_decay*>(&decay) != nullptr){
        return "Exponential";
    }
    if(dynamic_cast<const Time_based_decay*>(&decay) != nullptr){
        return "Time-based";
    }
    if(dynamic_cast<const Step_decay*>(&decay) != nullptr){
        return "Step";
    }
    if(dynamic_cast<const Cosine_annealing*>(&decay) != nullptr){
        return "Cosine Annealing";
    }
    return "Unknown";
}

std::string loss_to_string(const Loss &loss){
    switch(loss.kind){
        case LossKind::Simple:
            return "Simple";
        case LossKind::L1:
            return "L1";
        case LossKind::L2:
            return "L2";
        case LossKind::SmoothL1:
            return "SmoothL1";
        case LossKind::Huber:
            return "Huber";
        case LossKind::CrossEntropy:
            return "CrossEntropy";
        case LossKind::LL:
            return "LL";
    }
    return "Unknown";
}

std::string reduction_to_string(Reduction reduction){
    switch(reduction){
        case Reduction::Sum:
            return "Sum";
        case Reduction::Mean:
            return "Mean";
    }
    return "Unknown";
}

} // namespace

namespace training_metadata {

ReportMetadata build_report_metadata(const std::string &model_name, const std::string &run_type, int requested_epochs, int total_examples, int num_classes, int training_type, int window, bool use_nesterov, float momentum, float target_loss, const Decay &learning_rate_decay, const Loss &loss, const Activation &hidden_activation, const Activation &output_activation){
    ReportMetadata meta{};
    meta.model_name = model_name;
    meta.dataset_description = "Dataset caricato da CLI";
    meta.run_type = run_type;
    meta.requested_epochs = requested_epochs;
    meta.total_examples = total_examples;
    meta.num_classes = num_classes;
    meta.training_variant = training_variant_to_string(training_type);
    meta.training_window_label = training_window_label_to_string(training_type);
    meta.training_window = window;
    meta.use_nesterov = use_nesterov;
    meta.momentum = momentum;
    meta.target_loss = target_loss;
    meta.decay_type = decay_to_string(learning_rate_decay);
    meta.initial_learning_rate = learning_rate_decay.fn(0);
    meta.hidden_activation = activation_name(hidden_activation);
    meta.output_activation = activation_name(output_activation);
    meta.loss_name = loss_to_string(loss);
    meta.loss_reduction = reduction_to_string(loss.reduction());
    return meta;
}

TrainingSnapshotMetadata build_training_snapshot_metadata(const std::string &model_name, int training_method, int training_type, int window, bool use_nesterov, float momentum, float target_loss, int requested_epochs, const Decay &learning_rate_decay, const Loss &loss, const TrainingRuntimeState &runtime_state, bool training_finalized){
    TrainingSnapshotMetadata metadata{};
    metadata.model_name = model_name;
    metadata.training_method = training_method;
    metadata.training_type = training_type;
    metadata.training_window = window;
    metadata.use_nesterov = use_nesterov;
    metadata.momentum = momentum;
    metadata.target_loss = target_loss;
    metadata.requested_epochs = requested_epochs;
    metadata.completed_epochs = runtime_state.completed_epochs;
    metadata.optimizer_steps = runtime_state.optimizer_steps;
    metadata.training_finalized = training_finalized;
    metadata.optimizer_velocity = runtime_state.velocity;
    metadata.decay_kind = learning_rate_decay.kind();
    metadata.initial_learning_rate = learning_rate_decay.initial_lr();
    metadata.decay_rate = learning_rate_decay.decay_rate();
    metadata.decay_step_size = learning_rate_decay.step_size();
    metadata.cosine_final_lr = learning_rate_decay.final_lr();
    metadata.cosine_max_epoch = learning_rate_decay.max_epoch();
    metadata.loss_kind = loss.kind;
    metadata.loss_reduction = loss.reduction();
    metadata.loss_beta = loss.beta;
    return metadata;
}

} // namespace training_metadata
