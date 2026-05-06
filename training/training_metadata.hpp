#pragma once

#include "IO/model_io.hpp"
#include "IO/report_io.hpp"
#include "math/activations.hpp"
#include "math/decay.hpp"
#include "math/losses.hpp"

#include <string>

namespace training_metadata {

ReportMetadata build_report_metadata(const std::string &model_name, const std::string &run_type, int requested_epochs, int total_examples, int num_classes, int training_type, int window, bool use_nesterov, float momentum, float target_loss, const Decay &learning_rate_decay, const Loss &loss, const Activation &hidden_activation, const Activation &output_activation);
TrainingSnapshotMetadata build_training_snapshot_metadata(const std::string &model_name, int training_method, int training_type, int window, bool use_nesterov, float momentum, float target_loss, int requested_epochs, const Decay &learning_rate_decay, const Loss &loss, const TrainingRuntimeState &runtime_state, bool finalized_by_loss, bool finalized_by_validation);

} // namespace training_metadata
