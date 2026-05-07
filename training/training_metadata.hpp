#pragma once

#include "IO/report_io.hpp"
#include "math/activations.hpp"
#include "math/decay.hpp"
#include "math/losses.hpp"

#include <string>

namespace training_metadata {

ReportMetadata build_report_metadata(const std::string &model_name, const std::string &run_type, int requested_epochs, int total_examples, int num_classes, int training_type, int window, bool use_nesterov, float momentum, float target_loss, const Decay &learning_rate_decay, const Loss &loss, const Activation &hidden_activation, const Activation &output_activation);

} // namespace training_metadata
