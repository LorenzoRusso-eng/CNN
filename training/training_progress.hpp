#pragma once

#include "core/core_definitions.hpp"

#include <vector>

namespace training_progress {

void print_training_banner();
void print_epoch_start(int epoch_index, int total_epochs);
void print_epoch_progress(int epoch_index, int total_epochs, float loss);
bool should_stop_training(float target_loss, float loss);
TrainingSummary make_summary(int executed_epochs, float final_loss, const std::vector<double> &epoch_times_seconds, double total_training_seconds, bool stopped_early);
bool update_progress_and_check_stop(float epoch_loss, int epoch_index, float target_loss, float &last_loss, int &executed_epochs);

} // namespace training_progress
