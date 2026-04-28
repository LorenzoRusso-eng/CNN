#include "training/training_progress.hpp"

#include <algorithm>
#include <iostream>

namespace training_progress {

void print_training_banner(){
    std::cout << "Inizio training..." << std::endl;
    std::cout << "Inizializzazione dei gradienti e dello stato dell'ottimizzatore..." << std::endl;
}

void print_epoch_start(int epoch_index, int total_epochs){
    if(epoch_index == 0){
        std::cout << "Inizio epoca 1/" << total_epochs << "..." << std::endl;
    }
}

void print_epoch_progress(int epoch_index, int total_epochs, float loss){
    const int completed_epochs = epoch_index + 1;
    const int progress_step = std::max(1, total_epochs / 10);
    if(completed_epochs == 1 || completed_epochs == total_epochs || completed_epochs % progress_step == 0){
        std::cout << "Epoca completata: " << completed_epochs << "/" << total_epochs
                  << " | loss=" << loss << std::endl;
    }
}

bool should_stop_training(float target_loss, float loss){
    if(target_loss >= 0.0f && loss <= target_loss){
        std::cout << "Costo raggiunto: " << loss << " <= " << target_loss << std::endl;
        return true;
    }
    return false;
}

TrainingSummary make_summary(int executed_epochs, float final_loss, const std::vector<double> &epoch_times_seconds, double total_training_seconds, bool stopped_early){
    TrainingSummary summary{};
    summary.epochs_completed = executed_epochs;
    summary.final_loss = final_loss;
    summary.epoch_times_seconds = epoch_times_seconds;
    summary.total_training_seconds = total_training_seconds;
    summary.stopped_early = stopped_early;
    return summary;
}

bool update_progress_and_check_stop(float epoch_loss, int epoch_index, float target_loss, float &last_loss, int &executed_epochs){
    last_loss = epoch_loss;
    executed_epochs = epoch_index + 1;
    return should_stop_training(target_loss, epoch_loss);
}

} // namespace training_progress
