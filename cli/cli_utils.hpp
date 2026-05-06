#pragma once

// Questo file contiene le utility CLI per input validato e selezione di opzioni di training.

#include <limits>
#include <string>
#include <vector>

#include "core/layer.hpp"
#include "math/activations.hpp"
#include "math/losses.hpp"
#include "math/decay.hpp"

int read_bounded_int(const std::string &message, int min_value, int max_value = std::numeric_limits<int>::max(), const std::string &invalid_message = "Scelta non valida.");
float read_bounded_float(const std::string &message, float min_value, float max_value = std::numeric_limits<float>::max(), const std::string &invalid_message = "Scelta non valida.");
Pooling_type read_pooling_type();
Reduction choose_loss_reduction();
int choose_training_type();
int choose_training_method();
Activation choose_hidden_activation();
Activation choose_output_activation(bool final_is_softmax);
const Loss& choose_loss_function(bool final_is_softmax);
const Decay& choose_learning_rate_decay(float initial_lr, int num_epochs);
