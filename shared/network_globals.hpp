#pragma once

// Questo file contiene le dichiarazioni delle istanze globali condivise.

#include "math/activations.hpp"
#include "math/decay.hpp"
#include "math/losses.hpp"

extern Constant_decay constant_decay;
extern Exponential_decay exponential_decay;
extern Time_based_decay time_based_decay;
extern Step_decay step_decay;
extern Cosine_annealing cosine_annealing;
