#include "shared/network_globals.hpp"

// Questo file contiene le definizioni delle istanze globali condivise.

Activation identity{ActivationKind::Identity};
Activation sigmoid{ActivationKind::Sigmoid};
Activation tanh_activation{ActivationKind::Tanh};
Loss simple_loss{LossKind::Simple};
Loss l1_loss{LossKind::L1};
Loss l2_loss{LossKind::L2};
Activation relu{ActivationKind::ReLU};
Activation leaky_relu{ActivationKind::LeakyReLU};
Activation elu{ActivationKind::ELU};
Activation softplus{ActivationKind::Softplus};
Activation swish{ActivationKind::Swish};
Activation mish{ActivationKind::Mish};
Activation gelu{ActivationKind::GELU};
Loss smooth_l1_loss{LossKind::SmoothL1};
Loss huber_loss{LossKind::Huber};
Loss cross_entropy{LossKind::CrossEntropy};
Loss ll_loss{LossKind::LL};
Constant_decay constant_decay;
Exponential_decay exponential_decay;
Time_based_decay time_based_decay;
Step_decay step_decay;
Cosine_annealing cosine_annealing;
