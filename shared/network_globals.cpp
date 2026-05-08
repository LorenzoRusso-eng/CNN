#include "shared/network_globals.hpp"

// Questo file contiene le definizioni delle istanze globali condivise.

Activation identity{ActivationKind::Identity};
Activation sigmoid{ActivationKind::Sigmoid};
Activation tanh_activation{ActivationKind::Tanh};
Activation relu{ActivationKind::ReLU};
Activation leaky_relu{ActivationKind::LeakyReLU};
Activation elu{ActivationKind::ELU};
Activation softplus{ActivationKind::Softplus};
Activation swish{ActivationKind::Swish};
Activation mish{ActivationKind::Mish};
Activation gelu{ActivationKind::GELU};
