#include "IO/activation_codec.hpp"

#include "shared/network_globals.hpp"

#include <stdexcept>

std::string activation_to_snapshot_name(const Activation &activation){
    if(&activation == &identity) return "Identity";
    if(&activation == &sigmoid) return "Sigmoid";
    if(&activation == &tanh_activation) return "Tanh";
    if(&activation == &relu) return "ReLU";
    if(&activation == &leaky_relu) return "LeakyReLU";
    if(&activation == &elu) return "ELU";
    if(&activation == &softplus) return "Softplus";
    if(&activation == &swish) return "Swish";
    if(&activation == &mish) return "Mish";
    if(&activation == &gelu) return "GELU";
    throw std::invalid_argument("Attivazione non supportata per il salvataggio nello snapshot");
}

const Activation &activation_from_snapshot_name(const std::string &activation_name){
    if(activation_name == "Identity") return identity;
    if(activation_name == "Sigmoid") return sigmoid;
    if(activation_name == "Tanh") return tanh_activation;
    if(activation_name == "ReLU") return relu;
    if(activation_name == "LeakyReLU") return leaky_relu;
    if(activation_name == "ELU") return elu;
    if(activation_name == "Softplus") return softplus;
    if(activation_name == "Swish") return swish;
    if(activation_name == "Mish") return mish;
    if(activation_name == "GELU") return gelu;
    throw std::invalid_argument("Attivazione non riconosciuta nello snapshot: " + activation_name);
}
