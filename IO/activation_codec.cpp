#include "IO/activation_codec.hpp"

#include "shared/network_globals.hpp"

#include <limits>
#include <sstream>
#include <stdexcept>

std::string activation_to_snapshot_name(const Activation &activation){
    std::ostringstream out;
    out.precision(std::numeric_limits<float>::max_digits10);
    out << activation_name(activation) << " " << activation.alpha << " " << activation.beta;
    return out.str();
}

Activation activation_from_snapshot_name(const std::string &activation_snapshot){
    std::istringstream in(activation_snapshot);
    std::string activation_name;
    in >> activation_name;

    Activation activation;
    if(activation_name == "Identity") activation = identity;
    else if(activation_name == "Sigmoid") activation = sigmoid;
    else if(activation_name == "Tanh") activation = tanh_activation;
    else if(activation_name == "ReLU") activation = relu;
    else if(activation_name == "LeakyReLU") activation = leaky_relu;
    else if(activation_name == "ELU") activation = elu;
    else if(activation_name == "Softplus") activation = softplus;
    else if(activation_name == "Swish") activation = swish;
    else if(activation_name == "Mish") activation = mish;
    else if(activation_name == "GELU") activation = gelu;
    else throw std::invalid_argument("Attivazione non riconosciuta nello snapshot: " + activation_snapshot);

    float alpha = activation.alpha;
    float beta = activation.beta;
    if(in >> alpha){
        activation.alpha = alpha;
        if(in >> beta){
            activation.beta = beta;
        }
    }
    return activation;
}
