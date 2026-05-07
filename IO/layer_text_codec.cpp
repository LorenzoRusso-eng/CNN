#include "IO/layer_text_codec.hpp"

#include <stdexcept>

namespace layer_text {

Layer_type parse_layer_type(const std::string &type){
    if(type == "Input") return Layer_type::Input;
    if(type == "Dense") return Layer_type::Dense;
    if(type == "Conv") return Layer_type::Conv;
    if(type == "Pooling") return Layer_type::Pooling;
    if(type == "Flatten") return Layer_type::Flatten;
    if(type == "Softmax") return Layer_type::Softmax;
    throw std::invalid_argument("Tipo layer non riconosciuto nel file: " + type);
}

Pooling_type parse_pooling_type(const std::string &type){
    if(type == "Max") return Pooling_type::Max;
    if(type == "Average") return Pooling_type::Average;
    if(type == "L2") return Pooling_type::L2;
    throw std::invalid_argument("Tipo pooling non riconosciuto nel file: " + type);
}

std::string layer_type_to_string(Layer_type type){
    switch(type){
        case Layer_type::Input:
            return "Input";
        case Layer_type::Dense:
            return "Dense";
        case Layer_type::Conv:
            return "Conv";
        case Layer_type::Pooling:
            return "Pooling";
        case Layer_type::Flatten:
            return "Flatten";
        case Layer_type::Softmax:
            return "Softmax";
    }
    return "Unknown";
}

std::string pooling_type_to_string(Pooling_type type){
    switch(type){
        case Pooling_type::Max:
            return "Max";
        case Pooling_type::Average:
            return "Average";
        case Pooling_type::L2:
            return "L2";
    }
    return "Unknown";
}

} // namespace layer_text
