#pragma once

// Questo file contiene helper di validazione e calcolo delle shape.

#include "core/core_definitions.hpp"

#include <stdexcept>
#include <string>

inline bool dims_match(const int lhs[3], const int rhs[3]){
    return lhs[0] == rhs[0] && lhs[1] == rhs[1] && lhs[2] == rhs[2];
}

inline void require_condition(bool condition, const std::string &message){
    if(!condition){
        throw std::invalid_argument(message);
    }
}

inline void validate_tensor_shape(const Tensor &tensor, const int expected_dim[3], const std::string &context){
    require_condition(tensor.height == expected_dim[0], context + ": dimensione asse 0 incoerente");
    require_condition(tensor.width == expected_dim[1], context + ": dimensione asse 1 incoerente");
    require_condition(tensor.channels == expected_dim[2], context + ": dimensione asse 2 incoerente");
}

inline void validate_positive_dims(const int dim[3], const std::string &context){
    require_condition(dim[0] > 0 && dim[1] > 0 && dim[2] > 0, context + ": tutte le dimensioni devono essere positive");
}

inline int compute_spatial_output_dim(int input_size, int kernel_size, int stride_size, int padding_size, const std::string &context, const std::string &axis){
    require_condition(kernel_size > 0, context + ": kernel " + axis + " deve essere positivo");
    require_condition(stride_size > 0, context + ": stride " + axis + " deve essere positivo");
    require_condition(padding_size >= 0, context + ": padding " + axis + " non puo' essere negativo");

    const int numerator = input_size + 2 * padding_size - kernel_size;
    require_condition(numerator >= 0, context + ": output " + axis + " negativo con i parametri scelti");
    require_condition(numerator % stride_size == 0, context + ": output " + axis + " non intero con i parametri scelti");
    return numerator / stride_size + 1;
}
