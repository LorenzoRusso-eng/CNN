#pragma once

// Questo file contiene le validazioni di coerenza tra layer consecutivi.

#include "core/layer.hpp"

inline void validate_dense_layout(const Layer &current, const Layer &previous, const std::string &context){
    require_condition(current.type == Layer_type::Dense, context + ": il layer non e' Dense");
    validate_positive_dims(current.dim_layer, context);
    validate_positive_dims(previous.dim_layer, context + " previous");
    require_condition(dims_match(current.input_dim, previous.dim_layer), context + ": input_dim non coincide con il layer precedente");
    const int expected_input_size = previous.dim_layer[0] * previous.dim_layer[1] * previous.dim_layer[2];
    const int expected_output_size = current.dim_layer[0] * current.dim_layer[1] * current.dim_layer[2];
    require_condition(current.dense_input_size == expected_input_size, context + ": dense_input_size incoerente");
    require_condition(current.dense_output_size == expected_output_size, context + ": dense_output_size incoerente");
    require_condition(
        static_cast<int>(current.dense_params.weights.size()) == expected_output_size * expected_input_size,
        context + ": numero pesi dense incoerente"
    );
    require_condition(static_cast<int>(current.dense_params.bias.size()) == expected_output_size, context + ": dense_params.bias incoerente");
}

inline void validate_conv_layout(const Layer &current, const Layer &previous, const std::string &context){
    require_condition(current.type == Layer_type::Conv, context + ": il layer non e' Conv");
    validate_positive_dims(current.dim_layer, context);
    validate_positive_dims(previous.dim_layer, context + " previous");
    require_condition(dims_match(current.input_dim, previous.dim_layer), context + ": input_dim non coincide con il layer precedente");
    require_condition(current.kernel_dim[2] == previous.dim_layer[2], context + ": il numero di canali del kernel deve coincidere con i canali in input");

    const int expected_h = compute_spatial_output_dim(
        previous.dim_layer[0],
        current.kernel_dim[0],
        current.stride[0],
        current.padding[0],
        context,
        "height"
    );
    const int expected_w = compute_spatial_output_dim(
        previous.dim_layer[1],
        current.kernel_dim[1],
        current.stride[1],
        current.padding[1],
        context,
        "width"
    );

    require_condition(current.dim_layer[0] == expected_h, context + ": altezza output incoerente");
    require_condition(current.dim_layer[1] == expected_w, context + ": larghezza output incoerente");
    require_condition(static_cast<int>(current.conv_params.filters.size()) == current.conv_filter_count(), context + ": numero pesi filtri incoerente");
    require_condition(static_cast<int>(current.conv_params.bias.size()) == current.dim_layer[2], context + ": numero bias filtri incoerente");
}

inline void validate_pooling_layout(const Layer &current, const Layer &previous, const std::string &context){
    require_condition(current.type == Layer_type::Pooling, context + ": il layer non e' Pooling");
    validate_positive_dims(current.dim_layer, context);
    validate_positive_dims(previous.dim_layer, context + " previous");
    require_condition(dims_match(current.input_dim, previous.dim_layer), context + ": input_dim non coincide con il layer precedente");
    require_condition(current.dim_layer[2] == previous.dim_layer[2], context + ": il pooling deve preservare il numero di canali");

    const int expected_h = compute_spatial_output_dim(
        previous.dim_layer[0],
        current.kernel_dim[0],
        current.stride[0],
        current.padding[0],
        context,
        "height"
    );
    const int expected_w = compute_spatial_output_dim(
        previous.dim_layer[1],
        current.kernel_dim[1],
        current.stride[1],
        current.padding[1],
        context,
        "width"
    );
    require_condition(current.dim_layer[0] == expected_h, context + ": altezza output incoerente");
    require_condition(current.dim_layer[1] == expected_w, context + ": larghezza output incoerente");
}

inline void validate_flatten_layout(const Layer &current, const Layer &previous, const std::string &context){
    require_condition(current.type == Layer_type::Flatten, context + ": il layer non e' Flatten");
    validate_positive_dims(previous.dim_layer, context + " previous");
    require_condition(dims_match(current.input_dim, previous.dim_layer), context + ": input_dim non coincide con il layer precedente");
    require_condition(current.dim_layer[1] == 1 && current.dim_layer[2] == 1, context + ": il flatten deve avere shape [N,1,1]");
    require_condition(
        current.dim_layer[0] == previous.dim_layer[0] *
                                previous.dim_layer[1] *
                                previous.dim_layer[2],
        context + ": dimensione flatten incoerente"
    );
}

inline void validate_lrn_layout(const Layer &current, const Layer &previous, const std::string &context){
    require_condition(current.type == Layer_type::LRN, context + ": il layer non e' LRN");
    validate_positive_dims(current.dim_layer, context);
    validate_positive_dims(previous.dim_layer, context + " previous");
    require_condition(dims_match(current.input_dim, previous.dim_layer), context + ": input_dim non coincide con il layer precedente");
    require_condition(dims_match(current.dim_layer, previous.dim_layer), context + ": LRN richiede stessa shape tra input e output");
    require_condition(current.lrn_local_size > 0, context + ": local_size deve essere positivo");
    require_condition(current.lrn_alpha >= 0.0f, context + ": alpha non puo' essere negativo");
    require_condition(current.lrn_beta >= 0.0f, context + ": beta non puo' essere negativo");
    require_condition(current.lrn_k > 0.0f, context + ": k deve essere positivo");
}

inline void validate_softmax_layout(const Layer &current, const Layer &previous, const std::string &context){
    require_condition(current.type == Layer_type::Softmax, context + ": il layer non e' Softmax");
    validate_positive_dims(current.dim_layer, context);
    validate_positive_dims(previous.dim_layer, context + " previous");
    require_condition(dims_match(current.input_dim, previous.dim_layer), context + ": input_dim non coincide con il layer precedente");
    require_condition(dims_match(current.dim_layer, previous.dim_layer), context + ": Softmax richiede stessa shape tra input e output");
}

inline void validate_layer_connection(const Layer &current, const Layer &previous, const std::string &context){
    switch(current.type){
        case Layer_type::Dense:
            validate_dense_layout(current, previous, context);
            break;
        case Layer_type::Conv:
            validate_conv_layout(current, previous, context);
            break;
        case Layer_type::Pooling:
            validate_pooling_layout(current, previous, context);
            break;
        case Layer_type::Flatten:
            validate_flatten_layout(current, previous, context);
            break;
        case Layer_type::LRN:
            validate_lrn_layout(current, previous, context);
            break;
        case Layer_type::Softmax:
            validate_softmax_layout(current, previous, context);
            break;
        case Layer_type::Input:
            throw std::invalid_argument(context + ": Input non puo' essere validato come layer successivo");
    }
}

inline void validate_architecture(const LayerList &architecture, int num_layers, const std::string &context){
    require_condition(num_layers > 0, context + ": num_layers deve essere positivo");
    require_condition(static_cast<int>(architecture.size()) == num_layers, context + ": num_layers non coincide con la dimensione dell'architettura");
    require_condition(architecture[0].type == Layer_type::Input, context + ": il primo layer deve essere Input");
    validate_positive_dims(architecture[0].dim_layer, context + " input");

    for(int l=1; l<num_layers; l++){
        require_condition(architecture[l].type != Layer_type::Input, context + ": Input e' supportato solo come primo layer");
        if(architecture[l].type == Layer_type::Softmax){
            require_condition(l == num_layers - 1, context + ": Softmax e' supportato solo come layer finale");
        }
        validate_layer_connection(architecture[l], architecture[l - 1], context + " layer " + std::to_string(l));
    }
}

inline void validate_dataset_indices_io_shapes(const LayerList &architecture, int num_layers, const Dataset4D &input, const Dataset4D &output, const std::vector<int> &indices, const std::string &context){
    require_condition(static_cast<int>(input.size()) == static_cast<int>(output.size()), context + ": dataset input/output con cardinalita' differente");

    for(size_t position=0; position<indices.size(); position++){
        const int sample_index = indices[position];
        require_condition(sample_index >= 0 && sample_index < static_cast<int>(input.size()), context + ": indice sample fuori range");
        validate_tensor_shape(input[sample_index], architecture[0].dim_layer, context + " input sample " + std::to_string(sample_index));
        validate_tensor_shape(output[sample_index], architecture[num_layers - 1].dim_layer, context + " output sample " + std::to_string(sample_index));
    }
}
