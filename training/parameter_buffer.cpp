#include "training/parameter_buffer.hpp"

#include <algorithm>

namespace training_buffers {

void init_parameter_buffer(const LayerList &architecture, ParameterBuffer &buffer){
    const int num_layers = static_cast<int>(architecture.size());
    buffer.dense_weights.resize(num_layers);
    buffer.dense_biases.resize(num_layers);
    buffer.conv_weights.resize(num_layers);
    buffer.conv_biases.resize(num_layers);

    for(int l=0; l<num_layers; l++){
        const Layer &layer = architecture[l];

        if(layer.type == Layer_type::Dense){
            buffer.dense_weights[l].assign(static_cast<std::size_t>(layer.dense_output_size) * static_cast<std::size_t>(layer.dense_input_size), 0.0f);
            buffer.dense_biases[l].resize(layer.dense_output_size, 0.0f);
        } else {
            buffer.dense_weights[l].clear();
            buffer.dense_biases[l].clear();
        }

        if(layer.type == Layer_type::Conv){
            buffer.conv_weights[l].assign(static_cast<std::size_t>(layer.conv_filter_count()), 0.0f);
            buffer.conv_biases[l].resize(layer.dim_layer[2], 0.0f);
        } else {
            buffer.conv_weights[l].clear();
            buffer.conv_biases[l].clear();
        }
    }
}

void zero_parameter_buffer(const LayerList &architecture, ParameterBuffer &buffer){
    for(size_t l=0; l<architecture.size(); l++){
        if(architecture[l].type == Layer_type::Dense){
            std::fill(buffer.dense_weights[l].begin(), buffer.dense_weights[l].end(), 0.0f);
            std::fill(buffer.dense_biases[l].begin(), buffer.dense_biases[l].end(), 0.0f);
        }

        if(architecture[l].type == Layer_type::Conv){
            std::fill(buffer.conv_biases[l].begin(), buffer.conv_biases[l].end(), 0.0f);
            std::fill(buffer.conv_weights[l].begin(), buffer.conv_weights[l].end(), 0.0f);
        }
    }
}

} // namespace training_buffers
