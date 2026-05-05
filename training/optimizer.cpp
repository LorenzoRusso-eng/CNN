#include "training/optimizer.hpp"

namespace training_buffers {

void optimizer_step(LayerList &architecture, int num_layers, const ParameterBuffer &gradients, ParameterBuffer &velocity, float learning_rate, float momentum, float gradient_scale){
    for(int l=1; l<num_layers; l++){
        Layer &current = architecture[l];
        switch(current.type){
            case Layer_type::Dense: {
                auto &weight_data = current.dense_params.weights;
                auto &velocity_weight_data = velocity.dense_weights[l];
                const auto &gradient_weight_data = gradients.dense_weights[l];

                for(int idx=0; idx<static_cast<int>(weight_data.size()); idx++){
                    const std::size_t data_index = static_cast<std::size_t>(idx);
                    velocity_weight_data[data_index] =
                        momentum * velocity_weight_data[data_index] +
                        learning_rate * gradient_scale * gradient_weight_data[data_index];
                    weight_data[data_index] += velocity_weight_data[data_index];
                }

                auto &bias_data = current.dense_params.bias;
                auto &velocity_bias_data = velocity.dense_biases[l];
                const auto &gradient_bias_data = gradients.dense_biases[l];

                for(int idx=0; idx<static_cast<int>(bias_data.size()); idx++){
                    const std::size_t data_index = static_cast<std::size_t>(idx);
                    velocity_bias_data[data_index] =
                        momentum * velocity_bias_data[data_index] +
                        learning_rate * gradient_scale * gradient_bias_data[data_index];
                    bias_data[data_index] += velocity_bias_data[data_index];
                }
                break;
            }

            case Layer_type::Conv: {
                auto &weight_data = current.conv_params.filters;
                auto &velocity_weight_data = velocity.conv_weights[l];
                const auto &gradient_weight_data = gradients.conv_weights[l];

                for(int idx=0; idx<static_cast<int>(weight_data.size()); idx++){
                    const std::size_t data_index = static_cast<std::size_t>(idx);
                    velocity_weight_data[data_index] =
                        momentum * velocity_weight_data[data_index] +
                        learning_rate * gradient_scale * gradient_weight_data[data_index];
                    weight_data[data_index] += velocity_weight_data[data_index];
                }

                auto &bias_data = current.conv_params.bias;
                auto &velocity_bias_data = velocity.conv_biases[l];
                const auto &gradient_bias_data = gradients.conv_biases[l];

                for(int idx=0; idx<static_cast<int>(bias_data.size()); idx++){
                    const std::size_t data_index = static_cast<std::size_t>(idx);
                    velocity_bias_data[data_index] =
                        momentum * velocity_bias_data[data_index] +
                        learning_rate * gradient_scale * gradient_bias_data[data_index];
                    bias_data[data_index] += velocity_bias_data[data_index];
                }
                break;
            }

            case Layer_type::Pooling:
            case Layer_type::Flatten:
            case Layer_type::LRN:
            case Layer_type::Softmax:
            case Layer_type::Input:
                break;
        }
    }
}

} // namespace training_buffers
