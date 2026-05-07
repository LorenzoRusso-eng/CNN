#pragma once

// Questo file contiene la definizione del layer e della sua configurazione strutturale.

#include "core/core_definitions.hpp"
#include "core/shape_utils.hpp"

#include <cmath>
#include <random>

enum class Layer_type {
    Input,
    Dense,
    Conv,
    Pooling,
    Flatten,
    Softmax
};

enum class Pooling_type {
    Max,
    Average,
    L2
};

struct DenseParameters{
    std::vector<float> weights;
    std::vector<float> bias;

    void clear(){
        weights.clear();
        bias.clear();
    }
};

struct ConvParameters{
    std::vector<float> filters;
    std::vector<float> bias;

    void clear(){
        filters.clear();
        bias.clear();
    }
};

struct LayerRuntime{
    Tensor a;
    Tensor y;
    Tensor delta;
    std::vector<int> pooling_argmax;
    std::vector<float> conv_im2col;
    std::vector<float> backprop_cost_from_next;
    std::vector<float> reduction_ones;

    void clear(){
        a.clear();
        y.clear();
        delta.clear();
        pooling_argmax.clear();
        conv_im2col.clear();
        backprop_cost_from_next.clear();
        reduction_ones.clear();
    }

    void resize(const int dim[3], bool allocate_activation = true){
        if(allocate_activation){
            a = Tensor(dim[0], dim[1], dim[2], 0.0f);
        } else {
            a.clear();
        }
        y = Tensor(dim[0], dim[1], dim[2], 0.0f);
        delta = Tensor(dim[0], dim[1], dim[2], 0.0f);
    }
};

struct BatchLayerRuntime{
    BatchTensor a;
    BatchTensor y;
    BatchTensor delta;
    std::vector<int> pooling_argmax;
    std::vector<float> conv_im2col;
    std::vector<float> backprop_cost_from_next;
    std::vector<float> reduction_ones;

    void clear(){
        a.clear();
        y.clear();
        delta.clear();
        pooling_argmax.clear();
        conv_im2col.clear();
        backprop_cost_from_next.clear();
        reduction_ones.clear();
    }

    void resize(int batch_size, const int dim[3], bool allocate_activation = true){
        if(allocate_activation){
            a = BatchTensor(batch_size, dim[0], dim[1], dim[2], 0.0f);
        } else {
            a.clear();
        }
        y = BatchTensor(batch_size, dim[0], dim[1], dim[2], 0.0f);
        delta = BatchTensor(batch_size, dim[0], dim[1], dim[2], 0.0f);
    }
};

class Layer{
    public:
        int dim_layer[3] = {0, 0, 0};
        int input_dim[3] = {0, 0, 0};

        Layer_type type = Layer_type::Dense;

        int kernel_dim[3] = {0, 0, 0};
        int stride[2] = {1, 1};
        int padding[2] = {0, 0};
        Pooling_type pooling_type = Pooling_type::Max;

        // Parametri e stato separati per i layer dense:
        // dense_params.weights[out_idx * dense_input_size + in_idx]
        int dense_input_size = 0;
        int dense_output_size = 0;
        DenseParameters dense_params;

        // Parametri separati per i layer convoluzionali:
        // conv_params.filters[out_channel][kernel_h][kernel_w][in_channel]
        ConvParameters conv_params;

        void set_dims(const int dim[3]){
            for(int i=0; i<3; i++)
                dim_layer[i] = dim[i];
        }

        void set_input_dims(const int dim[3]){
            for(int i=0; i<3; i++)
                input_dim[i] = dim[i];
        }

        int flat_index(int i, int j, int k) const{
            return (i * dim_layer[1] + j) * dim_layer[2] + k;
        }

        int input_flat_index(int i, int j, int k) const{
            return (i * input_dim[1] + j) * input_dim[2] + k;
        }

        int flat_output_size() const{
            return dim_layer[0] * dim_layer[1] * dim_layer[2];
        }

        std::size_t conv_filter_index(int out_k, int kh, int kw, int in_k) const{
            return (((static_cast<std::size_t>(out_k) * static_cast<std::size_t>(kernel_dim[0])) +
                     static_cast<std::size_t>(kh)) *
                    static_cast<std::size_t>(kernel_dim[1]) +
                    static_cast<std::size_t>(kw)) *
                   static_cast<std::size_t>(kernel_dim[2]) +
                   static_cast<std::size_t>(in_k);
        }

        int conv_filter_count() const{
            return dim_layer[2] * kernel_dim[0] * kernel_dim[1] * kernel_dim[2];
        }

        std::size_t dense_weight_index(int out_idx, int in_idx) const{
            return static_cast<std::size_t>(out_idx) * static_cast<std::size_t>(dense_input_size) + static_cast<std::size_t>(in_idx);
        }

        float &dense_weight_at(int out_idx, int in_idx){
            return dense_params.weights[dense_weight_index(out_idx, in_idx)];
        }

        const float &dense_weight_at(int out_idx, int in_idx) const{
            return dense_params.weights[dense_weight_index(out_idx, in_idx)];
        }

        float &conv_filter_at(int out_k, int kh, int kw, int in_k){
            return conv_params.filters[conv_filter_index(out_k, kh, kw, in_k)];
        }

        const float &conv_filter_at(int out_k, int kh, int kw, int in_k) const{
            return conv_params.filters[conv_filter_index(out_k, kh, kw, in_k)];
        }

        void reset_dense_storage(){
            dense_input_size = 0;
            dense_output_size = 0;
            dense_params.clear();
        }

        void reset_conv_storage(){
            conv_params.clear();
        }

        void reset_spatial_config(){
            kernel_dim[0] = 0;
            kernel_dim[1] = 0;
            kernel_dim[2] = 0;
            stride[0] = stride[1] = 1;
            padding[0] = padding[1] = 0;
        }

        void init_dense(int dim[3], int p_dim[3]){
            type = Layer_type::Dense;

            static std::random_device rd;
            static std::mt19937 gen(rd());

            validate_positive_dims(dim, "init_dense output");
            validate_positive_dims(p_dim, "init_dense input");

            set_dims(dim);
            set_input_dims(p_dim);
            reset_dense_storage();
            reset_conv_storage();
            reset_spatial_config();

            dense_input_size = p_dim[0] * p_dim[1] * p_dim[2];
            dense_output_size = dim[0] * dim[1] * dim[2];
            dense_params.weights.assign(static_cast<std::size_t>(dense_output_size) * static_cast<std::size_t>(dense_input_size), 0.0f);
            dense_params.bias.resize(dense_output_size, 0.0f);

            const int fan_in = dense_input_size;
            std::normal_distribution<float> dist(0.0f, std::sqrt(1.0f/static_cast<float>(fan_in)));

            for(std::size_t weight_index = 0; weight_index < dense_params.weights.size(); weight_index++){
                dense_params.weights[weight_index] = dist(gen);
            }
        }

        void init_input(int dim[3]){
            type = Layer_type::Input;

            validate_positive_dims(dim, "init_input");

            set_dims(dim);
            const int no_input[3] = {0, 0, 0};
            set_input_dims(no_input);
            reset_dense_storage();
            reset_conv_storage();
            reset_spatial_config();
        }

        void init_conv(int dim[3], int in_dim[3], int kernel[3], int stride_in[2], int padding_in[2]){
            type = Layer_type::Conv;

            static std::random_device rd;
            static std::mt19937 gen(rd());

            validate_positive_dims(dim, "init_conv output");
            validate_positive_dims(in_dim, "init_conv input");
            require_condition(kernel[0] > 0 && kernel[1] > 0 && kernel[2] > 0, "init_conv: kernel deve avere dimensioni positive");
            require_condition(stride_in[0] > 0 && stride_in[1] > 0, "init_conv: stride deve essere positiva");
            require_condition(padding_in[0] >= 0 && padding_in[1] >= 0, "init_conv: padding non puo' essere negativo");
            require_condition(kernel[2] == in_dim[2], "init_conv: i canali del kernel devono coincidere con quelli in input");
            require_condition(
                dim[0] == compute_spatial_output_dim(
                    in_dim[0],
                    kernel[0],
                    stride_in[0],
                    padding_in[0],
                    "init_conv",
                    "height"
                ),
                "init_conv: altezza output incoerente"
            );
            require_condition(
                dim[1] == compute_spatial_output_dim(
                    in_dim[1],
                    kernel[1],
                    stride_in[1],
                    padding_in[1],
                    "init_conv",
                    "width"
                ),
                "init_conv: larghezza output incoerente"
            );

            
            set_dims(dim);
            set_input_dims(in_dim);
            reset_dense_storage();
            reset_conv_storage();
            reset_spatial_config();

            for(int i=0; i<3; i++)
                kernel_dim[i] = kernel[i];
            for(int i=0; i<2; i++){
                stride[i] = stride_in[i];
                padding[i] = padding_in[i];
            }

            const int out_channels = dim[2];
            const int fan_in = kernel[0] * kernel[1] * kernel[2];
            std::normal_distribution<float> dist(0.0f, std::sqrt(1.0f/static_cast<float>(fan_in)));

            conv_params.filters.assign(static_cast<std::size_t>(conv_filter_count()), 0.0f);
            conv_params.bias.resize(out_channels, 0.0f);

            for(std::size_t filter_index = 0; filter_index < conv_params.filters.size(); filter_index++){
                conv_params.filters[filter_index] = dist(gen);
            }
        }

        void init_pooling(int dim[3], int in_dim[3], int pool_window[3], int stride_in[2], int padding_in[2], Pooling_type pool_type){
            type = Layer_type::Pooling;
            pooling_type = pool_type;

            validate_positive_dims(dim, "init_pooling output");
            validate_positive_dims(in_dim, "init_pooling input");
            require_condition(pool_window[0] > 0 && pool_window[1] > 0, "init_pooling: finestra pooling deve avere dimensioni positive");
            require_condition(stride_in[0] > 0 && stride_in[1] > 0, "init_pooling: stride deve essere positiva");
            require_condition(padding_in[0] >= 0 && padding_in[1] >= 0, "init_pooling: padding non puo' essere negativo");
            require_condition(dim[2] == in_dim[2], "init_pooling: il pooling deve preservare il numero di canali");
            require_condition(
                dim[0] == compute_spatial_output_dim(
                    in_dim[0],
                    pool_window[0],
                    stride_in[0],
                    padding_in[0],
                    "init_pooling",
                    "height"
                ),
                "init_pooling: altezza output incoerente"
            );
            require_condition(
                dim[1] == compute_spatial_output_dim(
                    in_dim[1],
                    pool_window[1],
                    stride_in[1],
                    padding_in[1],
                    "init_pooling",
                    "width"
                ),
                "init_pooling: larghezza output incoerente"
            );

            set_dims(dim);
            set_input_dims(in_dim);
            reset_dense_storage();
            reset_conv_storage();
            reset_spatial_config();

            for(int i=0; i<3; i++)
                kernel_dim[i] = pool_window[i];
            for(int i=0; i<2; i++){
                stride[i] = stride_in[i];
                padding[i] = padding_in[i];
            }
        }

        void init_flatten(int dim[3], int in_dim[3]){
            type = Layer_type::Flatten;

            validate_positive_dims(in_dim, "init_flatten input");
            require_condition(dim[1] == 1 && dim[2] == 1, "init_flatten: il flatten deve avere shape [N,1,1]");
            require_condition(dim[0] == in_dim[0] * in_dim[1] * in_dim[2], "init_flatten: dimensione output incoerente");

            set_dims(dim);
            set_input_dims(in_dim);
            reset_dense_storage();
            reset_conv_storage();
            reset_spatial_config();
        }

        void init_softmax(int dim[3], int in_dim[3]){
            type = Layer_type::Softmax;

            validate_positive_dims(dim, "init_softmax output");
            validate_positive_dims(in_dim, "init_softmax input");
            require_condition(dims_match(dim, in_dim), "init_softmax: input e output devono avere la stessa shape");

            set_dims(dim);
            set_input_dims(in_dim);
            reset_dense_storage();
            reset_conv_storage();
            reset_spatial_config();
        }

};

using LayerList = std::vector<Layer>;
using RuntimeList = std::vector<LayerRuntime>;
using BatchRuntimeList = std::vector<BatchLayerRuntime>;

inline void init_layer_runtime(const Layer &layer, LayerRuntime &runtime){
    runtime.clear();
    switch(layer.type){
        case Layer_type::Dense:
            runtime.resize(layer.dim_layer);
            break;
        case Layer_type::Conv:
            runtime.resize(layer.dim_layer);
            break;
        case Layer_type::Input:
        case Layer_type::Pooling:
        case Layer_type::Flatten:
            runtime.resize(layer.dim_layer, false);
            if(layer.type == Layer_type::Pooling){
                runtime.pooling_argmax.assign(static_cast<std::size_t>(layer.flat_output_size()), -1);
            }
            break;
        case Layer_type::Softmax:
            runtime.resize(layer.dim_layer, true);
            break;
    }
}

inline void init_runtime_buffers(const LayerList &architecture, RuntimeList &runtime){
    runtime.resize(architecture.size());
    for(std::size_t layer_index = 0; layer_index < architecture.size(); layer_index++){
        init_layer_runtime(architecture[layer_index], runtime[layer_index]);
    }
}

inline void init_batch_layer_runtime(const Layer &layer, BatchLayerRuntime &runtime, int batch_size){
    const int flat_size = layer.flat_output_size();
    const bool needs_activation = !(layer.type == Layer_type::Input || layer.type == Layer_type::Pooling || layer.type == Layer_type::Flatten);
    const bool shape_changed = (runtime.y.batch_size != batch_size) ||
                               (runtime.y.height != layer.dim_layer[0]) ||
                               (runtime.y.width != layer.dim_layer[1]) ||
                               (runtime.y.channels != layer.dim_layer[2]);

    if(shape_changed){
        runtime.clear();
        runtime.resize(batch_size, layer.dim_layer, needs_activation);
    }

    if(layer.type == Layer_type::Pooling){
        const std::size_t required_size = static_cast<std::size_t>(batch_size) * static_cast<std::size_t>(flat_size);
        if(runtime.pooling_argmax.size() != required_size){
            runtime.pooling_argmax.assign(required_size, -1);
        } else {
            std::fill(runtime.pooling_argmax.begin(), runtime.pooling_argmax.end(), -1);
        }
    }
}

inline void init_batch_runtime_buffers(const LayerList &architecture, BatchRuntimeList &runtime, int batch_size){
    runtime.resize(architecture.size());
    for(std::size_t layer_index = 0; layer_index < architecture.size(); layer_index++){
        init_batch_layer_runtime(architecture[layer_index], runtime[layer_index], batch_size);
    }
}

inline const std::vector<float> &runtime_output_buffer(const Layer &layer, const LayerRuntime &runtime){
    switch(layer.type){
        case Layer_type::Dense:
        case Layer_type::Conv:
        case Layer_type::Input:
        case Layer_type::Pooling:
        case Layer_type::Flatten:
        case Layer_type::Softmax:
            return runtime.y.data;
    }

    return runtime.y.data;
}

inline const std::vector<float> &runtime_activation_buffer(const Layer &layer, const LayerRuntime &runtime){
    switch(layer.type){
        case Layer_type::Dense:
        case Layer_type::Conv:
            return runtime.a.data;
        case Layer_type::Input:
        case Layer_type::Pooling:
        case Layer_type::Flatten:
            return runtime.y.data;
        case Layer_type::Softmax:
            return runtime.a.data;
    }

    return runtime.a.data;
}

inline const std::vector<float> &runtime_delta_buffer(const Layer &layer, const LayerRuntime &runtime){
    switch(layer.type){
        case Layer_type::Dense:
        case Layer_type::Conv:
        case Layer_type::Input:
        case Layer_type::Pooling:
        case Layer_type::Flatten:
        case Layer_type::Softmax:
            return runtime.delta.data;
    }

    return runtime.delta.data;
}
