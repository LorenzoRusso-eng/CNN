#include "engine/forward.hpp"
#include "math/activation_kernels.hpp"
#include "shared/openmp_utils.hpp"
#include "engine/im2col.hpp"

// Questo file contiene l'implementazione del forward pass layer per layer.
#include <algorithm>
#include <cmath>
#include <limits>
#include <openblas/cblas.h>

namespace {

inline bool use_lookahead_params(const ParameterBuffer *velocity, float momentum){
    return velocity != nullptr && momentum != 0.0f;
}

void forward_dense_layer(int layer_index, const Layer &current, LayerRuntime &current_runtime, const Layer &previous, const LayerRuntime &previous_runtime, const Activation &act, const ParameterBuffer *velocity, float momentum){

    const auto &previous_output = runtime_output_buffer(previous, previous_runtime);
    const float *previous_output_data = previous_output.data();
    const float *weight_data = current.dense_params.weights.data();
    const float *bias_data = current.dense_params.bias.data();
    const int in_features = current.dense_input_size;
    const int out_features = current.dense_output_size;
    const bool use_lookahead = use_lookahead_params(velocity, momentum);

    if(!use_lookahead){

        cblas_sgemv(
            CblasRowMajor, CblasNoTrans,
            out_features, in_features,
            1.0f, weight_data, in_features,
            previous_output_data, 1,
            0.0f, current_runtime.dense_state.a.data(), 1
        );
        activation_kernels::dispatch_forward_op(act, [&](const auto &forward_op){
            activation_kernels::apply_bias_activation_buffer(
                current_runtime.dense_state.a.data(),
                current_runtime.dense_state.y.data(),
                bias_data,
                out_features,
                forward_op
            );
        });
        return;
    }

    const float *velocity_weight_data = velocity->dense_weights[layer_index].data();
    const float *velocity_bias_data = velocity->dense_biases[layer_index].data();

    cblas_sgemv(
        CblasRowMajor, CblasNoTrans,
        out_features, in_features,
        1.0f, weight_data, in_features,
        previous_output_data, 1,
        0.0f, current_runtime.dense_state.a.data(), 1
    );
    cblas_sgemv(
        CblasRowMajor, CblasNoTrans,
        out_features, in_features,
        momentum, velocity_weight_data, in_features,
        previous_output_data, 1,
        1.0f, current_runtime.dense_state.a.data(), 1
    );
    activation_kernels::dispatch_forward_op(act, [&](const auto &forward_op){
        activation_kernels::apply_lookahead_bias_activation_buffer(
            current_runtime.dense_state.a.data(),
            current_runtime.dense_state.y.data(),
            bias_data,
            velocity_bias_data,
            momentum,
            out_features,
            forward_op
        );
    });
}

void forward_conv_layer(int layer_index, const Layer &current, LayerRuntime &current_runtime, const Layer &previous, const LayerRuntime &previous_runtime, const Activation &act, const ParameterBuffer *velocity, float momentum){
    const float *bias_data = current.conv_params.bias.data();
    const int num_filters = current.dim_layer[2];
    const int in_features = current.kernel_dim[0] * current.kernel_dim[1] * current.kernel_dim[2];
    const int out_features = current.dim_layer[0] * current.dim_layer[1];
    const float *filters = current.conv_params.filters.data();
    const bool use_lookahead = use_lookahead_params(velocity, momentum);
    im2col(previous, current, previous_runtime, current_runtime.conv_im2col);
    const float *patched_input = current_runtime.conv_im2col.data();
    
    if(!use_lookahead){
        cblas_sgemm(
            CblasRowMajor, CblasTrans, CblasTrans,
            out_features, num_filters, in_features,
            1.0f, patched_input, out_features,
            filters, in_features,
            0.0f, current_runtime.conv_state.a.data(), num_filters
        );
        
        activation_kernels::dispatch_forward_op(act, [&](const auto &forward_op){
            activation_kernels::apply_repeated_bias_activation_buffer(
                current_runtime.conv_state.a.data(),
                current_runtime.conv_state.y.data(),
                bias_data,
                out_features,
                num_filters,
                forward_op
            );
        });
        return;
    }

    const float *velocity_filter_data = velocity->conv_weights[layer_index].data();
    const float *velocity_bias_data = velocity->conv_biases[layer_index].data();
    
    cblas_sgemm(
        CblasRowMajor, CblasTrans, CblasTrans,
        out_features, num_filters, in_features,
        1.0f, patched_input, out_features,
        filters, in_features,
        0.0f, current_runtime.conv_state.a.data(), num_filters
    );
    cblas_sgemm(
        CblasRowMajor, CblasTrans, CblasTrans,
        out_features, num_filters, in_features,
        momentum, patched_input, out_features,
        velocity_filter_data, in_features,
        1.0f, current_runtime.conv_state.a.data(), num_filters
    );
    activation_kernels::dispatch_forward_op(act, [&](const auto &forward_op){
        activation_kernels::apply_repeated_lookahead_bias_activation_buffer(
            current_runtime.conv_state.a.data(),
            current_runtime.conv_state.y.data(),
            bias_data,
            velocity_bias_data,
            momentum,
            out_features,
            num_filters,
            forward_op
        );
    });
}

void forward_pooling_layer(const Layer &current, LayerRuntime &current_runtime, const Layer &previous, const LayerRuntime &previous_runtime, bool enable_parallel){
    const auto &previous_output = runtime_output_buffer(previous, previous_runtime);
    const float *previous_output_data = previous_output.data();
    const int previous_height = previous.dim_layer[0];
    const int previous_width = previous.dim_layer[1];
    const int output_channels = current.dim_layer[2];
    const int output_height = current.dim_layer[0];
    const int output_width = current.dim_layer[1];
    const int kernel_height = current.kernel_dim[0];
    const int kernel_width = current.kernel_dim[1];
    const int stride_h = current.stride[0];
    const int stride_w = current.stride[1];
    const int padding_h = current.padding[0];
    const int padding_w = current.padding[1];
    const std::size_t input_row_stride = static_cast<std::size_t>(previous_width) * static_cast<std::size_t>(output_channels);
    const std::size_t output_row_stride = static_cast<std::size_t>(output_width) * static_cast<std::size_t>(output_channels);
    const bool has_padding = (padding_h != 0) || (padding_w != 0);
    const std::size_t pooling_work_items = static_cast<std::size_t>(output_height) *
                                           static_cast<std::size_t>(output_width) *
                                           static_cast<std::size_t>(output_channels) *
                                           static_cast<std::size_t>(std::max(1, kernel_height * kernel_width));
    const bool use_parallel_pooling = enable_parallel && pooling_work_items > 4096;

    if(current.pooling_type == Pooling_type::Max && current_runtime.pooling_argmax.size() != current_runtime.state.y.size()){
        current_runtime.pooling_argmax.assign(current_runtime.state.y.size(), -1);
    }

    switch(current.pooling_type){
        case Pooling_type::Max: {
            if(!has_padding){
                NN_OMP_PARALLEL_FOR_COLLAPSE_IF(2, use_parallel_pooling)
                for(int out_i=0; out_i<output_height; out_i++){
                    for(int out_j=0; out_j<output_width; out_j++){
                        const std::size_t output_base =
                            static_cast<std::size_t>(out_i) * output_row_stride +
                            static_cast<std::size_t>(out_j) * static_cast<std::size_t>(output_channels);
                        float *y_out = current_runtime.state.y.data() + static_cast<std::ptrdiff_t>(output_base);
                        int *argmax_out = current_runtime.pooling_argmax.data() + static_cast<std::ptrdiff_t>(output_base);
                        const int input_origin_i = out_i * stride_h;
                        const int input_origin_j = out_j * stride_w;

                        NN_OMP_SIMD
                        for(int out_k=0; out_k<output_channels; out_k++){
                            y_out[out_k] = -std::numeric_limits<float>::infinity();
                            argmax_out[out_k] = -1;
                        }

                        for(int kh=0; kh<kernel_height; kh++){
                            const int in_i = input_origin_i + kh;
                            const std::size_t input_row_base = static_cast<std::size_t>(in_i) * input_row_stride;
                            for(int kw=0; kw<kernel_width; kw++){
                                const int in_j = input_origin_j + kw;
                                const std::size_t input_base =
                                    input_row_base +
                                    static_cast<std::size_t>(in_j) *
                                    static_cast<std::size_t>(output_channels);
                                const float *src = previous_output_data + static_cast<std::ptrdiff_t>(input_base);
                                for(int out_k=0; out_k<output_channels; out_k++){
                                    if(src[out_k] > y_out[out_k]){
                                        y_out[out_k] = src[out_k];
                                        argmax_out[out_k] = static_cast<int>(input_base) + out_k;
                                    }
                                }
                            }
                        }

                    }
                }
            } else {
                NN_OMP_PARALLEL_FOR_COLLAPSE_IF(2, use_parallel_pooling)
                for(int out_i=0; out_i<output_height; out_i++){
                    for(int out_j=0; out_j<output_width; out_j++){
                        const std::size_t output_base =
                            static_cast<std::size_t>(out_i) * output_row_stride +
                            static_cast<std::size_t>(out_j) * static_cast<std::size_t>(output_channels);
                        float *y_out = current_runtime.state.y.data() + static_cast<std::ptrdiff_t>(output_base);
                        int *argmax_out = current_runtime.pooling_argmax.data() + static_cast<std::ptrdiff_t>(output_base);
                        const int input_origin_i = out_i * stride_h - padding_h;
                        const int input_origin_j = out_j * stride_w - padding_w;
                        const int kh_begin = std::max(0, -input_origin_i);
                        const int kh_end = std::min(kernel_height, previous_height - input_origin_i);
                        const int kw_begin = std::max(0, -input_origin_j);
                        const int kw_end = std::min(kernel_width, previous_width - input_origin_j);

                        if(kh_begin >= kh_end || kw_begin >= kw_end){
                            NN_OMP_SIMD
                            for(int out_k=0; out_k<output_channels; out_k++){
                                y_out[out_k] = 0.0f;
                                argmax_out[out_k] = -1;
                            }
                            continue;
                        }

                        NN_OMP_SIMD
                        for(int out_k=0; out_k<output_channels; out_k++){
                            y_out[out_k] = -std::numeric_limits<float>::infinity();
                            argmax_out[out_k] = -1;
                        }

                        for(int kh=kh_begin; kh<kh_end; kh++){
                            const int in_i = input_origin_i + kh;
                            const std::size_t input_row_base = static_cast<std::size_t>(in_i) * input_row_stride;
                            for(int kw=kw_begin; kw<kw_end; kw++){
                                const int in_j = input_origin_j + kw;
                                const std::size_t input_base =
                                    input_row_base +
                                    static_cast<std::size_t>(in_j) *
                                    static_cast<std::size_t>(output_channels);
                                const float *src = previous_output_data + static_cast<std::ptrdiff_t>(input_base);
                                for(int out_k=0; out_k<output_channels; out_k++){
                                    if(src[out_k] > y_out[out_k]){
                                        y_out[out_k] = src[out_k];
                                        argmax_out[out_k] = static_cast<int>(input_base) + out_k;
                                    }
                                }
                            }
                        }

                    }
                }
            }
            break;
        }

        case Pooling_type::Average: {
            if(!has_padding){
                const float inv_count = 1.0f / static_cast<float>(kernel_height * kernel_width);
                NN_OMP_PARALLEL_FOR_COLLAPSE_IF(2, use_parallel_pooling)
                for(int out_i=0; out_i<output_height; out_i++){
                    for(int out_j=0; out_j<output_width; out_j++){
                        const std::size_t output_base =
                            static_cast<std::size_t>(out_i) * output_row_stride +
                            static_cast<std::size_t>(out_j) * static_cast<std::size_t>(output_channels);
                        float *y_out = current_runtime.state.y.data() + static_cast<std::ptrdiff_t>(output_base);
                        const int input_origin_i = out_i * stride_h;
                        const int input_origin_j = out_j * stride_w;

                        NN_OMP_SIMD
                        for(int out_k=0; out_k<output_channels; out_k++){
                            y_out[out_k] = 0.0f;
                        }

                        for(int kh=0; kh<kernel_height; kh++){
                            const int in_i = input_origin_i + kh;
                            const std::size_t input_row_base = static_cast<std::size_t>(in_i) * input_row_stride;
                            for(int kw=0; kw<kernel_width; kw++){
                                const int in_j = input_origin_j + kw;
                                const std::size_t input_base =
                                    input_row_base +
                                    static_cast<std::size_t>(in_j) *
                                    static_cast<std::size_t>(output_channels);
                                const float *src = previous_output_data + static_cast<std::ptrdiff_t>(input_base);
                                NN_OMP_SIMD
                                for(int out_k=0; out_k<output_channels; out_k++){
                                    y_out[out_k] += src[out_k];
                                }
                            }
                        }

                        NN_OMP_SIMD
                        for(int out_k=0; out_k<output_channels; out_k++){
                            const float pooled_value = y_out[out_k] * inv_count;
                            y_out[out_k] = pooled_value;
                        }
                    }
                }
            } else {
                NN_OMP_PARALLEL_FOR_COLLAPSE_IF(2, use_parallel_pooling)
                for(int out_i=0; out_i<output_height; out_i++){
                    for(int out_j=0; out_j<output_width; out_j++){
                        const std::size_t output_base =
                            static_cast<std::size_t>(out_i) * output_row_stride +
                            static_cast<std::size_t>(out_j) * static_cast<std::size_t>(output_channels);
                        float *y_out = current_runtime.state.y.data() + static_cast<std::ptrdiff_t>(output_base);
                        const int input_origin_i = out_i * stride_h - padding_h;
                        const int input_origin_j = out_j * stride_w - padding_w;
                        const int kh_begin = std::max(0, -input_origin_i);
                        const int kh_end = std::min(kernel_height, previous_height - input_origin_i);
                        const int kw_begin = std::max(0, -input_origin_j);
                        const int kw_end = std::min(kernel_width, previous_width - input_origin_j);
                        const int valid_count = (kh_end - kh_begin) * (kw_end - kw_begin);

                        if(valid_count <= 0){
                            NN_OMP_SIMD
                            for(int out_k=0; out_k<output_channels; out_k++){
                                y_out[out_k] = 0.0f;
                            }
                            continue;
                        }

                        NN_OMP_SIMD
                        for(int out_k=0; out_k<output_channels; out_k++){
                            y_out[out_k] = 0.0f;
                        }

                        for(int kh=kh_begin; kh<kh_end; kh++){
                            const int in_i = input_origin_i + kh;
                            const std::size_t input_row_base = static_cast<std::size_t>(in_i) * input_row_stride;
                            for(int kw=kw_begin; kw<kw_end; kw++){
                                const int in_j = input_origin_j + kw;
                                const std::size_t input_base =
                                    input_row_base +
                                    static_cast<std::size_t>(in_j) *
                                    static_cast<std::size_t>(output_channels);
                                const float *src = previous_output_data + static_cast<std::ptrdiff_t>(input_base);
                                NN_OMP_SIMD
                                for(int out_k=0; out_k<output_channels; out_k++){
                                    y_out[out_k] += src[out_k];
                                }
                            }
                        }

                        const float inv_count = 1.0f / static_cast<float>(valid_count);
                        NN_OMP_SIMD
                        for(int out_k=0; out_k<output_channels; out_k++){
                            const float pooled_value = y_out[out_k] * inv_count;
                            y_out[out_k] = pooled_value;
                        }
                    }
                }
            }
            break;
        }

        case Pooling_type::L2: {
            if(!has_padding){
                NN_OMP_PARALLEL_FOR_COLLAPSE_IF(2, use_parallel_pooling)
                for(int out_i=0; out_i<output_height; out_i++){
                    for(int out_j=0; out_j<output_width; out_j++){
                        const std::size_t output_base =
                            static_cast<std::size_t>(out_i) * output_row_stride +
                            static_cast<std::size_t>(out_j) * static_cast<std::size_t>(output_channels);
                        float *y_out = current_runtime.state.y.data() + static_cast<std::ptrdiff_t>(output_base);
                        const int input_origin_i = out_i * stride_h;
                        const int input_origin_j = out_j * stride_w;

                        NN_OMP_SIMD
                        for(int out_k=0; out_k<output_channels; out_k++){
                            y_out[out_k] = 0.0f;
                        }

                        for(int kh=0; kh<kernel_height; kh++){
                            const int in_i = input_origin_i + kh;
                            const std::size_t input_row_base = static_cast<std::size_t>(in_i) * input_row_stride;
                            for(int kw=0; kw<kernel_width; kw++){
                                const int in_j = input_origin_j + kw;
                                const std::size_t input_base =
                                    input_row_base +
                                    static_cast<std::size_t>(in_j) *
                                    static_cast<std::size_t>(output_channels);
                                const float *src = previous_output_data + static_cast<std::ptrdiff_t>(input_base);
                                NN_OMP_SIMD
                                for(int out_k=0; out_k<output_channels; out_k++){
                                    y_out[out_k] += src[out_k] * src[out_k];
                                }
                            }
                        }

                        NN_OMP_SIMD
                        for(int out_k=0; out_k<output_channels; out_k++){
                            const float pooled_value = std::sqrt(y_out[out_k]);
                            y_out[out_k] = pooled_value;
                        }
                    }
                }
            } else {
                NN_OMP_PARALLEL_FOR_COLLAPSE_IF(2, use_parallel_pooling)
                for(int out_i=0; out_i<output_height; out_i++){
                    for(int out_j=0; out_j<output_width; out_j++){
                        const std::size_t output_base =
                            static_cast<std::size_t>(out_i) * output_row_stride +
                            static_cast<std::size_t>(out_j) * static_cast<std::size_t>(output_channels);
                        float *y_out = current_runtime.state.y.data() + static_cast<std::ptrdiff_t>(output_base);
                        const int input_origin_i = out_i * stride_h - padding_h;
                        const int input_origin_j = out_j * stride_w - padding_w;
                        const int kh_begin = std::max(0, -input_origin_i);
                        const int kh_end = std::min(kernel_height, previous_height - input_origin_i);
                        const int kw_begin = std::max(0, -input_origin_j);
                        const int kw_end = std::min(kernel_width, previous_width - input_origin_j);

                        if(kh_begin >= kh_end || kw_begin >= kw_end){
                            NN_OMP_SIMD
                            for(int out_k=0; out_k<output_channels; out_k++){
                                y_out[out_k] = 0.0f;
                            }
                            continue;
                        }

                        NN_OMP_SIMD
                        for(int out_k=0; out_k<output_channels; out_k++){
                            y_out[out_k] = 0.0f;
                        }

                        for(int kh=kh_begin; kh<kh_end; kh++){
                            const int in_i = input_origin_i + kh;
                            const std::size_t input_row_base = static_cast<std::size_t>(in_i) * input_row_stride;
                            for(int kw=kw_begin; kw<kw_end; kw++){
                                const int in_j = input_origin_j + kw;
                                const std::size_t input_base =
                                    input_row_base +
                                    static_cast<std::size_t>(in_j) *
                                    static_cast<std::size_t>(output_channels);
                                const float *src = previous_output_data + static_cast<std::ptrdiff_t>(input_base);
                                NN_OMP_SIMD
                                for(int out_k=0; out_k<output_channels; out_k++){
                                    y_out[out_k] += src[out_k] * src[out_k];
                                }
                            }
                        }

                        NN_OMP_SIMD
                        for(int out_k=0; out_k<output_channels; out_k++){
                            const float pooled_value = std::sqrt(y_out[out_k]);
                            y_out[out_k] = pooled_value;
                        }
                    }
                }
            }
            break;
        }
    }
}

void forward_flatten_layer(const Layer &current, LayerRuntime &current_runtime, const Layer &previous, const LayerRuntime &previous_runtime, bool enable_parallel){
    const auto &previous_output = runtime_output_buffer(previous, previous_runtime);
    auto &output = current_runtime.state.y;

    NN_OMP_PARALLEL_FOR_IF(enable_parallel && previous_output.size() > 4096)
    for(int idx = 0; idx < static_cast<int>(previous_output.size()); idx++){
        const std::size_t data_index = static_cast<std::size_t>(idx);
        output[data_index] = previous_output[data_index];
    }
}

void forward_lrn_layer(const Layer &current, LayerRuntime &current_runtime, const Layer &previous, const LayerRuntime &previous_runtime, bool enable_parallel){
    const int local_size = std::max(1, current.lrn_local_size);
    const int radius = local_size / 2;
    const float alpha_over_size = current.lrn_alpha / static_cast<float>(local_size);
    const float lrn_k = current.lrn_k;
    const float lrn_beta = current.lrn_beta;
    const auto &previous_output = runtime_output_buffer(previous, previous_runtime);
    const float *previous_output_data = previous_output.data();
    float *current_a = current_runtime.state.a.data();
    float *current_y = current_runtime.state.y.data();
    const int width = current.dim_layer[1];
    const int channels = current.dim_layer[2];
    const std::size_t row_stride = static_cast<std::size_t>(width) * static_cast<std::size_t>(channels);
    const int initial_right = std::min(channels - 1, radius);
    const std::size_t lrn_work_items = static_cast<std::size_t>(current.dim_layer[0]) *
                                       static_cast<std::size_t>(current.dim_layer[1]) *
                                       static_cast<std::size_t>(channels) *
                                       static_cast<std::size_t>(local_size);
    const bool use_parallel_lrn = enable_parallel && lrn_work_items > 4096;

    NN_OMP_PARALLEL_FOR_COLLAPSE_IF(2, use_parallel_lrn)
    for(int i=0; i<current.dim_layer[0]; i++){
        for(int j=0; j<current.dim_layer[1]; j++){
            const std::size_t pixel_offset = static_cast<std::size_t>(i) * row_stride + static_cast<std::size_t>(j) * static_cast<std::size_t>(channels);
            const float *src = previous_output_data + static_cast<std::ptrdiff_t>(pixel_offset);
            float *a_out = current_a + static_cast<std::ptrdiff_t>(pixel_offset);
            float *y_out = current_y + static_cast<std::ptrdiff_t>(pixel_offset);

            NN_OMP_SIMD
            for(int c = 0; c < channels; c++){
                a_out[c] = src[c];
            }

            float squared_sum = 0.0f;
            NN_OMP_SIMD_REDUCTION_PLUS(squared_sum)
            for(int c = 0; c <= initial_right; c++){
                const float value = src[c];
                squared_sum += value * value;
            }

            for(int k = 0; k < channels; k++){
                const float scale = lrn_k + alpha_over_size * squared_sum;
                y_out[k] = a_out[k] / std::pow(scale, lrn_beta);

                const int leaving = k - radius;
                const int entering = k + radius + 1;
                if(leaving >= 0){
                    const float value = src[leaving];
                    squared_sum -= value * value;
                }
                if(entering < channels){
                    const float value = src[entering];
                    squared_sum += value * value;
                }
            }
        }
    }
}

void forward_softmax_layer(const Layer &current, LayerRuntime &current_runtime, const Layer &previous, const LayerRuntime &previous_runtime){
    const auto &previous_output = runtime_output_buffer(previous, previous_runtime);
    const int output_size = current.flat_output_size();
    const float *src = previous_output.data();
    float *a_out = current_runtime.state.a.data();
    float *y_out = current_runtime.state.y.data();

    float max_logit = -std::numeric_limits<float>::infinity();
    NN_OMP_SIMD
    for(int idx=0; idx<output_size; idx++){
        a_out[idx] = src[idx];
    }

    NN_OMP_SIMD_REDUCTION_MAX(max_logit)
    for(int idx=0; idx<output_size; idx++){
        max_logit = std::max(max_logit, a_out[idx]);
    }

    float sum_exp = 0.0f;
    NN_OMP_SIMD_REDUCTION_PLUS(sum_exp)
    for(int idx=0; idx<output_size; idx++){
        const float stabilized = a_out[idx] - max_logit;
        const float exp_value = std::exp(stabilized);
        y_out[idx] = exp_value;
        sum_exp += exp_value;
    }

    require_condition(sum_exp > 0.0f, "forward_softmax_layer: somma delle esponenziali non valida");
    const float inv_sum_exp = 1.0f / sum_exp;

    NN_OMP_SIMD
    for(int idx=0; idx<output_size; idx++){
        y_out[idx] *= inv_sum_exp;
    }
}

} // namespace

void feed_input(const Tensor3D &input, const Layer &first, LayerRuntime &first_runtime){
    validate_tensor3d_shape(input, first.dim_layer, "feed_input");
    std::copy(input.data.begin(), input.data.end(), first_runtime.state.y.begin());
}

void forwardprop(const LayerList &architecture, RuntimeList &runtime, int num_layers, const Activation &hidden_activation, const Activation &output_activation, ExecutionPolicy policy, const ParameterBuffer *velocity, float momentum){
    const bool enable_parallel = policy.allows_intra_example_parallelism();
    for(int l=1; l<num_layers; l++){
        const Layer &current = architecture[l];
        LayerRuntime &current_runtime = runtime[l];
        const Layer &previous = architecture[l-1];
        const LayerRuntime &previous_runtime = runtime[l-1];
        const bool next_is_softmax = (l < num_layers - 1) && (architecture[l + 1].type == Layer_type::Softmax);
        const Activation &act = next_is_softmax ? identity : (l < num_layers - 1) ? hidden_activation : output_activation;

        switch(current.type){
            case Layer_type::Dense:
                forward_dense_layer(l, current, current_runtime, previous, previous_runtime, act, velocity, momentum);
                break;
            case Layer_type::Conv:
                forward_conv_layer(l, current, current_runtime, previous, previous_runtime, act, velocity, momentum);
                break;
            case Layer_type::Pooling:
                forward_pooling_layer(current, current_runtime, previous, previous_runtime, enable_parallel);
                break;
            case Layer_type::Flatten:
                forward_flatten_layer(current, current_runtime, previous, previous_runtime, enable_parallel);
                break;
            case Layer_type::LRN:
                forward_lrn_layer(current, current_runtime, previous, previous_runtime, enable_parallel);
                break;
            case Layer_type::Softmax:
                forward_softmax_layer(current, current_runtime, previous, previous_runtime);
                break;
            case Layer_type::Input:
                break;
        }
    }
}
