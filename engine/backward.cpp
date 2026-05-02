#include "engine/backward.hpp"
#include "math/activation_kernels.hpp"
#include "engine/im2col.hpp"
#include "shared/openmp_utils.hpp"

// Questo file contiene l'implementazione del backward pass e del routing dei delta.
#include <algorithm>
#include <cmath>
#include <limits>
#include <openblas/cblas.h>

namespace {

inline bool use_lookahead_params(const ParameterBuffer *velocity, float momentum){
    return velocity != nullptr && momentum != 0.0f;
}

void build_cost_from_next_dense_all(const Layer &current, const Layer &next, const LayerRuntime &next_runtime, int next_layer_index, const ParameterBuffer *velocity, float momentum, std::vector<float> &out_cost){
    const int current_features = current.flat_output_size();
    const int next_out_features = next.dense_output_size;
    const int next_in_features = next.dense_input_size;
    out_cost.assign(static_cast<std::size_t>(current_features), 0.0f);

    cblas_sgemv(
        CblasRowMajor, CblasTrans,
        next_out_features, next_in_features,
        -1.0f, next.dense_params.weights.data(), next_in_features,
        next_runtime.delta.data.data(), 1,
        0.0f, out_cost.data(), 1
    );

    if(use_lookahead_params(velocity, momentum)){
        cblas_sgemv(
            CblasRowMajor, CblasTrans,
            next_out_features, next_in_features,
            -momentum, velocity->dense_weights[next_layer_index].data(), next_in_features,
            next_runtime.delta.data.data(), 1,
            1.0f, out_cost.data(), 1
        );
    }
}

void build_cost_from_next_conv_all(const Layer &current, LayerRuntime &current_runtime, const Layer &next, const LayerRuntime &next_runtime, int next_layer_index, const ParameterBuffer *velocity, float momentum, std::vector<float> &out_cost){
    const int current_channels = current.dim_layer[2];
    const int next_height = next.dim_layer[0];
    const int next_width = next.dim_layer[1];
    const int next_channels = next.dim_layer[2];
    const int kernel_height = next.kernel_dim[0];
    const int kernel_width = next.kernel_dim[1];
    const int in_features = kernel_height * kernel_width * current_channels;
    const int patches = next_height * next_width;

    out_cost.assign(static_cast<std::size_t>(current.flat_output_size()), 0.0f);
    current_runtime.conv_im2col.resize(static_cast<std::size_t>(patches) * static_cast<std::size_t>(in_features));

    cblas_sgemm(
        CblasRowMajor, CblasNoTrans, CblasNoTrans,
        patches, in_features, next_channels,
        1.0f, next_runtime.delta.data.data(), next_channels,
        next.conv_params.filters.data(), in_features,
        0.0f, current_runtime.conv_im2col.data(), in_features
    );

    if(use_lookahead_params(velocity, momentum)){
        cblas_sgemm(
            CblasRowMajor, CblasNoTrans, CblasNoTrans,
            patches, in_features, next_channels,
            momentum, next_runtime.delta.data.data(), next_channels,
            velocity->conv_weights[next_layer_index].data(), in_features,
            1.0f, current_runtime.conv_im2col.data(), in_features
        );
    }

    col2im(current, next, current_runtime.conv_im2col.data(), out_cost, -1.0f);
}

void build_cost_from_next_pool_all(const Layer &current, const LayerRuntime &current_runtime, const Layer &next, const LayerRuntime &next_runtime, std::vector<float> &out_cost){
    const auto &current_output = runtime_output_buffer(current, current_runtime);
    const auto &next_output = runtime_output_buffer(next, next_runtime);
    const auto &next_delta = runtime_delta_buffer(next, next_runtime);
    const int current_height = current.dim_layer[0];
    const int current_width = current.dim_layer[1];
    const int channels = current.dim_layer[2];
    const int output_height = next.dim_layer[0];
    const int output_width = next.dim_layer[1];
    const int kernel_height = next.kernel_dim[0];
    const int kernel_width = next.kernel_dim[1];
    const int stride_h = next.stride[0];
    const int stride_w = next.stride[1];
    const int padding_h = next.padding[0];
    const int padding_w = next.padding[1];
    const bool has_padding = (padding_h != 0) || (padding_w != 0);
    const std::size_t pool_backprop_work_items = static_cast<std::size_t>(output_height) *
                                                 static_cast<std::size_t>(output_width) *
                                                 static_cast<std::size_t>(channels) *
                                                 static_cast<std::size_t>(std::max(1, kernel_height * kernel_width));
    const bool use_parallel_pool_backprop = pool_backprop_work_items > 4096;

    out_cost.assign(static_cast<std::size_t>(current.flat_output_size()), 0.0f);

    switch(next.pooling_type){
        case Pooling_type::Average:
            if(!has_padding){
                const float inv_count = 1.0f / static_cast<float>(kernel_height * kernel_width);

                NN_OMP_PARALLEL_FOR_IF(use_parallel_pool_backprop)
                for(int c=0; c<channels; c++){
                    for(int out_i=0; out_i<output_height; out_i++){
                        const int start_i = out_i * stride_h;
                        for(int out_j=0; out_j<output_width; out_j++){
                            const int start_j = out_j * stride_w;
                            const float g = next_delta[static_cast<std::size_t>(next.flat_index(out_i, out_j, c))] * inv_count;
                            for(int kh=0; kh<kernel_height; kh++){
                                const int in_i = start_i + kh;
                                for(int kw=0; kw<kernel_width; kw++){
                                    const int in_j = start_j + kw;
                                    out_cost[static_cast<std::size_t>(current.flat_index(in_i, in_j, c))] -= g;
                                }
                            }
                        }
                    }
                }
            } else {

                NN_OMP_PARALLEL_FOR_IF(use_parallel_pool_backprop)
                for(int c=0; c<channels; c++){
                    for(int out_i=0; out_i<output_height; out_i++){
                        const int start_i = out_i * stride_h - padding_h;
                        const int kh_begin = std::max(0, -start_i);
                        const int kh_end = std::min(kernel_height, current_height - start_i);

                        for(int out_j=0; out_j<output_width; out_j++){
                            const int start_j = out_j * stride_w - padding_w;
                            const int kw_begin = std::max(0, -start_j);
                            const int kw_end = std::min(kernel_width, current_width - start_j);
                            const int valid_count = (kh_end - kh_begin) * (kw_end - kw_begin);
                            if(valid_count <= 0){
                                continue;
                            }

                            const float inv = 1.0f / static_cast<float>(valid_count);
                            const float g = next_delta[static_cast<std::size_t>(next.flat_index(out_i, out_j, c))] * inv;
                            for(int kh=kh_begin; kh<kh_end; kh++){
                                const int in_i = start_i + kh;
                                for(int kw=kw_begin; kw<kw_end; kw++){
                                    const int in_j = start_j + kw;
                                    out_cost[static_cast<std::size_t>(current.flat_index(in_i, in_j, c))] -= g;
                                }
                            }
                        }
                    }
                }
            }
            break;

        case Pooling_type::Max:
            NN_OMP_PARALLEL_FOR_IF(use_parallel_pool_backprop)
            for(int c=0; c<channels; c++){
                for(int out_i=0; out_i<output_height; out_i++){
                    for(int out_j=0; out_j<output_width; out_j++){
                        const int out_idx = next.flat_index(out_i, out_j, c);
                        const int winner = next_runtime.pooling_argmax[static_cast<std::size_t>(out_idx)];
                        if(winner >= 0){
                            out_cost[static_cast<std::size_t>(winner)] -= next_delta[static_cast<std::size_t>(out_idx)];
                        }
                    }
                }
            }
            break;

        case Pooling_type::L2:
            if(!has_padding){
                const int row_stride = current_width * channels;
                const int pixel_stride = channels;

                NN_OMP_PARALLEL_FOR_IF(use_parallel_pool_backprop)
                for(int c=0; c<channels; c++){
                    for(int out_i=0; out_i<output_height; out_i++){
                        const int start_i = out_i * stride_h;
                        for(int out_j=0; out_j<output_width; out_j++){
                            const int out_idx = next.flat_index(out_i, out_j, c);
                            const float pooled_norm = next_output[static_cast<std::size_t>(out_idx)];
                            if(pooled_norm <= 0.0f){
                                continue;
                            }

                            const float g_over_norm = next_delta[static_cast<std::size_t>(out_idx)] / pooled_norm;
                            const int start_j = out_j * stride_w;

                            for(int kh=0; kh<kernel_height; kh++){
                                const int in_i = start_i + kh;
                                const int row_base = in_i * row_stride;
                                for(int kw=0; kw<kernel_width; kw++){
                                    const int in_j = start_j + kw;
                                    const int in_idx = row_base + in_j * pixel_stride + c;
                                    out_cost[static_cast<std::size_t>(in_idx)] -= g_over_norm * current_output[static_cast<std::size_t>(in_idx)];
                                }
                            }
                        }
                    }
                }
            } else {
                const int row_stride = current_width * channels;
                const int pixel_stride = channels;

                NN_OMP_PARALLEL_FOR_IF(use_parallel_pool_backprop)
                for(int c=0; c<channels; c++){
                    for(int out_i=0; out_i<output_height; out_i++){
                        const int start_i = out_i * stride_h - padding_h;
                        const int kh_begin = std::max(0, -start_i);
                        const int kh_end = std::min(kernel_height, current_height - start_i);
                        if(kh_begin >= kh_end){
                            continue;
                        }

                        for(int out_j=0; out_j<output_width; out_j++){
                            const int start_j = out_j * stride_w - padding_w;
                            const int kw_begin = std::max(0, -start_j);
                            const int kw_end = std::min(kernel_width, current_width - start_j);
                            if(kw_begin >= kw_end){
                                continue;
                            }

                            const int out_idx = next.flat_index(out_i, out_j, c);
                            const float pooled_norm = next_output[static_cast<std::size_t>(out_idx)];
                            if(pooled_norm <= 0.0f){
                                continue;
                            }

                            const float g_over_norm = next_delta[static_cast<std::size_t>(out_idx)] / pooled_norm;

                            for(int kh=kh_begin; kh<kh_end; kh++){
                                const int in_i = start_i + kh;
                                const int row_base = in_i * row_stride;
                                for(int kw=kw_begin; kw<kw_end; kw++){
                                    const int in_j = start_j + kw;
                                    const int in_idx = row_base + in_j * pixel_stride + c;
                                    out_cost[static_cast<std::size_t>(in_idx)] -= g_over_norm * current_output[static_cast<std::size_t>(in_idx)];
                                }
                            }
                        }
                    }
                }
            }
            break;
    }
}

void build_cost_from_next_flatten_all(const Layer &current, const LayerRuntime &next_runtime, std::vector<float> &out_cost){
    out_cost.assign(static_cast<std::size_t>(current.flat_output_size()), 0.0f);
    const int n = std::min(current.flat_output_size(), static_cast<int>(next_runtime.delta.data.size()));
    NN_OMP_PARALLEL_FOR_IF(n > 4096)
    for(int idx=0; idx<n; idx++){
        out_cost[static_cast<std::size_t>(idx)] = -next_runtime.delta.data[static_cast<std::size_t>(idx)];
    }
}

void build_cost_from_next_lrn_all(const Layer &current, const LayerRuntime &current_runtime, const Layer &next, const LayerRuntime &next_runtime, std::vector<float> &out_cost){
    const auto &current_output = runtime_output_buffer(current, current_runtime);
    const auto &next_activation = runtime_activation_buffer(next, next_runtime);
    const auto &next_delta = runtime_delta_buffer(next, next_runtime);
    const int height = current.dim_layer[0];
    const int width = current.dim_layer[1];
    const int channels = current.dim_layer[2];
    const int local_size = std::max(1, next.lrn_local_size);
    const int radius = local_size / 2;
    const float alpha_over_size = next.lrn_alpha / static_cast<float>(local_size);
    const float two_alpha_over_size = 2.0f * alpha_over_size;
    const std::size_t lrn_backprop_work_items = static_cast<std::size_t>(height) *
                                                static_cast<std::size_t>(width) *
                                                static_cast<std::size_t>(channels) *
                                                static_cast<std::size_t>(local_size);

    out_cost.assign(static_cast<std::size_t>(current.flat_output_size()), 0.0f);

    NN_OMP_PARALLEL_FOR_COLLAPSE_IF(2, lrn_backprop_work_items > 4096)
    for(int i=0; i<height; i++){
        for(int j=0; j<width; j++){
            const int idx_base = (i * width + j) * channels;
            for(int k=0; k<channels; k++){
                const float current_value = current_output[static_cast<std::size_t>(idx_base + k)];
                float cost_der = 0.0f;

                int start_c = 0;
                int end_c = std::min(channels - 1, radius);
                float squared_sum = 0.0f;
                for(int c=start_c; c<=end_c; c++){
                    const float neighbor = current_output[static_cast<std::size_t>(idx_base + c)];
                    squared_sum += neighbor * neighbor;
                }

                for(int out_k=0; out_k<channels; out_k++){
                    const int prev_start_c = start_c;
                    const int prev_end_c = end_c;
                    start_c = std::max(0, out_k - radius);
                    end_c = std::min(channels - 1, out_k + radius);

                    if(out_k > 0){
                        if(start_c > prev_start_c){
                            const float removed = current_output[static_cast<std::size_t>(idx_base + prev_start_c)];
                            squared_sum -= removed * removed;
                        }
                        if(end_c > prev_end_c){
                            const float added = current_output[static_cast<std::size_t>(idx_base + end_c)];
                            squared_sum += added * added;
                        }
                    }

                    if(k < start_c || k > end_c){
                        continue;
                    }

                    const float scale = next.lrn_k + alpha_over_size * squared_sum;
                    const float normalized_grad = std::pow(scale, -next.lrn_beta);
                    const int out_idx = idx_base + out_k;
                    float coeff =
                        -next.lrn_beta *
                        next_activation[static_cast<std::size_t>(out_idx)] *
                        std::pow(scale, -next.lrn_beta - 1.0f) *
                        two_alpha_over_size *
                        current_value;
                    if(k == out_k){
                        coeff += normalized_grad;
                    }

                    cost_der -= next_delta[static_cast<std::size_t>(out_idx)] * coeff;
                }

                out_cost[static_cast<std::size_t>(idx_base + k)] = cost_der;
            }
        }
    }
}

void build_cost_from_next_softmax_all(const Layer &current, const LayerRuntime &next_runtime, std::vector<float> &out_cost){
    out_cost.assign(static_cast<std::size_t>(current.flat_output_size()), 0.0f);
    const int n = std::min(current.flat_output_size(), static_cast<int>(next_runtime.delta.data.size()));
    NN_OMP_PARALLEL_FOR_IF(n > 4096)
    for(int idx=0; idx<n; idx++){
        out_cost[static_cast<std::size_t>(idx)] = -next_runtime.delta.data[static_cast<std::size_t>(idx)];
    }
}

void build_cost_from_next_layer_all(const Layer &current, LayerRuntime &current_runtime, const Layer &next, const LayerRuntime &next_runtime, int next_layer_index, const ParameterBuffer *velocity, float momentum, std::vector<float> &out_cost){
    switch(next.type){
        case Layer_type::Dense:
            build_cost_from_next_dense_all(current, next, next_runtime, next_layer_index, velocity, momentum, out_cost);
            break;
        case Layer_type::Conv:
            build_cost_from_next_conv_all(current, current_runtime, next, next_runtime, next_layer_index, velocity, momentum, out_cost);
            break;
        case Layer_type::Pooling:
            build_cost_from_next_pool_all(current, current_runtime, next, next_runtime, out_cost);
            break;
        case Layer_type::Flatten:
            build_cost_from_next_flatten_all(current, next_runtime, out_cost);
            break;
        case Layer_type::LRN:
            build_cost_from_next_lrn_all(current, current_runtime, next, next_runtime, out_cost);
            break;
        case Layer_type::Softmax:
            build_cost_from_next_softmax_all(current, next_runtime, out_cost);
            break;
        case Layer_type::Input:
            out_cost.assign(static_cast<std::size_t>(current.flat_output_size()), 0.0f);
            break;
    }
}

void backprop_dense_layer(int layer_index, const Layer &current, LayerRuntime &current_runtime, const Layer &previous, const LayerRuntime &previous_runtime, const Layer *next, const LayerRuntime *next_runtime, ParameterBuffer &gradients, bool is_output, const Loss &loss, const Tensor &desired_output, int output_size, const Activation &act, bool enable_parallel, const ParameterBuffer *velocity, float momentum){
    const auto &previous_output = runtime_output_buffer(previous, previous_runtime);
    const float *previous_output_data = previous_output.data();
    float *gradient_weight_data = gradients.dense_weights[layer_index].data();
    float *delta_data = current_runtime.delta.data.data();
    const int out_features = current.dense_output_size;
    const int in_features = current.dense_input_size;
    const bool use_parallel_dense_delta = enable_parallel && out_features > 4096;

    if(is_output){
        loss_kernels::dispatch_derivative_op(loss, output_size, [&](const auto &loss_derivative_op){
            activation_kernels::dispatch_derivative_op(act, [&](const auto &derivative_op){
                NN_OMP_PARALLEL_FOR_IF(use_parallel_dense_delta)
                for(int out_idx=0; out_idx<out_features; out_idx++){
                    const float output_value = current_runtime.y.data[out_idx];
                    const float cost_der = loss_derivative_op(output_value, desired_output.data[static_cast<std::size_t>(out_idx)]);
                    const float delta = -cost_der * derivative_op(current_runtime.a.data[out_idx]);
                    delta_data[out_idx] = delta;
                }
            });
        });
    } else {
        auto &cost_from_next = current_runtime.backprop_cost_from_next;
        build_cost_from_next_layer_all(current, current_runtime, *next, *next_runtime, layer_index + 1, velocity, momentum, cost_from_next);

        activation_kernels::dispatch_derivative_op(act, [&](const auto &derivative_op){
            NN_OMP_PARALLEL_FOR_IF(use_parallel_dense_delta)
            for(int out_idx=0; out_idx<out_features; out_idx++){
                const float cost_der = cost_from_next[static_cast<std::size_t>(out_idx)];
                const float delta = -cost_der * derivative_op(current_runtime.a.data[out_idx]);
                delta_data[out_idx] = delta;
            }
        });
    }

    std::copy(delta_data, delta_data + out_features, gradients.dense_biases[layer_index].begin());
    cblas_sgemm(
        CblasRowMajor, CblasNoTrans, CblasNoTrans,
        out_features, in_features, 1,
        1.0f, delta_data, 1,
        previous_output_data, in_features,
        0.0f, gradient_weight_data, in_features
    );
}

void backprop_conv_layer(int layer_index, const Layer &current, LayerRuntime &current_runtime, const Layer &previous, const LayerRuntime &previous_runtime, const Layer *next, const LayerRuntime *next_runtime, ParameterBuffer &gradients, bool is_output, const Loss &loss, const Tensor &desired_output, int output_size, const Activation &act, bool enable_parallel, const ParameterBuffer *velocity, float momentum){
    float *gradient_weight_data = gradients.conv_weights[layer_index].data();
    float *delta_data = current_runtime.delta.data.data();
    const int out_height = current.dim_layer[0];
    const int output_width = current.dim_layer[1];
    const int out_channels = current.dim_layer[2];
    const int kernel_height = current.kernel_dim[0];
    const int kernel_width = current.kernel_dim[1];
    const int previous_channels = current.kernel_dim[2];
    const int in_features = kernel_height * kernel_width * previous_channels;
    const int out_features = out_height * output_width;
    const int flat_size = out_features * out_channels;
    auto &cost_from_next = current_runtime.backprop_cost_from_next;
    const bool use_parallel_conv_output_delta =
        enable_parallel &&
        static_cast<std::size_t>(out_height) *
        static_cast<std::size_t>(output_width) *
        static_cast<std::size_t>(out_channels) > 4096;
    const bool use_parallel_conv_hidden_delta = enable_parallel && flat_size > 4096;

    if(!is_output && next != nullptr && next_runtime != nullptr){
        build_cost_from_next_layer_all(current, current_runtime, *next, *next_runtime, layer_index + 1, velocity, momentum, cost_from_next);
    }

    if(is_output){
        loss_kernels::dispatch_derivative_op(loss, output_size, [&](const auto &loss_derivative_op){
            activation_kernels::dispatch_derivative_op(act, [&](const auto &derivative_op){
                NN_OMP_PARALLEL_FOR_COLLAPSE_IF(3, use_parallel_conv_output_delta)
                for(int out_i=0; out_i<out_height; out_i++){
                    for(int out_j=0; out_j<output_width; out_j++){
                        for(int out_k=0; out_k<out_channels; out_k++){
                            const int out_idx = (out_i * output_width + out_j) * out_channels + out_k;
                            const float output_value = current_runtime.y.data[static_cast<std::size_t>(out_idx)];
                            const float activation_value = current_runtime.a.data[static_cast<std::size_t>(out_idx)];
                            const float cost_der = loss_derivative_op(
                                output_value,
                                desired_output.data[static_cast<std::size_t>(desired_output.index(out_i, out_j, out_k))]
                            );
                            const float delta = -cost_der * derivative_op(activation_value);
                            delta_data[out_idx] = delta;
                        }
                    }
                }
            });
        });
    } else {
        activation_kernels::dispatch_derivative_op(act, [&](const auto &derivative_op){
            NN_OMP_PARALLEL_FOR_IF(use_parallel_conv_hidden_delta)
            for(int idx=0; idx<flat_size; idx++){
                const float activation_value = current_runtime.a.data[static_cast<std::size_t>(idx)];
                const float cost_der = cost_from_next[static_cast<std::size_t>(idx)];
                const float delta = -cost_der * derivative_op(activation_value);
                delta_data[idx] = delta;
            }
        });
    }

    auto &ones = current_runtime.reduction_ones;
    ones.assign(static_cast<std::size_t>(out_features), 1.0f);
    cblas_sgemv(
        CblasRowMajor, CblasTrans,
        out_features, out_channels,
        1.0f, delta_data, out_channels,
        ones.data(), 1,
        0.0f, gradients.conv_biases[layer_index].data(), 1
    );

    im2col(previous, current, previous_runtime, current_runtime.conv_im2col);
    cblas_sgemm(
        CblasRowMajor, CblasTrans, CblasTrans,
        out_channels, in_features, out_features,
        1.0f, delta_data, out_channels,
        current_runtime.conv_im2col.data(), out_features,
        0.0f, gradient_weight_data, in_features
    );
}

void backprop_pooling_layer(const Layer &current, LayerRuntime &current_runtime, const Layer *next, const LayerRuntime *next_runtime, int next_layer_index, bool is_output, const Loss &loss, const Tensor &desired_output, int output_size, bool enable_parallel, const ParameterBuffer *velocity, float momentum){
    const int flat_size = current.flat_output_size();
    float *delta_data = current_runtime.delta.data.data();
    const float *output_data = current_runtime.y.data.data();
    const bool use_parallel_delta = enable_parallel && flat_size > 4096;

    if(is_output){
        loss_kernels::dispatch_derivative_op(loss, output_size, [&](const auto &loss_derivative_op){
            NN_OMP_PARALLEL_FOR_IF(use_parallel_delta)
            for(int idx=0; idx<flat_size; idx++){
                const float output_value = output_data[static_cast<std::size_t>(idx)];
                delta_data[static_cast<std::size_t>(idx)] = -loss_derivative_op(output_value, desired_output.data[static_cast<std::size_t>(idx)]);
            }
        });
        return;
    }

    auto &cost_from_next = current_runtime.backprop_cost_from_next;
    build_cost_from_next_layer_all(current, current_runtime, *next, *next_runtime, next_layer_index, velocity, momentum, cost_from_next);
    const float *cost_data = cost_from_next.data();

    NN_OMP_PARALLEL_FOR_IF(use_parallel_delta)
    for(int idx=0; idx<flat_size; idx++){
        delta_data[static_cast<std::size_t>(idx)] = -cost_data[static_cast<std::size_t>(idx)];
    }
}

void backprop_flatten_layer(const Layer &current, LayerRuntime &current_runtime, const Layer *next, const LayerRuntime *next_runtime, int next_layer_index, bool is_output, const Loss &loss, const Tensor &desired_output, int output_size, bool enable_parallel, const ParameterBuffer *velocity, float momentum){
    const int flat_size = current.flat_output_size();
    float *delta_data = current_runtime.delta.data.data();
    const float *output_data = current_runtime.y.data.data();
    const bool use_parallel_delta = enable_parallel && flat_size > 4096;

    if(is_output){
        loss_kernels::dispatch_derivative_op(loss, output_size, [&](const auto &loss_derivative_op){
            NN_OMP_PARALLEL_FOR_IF(use_parallel_delta)
            for(int idx=0; idx<flat_size; idx++){
                const float output_value = output_data[static_cast<std::size_t>(idx)];
                delta_data[static_cast<std::size_t>(idx)] = -loss_derivative_op(output_value, desired_output.data[static_cast<std::size_t>(idx)]);
            }
        });
        return;
    }

    auto &cost_from_next = current_runtime.backprop_cost_from_next;
    build_cost_from_next_layer_all(current, current_runtime, *next, *next_runtime, next_layer_index, velocity, momentum, cost_from_next);
    const float *cost_data = cost_from_next.data();

    NN_OMP_PARALLEL_FOR_IF(use_parallel_delta)
    for(int idx=0; idx<flat_size; idx++){
        delta_data[static_cast<std::size_t>(idx)] = -cost_data[static_cast<std::size_t>(idx)];
    }
}

void backprop_lrn_layer(const Layer &current, LayerRuntime &current_runtime, const Layer *next, const LayerRuntime *next_runtime, int next_layer_index, bool is_output, const Loss &loss, const Tensor &desired_output, int output_size, bool enable_parallel, const ParameterBuffer *velocity, float momentum){
    const int flat_size = current.flat_output_size();
    float *delta_data = current_runtime.delta.data.data();
    const float *output_data = current_runtime.y.data.data();
    const bool use_parallel_delta = enable_parallel && flat_size > 4096;

    if(is_output){
        loss_kernels::dispatch_derivative_op(loss, output_size, [&](const auto &loss_derivative_op){
            NN_OMP_PARALLEL_FOR_IF(use_parallel_delta)
            for(int idx=0; idx<flat_size; idx++){
                const float output_value = output_data[static_cast<std::size_t>(idx)];
                delta_data[static_cast<std::size_t>(idx)] = -loss_derivative_op(output_value, desired_output.data[static_cast<std::size_t>(idx)]);
            }
        });
        return;
    }

    auto &cost_from_next = current_runtime.backprop_cost_from_next;
    build_cost_from_next_layer_all(current, current_runtime, *next, *next_runtime, next_layer_index, velocity, momentum, cost_from_next);
    const float *cost_data = cost_from_next.data();

    NN_OMP_PARALLEL_FOR_IF(use_parallel_delta)
    for(int idx=0; idx<flat_size; idx++){
        delta_data[static_cast<std::size_t>(idx)] = -cost_data[static_cast<std::size_t>(idx)];
    }
}

void backprop_softmax_layer(const Layer &current, LayerRuntime &current_runtime, const Layer *next, const LayerRuntime *next_runtime, bool is_output, const Loss &loss, const Tensor &desired_output, int output_size, bool enable_parallel){
    (void)next;
    (void)next_runtime;
    (void)is_output;
    (void)loss;
    (void)output_size;

    NN_OMP_PARALLEL_FOR_IF(enable_parallel && current.flat_output_size() > 4096)
    for(int idx=0; idx<current.flat_output_size(); idx++){
        current_runtime.delta.data[static_cast<std::size_t>(idx)] =
            desired_output.data[static_cast<std::size_t>(idx)] -
            current_runtime.y.data[static_cast<std::size_t>(idx)];
    }
}

} // namespace

void backprop(const LayerList &architecture, RuntimeList &runtime, int num_layers, ParameterBuffer &gradients, const Loss &loss, float &loss_value, const Tensor &desired_output, const Activation &hidden_activation, const Activation &output_activation, ExecutionPolicy policy, const ParameterBuffer *velocity, float momentum){
    const int output_size = architecture[num_layers-1].dim_layer[0] * architecture[num_layers-1].dim_layer[1] * architecture[num_layers-1].dim_layer[2];
    const bool enable_parallel = policy.allows_intra_example_parallelism();
    for(int l=num_layers-1; l>0; l--){
        const Layer &current = architecture[l];
        LayerRuntime &current_runtime = runtime[l];
        const Layer &previous = architecture[l-1];
        const LayerRuntime &previous_runtime = runtime[l-1];
        const Layer *next = (l < num_layers-1) ? &architecture[l+1] : nullptr;
        const LayerRuntime *next_runtime = (l < num_layers-1) ? &runtime[l+1] : nullptr;
        const bool is_output = (l == num_layers-1);
        const bool next_is_softmax = (next != nullptr) && (next->type == Layer_type::Softmax);
        const Activation &act = next_is_softmax ? identity : (l<num_layers-1) ? hidden_activation : output_activation;

        switch(current.type){
            case Layer_type::Dense:
                backprop_dense_layer(
                    l,
                    current, current_runtime,
                    previous, previous_runtime,
                    next, next_runtime,
                    gradients,
                    is_output, loss, desired_output, output_size, act,
                    enable_parallel, velocity, momentum
                );
                break;
            case Layer_type::Conv:
                backprop_conv_layer(
                    l,
                    current, current_runtime,
                    previous, previous_runtime,
                    next, next_runtime,
                    gradients,
                    is_output, loss, desired_output, output_size, act,
                    enable_parallel, velocity, momentum
                );
                break;
            case Layer_type::Pooling:
                backprop_pooling_layer(
                    current, current_runtime,
                    next, next_runtime, l + 1,
                    is_output, loss, desired_output, output_size,
                    enable_parallel, velocity, momentum
                );
                break;
            case Layer_type::Flatten:
                backprop_flatten_layer(
                    current, current_runtime,
                    next, next_runtime, l + 1,
                    is_output, loss, desired_output, output_size,
                    enable_parallel, velocity, momentum
                );
                break;
            case Layer_type::LRN:
                backprop_lrn_layer(
                    current, current_runtime,
                    next, next_runtime, l + 1,
                    is_output, loss, desired_output, output_size,
                    enable_parallel, velocity, momentum
                );
                break;
            case Layer_type::Softmax:
                backprop_softmax_layer(current, current_runtime, next, next_runtime, is_output, loss, desired_output, output_size, enable_parallel);
                break;
            case Layer_type::Input:
                break;
        }
    }

    loss_value = loss.fn(architecture[num_layers-1], runtime[num_layers-1], desired_output);
}
