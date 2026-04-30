#include "engine/backward.hpp"
#include "math/activation_kernels.hpp"
#include "engine/im2col.hpp"
#include "shared/openmp_utils.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <openblas/cblas.h>

namespace {
    
inline bool use_lookahead_params_batch(const ParameterBuffer *velocity, float momentum){
    return velocity != nullptr && momentum != 0.0f;
}

void build_cost_from_next_dense_all_batch(const Layer &current, const Layer &next, const BatchLayerRuntime &next_runtime, int next_layer_index, const ParameterBuffer *velocity, float momentum, std::vector<float> &out_cost){
    const int batch_size = next_runtime.y.batch_size;
    const int out_features = next.dense_output_size;
    const int in_features = current.flat_output_size();
    const float *delta_data = next_runtime.delta.data.data();
    const float *weights_data = next.dense_params.weights.data();

    cblas_sgemm(
        CblasRowMajor,
        CblasNoTrans,
        CblasNoTrans,
        batch_size,
        in_features,
        out_features,
        -1.0f,
        delta_data,
        out_features,
        weights_data,
        in_features,
        0.0f,
        out_cost.data(),
        in_features
    );

    if(use_lookahead_params_batch(velocity, momentum)){
        const float *velocity_weights_data = velocity->dense_weights[next_layer_index].data();
        cblas_sgemm(
            CblasRowMajor,
            CblasNoTrans,
            CblasNoTrans,
            batch_size,
            in_features,
            out_features,
            -momentum,
            delta_data,
            out_features,
            velocity_weights_data,
            in_features,
            1.0f,
            out_cost.data(),
            in_features
        );
    }
}

void build_cost_from_next_conv_all_batch(const Layer &current, BatchLayerRuntime &current_runtime, const Layer &next, const BatchLayerRuntime &next_runtime, int next_layer_index, const ParameterBuffer *velocity, float momentum, std::vector<float> &out_cost){
    const int batch_size = current_runtime.y.batch_size;
    const int out_channels = next.dim_layer[2];
    const int patches_per_sample = next.dim_layer[0] * next.dim_layer[1];
    const int total_patches = batch_size * patches_per_sample;
    const int kernel_height = next.kernel_dim[0];
    const int kernel_width = next.kernel_dim[1];
    const int in_channels = next.kernel_dim[2];
    const int in_features = kernel_height * kernel_width * in_channels;
    const float *delta_data = next_runtime.delta.data.data();
    const float *filter_data = next.conv_params.filters.data();

    current_runtime.conv_im2col.resize(static_cast<std::size_t>(total_patches) * static_cast<std::size_t>(in_features));

    cblas_sgemm(
        CblasRowMajor,
        CblasNoTrans,
        CblasNoTrans,
        total_patches,
        in_features,
        out_channels,
        1.0f,
        delta_data,
        out_channels,
        filter_data,
        in_features,
        0.0f,
        current_runtime.conv_im2col.data(),
        in_features
    );

    if(use_lookahead_params_batch(velocity, momentum)){
        const float *velocity_filter_data = velocity->conv_weights[next_layer_index].data();
        cblas_sgemm(
            CblasRowMajor,
            CblasNoTrans,
            CblasNoTrans,
            total_patches,
            in_features,
            out_channels,
            momentum,
            delta_data,
            out_channels,
            velocity_filter_data,
            in_features,
            1.0f,
            current_runtime.conv_im2col.data(),
            in_features
        );
    }

    col2im_batch(current, next, batch_size, current_runtime.conv_im2col.data(), out_cost, -1.0f);
}

void build_cost_from_next_pool_all_batch(const Layer &current, const BatchLayerRuntime &current_runtime, const Layer &next, const BatchLayerRuntime &next_runtime, std::vector<float> &out_cost){
    const int batch_size = current_runtime.y.batch_size;
    const float *current_output_data = current_runtime.y.data.data();
    const float *next_output_data = next_runtime.y.data.data();
    const float *next_delta_data = next_runtime.delta.data.data();
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
    const int batch_channels = batch_size * channels;
    const std::size_t next_sample_stride = static_cast<std::size_t>(output_height * output_width * channels);
    const std::size_t current_sample_stride = static_cast<std::size_t>(current_height * current_width * channels);
    const std::size_t pool_backprop_work_items = static_cast<std::size_t>(batch_size) *
                                                 static_cast<std::size_t>(output_height) *
                                                 static_cast<std::size_t>(output_width) *
                                                 static_cast<std::size_t>(channels) *
                                                 static_cast<std::size_t>(std::max(1, kernel_height * kernel_width));
    const bool use_parallel_pool_backprop = pool_backprop_work_items > 4096;

    switch(next.pooling_type){
        case Pooling_type::Average:
            if(!has_padding){
                const float inv_count = 1.0f / static_cast<float>(kernel_height * kernel_width);
                NN_OMP_PARALLEL_FOR_IF(use_parallel_pool_backprop)
                for(int bc=0; bc<batch_channels; bc++){
                    const int batch_idx = bc / channels;
                    const int c = bc % channels;
                    const std::size_t next_base = static_cast<std::size_t>(batch_idx) * next_sample_stride;
                    const std::size_t current_base = static_cast<std::size_t>(batch_idx) * current_sample_stride;

                    for(int out_i=0; out_i<output_height; out_i++){
                        const int start_i = out_i * stride_h;
                        for(int out_j=0; out_j<output_width; out_j++){
                            const int start_j = out_j * stride_w;
                            const float g = next_delta_data[next_base + static_cast<std::size_t>(next.flat_index(out_i, out_j, c))] * inv_count;

                            for(int kh=0; kh<kernel_height; kh++){
                                const int in_i = start_i + kh;
                                for(int kw=0; kw<kernel_width; kw++){
                                    const int in_j = start_j + kw;
                                    out_cost[current_base + static_cast<std::size_t>(current.flat_index(in_i, in_j, c))] -= g;
                                }
                            }
                        }
                    }
                }
            } else {
                NN_OMP_PARALLEL_FOR_IF(use_parallel_pool_backprop)
                for(int bc=0; bc<batch_channels; bc++){
                    const int batch_idx = bc / channels;
                    const int c = bc % channels;
                    const std::size_t next_base = static_cast<std::size_t>(batch_idx) * next_sample_stride;
                    const std::size_t current_base = static_cast<std::size_t>(batch_idx) * current_sample_stride;

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
                            const float g = next_delta_data[next_base + static_cast<std::size_t>(next.flat_index(out_i, out_j, c))] * inv;

                            for(int kh=kh_begin; kh<kh_end; kh++){
                                const int in_i = start_i + kh;
                                for(int kw=kw_begin; kw<kw_end; kw++){
                                    const int in_j = start_j + kw;
                                    out_cost[current_base + static_cast<std::size_t>(current.flat_index(in_i, in_j, c))] -= g;
                                }
                            }
                        }
                    }
                }
            }
            break;

        case Pooling_type::Max:
            NN_OMP_PARALLEL_FOR_IF(use_parallel_pool_backprop)
            for(int bc=0; bc<batch_channels; bc++){
                const int batch_idx = bc / channels;
                const int c = bc % channels;
                const std::size_t next_base = static_cast<std::size_t>(batch_idx) * next_sample_stride;
                const std::size_t current_base = static_cast<std::size_t>(batch_idx) * current_sample_stride;

                for(int out_i=0; out_i<output_height; out_i++){
                    for(int out_j=0; out_j<output_width; out_j++){
                        const int out_idx = next.flat_index(out_i, out_j, c);
                        const int winner = next_runtime.pooling_argmax[next_base + static_cast<std::size_t>(out_idx)];
                        if(winner >= 0){
                            out_cost[current_base + static_cast<std::size_t>(winner)] -= next_delta_data[next_base + static_cast<std::size_t>(out_idx)];
                        }
                    }
                }
            }
            break;

        case Pooling_type::L2:
            if(!has_padding){
                NN_OMP_PARALLEL_FOR_IF(use_parallel_pool_backprop)
                for(int bc=0; bc<batch_channels; bc++){
                    const int batch_idx = bc / channels;
                    const int c = bc % channels;
                    const std::size_t next_base = static_cast<std::size_t>(batch_idx) * next_sample_stride;
                    const std::size_t current_base = static_cast<std::size_t>(batch_idx) * current_sample_stride;

                    for(int out_i=0; out_i<output_height; out_i++){
                        const int start_i = out_i * stride_h;
                        for(int out_j=0; out_j<output_width; out_j++){
                            const int start_j = out_j * stride_w;
                            const int out_idx = next.flat_index(out_i, out_j, c);
                            const float pooled_norm = next_output_data[next_base + static_cast<std::size_t>(out_idx)];
                            if(pooled_norm <= 0.0f){
                                continue;
                            }

                            const float g_over_norm = next_delta_data[next_base + static_cast<std::size_t>(out_idx)] / pooled_norm;

                            for(int kh=0; kh<kernel_height; kh++){
                                const int in_i = start_i + kh;
                                for(int kw=0; kw<kernel_width; kw++){
                                    const int in_j = start_j + kw;
                                    const int in_idx = current.flat_index(in_i, in_j, c);
                                    out_cost[current_base + static_cast<std::size_t>(in_idx)] -=
                                        g_over_norm *
                                        current_output_data[current_base + static_cast<std::size_t>(in_idx)];
                                }
                            }
                        }
                    }
                }
            } else {
                NN_OMP_PARALLEL_FOR_IF(use_parallel_pool_backprop)
                for(int bc=0; bc<batch_channels; bc++){
                    const int batch_idx = bc / channels;
                    const int c = bc % channels;
                    const std::size_t next_base = static_cast<std::size_t>(batch_idx) * next_sample_stride;
                    const std::size_t current_base = static_cast<std::size_t>(batch_idx) * current_sample_stride;

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
                            const float pooled_norm = next_output_data[next_base + static_cast<std::size_t>(out_idx)];
                            if(pooled_norm <= 0.0f){
                                continue;
                            }

                            const float g_over_norm = next_delta_data[next_base + static_cast<std::size_t>(out_idx)] / pooled_norm;

                            for(int kh=kh_begin; kh<kh_end; kh++){
                                const int in_i = start_i + kh;
                                for(int kw=kw_begin; kw<kw_end; kw++){
                                    const int in_j = start_j + kw;
                                    const int in_idx = current.flat_index(in_i, in_j, c);
                                    out_cost[current_base + static_cast<std::size_t>(in_idx)] -=
                                        g_over_norm *
                                        current_output_data[current_base + static_cast<std::size_t>(in_idx)];
                                }
                            }
                        }
                    }
                }
            }
            break;
    }
}

void build_cost_from_next_flatten_all_batch(const Layer &current, const Layer &next, const BatchLayerRuntime &next_runtime, std::vector<float> &out_cost){
    const int batch_size = next_runtime.y.batch_size;
    const int current_flat_size = current.flat_output_size();
    const float *next_delta_data = next_runtime.delta.data.data();
    const int next_flat_size = next.flat_output_size();
    const int n = std::min(current_flat_size, next_flat_size);

    NN_OMP_PARALLEL_FOR_COLLAPSE_IF(2, batch_size * n > 4096)
    for(int batch_idx = 0; batch_idx < batch_size; batch_idx++){
        for(int idx = 0; idx < n; idx++){
            const std::size_t next_base = static_cast<std::size_t>(batch_idx) * static_cast<std::size_t>(next_flat_size);
            const std::size_t current_base = static_cast<std::size_t>(batch_idx) * static_cast<std::size_t>(current_flat_size);
            out_cost[current_base + static_cast<std::size_t>(idx)] = -next_delta_data[next_base + static_cast<std::size_t>(idx)];
        }
    }
}

void build_cost_from_next_lrn_all_batch(const Layer &current, const BatchLayerRuntime &current_runtime, const Layer &next, const BatchLayerRuntime &next_runtime, std::vector<float> &out_cost){
    const int batch_size = current_runtime.y.batch_size;
    const int current_flat_size = current.flat_output_size();
    const float *current_output_data = current_runtime.y.data.data();
    const float *next_activation_data = next_runtime.a.data.data();
    const float *next_delta_data = next_runtime.delta.data.data();
    const int height = current.dim_layer[0];
    const int width = current.dim_layer[1];
    const int channels = current.dim_layer[2];
    const int local_size = std::max(1, next.lrn_local_size);
    const int radius = local_size / 2;
    const float alpha_over_size = next.lrn_alpha / static_cast<float>(local_size);
    const float two_alpha_over_size = 2.0f * alpha_over_size;
    const std::size_t lrn_backprop_work_items = static_cast<std::size_t>(batch_size) *
                                                static_cast<std::size_t>(height) *
                                                static_cast<std::size_t>(width) *
                                                static_cast<std::size_t>(channels) *
                                                static_cast<std::size_t>(local_size);

    NN_OMP_PARALLEL_FOR_COLLAPSE_IF(3, lrn_backprop_work_items > 4096)
    for(int batch_idx = 0; batch_idx < batch_size; batch_idx++){
        for(int i = 0; i < height; i++){
            for(int j = 0; j < width; j++){
                const std::size_t batch_offset_current = static_cast<std::size_t>(batch_idx) * static_cast<std::size_t>(current_flat_size);
                const std::size_t batch_offset_next = static_cast<std::size_t>(batch_idx) * static_cast<std::size_t>(next.flat_output_size());
                const int pixel_base = (i * width + j) * channels;

                for(int k = 0; k < channels; k++){
                    const int idx_base = pixel_base + k;
                    const float current_value = current_output_data[batch_offset_current + static_cast<std::size_t>(idx_base)];

                    float coeff = 0.0f;
                    int start_c = 0;
                    int end_c = std::min(channels - 1, radius);
                    float squared_sum = 0.0f;
                    for(int c = start_c; c <= end_c; c++){
                        const float neighbor = current_output_data[batch_offset_current + static_cast<std::size_t>(pixel_base + c)];
                        squared_sum += neighbor * neighbor;
                    }

                    for(int out_k = 0; out_k < channels; out_k++){
                        const int prev_start_c = start_c;
                        const int prev_end_c = end_c;
                        start_c = std::max(0, out_k - radius);
                        end_c = std::min(channels - 1, out_k + radius);

                        if(out_k > 0){
                            if(start_c > prev_start_c){
                                const float removed = current_output_data[batch_offset_current + static_cast<std::size_t>(pixel_base + prev_start_c)];
                                squared_sum -= removed * removed;
                            }
                            if(end_c > prev_end_c){
                                const float added = current_output_data[batch_offset_current + static_cast<std::size_t>(pixel_base + end_c)];
                                squared_sum += added * added;
                            }
                        }

                        if(k < start_c || k > end_c){
                            continue;
                        }

                        const float scale = next.lrn_k + alpha_over_size * squared_sum;
                        const float normalized_grad = std::pow(scale, -next.lrn_beta);
                        const int out_idx_flat = next.flat_index(i, j, out_k);
                        float local_coeff =
                            -next.lrn_beta *
                            next_activation_data[batch_offset_next + static_cast<std::size_t>(out_idx_flat)] *
                            std::pow(scale, -next.lrn_beta - 1.0f) *
                            two_alpha_over_size *
                            current_value;
                        if(k == out_k){
                            local_coeff += normalized_grad;
                        }

                        coeff -= next_delta_data[batch_offset_next + static_cast<std::size_t>(out_idx_flat)] * local_coeff;
                    }

                    out_cost[batch_offset_current + static_cast<std::size_t>(idx_base)] = coeff;
                }
            }
        }
    }
}

void build_cost_from_next_softmax_all_batch(const Layer &current, const Layer &next, const BatchLayerRuntime &next_runtime, std::vector<float> &out_cost){
    const int batch_size = next_runtime.y.batch_size;
    const int current_flat_size = current.flat_output_size();
    const float *next_delta_data = next_runtime.delta.data.data();
    const int next_flat_size = next.flat_output_size();

    NN_OMP_PARALLEL_FOR_COLLAPSE_IF(2, batch_size * current_flat_size > 4096)
    for(int batch_idx = 0; batch_idx < batch_size; batch_idx++){
        for(int idx = 0; idx < current_flat_size; idx++){
            const std::size_t next_base = static_cast<std::size_t>(batch_idx) * static_cast<std::size_t>(next_flat_size);
            const std::size_t current_base = static_cast<std::size_t>(batch_idx) * static_cast<std::size_t>(current_flat_size);
            out_cost[current_base + static_cast<std::size_t>(idx)] = -next_delta_data[next_base + static_cast<std::size_t>(idx)];
        }
    }
}

void build_cost_from_next_layer_all_batch(const Layer &current, BatchLayerRuntime &current_runtime, const Layer &next, const BatchLayerRuntime &next_runtime, int next_layer_index, const ParameterBuffer *velocity, float momentum, std::vector<float> &out_cost){
    const int batch_size = current_runtime.y.batch_size;
    const int current_flat_size = current.flat_output_size();
    const int total_elements = batch_size * current_flat_size;
    
    out_cost.assign(static_cast<std::size_t>(total_elements), 0.0f);

    switch(next.type){
        case Layer_type::Dense:
            build_cost_from_next_dense_all_batch(current, next, next_runtime, next_layer_index, velocity, momentum, out_cost);
            break;

        case Layer_type::Conv:
            build_cost_from_next_conv_all_batch(current, current_runtime, next, next_runtime, next_layer_index, velocity, momentum, out_cost);
            break;

        case Layer_type::Pooling:
            build_cost_from_next_pool_all_batch(current, current_runtime, next, next_runtime, out_cost);
            break;

        case Layer_type::Flatten:
            build_cost_from_next_flatten_all_batch(current, next, next_runtime, out_cost);
            break;

        case Layer_type::LRN:
            build_cost_from_next_lrn_all_batch(current, current_runtime, next, next_runtime, out_cost);
            break;

        case Layer_type::Softmax:
            build_cost_from_next_softmax_all_batch(current, next, next_runtime, out_cost);
            break;

        case Layer_type::Input:
            break;
    }
}


void backprop_pooling_layer_batch(const Layer &current, BatchLayerRuntime &current_runtime, const Layer *next, const BatchLayerRuntime *next_runtime, int next_layer_index, bool is_output, const Loss &loss, const BatchTensor &desired_output, int output_size, bool enable_parallel, const ParameterBuffer *velocity, float momentum);
void backprop_flatten_layer_batch(const Layer &current, BatchLayerRuntime &current_runtime, const Layer *next, const BatchLayerRuntime *next_runtime, int next_layer_index, bool is_output, const Loss &loss, const BatchTensor &desired_output, int output_size, bool enable_parallel, const ParameterBuffer *velocity, float momentum);
void backprop_lrn_layer_batch(const Layer &current, BatchLayerRuntime &current_runtime, const Layer *next, const BatchLayerRuntime *next_runtime, int next_layer_index, bool is_output, const Loss &loss, const BatchTensor &desired_output, int output_size, bool enable_parallel, const ParameterBuffer *velocity, float momentum);
void backprop_softmax_layer_batch(const Layer &current, BatchLayerRuntime &current_runtime, const BatchTensor &desired_output, bool enable_parallel);

void backprop_dense_layer_batch(int layer_index, const Layer &current, BatchLayerRuntime &current_runtime, const Layer &previous, const BatchLayerRuntime &previous_runtime, const Layer *next, const BatchLayerRuntime *next_runtime, ParameterBuffer &gradients, bool is_output, const Loss &loss, const BatchTensor &desired_output, int output_size, const Activation &act, bool enable_parallel, const ParameterBuffer *velocity, float momentum){
    (void)previous;
    const float *previous_output_data = previous_runtime.y.data.data();
    float *current_delta_data = current_runtime.delta.data.data();
    float *gradient_weight_data = gradients.dense_weights[layer_index].data();
    const int batch_size = current_runtime.y.batch_size;
    const int in_features = current.dense_input_size;
    const int out_features = current.dense_output_size;
    const bool use_parallel_dense_delta = enable_parallel && static_cast<std::size_t>(batch_size) * static_cast<std::size_t>(out_features) > 4096;

    if(is_output){
        loss_kernels::dispatch_derivative_op(loss, output_size, [&](const auto &loss_derivative_op){
            activation_kernels::dispatch_derivative_op(act, [&](const auto &derivative_op){
                NN_OMP_PARALLEL_FOR_COLLAPSE_IF(2, use_parallel_dense_delta)
                for(int batch_index = 0; batch_index < batch_size; batch_index++){
                    for(int out_idx = 0; out_idx < out_features; out_idx++){
                        const std::size_t output_index =
                            static_cast<std::size_t>(batch_index) *
                            static_cast<std::size_t>(out_features) +
                            static_cast<std::size_t>(out_idx);
                        const float output_value = current_runtime.y.data[output_index];
                        const float cost_der = loss_derivative_op(output_value, desired_output.data[output_index]);
                        current_delta_data[output_index] = -cost_der * derivative_op(current_runtime.a.data[output_index]);
                    }
                }
            });
        });
    } else {
        activation_kernels::dispatch_derivative_op(act, [&](const auto &derivative_op){
            std::vector<float>& cost_from_next = current_runtime.backprop_cost_from_next;
            build_cost_from_next_layer_all_batch(current, current_runtime, *next, *next_runtime, layer_index + 1, velocity, momentum, cost_from_next);

            NN_OMP_PARALLEL_FOR_COLLAPSE_IF(2, use_parallel_dense_delta)
            for(int batch_index = 0; batch_index < batch_size; batch_index++){
                for(int out_idx = 0; out_idx < out_features; out_idx++){
                    const std::size_t output_index =
                        static_cast<std::size_t>(batch_index) *
                        static_cast<std::size_t>(out_features) +
                        static_cast<std::size_t>(out_idx);
                    const float cost_der = cost_from_next[output_index];
                    current_delta_data[output_index] = -cost_der * derivative_op(current_runtime.a.data[output_index]);
                }
            }
        });
    }

    for(int out_idx = 0; out_idx < out_features; out_idx++){
        float bias_grad = 0.0f;
        NN_OMP_SIMD_REDUCTION_PLUS(bias_grad)
        for(int batch_index = 0; batch_index < batch_size; batch_index++){
            const std::size_t output_index = static_cast<std::size_t>(batch_index) * static_cast<std::size_t>(out_features) + static_cast<std::size_t>(out_idx);
            bias_grad += current_delta_data[output_index];
        }
        gradients.dense_biases[layer_index][out_idx] = bias_grad;
    }

    cblas_sgemm(
        CblasRowMajor,
        CblasTrans,
        CblasNoTrans,
        out_features,
        in_features,
        batch_size,
        1.0f,
        current_delta_data,
        out_features,
        previous_output_data,
        in_features,
        0.0f,
        gradient_weight_data,
        in_features
    );
}

void backprop_conv_layer_batch(int layer_index, const Layer &current, BatchLayerRuntime &current_runtime, const Layer &previous, const BatchLayerRuntime &previous_runtime, const Layer *next, const BatchLayerRuntime *next_runtime, ParameterBuffer &gradients, bool is_output, const Loss &loss, const BatchTensor &desired_output, int output_size, const Activation &act, bool enable_parallel, const ParameterBuffer *velocity, float momentum){
    float *gradient_weight_data = gradients.conv_weights[layer_index].data();
    float *current_delta_data = current_runtime.delta.data.data();
    const int batch_size = current_runtime.y.batch_size;
    const int out_height = current.dim_layer[0];
    const int output_width = current.dim_layer[1];
    const int out_channels = current.dim_layer[2];
    const int kernel_height = current.kernel_dim[0];
    const int kernel_width = current.kernel_dim[1];
    const int previous_channels = current.kernel_dim[2];
    const int in_features = kernel_height * kernel_width * previous_channels;
    const int current_flat_size = current.flat_output_size();
    const int patches_per_sample = out_height * output_width;
    const int total_patches = batch_size * patches_per_sample;
    const bool use_parallel_conv_output_delta =
        enable_parallel &&
        static_cast<std::size_t>(batch_size) *
        static_cast<std::size_t>(out_height) *
        static_cast<std::size_t>(output_width) *
        static_cast<std::size_t>(out_channels) > 4096;
    const bool use_parallel_conv_hidden_delta = enable_parallel && static_cast<std::size_t>(batch_size) * static_cast<std::size_t>(current_flat_size) > 4096;

    if(is_output){
        loss_kernels::dispatch_derivative_op(loss, output_size, [&](const auto &loss_derivative_op){
            activation_kernels::dispatch_derivative_op(act, [&](const auto &derivative_op){
                NN_OMP_PARALLEL_FOR_COLLAPSE_IF(4, use_parallel_conv_output_delta)
                for(int batch_index = 0; batch_index < batch_size; batch_index++){
                    for(int out_i = 0; out_i < out_height; out_i++){
                        for(int out_j = 0; out_j < output_width; out_j++){
                            for(int out_k = 0; out_k < out_channels; out_k++){
                                const std::size_t output_index = static_cast<std::size_t>(batch_index) * static_cast<std::size_t>(current_flat_size) +
                                                                 static_cast<std::size_t>((out_i * output_width + out_j) * out_channels + out_k);
                                const float output_value = current_runtime.y.data[output_index];
                                const float activation_value = current_runtime.a.data[output_index];
                                const float cost_der = loss_derivative_op(output_value, desired_output.data[output_index]);
                                current_delta_data[output_index] = -cost_der * derivative_op(activation_value);
                            }
                        }
                    }
                }
            });
        });
    } else {
        activation_kernels::dispatch_derivative_op(act, [&](const auto &derivative_op){
            std::vector<float>& cost_from_next = current_runtime.backprop_cost_from_next;
            build_cost_from_next_layer_all_batch(current, current_runtime, *next, *next_runtime, layer_index + 1, velocity, momentum, cost_from_next);

            NN_OMP_PARALLEL_FOR_IF(use_parallel_conv_hidden_delta)
            for(int batch_index = 0; batch_index < batch_size; batch_index++){
                const int batch_offset = batch_index * current_flat_size;
                for(int idx = 0; idx < current_flat_size; idx++){
                    const std::size_t output_index = static_cast<std::size_t>(batch_offset + idx);
                    const float cost_der = cost_from_next[output_index];
                    current_delta_data[output_index] = -cost_der * derivative_op(current_runtime.a.data[output_index]);
                }
            }
        });
    }

    for(int out_k = 0; out_k < out_channels; out_k++){
        float bias_grad = 0.0f;
        NN_OMP_SIMD_REDUCTION_PLUS(bias_grad)
        for(int patch_idx = 0; patch_idx < total_patches; patch_idx++){
            bias_grad += current_delta_data[static_cast<std::size_t>(patch_idx) * static_cast<std::size_t>(out_channels) + static_cast<std::size_t>(out_k)];
        }
        gradients.conv_biases[layer_index][out_k] = bias_grad;
    }

    im2col_batch(previous, current, previous_runtime, current_runtime.conv_im2col);
    cblas_sgemm(
        CblasRowMajor,
        CblasTrans,
        CblasTrans,
        out_channels,
        in_features,
        total_patches,
        1.0f,
        current_delta_data,
        out_channels,
        current_runtime.conv_im2col.data(),
        total_patches,
        0.0f,
        gradient_weight_data,
        in_features
    );
}

void backprop_pooling_layer_batch(const Layer &current, BatchLayerRuntime &current_runtime, const Layer *next, const BatchLayerRuntime *next_runtime, int next_layer_index, bool is_output, const Loss &loss, const BatchTensor &desired_output, int output_size, bool enable_parallel, const ParameterBuffer *velocity, float momentum){
    const int batch_size = current_runtime.y.batch_size;
    const int flat_size = current.flat_output_size();
    const bool use_parallel_delta = enable_parallel && static_cast<std::size_t>(batch_size) * static_cast<std::size_t>(flat_size) > 4096;

    if(is_output){
        loss_kernels::dispatch_derivative_op(loss, output_size, [&](const auto &loss_derivative_op){
            NN_OMP_PARALLEL_FOR_COLLAPSE_IF(2, use_parallel_delta)
            for(int batch_index = 0; batch_index < batch_size; batch_index++){
                for(int idx = 0; idx < flat_size; idx++){
                    const std::size_t data_index = static_cast<std::size_t>(batch_index) * static_cast<std::size_t>(flat_size) + static_cast<std::size_t>(idx);
                    const float output_value = current_runtime.y.data[data_index];
                    current_runtime.delta.data[data_index] = -loss_derivative_op(output_value, desired_output.data[data_index]);
                }
            }
        });
        return;
    }

    std::vector<float>& cost_from_next = current_runtime.backprop_cost_from_next;
    build_cost_from_next_layer_all_batch(current, current_runtime, *next, *next_runtime, next_layer_index, velocity, momentum, cost_from_next);

    NN_OMP_PARALLEL_FOR_COLLAPSE_IF(2, use_parallel_delta)
    for(int batch_index = 0; batch_index < batch_size; batch_index++){
        for(int idx = 0; idx < flat_size; idx++){
            const std::size_t data_index = static_cast<std::size_t>(batch_index) * static_cast<std::size_t>(flat_size) + static_cast<std::size_t>(idx);
            current_runtime.delta.data[data_index] = -cost_from_next[data_index];
        }
    }
}

void backprop_flatten_layer_batch(const Layer &current, BatchLayerRuntime &current_runtime, const Layer *next, const BatchLayerRuntime *next_runtime, int next_layer_index, bool is_output, const Loss &loss, const BatchTensor &desired_output, int output_size, bool enable_parallel, const ParameterBuffer *velocity, float momentum){
    const int batch_size = current_runtime.y.batch_size;
    const int flat_size = current.flat_output_size();
    const bool use_parallel_delta = enable_parallel && static_cast<std::size_t>(batch_size) * static_cast<std::size_t>(flat_size) > 4096;

    if(is_output){
        loss_kernels::dispatch_derivative_op(loss, output_size, [&](const auto &loss_derivative_op){
            NN_OMP_PARALLEL_FOR_COLLAPSE_IF(2, use_parallel_delta)
            for(int batch_index = 0; batch_index < batch_size; batch_index++){
                for(int idx = 0; idx < flat_size; idx++){
                    const std::size_t data_index = static_cast<std::size_t>(batch_index) * static_cast<std::size_t>(flat_size) + static_cast<std::size_t>(idx);
                    const float output_value = current_runtime.y.data[data_index];
                    current_runtime.delta.data[data_index] = -loss_derivative_op(output_value, desired_output.data[data_index]);
                }
            }
        });
        return;
    }

    std::vector<float>& cost_from_next = current_runtime.backprop_cost_from_next;
    build_cost_from_next_layer_all_batch(current, current_runtime, *next, *next_runtime, next_layer_index, velocity, momentum, cost_from_next);

    NN_OMP_PARALLEL_FOR_COLLAPSE_IF(2, use_parallel_delta)
    for(int batch_index = 0; batch_index < batch_size; batch_index++){
        for(int idx = 0; idx < flat_size; idx++){
            const std::size_t data_index = static_cast<std::size_t>(batch_index) * static_cast<std::size_t>(flat_size) + static_cast<std::size_t>(idx);
            current_runtime.delta.data[data_index] = -cost_from_next[data_index];
        }
    }
}

void backprop_lrn_layer_batch(const Layer &current, BatchLayerRuntime &current_runtime, const Layer *next, const BatchLayerRuntime *next_runtime, int next_layer_index, bool is_output, const Loss &loss, const BatchTensor &desired_output, int output_size, bool enable_parallel, const ParameterBuffer *velocity, float momentum){
    const int batch_size = current_runtime.y.batch_size;
    const int flat_size = current.flat_output_size();
    const bool use_parallel_delta = enable_parallel && static_cast<std::size_t>(batch_size) * static_cast<std::size_t>(flat_size) > 4096;

    if(is_output){
        loss_kernels::dispatch_derivative_op(loss, output_size, [&](const auto &loss_derivative_op){
            NN_OMP_PARALLEL_FOR_COLLAPSE_IF(2, use_parallel_delta)
            for(int batch_index = 0; batch_index < batch_size; batch_index++){
                for(int idx = 0; idx < flat_size; idx++){
                    const std::size_t data_index = static_cast<std::size_t>(batch_index) * static_cast<std::size_t>(flat_size) + static_cast<std::size_t>(idx);
                    const float output_value = current_runtime.y.data[data_index];
                    current_runtime.delta.data[data_index] = -loss_derivative_op(output_value, desired_output.data[data_index]);
                }
            }
        });
        return;
    }

    std::vector<float>& cost_from_next = current_runtime.backprop_cost_from_next;
    build_cost_from_next_layer_all_batch(current, current_runtime, *next, *next_runtime, next_layer_index, velocity, momentum, cost_from_next);

    NN_OMP_PARALLEL_FOR_COLLAPSE_IF(2, use_parallel_delta)
    for(int batch_index = 0; batch_index < batch_size; batch_index++){
        for(int idx = 0; idx < flat_size; idx++){
            const std::size_t data_index = static_cast<std::size_t>(batch_index) * static_cast<std::size_t>(flat_size) + static_cast<std::size_t>(idx);
            current_runtime.delta.data[data_index] = -cost_from_next[data_index];
        }
    }
}

void backprop_softmax_layer_batch(const Layer &current, BatchLayerRuntime &current_runtime, const BatchTensor &desired_output, bool enable_parallel){
    const int batch_size = current_runtime.y.batch_size;
    const int flat_size = current.flat_output_size();
    const bool use_parallel_delta = enable_parallel && static_cast<std::size_t>(batch_size) * static_cast<std::size_t>(flat_size) > 4096;

    NN_OMP_PARALLEL_FOR_COLLAPSE_IF(2, use_parallel_delta)
    for(int batch_index = 0; batch_index < batch_size; batch_index++){
        for(int idx = 0; idx < flat_size; idx++){
            const std::size_t data_index = static_cast<std::size_t>(batch_index) * static_cast<std::size_t>(flat_size) + static_cast<std::size_t>(idx);
            current_runtime.delta.data[data_index] = desired_output.data[data_index] - current_runtime.y.data[data_index];
        }
    }
}

} // namespace

void backprop_batch(const LayerList &architecture, BatchRuntimeList &runtime, int num_layers, ParameterBuffer &gradients, const Loss &loss, float &loss_value, const BatchTensor &desired_output, const Activation &hidden_activation, const Activation &output_activation, ExecutionPolicy policy, const ParameterBuffer *velocity, float momentum){
    const bool enable_parallel = policy.allows_batch_sample_parallelism() || policy.allows_intra_example_parallelism();
    const int output_size = architecture[num_layers - 1].flat_output_size();

    for(int l = num_layers - 1; l > 0; l--){
        const Layer &current = architecture[l];
        BatchLayerRuntime &current_runtime = runtime[l];
        const Layer &previous = architecture[l - 1];
        const BatchLayerRuntime &previous_runtime = runtime[l - 1];
        const Layer *next = (l < num_layers - 1) ? &architecture[l + 1] : nullptr;
        const BatchLayerRuntime *next_runtime = (l < num_layers - 1) ? &runtime[l + 1] : nullptr;
        const bool is_output = (l == num_layers - 1);
        const bool next_is_softmax = (next != nullptr) && (next->type == Layer_type::Softmax);
        const Activation &act = next_is_softmax ? identity : (l < num_layers - 1) ? hidden_activation : output_activation;

        switch(current.type){
            case Layer_type::Dense:
                backprop_dense_layer_batch(
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
                backprop_conv_layer_batch(
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
                backprop_pooling_layer_batch(
                    current, current_runtime,
                    next, next_runtime, l + 1,
                    is_output, loss, desired_output, output_size,
                    enable_parallel, velocity, momentum
                );
                break;
            case Layer_type::Flatten:
                backprop_flatten_layer_batch(
                    current, current_runtime,
                    next, next_runtime, l + 1,
                    is_output, loss, desired_output, output_size,
                    enable_parallel, velocity, momentum
                );
                break;
            case Layer_type::LRN:
                backprop_lrn_layer_batch(
                    current, current_runtime,
                    next, next_runtime, l + 1,
                    is_output, loss, desired_output, output_size,
                    enable_parallel, velocity, momentum
                );
                break;
            case Layer_type::Softmax:
                backprop_softmax_layer_batch(current, current_runtime, desired_output, enable_parallel);
                break;
            case Layer_type::Input:
                break;
        }
    }

    loss_value = loss.fn_batch(runtime[num_layers - 1], desired_output);
}
