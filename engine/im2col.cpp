#include "engine/im2col.hpp"
#include "core/layer.hpp"
#include "shared/openmp_utils.hpp"

// Questo file contiene l'implementazione del forward pass layer per layer.
#include <algorithm>
#include <vector>

void im2col(const Layer &previous, const Layer &current, const LayerRuntime &previous_runtime, std::vector<float> &col){
    const int kernel_h = current.kernel_dim[0];
    const int kernel_w = current.kernel_dim[1];
    const int input_c = current.kernel_dim[2];
    const int stride_h = current.stride[0];
    const int stride_w = current.stride[1];
    const int padding_h = current.padding[0];
    const int padding_w = current.padding[1];
    const int out_h = current.dim_layer[0];
    const int out_w = current.dim_layer[1];
    const int in_h = previous.dim_layer[0];
    const int in_w = previous.dim_layer[1];
    const float *input = runtime_output_buffer(previous, previous_runtime).data();
    const int patch_size = kernel_h * kernel_w * input_c;
    const int num_patches = out_h * out_w;
    const int work_items = out_h * out_w;
    const bool has_padding = (padding_h != 0) || (padding_w != 0);

    col.resize(static_cast<std::size_t>(patch_size) * static_cast<std::size_t>(num_patches));

    if(!has_padding){
        NN_OMP_PARALLEL_FOR_COLLAPSE_IF(2, work_items > 32)
        for(int out_i = 0; out_i < out_h; out_i++){
            for(int out_j = 0; out_j < out_w; out_j++){
                const int column = out_i * out_w + out_j;
                const int base_i = out_i * stride_h;
                const int base_j = out_j * stride_w;

                for(int kh = 0; kh < kernel_h; kh++){
                    const int in_i = base_i + kh;
                    for(int kw = 0; kw < kernel_w; kw++){
                        const int in_j = base_j + kw;
                        const int row = (kh * kernel_w + kw) * input_c;
                        const std::size_t input_base = static_cast<std::size_t>((in_i * in_w + in_j) * input_c);
                        const std::size_t col_base = static_cast<std::size_t>(row) * static_cast<std::size_t>(num_patches) + static_cast<std::size_t>(column);

                        NN_OMP_SIMD
                        for(int ic = 0; ic < input_c; ic++){
                            col[col_base +
                                static_cast<std::size_t>(ic) * static_cast<std::size_t>(num_patches)] =
                                input[input_base + static_cast<std::size_t>(ic)];
                        }
                    }
                }
            }
        }

        return;
    }

    NN_OMP_PARALLEL_FOR_COLLAPSE_IF(2, work_items > 32)
    for (int out_i = 0; out_i < out_h; out_i ++){
        for (int out_j = 0; out_j < out_w; out_j ++){
            const int column = out_i * out_w + out_j;

            for(int kh = 0; kh < kernel_h; kh ++){
                for(int kw = 0; kw < kernel_w; kw ++){
                    NN_OMP_SIMD
                    for(int ic = 0; ic < input_c; ic ++){
                        const int in_i = out_i * stride_h - padding_h + kh;
                        const int in_j = out_j * stride_w - padding_w + kw;
                        const int row = ((kh * kernel_w + kw) * input_c) + ic;

                        float value = 0.0f;
                        if(in_i >= 0 && in_i < in_h && in_j >= 0 && in_j < in_w){
                            value = input[(in_i * in_w + in_j) * input_c + ic];
                        }

                        col[row * num_patches + column] = value;
                    }
                }
            }
        }
    }

    return;
}

void im2col_batch(const Layer &previous, const Layer &current, const BatchLayerRuntime &previous_runtime, std::vector<float> &col){
    const int kernel_h = current.kernel_dim[0];
    const int kernel_w = current.kernel_dim[1];
    const int input_c = current.kernel_dim[2];
    const int stride_h = current.stride[0];
    const int stride_w = current.stride[1];
    const int padding_h = current.padding[0];
    const int padding_w = current.padding[1];
    const int out_h = current.dim_layer[0];
    const int out_w = current.dim_layer[1];
    const int in_h = previous.dim_layer[0];
    const int in_w = previous.dim_layer[1];
    const int batch_size = previous_runtime.y.batch_size;
    const int patch_size = kernel_h * kernel_w * input_c;
    const int patches_per_sample = out_h * out_w;
    const int total_patches = batch_size * patches_per_sample;
    const int work_items = batch_size * patches_per_sample;
    const bool has_padding = (padding_h != 0) || (padding_w != 0);
    const float *input = previous_runtime.y.data.data();

    col.resize(static_cast<std::size_t>(patch_size) * static_cast<std::size_t>(total_patches));

    if(!has_padding){
        NN_OMP_PARALLEL_FOR_IF(work_items > 32)
        for(int batch_index = 0; batch_index < batch_size; batch_index++){
            const std::size_t batch_input_base = previous_runtime.y.index(batch_index, 0);

            for(int out_i = 0; out_i < out_h; out_i++){
                for(int out_j = 0; out_j < out_w; out_j++){
                    const int column = batch_index * patches_per_sample + out_i * out_w + out_j;
                    const int base_i = out_i * stride_h;
                    const int base_j = out_j * stride_w;

                    for(int kh = 0; kh < kernel_h; kh++){
                        const int in_i = base_i + kh;
                        for(int kw = 0; kw < kernel_w; kw++){
                            const int in_j = base_j + kw;
                            const int row = (kh * kernel_w + kw) * input_c;
                            const std::size_t input_base =
                                batch_input_base +
                                static_cast<std::size_t>((in_i * in_w + in_j) * input_c);
                            const std::size_t col_base =
                                static_cast<std::size_t>(row) *
                                static_cast<std::size_t>(total_patches) +
                                static_cast<std::size_t>(column);

                            NN_OMP_SIMD
                            for(int ic = 0; ic < input_c; ic++){
                                col[col_base +
                                    static_cast<std::size_t>(ic) * static_cast<std::size_t>(total_patches)] =
                                    input[input_base + static_cast<std::size_t>(ic)];
                            }
                        }
                    }
                }
            }
        }

        return;
    }

    NN_OMP_PARALLEL_FOR_IF(work_items > 32)
    for(int batch_index = 0; batch_index < batch_size; batch_index++){
        const std::size_t batch_input_base = previous_runtime.y.index(batch_index, 0);

        for(int out_i = 0; out_i < out_h; out_i++){
            for(int out_j = 0; out_j < out_w; out_j++){
                const int column = batch_index * patches_per_sample + out_i * out_w + out_j;

                for(int kh = 0; kh < kernel_h; kh++){
                    for(int kw = 0; kw < kernel_w; kw++){
                        NN_OMP_SIMD
                        for(int ic = 0; ic < input_c; ic++){
                            const int in_i = out_i * stride_h - padding_h + kh;
                            const int in_j = out_j * stride_w - padding_w + kw;
                            const int row = ((kh * kernel_w + kw) * input_c) + ic;

                            float value = 0.0f;
                            if(in_i >= 0 && in_i < in_h && in_j >= 0 && in_j < in_w){
                                const std::size_t input_index =
                                    batch_input_base +
                                    static_cast<std::size_t>((in_i * in_w + in_j) * input_c + ic);
                                value = input[input_index];
                            }

                            col[static_cast<std::size_t>(row) *
                                static_cast<std::size_t>(total_patches) +
                                static_cast<std::size_t>(column)] = value;
                        }
                    }
                }
            }
        }
    }

    return;
}

void col2im(const Layer &current, const Layer &next, const float *col_grad, std::vector<float> &input_grad, float alpha){
    const int current_height = current.dim_layer[0];
    const int current_width = current.dim_layer[1];
    const int current_channels = current.dim_layer[2];
    const int next_height = next.dim_layer[0];
    const int next_width = next.dim_layer[1];
    const int kernel_height = next.kernel_dim[0];
    const int kernel_width = next.kernel_dim[1];
    const int stride_h = next.stride[0];
    const int stride_w = next.stride[1];
    const int padding_h = next.padding[0];
    const int padding_w = next.padding[1];
    const int in_features = kernel_height * kernel_width * current_channels;
    const bool has_padding = (padding_h != 0) || (padding_w != 0);

    input_grad.resize(static_cast<std::size_t>(current.flat_output_size()));

    if(!has_padding){
        for(int out_i = 0; out_i < next_height; out_i++){
            const int input_origin_i = out_i * stride_h;

            for(int out_j = 0; out_j < next_width; out_j++){
                const int patch_index = out_i * next_width + out_j;
                const int input_origin_j = out_j * stride_w;

                for(int kh = 0; kh < kernel_height; kh++){
                    const int in_i = input_origin_i + kh;
                    const std::size_t input_row_base =
                        static_cast<std::size_t>(in_i) *
                        static_cast<std::size_t>(current_width) *
                        static_cast<std::size_t>(current_channels);

                    for(int kw = 0; kw < kernel_width; kw++){
                        const int in_j = input_origin_j + kw;
                        const std::size_t input_base = input_row_base + static_cast<std::size_t>(in_j) * static_cast<std::size_t>(current_channels);
                        const std::size_t feature_base = static_cast<std::size_t>((kh * kernel_width + kw) * current_channels);

                        for(int c = 0; c < current_channels; c++){
                            const std::size_t col_index =
                                static_cast<std::size_t>(patch_index) *
                                static_cast<std::size_t>(in_features) +
                                feature_base +
                                static_cast<std::size_t>(c);
                            input_grad[input_base + static_cast<std::size_t>(c)] += alpha * col_grad[col_index];
                        }
                    }
                }
            }
        }

        return;
    }

    for(int out_i = 0; out_i < next_height; out_i++){
        const int input_origin_i = out_i * stride_h - padding_h;
        const int kh_begin = std::max(0, -input_origin_i);
        const int kh_end = std::min(kernel_height, current_height - input_origin_i);

        for(int out_j = 0; out_j < next_width; out_j++){
            const int patch_index = out_i * next_width + out_j;
            const int input_origin_j = out_j * stride_w - padding_w;
            const int kw_begin = std::max(0, -input_origin_j);
            const int kw_end = std::min(kernel_width, current_width - input_origin_j);

            for(int kh = kh_begin; kh < kh_end; kh++){
                const int in_i = input_origin_i + kh;
                const std::size_t input_row_base =
                    static_cast<std::size_t>(in_i) *
                    static_cast<std::size_t>(current_width) *
                    static_cast<std::size_t>(current_channels);

                for(int kw = kw_begin; kw < kw_end; kw++){
                    const int in_j = input_origin_j + kw;
                    const std::size_t input_base = input_row_base + static_cast<std::size_t>(in_j) * static_cast<std::size_t>(current_channels);
                    const std::size_t feature_base = static_cast<std::size_t>((kh * kernel_width + kw) * current_channels);

                    for(int c = 0; c < current_channels; c++){
                        const std::size_t col_index =
                            static_cast<std::size_t>(patch_index) *
                            static_cast<std::size_t>(in_features) +
                            feature_base +
                            static_cast<std::size_t>(c);
                        input_grad[input_base + static_cast<std::size_t>(c)] += alpha * col_grad[col_index];
                    }
                }
            }
        }
    }
}

void col2im_batch(const Layer &current, const Layer &next, int batch_size, const float *col_grad, std::vector<float> &input_grad, float alpha){
    const int current_height = current.dim_layer[0];
    const int current_width = current.dim_layer[1];
    const int current_channels = current.dim_layer[2];
    const int next_height = next.dim_layer[0];
    const int next_width = next.dim_layer[1];
    const int kernel_height = next.kernel_dim[0];
    const int kernel_width = next.kernel_dim[1];
    const int stride_h = next.stride[0];
    const int stride_w = next.stride[1];
    const int padding_h = next.padding[0];
    const int padding_w = next.padding[1];
    const int in_features = kernel_height * kernel_width * current_channels;
    const int patches_per_sample = next_height * next_width;
    const int current_flat_size = current.flat_output_size();
    const int work_items = batch_size * patches_per_sample;
    const bool has_padding = (padding_h != 0) || (padding_w != 0);

    input_grad.assign(static_cast<std::size_t>(batch_size) * static_cast<std::size_t>(current_flat_size), 0.0f);

    if(!has_padding){
        NN_OMP_PARALLEL_FOR_IF(work_items > 32)
        for(int batch_index = 0; batch_index < batch_size; batch_index++){
            const std::size_t batch_input_base = static_cast<std::size_t>(batch_index) * static_cast<std::size_t>(current_flat_size);

            for(int out_i = 0; out_i < next_height; out_i++){
                const int input_origin_i = out_i * stride_h;

                for(int out_j = 0; out_j < next_width; out_j++){
                    const int patch_index = batch_index * patches_per_sample + out_i * next_width + out_j;
                    const int input_origin_j = out_j * stride_w;

                    for(int kh = 0; kh < kernel_height; kh++){
                        const int in_i = input_origin_i + kh;
                        const std::size_t input_row_base =
                            batch_input_base +
                            static_cast<std::size_t>(in_i) *
                            static_cast<std::size_t>(current_width) *
                            static_cast<std::size_t>(current_channels);

                        for(int kw = 0; kw < kernel_width; kw++){
                            const int in_j = input_origin_j + kw;
                            const std::size_t input_base = input_row_base + static_cast<std::size_t>(in_j) * static_cast<std::size_t>(current_channels);
                            const std::size_t feature_base = static_cast<std::size_t>((kh * kernel_width + kw) * current_channels);

                            for(int c = 0; c < current_channels; c++){
                                const std::size_t col_index =
                                    static_cast<std::size_t>(patch_index) *
                                    static_cast<std::size_t>(in_features) +
                                    feature_base +
                                    static_cast<std::size_t>(c);
                                input_grad[input_base + static_cast<std::size_t>(c)] += alpha * col_grad[col_index];
                            }
                        }
                    }
                }
            }
        }

        return;
    }

    NN_OMP_PARALLEL_FOR_IF(work_items > 32)
    for(int batch_index = 0; batch_index < batch_size; batch_index++){
        const std::size_t batch_input_base = static_cast<std::size_t>(batch_index) * static_cast<std::size_t>(current_flat_size);

        for(int out_i = 0; out_i < next_height; out_i++){
            const int input_origin_i = out_i * stride_h - padding_h;
            const int kh_begin = std::max(0, -input_origin_i);
            const int kh_end = std::min(kernel_height, current_height - input_origin_i);

            for(int out_j = 0; out_j < next_width; out_j++){
                const int patch_index = batch_index * patches_per_sample + out_i * next_width + out_j;
                const int input_origin_j = out_j * stride_w - padding_w;
                const int kw_begin = std::max(0, -input_origin_j);
                const int kw_end = std::min(kernel_width, current_width - input_origin_j);

                for(int kh = kh_begin; kh < kh_end; kh++){
                    const int in_i = input_origin_i + kh;
                    const std::size_t input_row_base =
                        batch_input_base +
                        static_cast<std::size_t>(in_i) *
                        static_cast<std::size_t>(current_width) *
                        static_cast<std::size_t>(current_channels);

                    for(int kw = kw_begin; kw < kw_end; kw++){
                        const int in_j = input_origin_j + kw;
                        const std::size_t input_base = input_row_base + static_cast<std::size_t>(in_j) * static_cast<std::size_t>(current_channels);
                        const std::size_t feature_base = static_cast<std::size_t>((kh * kernel_width + kw) * current_channels);

                        for(int c = 0; c < current_channels; c++){
                            const std::size_t col_index =
                                static_cast<std::size_t>(patch_index) *
                                static_cast<std::size_t>(in_features) +
                                feature_base +
                                static_cast<std::size_t>(c);
                            input_grad[input_base + static_cast<std::size_t>(c)] += alpha * col_grad[col_index];
                        }
                    }
                }
            }
        }
    }
}
