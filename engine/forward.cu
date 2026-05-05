#include "engine/forward.hpp"
#include "core/cuda_backend.hpp"
#include "kernels/general.hpp"
#include "kernels/pooling.hpp"
#include "kernels/im2col.hpp"
#include "kernels/lrn.hpp"
#include "kernels/softmax.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <math_constants.h>


using cuda_backend::CudaBatchLayerRuntime;
using cuda_backend::CudaParameterBuffer;

namespace {

inline void copy_negated_device(
    const cuda_backend::CudaBatchTensor<float> &src,
    cuda_backend::CudaBatchTensor<float> &dst
){
    dst.resize(src.batch_size(), src.height(), src.width(), src.channels());
    const int total_size = static_cast<int>(src.size());
    if(total_size <= 0){
        return;
    }
    const int grid_dim = (total_size + 256 - 1) / 256;
    copy_negated <<< grid_dim, 256 >>> (src.data(), dst.data(), total_size);
}

inline bool use_lookahead_params_batch(const CudaParameterBuffer *velocity, float momentum){
    return velocity != nullptr && momentum != 0.0f;
}

void forward_dense_layer_batch(
    int layer_index, const Layer &current, CudaBatchLayerRuntime &current_runtime, const CudaParameterBuffer &cuda_params,
    const CudaBatchLayerRuntime &previous_runtime,
    const Activation &act,
    const CudaParameterBuffer *velocity, float momentum
){


    const float *weights_data = cuda_params.dense_weights[layer_index].data();
    const float *bias_data = cuda_params.dense_biases[layer_index].data();
    const float *input = previous_runtime.y.data();

    float *a = current_runtime.a.data();
    float *y = current_runtime.y.data();

    const int batch_size = previous_runtime.y.batch_size();

    const int in_features = current.dense_input_size;
    const int out_features = current.dense_output_size;
    const int output_size = static_cast<int>(current_runtime.y.size());

    const bool use_lookahead = use_lookahead_params_batch(velocity, momentum);

    int activation_grid_dim = (output_size + 256 -1) / 256;

    
    cublasHandle_t handle = static_cast<cublasHandle_t>(cuda_backend::current_cublas_handle());

    if(!use_lookahead){
        float alpha = 1.0f;
        float beta = 0.0f;
        cublasSgemm(
            handle,
            CUBLAS_OP_T, CUBLAS_OP_N,
            out_features, batch_size, in_features,
            &alpha,
            weights_data, in_features,
            input, in_features,
            &beta, a, out_features
        );
        apply_bias_activation <<< activation_grid_dim, 256 >>> (bias_data, a, y, output_size, out_features, act.kind);
        return;
    }

    const float *velocity_weight_data = velocity->dense_weights[layer_index].data();
    const float *velocity_bias_data = velocity->dense_biases[layer_index].data();

    float alpha = 1.0f;
    float beta = 0.0f;
    cublasSgemm(
        handle,
        CUBLAS_OP_T, CUBLAS_OP_N,
        out_features, batch_size, in_features,
        &alpha,
        weights_data, in_features,
        input, in_features,
        &beta, a, out_features
    );
    cublasSgemm(
        handle,
        CUBLAS_OP_T, CUBLAS_OP_N,
        out_features, batch_size, in_features,
        &momentum,
        velocity_weight_data, in_features,
        input, in_features,
        &alpha, a, out_features
    );
    apply_nesterov_bias_activation <<< activation_grid_dim, 256 >>> (
        bias_data, velocity_bias_data, a, y,
        output_size, out_features, momentum, 
        act.kind
    );
}

void forward_conv_layer_batch(
    int layer_index, const Layer &current, CudaBatchLayerRuntime &current_runtime, const CudaParameterBuffer &cuda_params,
    const CudaBatchLayerRuntime &previous_runtime,
    const Activation &act,
    const CudaParameterBuffer *velocity, float momentum
){

    const float *filter_data = cuda_params.conv_weights[layer_index].data();
    const float *bias_data = cuda_params.conv_biases[layer_index].data();
    const float *input = previous_runtime.y.data();

    float *col = current_runtime.conv_im2col.data();
    float *a = current_runtime.a.data();
    float *y = current_runtime.y.data();

    const int batch_size = previous_runtime.y.batch_size();
    const int input_height = previous_runtime.y.height();
    const int input_width = previous_runtime.y.width();

    const int im2col_height = current_runtime.conv_im2col.height();
    const int im2col_width = current_runtime.conv_im2col.width();

    const int num_filters = current.dim_layer[2];
    
    const int output_size = static_cast<int>(current_runtime.y.size());
    const int output_channels = current_runtime.y.channels();

    const int kernel_height = current.kernel_dim[0]; 
    const int kernel_width = current.kernel_dim[1];
    const int kernel_channels = current.kernel_dim[2];

    const int patch_size = current.kernel_dim[0] * current.kernel_dim[1] * current.kernel_dim[2];
    const int out_features = current.dim_layer[0] * current.dim_layer[1];
    const int total_patches = batch_size * out_features;


    const int padding_height = current.padding[0];
    const int padding_width = current.padding[1];
    const int stride_height = current.stride[0];
    const int stride_width = current.stride[1];
    const bool use_lookahead = use_lookahead_params_batch(velocity, momentum);

    dim3 KernelDim(16, 16);
    int Grid_dim_h = (im2col_height + 16 -1) / 16;
    int Grid_dim_w = (im2col_width + 16 -1) / 16;
    int activation_grid_dim = (output_size + 256 -1) / 256;
    dim3 GridDim(Grid_dim_h, Grid_dim_w, batch_size);

    im2col <<< GridDim, KernelDim >>>(
        input, input_height, input_width,
        kernel_height, kernel_width, kernel_channels, batch_size,
        col, patch_size, out_features,
        padding_height, padding_width, stride_height, stride_width
    );

    cublasHandle_t handle = static_cast<cublasHandle_t>(cuda_backend::current_cublas_handle());
    if(!use_lookahead){
        float alpha = 1.0f;
        float beta = 0.0f;
        cublasSgemm(
            handle,
            CUBLAS_OP_T, CUBLAS_OP_T,
            num_filters, total_patches, patch_size,
            &alpha,
            filter_data, patch_size,
            col, total_patches,
            &beta, a, num_filters
        );
        apply_bias_activation <<< activation_grid_dim, 256 >>> (bias_data, a, y, output_size, output_channels, act.kind);
        return;
    }

    const float *velocity_filter_data = velocity->conv_weights[layer_index].data();
    const float *velocity_bias_data = velocity->conv_biases[layer_index].data();

    float alpha = 1.0f;
    float beta = 0.0f;
    cublasSgemm(
        handle,
        CUBLAS_OP_T, CUBLAS_OP_T,
        num_filters, total_patches, patch_size,
        &alpha,
        filter_data, patch_size,
        col, total_patches,
        &beta, a, num_filters
    );
    cublasSgemm(
        handle,
        CUBLAS_OP_T, CUBLAS_OP_T,
        num_filters, total_patches, patch_size,
        &momentum,
        velocity_filter_data, patch_size,
        col, total_patches,
        &alpha, a, num_filters
    );
    apply_nesterov_bias_activation <<< activation_grid_dim, 256 >>> (
        bias_data, velocity_bias_data, a, y,
        output_size, output_channels, momentum, 
        act.kind
    );
}

void forward_pooling_layer_batch(
    const Layer &current, CudaBatchLayerRuntime &current_runtime,
    const Layer &previous, const CudaBatchLayerRuntime &previous_runtime
){
    const float *input = previous_runtime.y.data();
    float *output = current_runtime.y.data();
    int *output_index = current_runtime.pooling_argmax.data();

    const int batch_size = previous_runtime.y.batch_size();

    const int input_height = previous.dim_layer[0];
    const int input_width = previous.dim_layer[1];

    const int output_height = current.dim_layer[0];
    const int output_width = current.dim_layer[1];
    const int output_channels = current.dim_layer[2];
    
    const int kernel_height = current.kernel_dim[0];
    const int kernel_width = current.kernel_dim[1];

    const int stride_h = current.stride[0];
    const int stride_w = current.stride[1];
    const int padding_h = current.padding[0];
    const int padding_w = current.padding[1];

    dim3 KernelDim(16, 16);
    int Grid_dim_h = (output_height + 16 -1) / 16;
    int Grid_dim_w = (output_width + 16 -1) / 16;
    dim3 GridDim(Grid_dim_h, Grid_dim_w, batch_size * output_channels);

    switch(current.pooling_type){
        case Pooling_type::Max: {
            max_pooling_forward <<< GridDim, KernelDim >>> (
                input, input_height, input_width,
                kernel_height, kernel_width, output_channels, batch_size,
                output, output_index, output_height, output_width, 
                padding_h, padding_w, stride_h, stride_w
            );
            break;
        }

        case Pooling_type::Average: {
            average_pooling_forward <<< GridDim, KernelDim >>> (
                input, input_height, input_width,
                kernel_height, kernel_width, output_channels, batch_size,
                output, output_height, output_width, 
                padding_h, padding_w, stride_h, stride_w
            );
            break;
        }

        case Pooling_type::L2: {
            L2_pooling_forward <<< GridDim, KernelDim >>> (
                input, input_height, input_width,
                kernel_height, kernel_width, output_channels, batch_size,
                output, output_height, output_width, 
                padding_h, padding_w, stride_h, stride_w
            );
            break;
        }
    }
}

void forward_lrn_layer_batch(
    const Layer &current, CudaBatchLayerRuntime &current_runtime,
    const CudaBatchLayerRuntime &previous_runtime
){

    const int window_size = std::max(1, current.lrn_local_size);
    const float alpha_over_size = current.lrn_alpha / static_cast<float>(window_size);
    const float lrn_k = current.lrn_k;
    const float lrn_beta = current.lrn_beta;

    const float *input = previous_runtime.y.data();
    float *output = current_runtime.y.data();

    const int channels = current_runtime.y.channels();
    const int total_size = static_cast<int>(current_runtime.y.size());

    int Grid_dim = (total_size + 256 -1) / 256;
    
    lrn_forward <<< Grid_dim, 256 >>> (
        input, output,
        window_size, total_size, channels,
        alpha_over_size, lrn_beta, lrn_k
    );

}

void forward_softmax_layer_batch(
    const Layer &current, CudaBatchLayerRuntime &current_runtime,
    const CudaBatchLayerRuntime &previous_runtime
){

    const float *input = previous_runtime.y.data();
    float *output = current_runtime.y.data();

    const int batch_size = previous_runtime.y.batch_size();
    const int flat_size = current.flat_output_size();

    const int block_size = 256;
    const std::size_t shared_bytes = static_cast<std::size_t>(block_size) * sizeof(float);

    softmax_forward <<< batch_size, block_size, shared_bytes >>> (input, output, batch_size, flat_size);
}

} // namespace

void feed_input_batch(
    const Dataset4D &input, const std::vector<int> &indices,
    int start, int end,
    const Layer &first, CudaBatchLayerRuntime &first_runtime
){
    require_condition(start >= 0 && end >= start && end <= static_cast<int>(indices.size()), "feed_input_batch: range batch non valido");

    const int batch_size = end - start;
    const int flat_size = first.flat_output_size();
    if(first_runtime.y.batch_size() != batch_size ||
       first_runtime.y.height() != first.dim_layer[0] ||
       first_runtime.y.width() != first.dim_layer[1] ||
       first_runtime.y.channels() != first.dim_layer[2]){
       first_runtime.y.resize(batch_size, first.dim_layer[0], first.dim_layer[1], first.dim_layer[2]);
    }

    for(int batch_index = 0; batch_index < batch_size; batch_index++){
        const Tensor &src = input[indices[start + batch_index]];
        validate_tensor_shape(src, first.dim_layer, "feed_input_batch");
        cuda_backend::check_cuda(cudaMemcpy(
            first_runtime.y.data() + static_cast<std::ptrdiff_t>(batch_index) * flat_size, src.data.data(),
            static_cast<std::size_t>(flat_size) * sizeof(float), cudaMemcpyHostToDevice
        ), "feed_input_batch: cudaMemcpy failed");
    }
}

void fill_target_batch(
    const Dataset4D &output,
    const std::vector<int> &indices, int start, int end,
    const Layer &last, cuda_backend::CudaBatchTensor<float> &target_batch
){
    require_condition(start >= 0 && end >= start && end <= static_cast<int>(indices.size()), "fill_target_batch: range batch non valido");

    const int batch_size = end - start;
    const int flat_size = last.flat_output_size();
    target_batch.resize(batch_size, last.dim_layer[0], last.dim_layer[1], last.dim_layer[2]);

    for(int batch_index = 0; batch_index < batch_size; batch_index++){
        const Tensor &src = output[indices[start + batch_index]];
        validate_tensor_shape(src, last.dim_layer, "fill_target_batch");
        cuda_backend::check_cuda(cudaMemcpy(
            target_batch.data() + static_cast<std::ptrdiff_t>(batch_index) * flat_size, src.data.data(),
            static_cast<std::size_t>(flat_size) * sizeof(float), cudaMemcpyHostToDevice
        ), "feed_target_batch: cudaMemcpy failed");
    }
}

void forwardprop_batch(
    const LayerList &architecture, cuda_backend::CudaBatchRuntimeList &runtime, int num_layers, const CudaParameterBuffer &cuda_params,
    const Activation &hidden_activation, const Activation &output_activation,
    const CudaParameterBuffer *velocity, float momentum
){
    for(int l = 1; l < num_layers; l++){
        const Layer &current = architecture[l];
        const Layer &previous = architecture[l - 1];
        const Layer *next = (l < num_layers - 1) ? &architecture[l + 1] : nullptr;

        CudaBatchLayerRuntime &current_runtime = runtime[l];
        const CudaBatchLayerRuntime &previous_runtime = runtime[l - 1];
        const CudaBatchLayerRuntime *next_runtime = (l < num_layers - 1) ? &runtime[l + 1] : nullptr;

        const bool next_is_softmax = (l < num_layers - 1) && (architecture[l + 1].type == Layer_type::Softmax);
        const Activation &act = next_is_softmax ? identity : (l < num_layers - 1) ? hidden_activation : output_activation;

        switch(current.type){
            case Layer_type::Dense:
                forward_dense_layer_batch(
                    l, current, current_runtime, cuda_params,
                    previous_runtime, act,
                    velocity, momentum
                );
                break;
            case Layer_type::Conv:
                forward_conv_layer_batch(
                    l, current, current_runtime, cuda_params,
                    previous_runtime, act,
                    velocity, momentum
                );
                break;
            case Layer_type::Pooling:
                forward_pooling_layer_batch(
                    current, current_runtime,
                    previous, previous_runtime
                );
                break;
            case Layer_type::Flatten:
                current_runtime.y.copy_from_device(previous_runtime.y);
                break;
            case Layer_type::LRN:
                forward_lrn_layer_batch(
                    current, current_runtime,
                    previous_runtime
                );
                break;
            case Layer_type::Softmax:
                forward_softmax_layer_batch(
                    current, current_runtime,
                    previous_runtime
                );
                break;
            case Layer_type::Input:
                break;
        }
    }
}



