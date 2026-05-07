#include "engine/backward.hpp"
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
#include <cub/cub.cuh>
#include <math_constants.h>

using cuda_backend::CudaBatchLayerRuntime;
using cuda_backend::CudaParameterBuffer;

namespace{

inline bool use_lookahead_params_batch(const CudaParameterBuffer *velocity, float momentum){
    return velocity != nullptr && momentum != 0.0f;
}

void build_cost_from_next_dense_all_batch(
    const CudaBatchLayerRuntime &next_runtime, const CudaParameterBuffer &cuda_params, int next_layer_index,
    const CudaParameterBuffer *velocity, float momentum, 
    cuda_backend::CudaBatchTensor<float> &out_cost
){

    const int batch_size = next_runtime.y.batch_size();

    const float *delta_data = next_runtime.delta.data();
    const float *weights_data = cuda_params.dense_weights[next_layer_index].data();

    const int in_features = cuda_params.dense_weights[next_layer_index].width();
    const int out_features = cuda_params.dense_weights[next_layer_index].height();

    cublasHandle_t handle = static_cast<cublasHandle_t>(cuda_backend::current_cublas_handle());

    float alpha = 1.0f;
    float beta = 0.0f;

    cuda_backend::check_cublas(cublasSgemm(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        in_features, batch_size, out_features,
        &alpha,
        weights_data, in_features,
        delta_data, out_features,
        &beta,
        out_cost.data(), in_features
    ), "backward dense cost cublasSgemm");

    if(use_lookahead_params_batch(velocity, momentum)){
        const float *velocity_weights_data = velocity->dense_weights[next_layer_index].data();
        cuda_backend::check_cublas(cublasSgemm(
            handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            in_features, batch_size, out_features,
            &momentum,
            velocity_weights_data, in_features,
            delta_data, out_features,
            &alpha,
            out_cost.data(), in_features
        ), "backward dense lookahead cost cublasSgemm");
    }
}

void build_cost_from_next_conv_all_batch(
    CudaBatchLayerRuntime &current_runtime, 
    const Layer &next, const CudaBatchLayerRuntime &next_runtime, const CudaParameterBuffer &cuda_params, int next_layer_index,
    const CudaParameterBuffer *velocity, float momentum,
    cuda_backend::CudaBatchTensor<float> &out_cost
){

    const int batch_size = current_runtime.y.batch_size();
    const int num_filters = next.dim_layer[2];
    const int out_features = next.dim_layer[0] * next.dim_layer[1];
    const int total_out_size = batch_size * out_features;

    const int kernel_height = next.kernel_dim[0];
    const int kernel_width = next.kernel_dim[1];
    const int channels = next.kernel_dim[2];

    const int padding_height = next.padding[0];
    const int padding_width = next.padding[1];
    const int stride_height = next.stride[0];
    const int stride_width = next.stride[1];

    const int patch_size = kernel_height * kernel_width * channels;

    const float *delta_data = next_runtime.delta.data();
    const float *filter_data = cuda_params.conv_weights[next_layer_index].data();
    float *prev_delta = out_cost.data();
    
    const int prev_height = out_cost.height();
    const int prev_width = out_cost.width();

    current_runtime.conv_im2col.resize(batch_size, out_features, patch_size, 1);
    float *col = current_runtime.conv_im2col.data();

    cublasHandle_t handle = static_cast<cublasHandle_t>(cuda_backend::current_cublas_handle());
    float alpha = 1.0f;
    float beta = 0.0f;

    cuda_backend::check_cublas(cublasSgemm(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        patch_size, total_out_size, num_filters,
        &alpha,
        filter_data, patch_size,
        delta_data, num_filters,
        &beta,
        col, patch_size
    ), "backward conv cost cublasSgemm");

    if(use_lookahead_params_batch(velocity, momentum)){
        const float *velocity_filter_data = velocity->conv_weights[next_layer_index].data();
        cuda_backend::check_cublas(cublasSgemm(
            handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            patch_size, total_out_size, num_filters,
            &momentum,
            velocity_filter_data, patch_size,
            delta_data, num_filters,
            &alpha,
            col, patch_size
        ), "backward conv lookahead cost cublasSgemm");
    }

    dim3 KernelDim(16, 16);
    int Grid_dim_h = (prev_height + 16 -1) / 16;
    int Grid_dim_w = (prev_width + 16 -1) / 16;
    dim3 GridDim(Grid_dim_h, Grid_dim_w, batch_size * channels);

    col2im <<< GridDim, KernelDim  >>>(
        col, out_features, patch_size,
        kernel_height, kernel_width, channels, batch_size,
        prev_delta, prev_height, prev_width,
        padding_height, padding_width, stride_height, stride_width
    );
    cuda_backend::check_cuda_kernel("backward conv col2im");
}

void build_cost_from_next_pool_all_batch(
    CudaBatchLayerRuntime &current_runtime, 
    const Layer &next, const CudaBatchLayerRuntime &next_runtime,
    cuda_backend::CudaBatchTensor<float> &out_cost
){

    const float *prev_out = current_runtime.y.data();
    float *prev_delta = out_cost.data();
    const float *next_delta = next_runtime.delta.data();
    const float *next_out = next_runtime.y.data();
    const int *prev_winners = next_runtime.pooling_argmax.data();

    const int batch_size = current_runtime.y.batch_size();
    const int channels = current_runtime.y.channels();

    const int prev_height = current_runtime.y.height();
    const int prev_width = current_runtime.y.width();
    const int prev_total_size = static_cast<int>(out_cost.size());

    const int next_height = next.dim_layer[0];
    const int next_width = next.dim_layer[1];
    const int next_total_size = static_cast<int>(next_runtime.delta.size());
    
    const int kernel_height = next.kernel_dim[0];
    const int kernel_width = next.kernel_dim[1];

    const int stride_height = next.stride[0];
    const int stride_width = next.stride[1];
    const int padding_height = next.padding[0];
    const int padding_width = next.padding[1];

    dim3 KernelDim(16, 16);
    int Grid_dim_h_prev = (prev_height + 16 -1) / 16;
    int Grid_dim_w_prev = (prev_width + 16 -1) / 16;
    dim3 GridDimPrev(Grid_dim_h_prev, Grid_dim_w_prev, batch_size * channels);

    
    const int grid_size = (next_total_size + 256 - 1) / 256;
    const bool non_overlapping_pooling = padding_height == 0 && padding_width == 0 
                                        &&
                                        stride_height >= kernel_height && stride_width >= kernel_width;

    switch(next.pooling_type){
        case Pooling_type::Max:
            cuda_backend::check_cuda(
                cudaMemset(prev_delta, 0, out_cost.size() * sizeof(float)),
                "backward max pooling cudaMemset prev_delta"
            );
            if(non_overlapping_pooling){
                max_pooling_backward_no_overlap <<< grid_size, 256 >>> (
                    next_delta, prev_winners, next_total_size,
                    prev_delta, prev_total_size, batch_size
                );
                cuda_backend::check_cuda_kernel("backward max_pooling_backward_no_overlap");
            }
            else{
                max_pooling_backward_overlap <<< grid_size, 256 >>> (
                    next_delta, prev_winners, next_total_size,
                    prev_delta, prev_total_size, batch_size
                );
                cuda_backend::check_cuda_kernel("backward max_pooling_backward_overlap");
            }
            break;
        case Pooling_type::Average:
            average_pooling_backward <<< GridDimPrev, KernelDim >>> (
                next_delta, next_height, next_width,
                kernel_height, kernel_width, channels, batch_size,
                prev_delta, prev_height, prev_width,
                padding_height, padding_width, stride_height, stride_width
            );
            cuda_backend::check_cuda_kernel("backward average_pooling_backward");
            break;
        case Pooling_type::L2:
            L2_pooling_backward <<< GridDimPrev, KernelDim >>> (
                next_delta, next_out, next_height, next_width,
                kernel_height, kernel_width, channels, batch_size,
                prev_out, prev_delta, prev_height, prev_width,
                padding_height, padding_width, stride_height, stride_width
            );
            cuda_backend::check_cuda_kernel("backward L2_pooling_backward");
            break;
    }
}


void build_cost_from_next_lrn_all_batch(
    CudaBatchLayerRuntime &current_runtime, 
    const Layer &next, const CudaBatchLayerRuntime &next_runtime,
    cuda_backend::CudaBatchTensor<float> &out_cost
){

    const float *prev_out = current_runtime.y.data();
    float *prev_delta = out_cost.data();
    const float *next_out = next_runtime.y.data();
    const float *next_delta = next_runtime.delta.data();

    const int channels = current_runtime.y.channels();

    const int total_size = current_runtime.y.size();
    const int prev_total_size = static_cast<int>(out_cost.size());

    const int window_size = std::max(1, next.lrn_local_size);
    const float alpha_over_size = next.lrn_alpha / static_cast<float>(window_size);
    const float lrn_k = next.lrn_k;
    const float lrn_beta = next.lrn_beta;

    int GridDim = (prev_total_size + 256 -1) / 256;

    lrn_backward <<< GridDim, 256 >>>(
        next_delta,
        window_size, total_size, channels,
        prev_delta, prev_out,
        alpha_over_size, lrn_beta, lrn_k
    );
    cuda_backend::check_cuda_kernel("backward lrn_backward");

}

} // namespace

void build_cost_from_next_layer_all_batch(
    CudaBatchLayerRuntime &current_runtime,
    const Layer &next, const CudaBatchLayerRuntime &next_runtime, const CudaParameterBuffer &cuda_params, int next_layer_index,
    const CudaParameterBuffer *velocity, float momentum,
    cuda_backend::CudaBatchTensor<float> &out_cost
){
    const int batch_size = current_runtime.y.batch_size();
    const int height = current_runtime.y.height();
    const int width = current_runtime.y.width();
    const int channels = current_runtime.y.channels();
    
    out_cost.resize(batch_size, height, width, channels);

    switch(next.type){
        case Layer_type::Dense:
            build_cost_from_next_dense_all_batch(
                next_runtime, cuda_params, next_layer_index,
                velocity, momentum,
                out_cost
            );
            break;

        case Layer_type::Conv:
            build_cost_from_next_conv_all_batch(
                current_runtime,
                next, next_runtime, cuda_params, next_layer_index,
                velocity, momentum,
                out_cost
            );
            break;

        case Layer_type::Pooling:
            build_cost_from_next_pool_all_batch(
                current_runtime,
                next, next_runtime,
                out_cost
            );
            break;

        case Layer_type::Flatten:
            out_cost.copy_data_from_device(next_runtime.delta);
            break;

        case Layer_type::LRN:
            build_cost_from_next_lrn_all_batch(
                current_runtime,
                next, next_runtime, 
                out_cost);
            break;

        case Layer_type::Softmax:
            out_cost.copy_from_device(next_runtime.delta);
            break;

        case Layer_type::Input:
            break;
    }
}


