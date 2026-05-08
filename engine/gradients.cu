
#include "engine/gradients.hpp"
#include "engine/backward.hpp"
#include "core/cuda_backend.hpp"
#include "math/losses.hpp"
#include "kernels/general.hpp"
#include "kernels/pooling.hpp"
#include "kernels/im2col.hpp"
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

void backprop_dense_layer_batch(
    CudaBatchLayerRuntime &current_runtime,
    const CudaBatchLayerRuntime &previous_runtime,
    const Layer *next, CudaBatchLayerRuntime *next_runtime, const CudaParameterBuffer &cuda_params,
    CudaParameterBuffer &gradients, 
    const bool is_output, int layer_index,
    const Loss &loss, const Activation &act, const cuda_backend::CudaBatchTensor<float> &desired_output, 
    const CudaParameterBuffer *velocity, float momentum
){
    
    const int total_size = current_runtime.y.size();
    const int batch_size = current_runtime.y.batch_size();
    const int out_features = total_size / batch_size;
    const int in_features = previous_runtime.y.height() * previous_runtime.y.width() * previous_runtime.y.channels();

    const float *desired = desired_output.data();
    const float *output = current_runtime.y.data();
    const float *input = previous_runtime.y.data();
    const float *a = current_runtime.a.data();
    const float *ones = current_runtime.reduction_ones.data();

    float *delta = current_runtime.delta.data();

    float *bias_grad = gradients.dense_biases[layer_index].data();
    float *wheight_grad = gradients.dense_weights[layer_index].data();


    int GridDim_delta = (total_size + 256 -1) / 256;

    if(is_output){
        compute_delta_output <<< GridDim_delta, 256 >>>(
            output, desired, a,
            total_size, batch_size,
            act.kind, act.alpha, act.beta,
            loss.kind, loss.reduction(), loss.beta,
            delta
        );
        cuda_backend::check_cuda_kernel("dense compute_delta_output");
    } else {
        build_cost_from_next_layer_all_batch(
            current_runtime,
            *next, *next_runtime, cuda_params, layer_index + 1,
            velocity, momentum, 
            current_runtime.backprop_cost_from_next
        );
        const float *cost_from_next = current_runtime.backprop_cost_from_next.data();
        compute_delta_hidden <<< GridDim_delta, 256 >>>(
            cost_from_next, a,
            total_size, act.kind, act.alpha, act.beta,
            delta
        );
        cuda_backend::check_cuda_kernel("dense compute_delta_hidden");
    }

    cublasHandle_t handle = static_cast<cublasHandle_t>(cuda_backend::current_cublas_handle());
    float alpha = 1.0f / batch_size;
    float beta = 0.0f;

    if(batch_size == 1){
        cuda_backend::check_cuda(cudaMemcpy(
            bias_grad,
            delta,
            out_features * sizeof(float),
            cudaMemcpyDeviceToDevice
        ), "cudaMemcpy dense bias gradient");
    } else {
        cuda_backend::check_cublas(cublasSgemv(
            handle,
            CUBLAS_OP_N,
            out_features, batch_size,
            &alpha,
            delta, out_features,
            ones, 1,
            &beta,
            bias_grad, 1
        ), "dense bias gradient cublasSgemv");
    }

    cuda_backend::check_cublas(cublasSgemm(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_T,
        in_features, out_features, batch_size,
        &alpha,
        input, in_features,
        delta, out_features,
        &beta,
        wheight_grad, in_features
    ), "dense weight gradient cublasSgemm");
}

void backprop_conv_layer_batch(
    const Layer &current, CudaBatchLayerRuntime &current_runtime,
    const CudaBatchLayerRuntime &previous_runtime,
    const Layer *next, CudaBatchLayerRuntime *next_runtime, const CudaParameterBuffer &cuda_params,
    CudaParameterBuffer &gradients, 
    const bool is_output, int layer_index,
    const Loss &loss, const Activation &act, const cuda_backend::CudaBatchTensor<float> &desired_output, 
    const CudaParameterBuffer *velocity, float momentum
){
    
    const int total_size = current_runtime.y.size();
    const int batch_size = current_runtime.y.batch_size();
    
    const float *input = previous_runtime.y.data();
    const int input_height = previous_runtime.y.height();
    const int input_width = previous_runtime.y.width();

    const int num_filters = current_runtime.y.channels();
    const int out_height = current_runtime.y.height();
    const int out_width = current_runtime.y.width();

    const int out_patches = out_height * out_width;
    const int total_out_patches = out_patches * batch_size;

    const int kernel_height = current.kernel_shape[0];
    const int kernel_width = current.kernel_shape[1];
    const int kernel_channels = current.kernel_shape[2];
    const int patch_size = kernel_height * kernel_width * kernel_channels;

    const int padding_height = current.padding[0];
    const int padding_width = current.padding[1];
    const int stride_height = current.stride[0];
    const int stride_width = current.stride[1];

    const float *desired = desired_output.data();
    const float *output = current_runtime.y.data();
    const float *a = current_runtime.a.data();

    const float *ones = current_runtime.reduction_ones.data();

    float *delta = current_runtime.delta.data();

    float *bias_grad = gradients.conv_biases[layer_index].data();
    float *filter_grad = gradients.conv_weights[layer_index].data();

    int GridDim_delta = (total_size + 256 -1) / 256;

    if(is_output){
        compute_delta_output <<< GridDim_delta, 256 >>>(
            output, desired, a,
            total_size, batch_size,
            act.kind, act.alpha, act.beta,
            loss.kind, loss.reduction(), loss.beta,
            delta
        );
        cuda_backend::check_cuda_kernel("conv compute_delta_output");
    } else {
        build_cost_from_next_layer_all_batch(
            current_runtime,
            *next, *next_runtime, cuda_params, layer_index + 1,
            velocity, momentum, 
            current_runtime.backprop_cost_from_next
        );
        const float *cost_from_next = current_runtime.backprop_cost_from_next.data();
        compute_delta_hidden <<< GridDim_delta, 256 >>>(
            cost_from_next, a,
            total_size, act.kind, act.alpha, act.beta,
            delta
        );
        cuda_backend::check_cuda_kernel("conv compute_delta_hidden");
    }

    cublasHandle_t handle = static_cast<cublasHandle_t>(cuda_backend::current_cublas_handle());
    float alpha = 1.0f / batch_size;
    float beta = 0.0f;

    if(total_out_patches == 1){
        cuda_backend::check_cuda(cudaMemcpy(
            bias_grad,
            delta,
            num_filters * sizeof(float),
            cudaMemcpyDeviceToDevice
        ), "cudaMemcpy conv bias gradient");
    } else {
        cuda_backend::check_cublas(cublasSgemv(
            handle,
            CUBLAS_OP_N,
            num_filters, total_out_patches,
            &alpha,
            delta, num_filters,
            ones, 1,
            &beta,
            bias_grad, 1
        ), "conv bias gradient cublasSgemv");
    }

    current_runtime.conv_im2col.resize(batch_size, patch_size, out_patches, 1);
    float *col = current_runtime.conv_im2col.data();
    const int im2col_height = current_runtime.conv_im2col.height();
    const int im2col_width = current_runtime.conv_im2col.width();

    dim3 KernelDim(16, 16);
    int Grid_dim_h = (im2col_height + 16 -1) / 16;
    int Grid_dim_w = (im2col_width + 16 -1) / 16;
    dim3 GridDim_im2col(Grid_dim_h, Grid_dim_w, batch_size);

    im2col <<< GridDim_im2col, KernelDim >>>(
        input, input_height, input_width,
        kernel_height, kernel_width, kernel_channels, batch_size,
        col, patch_size, out_patches,
        padding_height, padding_width, stride_height, stride_width
    );
    cuda_backend::check_cuda_kernel("conv gradient im2col");

    cuda_backend::check_cublas(cublasSgemm(
        handle,
        CUBLAS_OP_T, CUBLAS_OP_T,
        patch_size, num_filters, total_out_patches,
        &alpha,
        col, total_out_patches,
        delta, num_filters,
        &beta,
        filter_grad, patch_size
    ), "conv filter gradient cublasSgemm");
}

} // namespace

void backprop_batch(
    const LayerList &architecture, cuda_backend::CudaBatchRuntimeList &runtime,
    CudaParameterBuffer &gradients, const CudaParameterBuffer &cuda_params,
    const Loss &loss, const cuda_backend::CudaBatchTensor<float> &desired_output,
    const Activation &hidden_activation, const Activation &output_activation,
    const CudaParameterBuffer *velocity, float momentum
){
    const int num_layers = static_cast<int>(architecture.size());

    for(int l = num_layers - 1; l > 0; l--){
        const Layer &current = architecture[l];
        const Layer *next = (l < num_layers - 1) ? &architecture[l + 1] : nullptr;

        CudaBatchLayerRuntime &current_runtime = runtime[l];
        const CudaBatchLayerRuntime &previous_runtime = runtime[l - 1];
        CudaBatchLayerRuntime *next_runtime = (l < num_layers - 1) ? &runtime[l + 1] : nullptr;

        const bool is_output = (l == num_layers - 1);
        const bool next_is_softmax = (next != nullptr) && (next->type == Layer_type::Softmax);
        const Activation &act = next_is_softmax ? identity : (l < num_layers - 1) ? hidden_activation : output_activation;

        switch(current.type){
            case Layer_type::Dense:
                backprop_dense_layer_batch(
                    current_runtime,
                    previous_runtime,
                    next, next_runtime, cuda_params,
                    gradients,
                    is_output, l,
                    loss, act, desired_output, 
                    velocity, momentum
                );
                break;
            case Layer_type::Conv:
                backprop_conv_layer_batch(
                    current, current_runtime,
                    previous_runtime,
                    next, next_runtime, cuda_params,
                    gradients,
                    is_output, l,
                    loss, act, desired_output, 
                    velocity, momentum
                );
                break;
            case Layer_type::Pooling:
                build_cost_from_next_layer_all_batch(
                    current_runtime,
                    *next, *next_runtime, cuda_params, l + 1,
                    velocity, momentum, 
                    current_runtime.backprop_cost_from_next
                );
                current_runtime.delta.alias_from(
                    current_runtime.backprop_cost_from_next,
                    current_runtime.y.batch_size(),
                    current.dim_layer[0],
                    current.dim_layer[1],
                    current.dim_layer[2]
                );
                break;
            case Layer_type::Flatten:
                build_cost_from_next_layer_all_batch(
                    current_runtime,
                    *next, *next_runtime, cuda_params, l + 1,
                    velocity, momentum, 
                    current_runtime.backprop_cost_from_next
                );
                current_runtime.delta.alias_from(
                    current_runtime.backprop_cost_from_next,
                    current_runtime.y.batch_size(),
                    current.dim_layer[0],
                    current.dim_layer[1],
                    current.dim_layer[2]
                );
                break;
            case Layer_type::Softmax: {
                int GridDim = (current_runtime.y.size() + 256 -1) / 256;
                softmax_backward <<< GridDim, 256 >>>(
                    current_runtime.y.data(), desired_output.data(), current_runtime.delta.data(), static_cast<int>(current_runtime.y.size())
                );
                cuda_backend::check_cuda_kernel("softmax_backward");
                break;
            }
            case Layer_type::Input:
                break;
        }
    }
}
