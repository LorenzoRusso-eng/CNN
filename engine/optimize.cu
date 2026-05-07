
#include "engine/optimize.hpp"
#include "core/cuda_backend.hpp"
#include "kernels/general.hpp"
#include "kernels/pooling.hpp"
#include "kernels/im2col.hpp"
#include "kernels/softmax.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cub/cub.cuh>
#include <math_constants.h>

using cuda_backend::CudaParameterBuffer;

void optimizer_step(
    LayerList &architecture, int num_layers,
    const CudaParameterBuffer &gradients, CudaParameterBuffer &velocity, CudaParameterBuffer &cuda_params,
    float learning_rate, float momentum
){
    for(int l=1; l<num_layers; l++){
        Layer &current = architecture[l];
        switch(current.type){
            case Layer_type::Dense: {
                float *weight_data = cuda_params.dense_weights[l].data();
                float *velocity_weight_data = velocity.dense_weights[l].data();

                float *bias_data = cuda_params.dense_biases[l].data();
                float *velocity_bias_data = velocity.dense_biases[l].data();

                const float *weight_grad = gradients.dense_weights[l].data();
                const float *bias_grad = gradients.dense_biases[l].data();

                int out_features = cuda_params.dense_weights[l].height();
                int in_features = cuda_params.dense_weights[l].width();
                int total_size = out_features * in_features;

                int GridDim = (total_size + 256 - 1) / 256;
                update_params <<< GridDim, 256>>>(
                    weight_grad, bias_grad,
                    weight_data, bias_data,
                    velocity_weight_data, velocity_bias_data,
                    learning_rate, momentum, total_size, in_features
                );
                cuda_backend::check_cuda_kernel("optimizer dense update_params");
                break;
            }

            case Layer_type::Conv: {
                float *filter_data = cuda_params.conv_weights[l].data();
                float *velocity_filter_data = velocity.conv_weights[l].data();

                float *bias_data = cuda_params.conv_biases[l].data();
                float *velocity_bias_data = velocity.conv_biases[l].data();

                const float *filter_grad = gradients.conv_weights[l].data();
                const float *bias_grad = gradients.conv_biases[l].data();

                int num_filters = cuda_params.conv_weights[l].height();
                int patch_size = cuda_params.conv_weights[l].width();
                int total_size = patch_size * num_filters;

                int GridDim = (total_size + 256 - 1) / 256;
                update_params <<< GridDim, 256>>>(
                    filter_grad, bias_grad,
                    filter_data, bias_data,
                    velocity_filter_data, velocity_bias_data,
                    learning_rate, momentum, total_size, patch_size
                );
                cuda_backend::check_cuda_kernel("optimizer conv update_params");
                break;
            }

            case Layer_type::Pooling:
            case Layer_type::Flatten:
            case Layer_type::Softmax:
            case Layer_type::Input:
                break;
        }
    }
}

void accumulate_scaled_gradients(
    const LayerList &architecture, int num_layers,
    CudaParameterBuffer &accumulated,
    const CudaParameterBuffer &chunk,
    float scale
){
    for(int l=1; l<num_layers; l++){
        const Layer &current = architecture[l];
        switch(current.type){
            case Layer_type::Dense: {
                float *accum_weight_data = accumulated.dense_weights[l].data();
                float *accum_bias_data = accumulated.dense_biases[l].data();

                const float *chunk_weight_grad = chunk.dense_weights[l].data();
                const float *chunk_bias_grad = chunk.dense_biases[l].data();

                int out_features = accumulated.dense_weights[l].height();
                int in_features = accumulated.dense_weights[l].width();
                int total_size = out_features * in_features;

                int GridDim = (total_size + 256 - 1) / 256;
                accumulate_scaled_params <<< GridDim, 256 >>>(
                    chunk_weight_grad, chunk_bias_grad,
                    accum_weight_data, accum_bias_data,
                    scale, total_size, in_features
                );
                cuda_backend::check_cuda_kernel("accumulate dense gradients");
                break;
            }

            case Layer_type::Conv: {
                float *accum_filter_data = accumulated.conv_weights[l].data();
                float *accum_bias_data = accumulated.conv_biases[l].data();

                const float *chunk_filter_grad = chunk.conv_weights[l].data();
                const float *chunk_bias_grad = chunk.conv_biases[l].data();

                int num_filters = accumulated.conv_weights[l].height();
                int patch_size = accumulated.conv_weights[l].width();
                int total_size = patch_size * num_filters;

                int GridDim = (total_size + 256 - 1) / 256;
                accumulate_scaled_params <<< GridDim, 256 >>>(
                    chunk_filter_grad, chunk_bias_grad,
                    accum_filter_data, accum_bias_data,
                    scale, total_size, patch_size
                );
                cuda_backend::check_cuda_kernel("accumulate conv gradients");
                break;
            }

            case Layer_type::Pooling:
            case Layer_type::Flatten:
            case Layer_type::Softmax:
            case Layer_type::Input:
                break;
        }
    }
}
