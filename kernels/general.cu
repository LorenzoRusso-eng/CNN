#include "kernels/general.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <math_constants.h>


__global__ void apply_bias_activation(
    const float *B, float *A, float *Y,
    int total_size, int bias_size, ActivationKind kind,
    float alpha, float beta
){

    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if( i < total_size){
        int bs = i % bias_size;
        A[i] += B[bs];
        Y[i] = cuda_apply_activation(A[i], kind, alpha, beta);
    }
    
}

__global__ void copy_negated(const float *In, float *Out, int total_size){
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if(i < total_size){
        Out[i] = -In[i];
    }
}

__global__ void apply_nesterov_bias_activation(
    const float *B, const float *VB, float *A, float *Y,
    int total_size, int bias_size, float momentum, ActivationKind kind,
    float alpha, float beta
){

    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if( i < total_size){
        int bs = i % bias_size;
        A[i] += B[bs] + momentum * VB[bs];
        Y[i] = cuda_apply_activation(A[i], kind, alpha, beta);
    }
    
}                             

__global__ void compute_delta_output(
    const float *output, const float *desired, const float *a,
    int total_size, int batch_size, ActivationKind actkind, float activation_alpha, float activation_beta,
    LossKind losskind, Reduction red, float loss_beta,
    float *delta
){
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if(i < total_size){
        int flat_size = total_size / batch_size;
        float loss_der = cuda_apply_cost_derivative(output[i], desired[i], losskind, red, flat_size, loss_beta);
        float act_der = cuda_apply_activation_derivative(a[i], actkind, activation_alpha, activation_beta);
        delta[i] = act_der * loss_der;
    }
}

__global__ void compute_delta_hidden(
    const float *cost_from_next, const float *a,
    int total_size, ActivationKind kind, float activation_alpha, float activation_beta,
    float *delta
){
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if(i < total_size){
        float act_der = cuda_apply_activation_derivative(a[i], kind, activation_alpha, activation_beta);
        delta[i] = act_der * cost_from_next[i];
    }
}

__global__ void compute_loss(
    const float *output, const float *desired,
    int total_size, int flat_size, LossKind kind, Reduction red, float loss_beta,
    float *loss_value
){
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if(i < total_size){
        loss_value[i] = cuda_apply_cost(output[i], desired[i], kind, red, flat_size, loss_beta);
    }
}

__global__ void update_params(
    const float *grad_w, const float *grad_b, float *param_w, float *param_b, float *velocity_w, float *velocity_b,
    float learning_rate, float momentum, int total_size, int in_features
){
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if(i < total_size){
        if(i % in_features == 0){
            int feature_idx = i / in_features;
            velocity_b[feature_idx] = momentum * velocity_b[feature_idx] - learning_rate * grad_b[feature_idx];
            param_b[feature_idx] += velocity_b[feature_idx];
        }

        velocity_w[i] = momentum * velocity_w[i] - learning_rate * grad_w[i];
        param_w[i] += velocity_w[i];
        
    }
}

__global__ void accumulate_scaled_params(
    const float *grad_w, const float *grad_b,
    float *accum_w, float *accum_b,
    float scale, int total_size, int in_features
){
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if(i < total_size){
        if(i % in_features == 0){
            int feature_idx = i / in_features;
            accum_b[feature_idx] += scale * grad_b[feature_idx];
        }

        accum_w[i] += scale * grad_w[i];
    }
}

