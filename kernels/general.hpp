#pragma once

#include "math/activations.hpp"
#include "math/losses.hpp"

#include <cuda_runtime.h>

__global__ void apply_bias_activation(
    const float *B, float *A, float *Y,
    int total_size, int bias_size, ActivationKind kind,
    float alpha = 0.0f, float beta = 0.0f
);

__global__ void copy_negated(const float *In, float *Out, int total_size);

__global__ void apply_nesterov_bias_activation(
    const float *B, const float *VB, float *A, float *Y,
    int total_size, int bias_size, float momentum, ActivationKind kind,
    float alpha = 0.0f, float beta = 0.0f
);

__global__ void compute_delta_output(
    const float *desired, const float *output, const float *a,
    int total_size, int batch_size, ActivationKind actkind, float activation_alpha, float activation_beta,
    LossKind losskind, Reduction red, float loss_beta,
    float *delta
);

__global__ void compute_delta_hidden(
    const float *cost_from_next, const float *a,
    int total_size, ActivationKind kind, float activation_alpha, float activation_beta,
    float *delta
);

__global__ void compute_loss(
    const float *output, const float *desired,
    int total_size, int flat_size, LossKind kind, Reduction red, float loss_beta,
    float *loss_value
);

__global__ void update_params(
    const float *grad_w, const float *grad_b, float *param_w, float *param_b, float *velocity_w, float *velocity_b,
    float learning_rate, float momentum, int total_size, int in_features
);

__global__ void accumulate_scaled_params(
    const float *grad_w, const float *grad_b,
    float *accum_w, float *accum_b,
    float scale, int total_size, int in_features
);
