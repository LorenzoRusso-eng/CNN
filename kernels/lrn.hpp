#pragma once

#include <cuda_runtime.h>

__global__ void lrn_forward(
    const float *In, float *Out,
    int window_size, int total_size, int channels,
    float alpha, float beta, float k
);

__global__ void lrn_backward(
    const float *Next_delta,
    int window_size, int total_size, int channels,
    float *Prev_cost_der, const float *Prev_out,
    float alpha, float beta, float k
);
