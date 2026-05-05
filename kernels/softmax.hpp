#pragma once

#include <cuda_runtime.h>

__global__ void softmax_forward(const float *In, float *Out, int batch_size, int flat_size);

__global__ void softmax_backward(const float *Output, const float *Desired, float *Delta, int total_size);
