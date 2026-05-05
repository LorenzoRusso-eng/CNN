#pragma once

#include <cuda_runtime.h>

__global__ void max_pooling_forward(
    const float *In, int In_h, int In_w,
    int ker_h, int ker_w, int channels, int batch_size,
    float *Out, int *Out_idx, int Out_h, int Out_w,
    int pad_h, int pad_w, int str_h, int str_w
);

__global__ void average_pooling_forward(
    const float *In, int In_h, int In_w,
    int ker_h, int ker_w, int channels, int batch_size,
    float *Out, int Out_h, int Out_w,
    int pad_h, int pad_w, int str_h, int str_w
);

__global__ void L2_pooling_forward(
    const float *In, int In_h, int In_w,
    int ker_h, int ker_w, int channels, int batch_size,
    float *Out, int Out_h, int Out_w,
    int pad_h, int pad_w, int str_h, int str_w
);

__global__ void max_pooling_backward_no_overlap(
    const float *Next, const int *Win_idx, int Next_total_size,
    float *Prev, int Prev_total_size, int Batch_size
);

__global__ void max_pooling_backward_overlap(
    const float *Next, const int *Win_idx, int Next_total_size,
    float *Prev, int Prev_total_size, int Batch_size
);

__global__ void average_pooling_backward(
    const float *Next, int Next_h, int Next_w,
    int ker_h, int ker_w, int channels, int batch_size,
    float *Prev, int Prev_h, int Prev_w,
    int pad_h, int pad_w, int str_h, int str_w
);

__global__ void L2_pooling_backward(
    const float *Next_delta, const float *Next_output, int Next_h, int Next_w,
    int ker_h, int ker_w, int channels, int batch_size,
    const float *Prev_out, float *Prev_cost_der, int Prev_h, int Prev_w,
    int pad_h, int pad_w, int str_h, int str_w
);
