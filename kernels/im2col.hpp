#pragma once

#include <cuda_runtime.h>

__global__ void im2col(
    const float *Im, int Im_h, int Im_w,
    int ker_h, int ker_w, int ker_c, int batch_size,
    float *Col, int Col_h, int Col_w,
    int pad_h, int pad_w, int str_h, int str_w
);

__global__ void col2im(
    const float *Col, int Col_h, int Col_w,
    int ker_h, int ker_w, int channels, int batch_size,
    float *Im, int Im_h, int Im_w,
    int pad_h, int pad_w, int str_h, int str_w
);
