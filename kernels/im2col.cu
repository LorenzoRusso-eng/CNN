#include "kernels/im2col.hpp"

#include <cmath>
#include <cstddef>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <math_constants.h>

__global__ void im2col(
    const float *Im, int Im_h, int Im_w,
    int ker_h, int ker_w, int ker_c, int batch_size,
    float *Col, int Col_h, int Col_w, 
    int pad_h, int pad_w, int str_h, int str_w
){

    int row = blockDim.x * blockIdx.x + threadIdx.x;
    int col = blockDim.y * blockIdx.y + threadIdx.y;
    int batch = blockIdx.z;

    if(row < Col_h && col < Col_w && batch < batch_size){
        int out_w = (Im_w + 2 * pad_w - ker_w) / str_w + 1;

        // col indicates where i am on the output
        int out_i = col / out_w;
        int out_j = col % out_w;

        // row indicates where i am on the kernel
        int ker_i = (row / ker_c) / ker_w;
        int ker_j = (row / ker_c) % ker_w;
        int ch = row % ker_c;

        // Computing indices of the input
        int in_i = ker_i + out_i * str_h - pad_h;
        int in_j = ker_j + out_j * str_w - pad_w;

        // Assigning value based on padding
        if(in_i < 0 || in_i >= Im_h || in_j < 0 || in_j >= Im_w){
            Col[(row * batch_size + batch) * Col_w + col] = 0;
        }
        else{
            Col[(row * batch_size + batch) * Col_w + col] = Im[((batch * Im_h + in_i) * Im_w + in_j) * ker_c + ch];
        }
    }

}

__global__ void col2im(
    const float *Col, int Col_h, int Col_w,
    int ker_h, int ker_w, int channels, int batch_size,
    float *Im, int Im_h, int Im_w, 
    int pad_h, int pad_w, int str_h, int str_w
){

    int row = blockDim.x * blockIdx.x + threadIdx.x;
    int col = blockDim.y * blockIdx.y + threadIdx.y;
    int bz = blockIdx.z;
    int ch = bz % channels;
    int batch = bz / channels;

    if(row < Im_h && col < Im_w && bz < batch_size * channels){

        int out_w = (Im_w + 2 * pad_w - ker_w) / str_w + 1;
        int out_h = (Im_h + 2 * pad_h - ker_h) / str_h + 1;

        float sum = 0;
        for(int ker_i = 0; ker_i < ker_h; ker_i++){

            int out_i_t = row + pad_h - ker_i;
            if(out_i_t < 0 || out_i_t % str_h != 0) continue;

            int out_i = out_i_t / str_h;
            if(out_i >= out_h) continue;


            for(int ker_j = 0; ker_j < ker_w; ker_j++){

                int out_j_t = (col + pad_w - ker_j);
                if(out_j_t < 0 || out_j_t % str_w != 0) continue;

                int out_j = out_j_t / str_w;
                if(out_j >= out_w) continue;

                int Col_i = out_i * out_w + out_j;
                int Col_j = (ker_i * ker_w + ker_j) * channels + ch;

                sum += Col[(Col_i + batch * Col_h) * Col_w + Col_j];

            }
        }
        Im[((batch * Im_h + row) * Im_w + col) * channels + ch] = sum;
    }
}
