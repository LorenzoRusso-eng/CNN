
#include "kernels/pooling.hpp"

#include <cmath>
#include <cstddef>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <math_constants.h>

__global__ void max_pooling_forward(
    const float *In, int In_h, int In_w,
    int ker_h, int ker_w, int channels, int batch_size,
    float *Out, int *Out_idx, int Out_h, int Out_w,
    int pad_h, int pad_w, int str_h, int str_w
){

    int row = blockDim.x * blockIdx.x + threadIdx.x;
    int col = blockDim.y * blockIdx.y + threadIdx.y;
    int bz = blockIdx.z;
    int batch = bz / channels;
    int ch = bz % channels;

    int global_out_index = ((batch * Out_h + row) * Out_w + col) * channels + ch;


    if(row < Out_h && col < Out_w && bz < channels * batch_size){

        int input_origin_i = row * str_h - pad_h;
        int input_origin_j = col * str_w - pad_w;

        int kh_begin = max(0, -input_origin_i);
        int kh_end = min(ker_h, In_h - input_origin_i);
        
        int kw_begin = max(0, -input_origin_j);
        int kw_end = min(ker_w, In_w - input_origin_j);

        float max_value = -CUDART_INF_F;
        int max_index = -1;

        if(kh_begin >= kh_end || kw_begin >= kw_end){
            Out[global_out_index] = 0.0f;
            Out_idx[global_out_index] = -1;
            return;
        }

        for(int ker_i = kh_begin; ker_i < kh_end; ker_i++){
            for(int ker_j = kw_begin; ker_j < kw_end; ker_j++){

                int In_i = ker_i + row * str_h - pad_h;
                int In_j = ker_j + col * str_w - pad_w;
                int global_in_index = ((batch * In_h + In_i) * In_w + In_j) * channels + ch;

                if(In[global_in_index] > max_value){
                    max_value = In[global_in_index];
                    max_index = global_in_index;
                }
                else{
                    continue;
                }

            }
        }

        Out[global_out_index] = max_value;
        Out_idx[global_out_index] = max_index;
        
    }
}


__global__ void average_pooling_forward(
    const float *In, int In_h, int In_w,
    int ker_h, int ker_w, int channels, int batch_size,
    float *Out, int Out_h, int Out_w,
    int pad_h, int pad_w, int str_h, int str_w
){
    
    int row = blockDim.x * blockIdx.x + threadIdx.x;
    int col = blockDim.y * blockIdx.y + threadIdx.y;
    int bz = blockIdx.z;
    int batch = bz / channels;
    int ch = bz % channels;

    int global_out_index = ((batch * Out_h + row) * Out_w + col) * channels + ch;


    if(row < Out_h && col < Out_w && bz < channels * batch_size){

        int input_origin_i = row * str_h - pad_h;
        int input_origin_j = col * str_w - pad_w;

        int kh_begin = max(0, -input_origin_i);
        int kh_end = min(ker_h, In_h - input_origin_i);
        
        int kw_begin = max(0, -input_origin_j);
        int kw_end = min(ker_w, In_w - input_origin_j);

        int valid_count = (kh_end - kh_begin) * (kw_end - kw_begin);
        float sum = 0;

        for(int ker_i = kh_begin; ker_i < kh_end; ker_i++){
            for(int ker_j = kw_begin; ker_j < kw_end; ker_j++){

                int In_i = ker_i + row * str_h - pad_h;
                int In_j = ker_j + col * str_w - pad_w;
                int global_in_index = ((batch * In_h + In_i) * In_w + In_j) * channels + ch;
                
                sum += In[global_in_index];
            }
        }

        Out[global_out_index] = (valid_count <= 0)? 0 : sum / valid_count;
    }
}

__global__ void L2_pooling_forward(
    const float *In, int In_h, int In_w,
    int ker_h, int ker_w, int channels, int batch_size,
    float *Out, int Out_h, int Out_w,
    int pad_h, int pad_w, int str_h, int str_w
){
    
    int row = blockDim.x * blockIdx.x + threadIdx.x;
    int col = blockDim.y * blockIdx.y + threadIdx.y;
    int bz = blockIdx.z;
    int batch = bz / channels;
    int ch = bz % channels;

    int global_out_index = ((batch * Out_h + row) * Out_w + col) * channels + ch;


    if(row < Out_h && col < Out_w && bz < channels * batch_size){

        int input_origin_i = row * str_h - pad_h;
        int input_origin_j = col * str_w - pad_w;

        int kh_begin = max(0, -input_origin_i);
        int kh_end = min(ker_h, In_h - input_origin_i);
        
        int kw_begin = max(0, -input_origin_j);
        int kw_end = min(ker_w, In_w - input_origin_j);

        float sum = 0;

        for(int ker_i = kh_begin; ker_i < kh_end; ker_i++){
            for(int ker_j = kw_begin; ker_j < kw_end; ker_j++){

                int In_i = ker_i + row * str_h - pad_h;
                int In_j = ker_j + col * str_w - pad_w;
                int global_in_index = ((batch * In_h + In_i) * In_w + In_j) * channels + ch;
                
                sum += In[global_in_index] * In[global_in_index];
            }
        }

        Out[global_out_index] = sqrtf(sum);
    }
}


__global__ void max_pooling_backward_no_overlap(
    const float *Next, const int *Win_idx, int Next_total_size,
    float *Prev, int Prev_total_size, int Batch_size
){

    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if(i < Next_total_size){

        if(Win_idx[i] < 0) return;

        Prev[Win_idx[i]] = Next[i];
    }
}

__global__ void max_pooling_backward_overlap(
    const float *Next, const int *Win_idx, int Next_total_size,
    float *Prev, int Prev_total_size, int Batch_size
){

    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if(i < Next_total_size){

        if(Win_idx[i] < 0) return;

        atomicAdd(&Prev[Win_idx[i]], Next[i]);
    }
}

__global__ void average_pooling_backward(
    const float *Next, int Next_h, int Next_w,
    int ker_h, int ker_w, int channels, int batch_size,
    float *Prev, int Prev_h, int Prev_w, 
    int pad_h, int pad_w, int str_h, int str_w
){

    int row = blockDim.x * blockIdx.x + threadIdx.x;
    int col = blockDim.y * blockIdx.y + threadIdx.y;
    int bz = blockIdx.z;
    int ch = bz % channels;
    int batch = bz / channels;

    if(row < Prev_h && col < Prev_w && bz < batch_size * channels){

        float sum = 0;
        for(int ker_i = 0; ker_i < ker_h; ker_i++){

            int Next_i_t = row + pad_h - ker_i;
            if(Next_i_t < 0 || Next_i_t % str_h != 0) continue;

            int Next_i = Next_i_t / str_h;
            if(Next_i >= Next_h) continue;


            for(int ker_j = 0; ker_j < ker_w; ker_j++){

                int Next_j_t = (col + pad_w - ker_j);
                if(Next_j_t < 0 || Next_j_t % str_w != 0) continue;

                int Next_j = Next_j_t / str_w;
                if(Next_j >= Next_w) continue;

                int input_origin_i = Next_i * str_h - pad_h;
                int input_origin_j = Next_j * str_w - pad_w;

                int kh_begin = max(0, -input_origin_i);
                int kh_end = min(ker_h, Prev_h - input_origin_i);

                int kw_begin = max(0, -input_origin_j);
                int kw_end = min(ker_w, Prev_w - input_origin_j);

                int valid_count = (kh_end - kh_begin) * (kw_end - kw_begin);
                if(valid_count <= 0) continue;

                sum += Next[((batch * Next_h + Next_i) * Next_w + Next_j) * channels + ch] / valid_count;

            }
        }
        Prev[((batch * Prev_h + row) * Prev_w + col) * channels + ch] = sum;
    }
}

__global__ void L2_pooling_backward(
    const float *Next_delta, const float *Next_output, int Next_h, int Next_w,
    int ker_h, int ker_w, int channels, int batch_size,
    const float *Prev_out, float *Prev_cost_der, int Prev_h, int Prev_w, 
    int pad_h, int pad_w, int str_h, int str_w
){

    int row = blockDim.x * blockIdx.x + threadIdx.x;
    int col = blockDim.y * blockIdx.y + threadIdx.y;
    int bz = blockIdx.z;
    int ch = bz % channels;
    int batch = bz / channels;

    if(row < Prev_h && col < Prev_w && bz < batch_size * channels){

        float sum = 0;
        int Prev_global_idx = ((batch * Prev_h + row) * Prev_w + col) * channels + ch;
        for(int ker_i = 0; ker_i < ker_h; ker_i++){

            int Next_i_t = row + pad_h - ker_i;
            if(Next_i_t < 0 || Next_i_t % str_h != 0) continue;

            int Next_i = Next_i_t / str_h;
            if(Next_i >= Next_h) continue;


            for(int ker_j = 0; ker_j < ker_w; ker_j++){

                int Next_j_t = (col + pad_w - ker_j);
                if(Next_j_t < 0 || Next_j_t % str_w != 0) continue;

                int Next_j = Next_j_t / str_w;
                if(Next_j >= Next_w) continue;

                int Next_global_idx = ((batch * Next_h + Next_i) * Next_w + Next_j) * channels + ch;
                
                if(Next_output[Next_global_idx] > 0){
                    sum += Next_delta[Next_global_idx] / Next_output[Next_global_idx];
                }
            }
        }
        Prev_cost_der[Prev_global_idx] = sum * Prev_out[Prev_global_idx];
    }
}
