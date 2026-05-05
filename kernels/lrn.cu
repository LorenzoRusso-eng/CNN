#include "kernels/lrn.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <math_constants.h>

__global__ void lrn_forward(
    const float *In, float *Out,
    int window_size, int total_size, int channels,
    float alpha, float beta, float k
){

    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if(i < total_size){

        int ch = i % channels;

        int window_half_size = window_size / 2;
        float squared_sum = 0;

        int origin = ch - window_half_size;
        int ending = ch + window_half_size;

        int start = (origin < 0)? -(window_half_size + origin) : -window_half_size;
        int end = (ending >= channels)? window_half_size + channels - ending -1 : window_half_size;

        for(int j = start; j <= end; j++){
            squared_sum += In[i +j] * In[i + j];
        }

        float scale = k + alpha * squared_sum;
        Out[i] = In[i] / powf(scale, beta);
    }
}

__global__ void lrn_backward(
    const float *Next_delta,
    int window_size, int total_size, int channels,
    float *Prev_cost_der, const float *Prev_out,
    float alpha, float beta, float k
){

    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if(i<total_size){

        int ch = i % channels;

        int window_half_size = window_size / 2;
        float sum = 0;

        int origin = ch - window_half_size;
        int ending = ch + window_half_size;

        int start = (origin < 0)? -(window_half_size + origin) : -window_half_size;
        int end = (ending >= channels)? window_half_size + channels - ending -1 : window_half_size;

        for(int j = start; j <= end; j++){

            int Next_i = i + j;
            int Next_ch = Next_i % channels;

            int Next_origin = Next_ch - window_half_size;
            int Next_ending = Next_ch + window_half_size;

            int Next_start = (Next_origin < 0)? -(window_half_size + Next_origin) : -window_half_size;
            int Next_end = (Next_ending >= channels)? window_half_size + channels - Next_ending -1 : window_half_size;

            float squared_sum = 0;
            for(int k = Next_start; k <= Next_end; k++){
                squared_sum += Prev_out[Next_i + k] * Prev_out[Next_i + k]; 
            }

            float scale = k + alpha * squared_sum;

            float local_coeff = - beta * Prev_out[Next_i] * 2 * alpha * Prev_out[i] * powf(scale, -beta - 1);

            if(j == 0){
                local_coeff += powf(scale, -beta);
            }
            
            sum += Next_delta[Next_i] * local_coeff;

        }
        Prev_cost_der[i] = -sum;
    }
}
