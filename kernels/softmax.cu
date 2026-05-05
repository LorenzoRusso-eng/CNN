#include "kernels/softmax.hpp"

#include <cmath>
#include <cstddef>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <math_constants.h>

__global__ void softmax_forward(const float *In, float *Out, int batch_size, int flat_size){
    extern __shared__ float shared[];

    int batch = blockIdx.x;
    int thread = threadIdx.x;

    if(batch < batch_size){

        int base = batch * flat_size;

        float local_max = -CUDART_INF_F;
        for(int i = thread; i < flat_size; i += blockDim.x){
            local_max = fmaxf(local_max, In[i + base]);
        }

        shared[thread] = local_max;
        __syncthreads();

        for(int stride = blockDim.x / 2; stride > 0; stride >>= 1){
            if(thread < stride){
                shared[thread] = fmaxf(shared[thread], shared[thread + stride]);
            }
            __syncthreads();
        }

        float max_value = shared[0];
        float local_sum = 0.0f;

        for(int i = thread; i < flat_size; i += blockDim.x){
            float exp_value = expf(In[i + base] - max_value);
            Out[i + base] = exp_value;
            local_sum += exp_value;
        }

        shared[thread] = local_sum;
        __syncthreads();

        for(int stride = blockDim.x / 2; stride > 0; stride >>= 1){
            if(thread < stride){
                shared[thread] += shared[thread + stride];
            }
            __syncthreads();
        }

        float inv_sum = 1.0f / shared[0];
        for(int i = thread; i < flat_size; i += blockDim.x){
            Out[i + base] *= inv_sum;
        }
    }
}

__global__ void softmax_backward(const float *Output, const float *Desired, float *Delta, int total_size){
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if(i < total_size){
        Delta[i] = Output[i] - Desired[i];
    }
}
