#pragma once

#include "math/losses.hpp"
#include "core/cuda_backend.hpp"

#include <cmath>
#include <cstddef>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <math_constants.h>

#ifdef __CUDACC__
__device__ inline float cuda_clamp_loss_output(float output_value, float min_value, float max_value){
    return fminf(fmaxf(output_value, min_value), max_value);
}

__device__ inline float cuda_apply_loss_reduction(float value, int contribution_count, Reduction reduction_mode){
    if(reduction_mode == Reduction::Mean){
        const int safe_count = contribution_count > 1 ? contribution_count : 1;
        return value / static_cast<float>(safe_count);
    }
    return value;
}

__device__ inline int cuda_loss_reduction_count(LossKind kind, int output_size){
    switch(kind){
        case LossKind::LL:
            return 1;
        default:
            return output_size;
    }
}

__device__ inline float cuda_apply_cost(
    float output_value, float desired_output,
    LossKind kind, Reduction reduction_mode,
    int output_size = 1, float beta = 1.0f
){
    float value = 0.0f;

    switch(kind){
        case LossKind::Simple: {
            const float diff = output_value - desired_output;
            value = 0.5f * diff * diff;
            break;
        }
        case LossKind::L1:
            value = fabsf(output_value - desired_output);
            break;
        case LossKind::L2: {
            const float diff = output_value - desired_output;
            value = diff * diff;
            break;
        }
        case LossKind::SmoothL1: {
            const float diff = output_value - desired_output;
            const float abs_diff = fabsf(diff);
            value = (abs_diff < beta) ? 0.5f * diff * diff / beta : abs_diff - 0.5f * beta;
            break;
        }
        case LossKind::Huber: {
            const float diff = output_value - desired_output;
            const float abs_diff = fabsf(diff);
            value = (abs_diff < beta) ? 0.5f * diff * diff : beta * (abs_diff - 0.5f * beta);
            break;
        }
        case LossKind::CrossEntropy: {
            const float clamped_y = cuda_clamp_loss_output(output_value, 1e-7f, 1.0f - 1e-7f);
            value = -desired_output * logf(clamped_y) - (1.0f - desired_output) * logf(1.0f - clamped_y);
            break;
        }
        case LossKind::LL: {
            const float clamped_y = cuda_clamp_loss_output(output_value, 1e-7f, 1.0f);
            value = -desired_output * logf(clamped_y);
            break;
        }
    }

    return cuda_apply_loss_reduction(value, cuda_loss_reduction_count(kind, output_size), reduction_mode);
}

__device__ inline float cuda_apply_cost_derivative(
    float output_value, float desired_output,
    LossKind kind, Reduction reduction_mode,
    int output_size = 1, float beta = 1.0f
){
    float derivative = 0.0f;

    switch(kind){
        case LossKind::Simple:
            derivative = output_value - desired_output;
            break;
        case LossKind::L1:
            if(output_value > desired_output){
                derivative = 1.0f;
            } else if(output_value < desired_output){
                derivative = -1.0f;
            }
            break;
        case LossKind::L2:
            derivative = 2.0f * (output_value - desired_output);
            break;
        case LossKind::SmoothL1: {
            const float diff = output_value - desired_output;
            derivative = (fabsf(diff) < beta) ? diff / beta : (diff > 0.0f ? 1.0f : -1.0f);
            break;
        }
        case LossKind::Huber: {
            const float diff = output_value - desired_output;
            derivative = (fabsf(diff) < beta) ? diff : (diff > 0.0f ? beta : -beta);
            break;
        }
        case LossKind::CrossEntropy: {
            const float clamped_y = cuda_clamp_loss_output(output_value, 1e-7f, 1.0f - 1e-7f);
            derivative = -(desired_output / clamped_y) + (1.0f - desired_output) / (1.0f - clamped_y);
            break;
        }
        case LossKind::LL: {
            const float clamped_y = cuda_clamp_loss_output(output_value, 1e-7f, 1.0f);
            return -(desired_output / clamped_y);
        }
    }

    return cuda_apply_loss_reduction(derivative, output_size, reduction_mode);
}
#endif

float Loss_fun(
    cuda_backend::CudaBatchLayerRuntime &output_runtime,
    const Loss &loss,
    const cuda_backend::CudaBatchTensor<float> &desired_output
);
