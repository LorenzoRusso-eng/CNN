#pragma once

#include "math/activations.hpp"
#include <algorithm>
#include <cmath>
#include <cstddef>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <math_constants.h>


__device__ float cuda_stable_sigmoid(float x){
    if(x >= 0.0f){
        float z = expf(-x);
        return 1.0f / (1.0f + z);
    } else {
        float z = expf(x);
        return z / (1.0f + z);
    }
}

__device__ float cuda_stable_softplus(float x){
    if(x > 0.0f){
        return x + log1pf(expf(-x));
    }
    return log1pf(expf(x));
}

__device__ float cuda_apply_activation(float x, ActivationKind kind, float alpha = 0.0f, float beta = 0.0f){
    switch(kind){
        case ActivationKind::Identity:
            return x;
        case ActivationKind::Sigmoid:
            return cuda_stable_sigmoid(x);
        case ActivationKind::Tanh:
            return tanhf(x);
        case ActivationKind::ReLU:
            return x > 0.0f ? x : 0.0f;
        case ActivationKind::LeakyReLU:
            return x > 0.0f ? x : 0.01f * x;
        case ActivationKind::ELU:
            return x > 0.0f ? x : alpha * (expf(x) - 1.0f);
        case ActivationKind::Softplus:
            return cuda_stable_softplus(x);
        case ActivationKind::Swish:
            return x * cuda_stable_sigmoid(beta * x);
        case ActivationKind::Mish:
            return x * tanhf(cuda_stable_softplus(x));
        case ActivationKind::GELU:
            return 0.5f * x * (1.0f + erff(x / sqrtf(2.0f)));
        default:
            return x;
    }
}

__device__ float cuda_apply_activation_derivative(float x, ActivationKind kind, float alpha = 0.0f, float beta = 0.0f){
    switch(kind){
        case ActivationKind::Identity:
            return 1.0f;
        case ActivationKind::Sigmoid: {
            const float sigmoid_value = cuda_stable_sigmoid(x);
            return sigmoid_value * (1.0f - sigmoid_value);
        }
        case ActivationKind::Tanh: {
            const float tanh_value = tanhf(x);
            return 1.0f - tanh_value * tanh_value;
        }
        case ActivationKind::ReLU:
            return x > 0.0f ? 1.0f : 0.0f;
        case ActivationKind::LeakyReLU:
            return x > 0.0f ? 1.0f : 0.01f;
        case ActivationKind::ELU:
            return x > 0.0f ? 1.0f : alpha * expf(x);
        case ActivationKind::Softplus:
            return cuda_stable_sigmoid(x);
        case ActivationKind::Swish: {
            const float sigmoid_value = cuda_stable_sigmoid(beta * x);
            const float swish_value = x * sigmoid_value;
            return beta * swish_value + sigmoid_value * (1.0f - beta * swish_value);
        }
        case ActivationKind::Mish: {
            const float softplus_value = cuda_stable_softplus(x);
            const float tanh_softplus = tanhf(softplus_value);
            const float sigmoid_value = cuda_stable_sigmoid(x);
            const float sech2_softplus = 1.0f - tanh_softplus * tanh_softplus;
            return tanh_softplus + x * sigmoid_value * sech2_softplus;
        }
        case ActivationKind::GELU:
            return 0.5f * (1.0f + erff(x / sqrtf(2.0f)))
                 + (x / sqrtf(2.0f * PI_F)) * expf(-0.5f * x * x);
        default:
            return 1.0f;
    }
}
