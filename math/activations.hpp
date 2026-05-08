#pragma once

#include "core/core_definitions.hpp"

#include <cmath>

#ifdef __CUDACC__
#include <cuda_runtime.h>
#include <math_constants.h>
#endif

inline float stable_sigmoid(float a) {
    if(a >= 0.0f) {
        const float z = std::exp(-a);
        return 1.0f / (1.0f + z);
    }

    const float z = std::exp(a);
    return z / (1.0f + z);
}

inline float stable_softplus(float a) {
    if(a > 0.0f) {
        return a + std::log1p(std::exp(-a));
    }

    return std::log1p(std::exp(a));
}

enum class ActivationKind {
    Identity,
    Sigmoid,
    Tanh,
    ReLU,
    LeakyReLU,
    ELU,
    Softplus,
    Swish,
    Mish,
    GELU
};

struct Activation {
    ActivationKind kind = ActivationKind::Identity;
    float alpha = 1.0f;
    float beta = 1.0f;

    bool outputs_in_unit_interval() const {
        return kind == ActivationKind::Sigmoid;
    }
};

inline const char *activation_name(const Activation &activation){
    switch(activation.kind){
        case ActivationKind::Identity:
            return "Identity";
        case ActivationKind::Sigmoid:
            return "Sigmoid";
        case ActivationKind::Tanh:
            return "Tanh";
        case ActivationKind::ReLU:
            return "ReLU";
        case ActivationKind::LeakyReLU:
            return "LeakyReLU";
        case ActivationKind::ELU:
            return "ELU";
        case ActivationKind::Softplus:
            return "Softplus";
        case ActivationKind::Swish:
            return "Swish";
        case ActivationKind::Mish:
            return "Mish";
        case ActivationKind::GELU:
            return "GELU";
    }
    return "Identity";
}

#ifdef __CUDACC__
__device__ inline float cuda_stable_sigmoid(float x){
    if(x >= 0.0f){
        float z = expf(-x);
        return 1.0f / (1.0f + z);
    } else {
        float z = expf(x);
        return z / (1.0f + z);
    }
}

__device__ inline float cuda_stable_softplus(float x){
    if(x > 0.0f){
        return x + log1pf(expf(-x));
    }
    return log1pf(expf(x));
}

__device__ inline float cuda_apply_activation(float x, ActivationKind kind, float alpha = 0.0f, float beta = 0.0f){
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

__device__ inline float cuda_apply_activation_derivative(float x, ActivationKind kind, float alpha = 0.0f, float beta = 0.0f){
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
#endif

extern Activation identity;
extern Activation sigmoid;
extern Activation tanh_activation;
extern Activation relu;
extern Activation leaky_relu;
extern Activation elu;
extern Activation softplus;
extern Activation swish;
extern Activation mish;
extern Activation gelu;
