#pragma once
// Questo file contiene le funzioni di attivazione senza dispatch virtuale.

#include <cmath>
#include "core/core_definitions.hpp"

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

    float fn(float a) const {
        switch(kind){
            case ActivationKind::Identity:
                return a;
            case ActivationKind::Sigmoid:
                return stable_sigmoid(a);
            case ActivationKind::Tanh:
                return std::tanh(a);
            case ActivationKind::ReLU:
                return (a > 0.0f) ? a : 0.0f;
            case ActivationKind::LeakyReLU:
                return (a > 0.0f) ? a : 0.01f * a;
            case ActivationKind::ELU:
                return (a > 0.0f) ? a : alpha * (std::exp(a) - 1.0f);
            case ActivationKind::Softplus:
                return stable_softplus(a);
            case ActivationKind::Swish:
                return a * stable_sigmoid(beta * a);
            case ActivationKind::Mish:
                return a * std::tanh(stable_softplus(a));
            case ActivationKind::GELU:
                return 0.5f * a * (1.0f + std::erf(a / std::sqrt(2.0f)));
        }
        return a;
    }

    float der(float a) const {
        switch(kind){
            case ActivationKind::Identity:
                (void)a;
                return 1.0f;
            case ActivationKind::Sigmoid: {
                const float s = stable_sigmoid(a);
                return s * (1.0f - s);
            }
            case ActivationKind::Tanh: {
                const float t = std::tanh(a);
                return 1.0f - t * t;
            }
            case ActivationKind::ReLU:
                return (a > 0.0f) ? 1.0f : 0.0f;
            case ActivationKind::LeakyReLU:
                return (a > 0.0f) ? 1.0f : 0.01f;
            case ActivationKind::ELU:
                return (a > 0.0f) ? 1.0f : alpha * std::exp(a);
            case ActivationKind::Softplus:
                return stable_sigmoid(a);
            case ActivationKind::Swish: {
                const float sigmoid_value = stable_sigmoid(beta * a);
                const float swish_value = a * sigmoid_value;
                return beta * swish_value + sigmoid_value * (1.0f - beta * swish_value);
            }
            case ActivationKind::Mish: {
                const float softplus_value = stable_softplus(a);
                const float tanh_softplus = std::tanh(softplus_value);
                const float sigmoid_value = stable_sigmoid(a);
                const float sech2_softplus = 1.0f - tanh_softplus * tanh_softplus;
                return tanh_softplus + a * sigmoid_value * sech2_softplus;
            }
            case ActivationKind::GELU:
                return 0.5f * (1.0f + std::erf(a / std::sqrt(2.0f)))
                     + (a / std::sqrt(2.0f * PI_F)) * std::exp(-0.5f * a * a);
        }
        return 1.0f;
    }

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
