#pragma once

#include "math/activations.hpp"
#include "shared/openmp_utils.hpp"

#include <cmath>
#include <cstddef>

namespace activation_kernels {

struct IdentityForwardOp {
    float operator()(float value) const { return value; }
};

struct SigmoidForwardOp {
    float operator()(float value) const { return stable_sigmoid(value); }
};

struct TanhForwardOp {
    float operator()(float value) const { return std::tanh(value); }
};

struct ReLUForwardOp {
    float operator()(float value) const { return (value > 0.0f) ? value : 0.0f; }
};

struct LeakyReLUForwardOp {
    float operator()(float value) const { return (value > 0.0f) ? value : 0.01f * value; }
};

struct ELUForwardOp {
    float alpha = 1.0f;

    float operator()(float value) const {
        return (value > 0.0f) ? value : alpha * (std::exp(value) - 1.0f);
    }
};

struct SoftplusForwardOp {
    float operator()(float value) const { return stable_softplus(value); }
};

struct SwishForwardOp {
    float beta = 1.0f;

    float operator()(float value) const {
        return value * stable_sigmoid(beta * value);
    }
};

struct MishForwardOp {
    float operator()(float value) const {
        return value * std::tanh(stable_softplus(value));
    }
};

struct GELUForwardOp {
    float operator()(float value) const {
        return 0.5f * value * (1.0f + std::erf(value / std::sqrt(2.0f)));
    }
};

struct IdentityDerivativeOp {
    float operator()(float) const { return 1.0f; }
};

struct SigmoidDerivativeOp {
    float operator()(float value) const {
        const float sigmoid_value = stable_sigmoid(value);
        return sigmoid_value * (1.0f - sigmoid_value);
    }
};

struct TanhDerivativeOp {
    float operator()(float value) const {
        const float tanh_value = std::tanh(value);
        return 1.0f - tanh_value * tanh_value;
    }
};

struct ReLUDerivativeOp {
    float operator()(float value) const { return (value > 0.0f) ? 1.0f : 0.0f; }
};

struct LeakyReLUDerivativeOp {
    float operator()(float value) const { return (value > 0.0f) ? 1.0f : 0.01f; }
};

struct ELUDerivativeOp {
    float alpha = 1.0f;

    float operator()(float value) const {
        return (value > 0.0f) ? 1.0f : alpha * std::exp(value);
    }
};

struct SoftplusDerivativeOp {
    float operator()(float value) const { return stable_sigmoid(value); }
};

struct SwishDerivativeOp {
    float beta = 1.0f;

    float operator()(float value) const {
        const float sigmoid_value = stable_sigmoid(beta * value);
        const float swish_value = value * sigmoid_value;
        return beta * swish_value + sigmoid_value * (1.0f - beta * swish_value);
    }
};

struct MishDerivativeOp {
    float operator()(float value) const {
        const float softplus_value = stable_softplus(value);
        const float tanh_softplus = std::tanh(softplus_value);
        const float sigmoid_value = stable_sigmoid(value);
        const float sech2_softplus = 1.0f - tanh_softplus * tanh_softplus;
        return tanh_softplus + value * sigmoid_value * sech2_softplus;
    }
};

struct GELUDerivativeOp {
    float operator()(float value) const {
        return 0.5f * (1.0f + std::erf(value / std::sqrt(2.0f)))
             + (value / std::sqrt(2.0f * PI_F)) * std::exp(-0.5f * value * value);
    }
};

template <class Fn>
inline void dispatch_forward_op(const Activation &act, Fn &&fn){
    switch(act.kind){
        case ActivationKind::Identity:
            fn(IdentityForwardOp{});
            return;
        case ActivationKind::Sigmoid:
            fn(SigmoidForwardOp{});
            return;
        case ActivationKind::Tanh:
            fn(TanhForwardOp{});
            return;
        case ActivationKind::ReLU:
            fn(ReLUForwardOp{});
            return;
        case ActivationKind::LeakyReLU:
            fn(LeakyReLUForwardOp{});
            return;
        case ActivationKind::ELU:
            fn(ELUForwardOp{act.alpha});
            return;
        case ActivationKind::Softplus:
            fn(SoftplusForwardOp{});
            return;
        case ActivationKind::Swish:
            fn(SwishForwardOp{act.beta});
            return;
        case ActivationKind::Mish:
            fn(MishForwardOp{});
            return;
        case ActivationKind::GELU:
            fn(GELUForwardOp{});
            return;
    }
}

template <class Fn>
inline void dispatch_derivative_op(const Activation &act, Fn &&fn){
    switch(act.kind){
        case ActivationKind::Identity:
            fn(IdentityDerivativeOp{});
            return;
        case ActivationKind::Sigmoid:
            fn(SigmoidDerivativeOp{});
            return;
        case ActivationKind::Tanh:
            fn(TanhDerivativeOp{});
            return;
        case ActivationKind::ReLU:
            fn(ReLUDerivativeOp{});
            return;
        case ActivationKind::LeakyReLU:
            fn(LeakyReLUDerivativeOp{});
            return;
        case ActivationKind::ELU:
            fn(ELUDerivativeOp{act.alpha});
            return;
        case ActivationKind::Softplus:
            fn(SoftplusDerivativeOp{});
            return;
        case ActivationKind::Swish:
            fn(SwishDerivativeOp{act.beta});
            return;
        case ActivationKind::Mish:
            fn(MishDerivativeOp{});
            return;
        case ActivationKind::GELU:
            fn(GELUDerivativeOp{});
            return;
    }
}

template <class ForwardOp>
inline void apply_activation_buffer(const float *a, float *y, int n, const ForwardOp &forward_op){
    NN_OMP_SIMD
    for(int idx = 0; idx < n; idx++){
        y[idx] = forward_op(a[idx]);
    }
}

template <class ForwardOp>
inline void apply_bias_activation_buffer(float *a, float *y, const float *bias, int n, const ForwardOp &forward_op){
    NN_OMP_SIMD
    for(int idx = 0; idx < n; idx++){
        const float value = a[idx] + bias[idx];
        a[idx] = value;
        y[idx] = forward_op(value);
    }
}

template <class ForwardOp>
inline void apply_lookahead_bias_activation_buffer(float *a, float *y, const float *bias, const float *velocity_bias, float momentum, int n, const ForwardOp &forward_op){
    NN_OMP_SIMD
    for(int idx = 0; idx < n; idx++){
        const float value = a[idx] + bias[idx] + momentum * velocity_bias[idx];
        a[idx] = value;
        y[idx] = forward_op(value);
    }
}

template <class ForwardOp>
inline void apply_repeated_bias_activation_buffer(float *a, float *y, const float *bias, int outer_count, int inner_count, const ForwardOp &forward_op){
    NN_OMP_PARALLEL_FOR_IF(outer_count * inner_count > 4096)
    for(int outer_index = 0; outer_index < outer_count; outer_index++){
        const std::size_t base = static_cast<std::size_t>(outer_index) * static_cast<std::size_t>(inner_count);

        NN_OMP_SIMD
        for(int inner_index = 0; inner_index < inner_count; inner_index++){
            const std::size_t data_index = base + static_cast<std::size_t>(inner_index);
            const float value = a[data_index] + bias[inner_index];
            a[data_index] = value;
            y[data_index] = forward_op(value);
        }
    }
}

template <class ForwardOp>
inline void apply_repeated_lookahead_bias_activation_buffer(float *a, float *y, const float *bias, const float *velocity_bias, float momentum, int outer_count, int inner_count, const ForwardOp &forward_op){
    NN_OMP_PARALLEL_FOR_IF(outer_count * inner_count > 4096)
    for(int outer_index = 0; outer_index < outer_count; outer_index++){
        const std::size_t base = static_cast<std::size_t>(outer_index) * static_cast<std::size_t>(inner_count);

        NN_OMP_SIMD
        for(int inner_index = 0; inner_index < inner_count; inner_index++){
            const std::size_t data_index = base + static_cast<std::size_t>(inner_index);
            const float value = a[data_index] + bias[inner_index] + momentum * velocity_bias[inner_index];
            a[data_index] = value;
            y[data_index] = forward_op(value);
        }
    }
}

} // namespace activation_kernels
