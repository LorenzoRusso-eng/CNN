#pragma once

// Questo file contiene le funzioni di loss senza dispatch virtuale.

#include "core/layer.hpp"
#include "core/shape_utils.hpp"

#include <algorithm>
#include <cmath>

enum class Reduction {
    Sum,
    Mean
};

enum class LossKind {
    Simple,
    L1,
    L2,
    SmoothL1,
    Huber,
    CrossEntropy,
    LL
};

struct Loss;

namespace loss_kernels {

inline float apply_loss_reduction(float total_loss, int contribution_count, Reduction reduction_mode){
    if(reduction_mode == Reduction::Mean){
        return total_loss / static_cast<float>(std::max(1, contribution_count));
    }
    return total_loss;
}

inline float apply_derivative_reduction(float derivative, int contribution_count, Reduction reduction_mode){
    if(reduction_mode == Reduction::Mean){
        return derivative / static_cast<float>(std::max(1, contribution_count));
    }
    return derivative;
}

struct SimpleLossValueOp {
    float operator()(float output_value, float desired_output) const {
        const float diff = output_value - desired_output;
        return 0.5f * diff * diff;
    }

    int reduction_count(int output_size) const { return output_size; }
};

struct L1LossValueOp {
    float operator()(float output_value, float desired_output) const {
        return std::abs(output_value - desired_output);
    }

    int reduction_count(int output_size) const { return output_size; }
};

struct L2LossValueOp {
    float operator()(float output_value, float desired_output) const {
        const float diff = output_value - desired_output;
        return diff * diff;
    }

    int reduction_count(int output_size) const { return output_size; }
};

struct SmoothL1LossValueOp {
    float beta = 1.0f;

    float operator()(float output_value, float desired_output) const {
        const float diff = output_value - desired_output;
        if(std::abs(diff) < beta){
            return 0.5f * diff * diff / beta;
        }
        return std::abs(diff) - 0.5f * beta;
    }

    int reduction_count(int output_size) const { return output_size; }
};

struct HuberLossValueOp {
    float beta = 1.0f;

    float operator()(float output_value, float desired_output) const {
        const float diff = output_value - desired_output;
        if(std::abs(diff) < beta){
            return 0.5f * diff * diff;
        }
        return beta * (std::abs(diff) - 0.5f * beta);
    }

    int reduction_count(int output_size) const { return output_size; }
};

struct CrossEntropyLossValueOp {
    float operator()(float output_value, float desired_output) const {
        constexpr float loss_epsilon = 1e-7f;
        const float clamped_y = std::clamp(output_value, loss_epsilon, 1.0f - loss_epsilon);
        return -desired_output * std::log(clamped_y) - (1.0f - desired_output) * std::log(1.0f - clamped_y);
    }

    int reduction_count(int output_size) const { return output_size; }
};

struct LLLossValueOp {
    float operator()(float output_value, float desired_output) const {
        constexpr float loss_epsilon = 1e-7f;
        const float clamped_y = std::clamp(output_value, loss_epsilon, 1.0f);
        return -desired_output * std::log(clamped_y);
    }

    int reduction_count(int) const { return 1; }
};

struct SimpleLossDerivativeOp {
    int output_size = 1;
    Reduction reduction_mode = Reduction::Sum;

    float operator()(float output_value, float desired_output) const {
        return apply_derivative_reduction(output_value - desired_output, output_size, reduction_mode);
    }
};

struct L1LossDerivativeOp {
    int output_size = 1;
    Reduction reduction_mode = Reduction::Sum;

    float operator()(float output_value, float desired_output) const {
        float derivative = 0.0f;
        if(output_value > desired_output){
            derivative = 1.0f;
        } else if(output_value < desired_output){
            derivative = -1.0f;
        }
        return apply_derivative_reduction(derivative, output_size, reduction_mode);
    }
};

struct L2LossDerivativeOp {
    int output_size = 1;
    Reduction reduction_mode = Reduction::Sum;

    float operator()(float output_value, float desired_output) const {
        return apply_derivative_reduction(2.0f * (output_value - desired_output), output_size, reduction_mode);
    }
};

struct SmoothL1LossDerivativeOp {
    float beta = 1.0f;
    int output_size = 1;
    Reduction reduction_mode = Reduction::Sum;

    float operator()(float output_value, float desired_output) const {
        const float diff = output_value - desired_output;
        if(std::abs(diff) < beta){
            return apply_derivative_reduction(diff / beta, output_size, reduction_mode);
        }
        return apply_derivative_reduction((diff > 0.0f) ? 1.0f : -1.0f, output_size, reduction_mode);
    }
};

struct HuberLossDerivativeOp {
    float beta = 1.0f;
    int output_size = 1;
    Reduction reduction_mode = Reduction::Sum;

    float operator()(float output_value, float desired_output) const {
        const float diff = output_value - desired_output;
        if(std::abs(diff) < beta){
            return apply_derivative_reduction(diff, output_size, reduction_mode);
        }
        return apply_derivative_reduction((diff > 0.0f) ? beta : -beta, output_size, reduction_mode);
    }
};

struct CrossEntropyLossDerivativeOp {
    int output_size = 1;
    Reduction reduction_mode = Reduction::Sum;

    float operator()(float output_value, float desired_output) const {
        constexpr float loss_epsilon = 1e-7f;
        const float clamped_y = std::clamp(output_value, loss_epsilon, 1.0f - loss_epsilon);
        return apply_derivative_reduction(
            -(desired_output / clamped_y) + (1.0f - desired_output) / (1.0f - clamped_y),
            output_size,
            reduction_mode
        );
    }
};

struct LLLossDerivativeOp {
    float operator()(float output_value, float desired_output) const {
        constexpr float loss_epsilon = 1e-7f;
        const float clamped_y = std::clamp(output_value, loss_epsilon, 1.0f);
        return -(desired_output / clamped_y);
    }
};

template <class LossValueOp>
inline float reduce_loss_buffer(const float *output, const float *desired_output, int output_size, const LossValueOp &loss_value_op, Reduction reduction_mode){
    float total_loss = 0.0f;

    for(int idx = 0; idx < output_size; idx++){
        const std::size_t data_index = static_cast<std::size_t>(idx);
        total_loss += loss_value_op(output[data_index], desired_output[data_index]);
    }

    return apply_loss_reduction(total_loss, loss_value_op.reduction_count(output_size), reduction_mode);
}

template <class Fn>
inline void dispatch_loss_value_op(const Loss &loss, Fn &&fn);

template <class Fn>
inline void dispatch_derivative_op(const Loss &loss, int output_size, Fn &&fn);

inline float compute_loss_values(const Loss &loss, const float *output, const float *desired_output, int output_size);
inline float compute_loss_batch(const Loss &loss, const BatchLayerRuntime &runtime, const BatchTensor &desired_output);
inline float compute_derivative(const Loss &loss, float output_value, float desired_output, int output_size = 1);

} // namespace loss_kernels

struct Loss {
    LossKind kind = LossKind::Simple;
    float beta = 1.0f;

    explicit Loss(LossKind kind_in = LossKind::Simple, Reduction reduction_mode = Reduction::Sum)
        : kind(kind_in), reduction_mode_(reduction_mode) {}

    float fn(const Layer &l, const LayerRuntime &runtime, const Tensor &desired_output) const{
        const auto &output = runtime_output_buffer(l, runtime);
        return loss_kernels::compute_loss_values(*this, output.data(), desired_output.data.data(), l.flat_output_size());
    }

    float fn_batch(const BatchLayerRuntime &runtime, const BatchTensor &desired_output) const{
        return loss_kernels::compute_loss_batch(*this, runtime, desired_output);
    }

    float der(float output_value, float desired_output, int output_size = 1) const{
        return loss_kernels::compute_derivative(*this, output_value, desired_output, output_size);
    }

    bool requires_unit_interval_output() const {
        return kind == LossKind::CrossEntropy;
    }

    bool supports_softmax_output() const {
        return kind == LossKind::LL;
    }

    void set_reduction(Reduction reduction_mode) {
        reduction_mode_ = reduction_mode;
    }

    Reduction reduction() const {
        return reduction_mode_;
    }

private:
    Reduction reduction_mode_ = Reduction::Sum;
};

namespace loss_kernels {

template <class Fn>
inline void dispatch_loss_value_op(const Loss &loss, Fn &&fn){
    switch(loss.kind){
        case LossKind::Simple:
            fn(SimpleLossValueOp{});
            return;
        case LossKind::L1:
            fn(L1LossValueOp{});
            return;
        case LossKind::L2:
            fn(L2LossValueOp{});
            return;
        case LossKind::SmoothL1:
            fn(SmoothL1LossValueOp{loss.beta});
            return;
        case LossKind::Huber:
            fn(HuberLossValueOp{loss.beta});
            return;
        case LossKind::CrossEntropy:
            fn(CrossEntropyLossValueOp{});
            return;
        case LossKind::LL:
            fn(LLLossValueOp{});
            return;
    }
}

template <class Fn>
inline void dispatch_derivative_op(const Loss &loss, int output_size, Fn &&fn){
    const Reduction reduction_mode = loss.reduction();

    switch(loss.kind){
        case LossKind::Simple:
            fn(SimpleLossDerivativeOp{output_size, reduction_mode});
            return;
        case LossKind::L1:
            fn(L1LossDerivativeOp{output_size, reduction_mode});
            return;
        case LossKind::L2:
            fn(L2LossDerivativeOp{output_size, reduction_mode});
            return;
        case LossKind::SmoothL1:
            fn(SmoothL1LossDerivativeOp{loss.beta, output_size, reduction_mode});
            return;
        case LossKind::Huber:
            fn(HuberLossDerivativeOp{loss.beta, output_size, reduction_mode});
            return;
        case LossKind::CrossEntropy:
            fn(CrossEntropyLossDerivativeOp{output_size, reduction_mode});
            return;
        case LossKind::LL:
            fn(LLLossDerivativeOp{});
            return;
    }
}

inline float compute_loss_values(const Loss &loss, const float *output, const float *desired_output, int output_size){
    float total_loss = 0.0f;

    dispatch_loss_value_op(loss, [&](const auto &loss_value_op){
        total_loss = reduce_loss_buffer(output, desired_output, output_size, loss_value_op, loss.reduction());
    });

    return total_loss;
}

inline float compute_loss_batch(const Loss &loss, const BatchLayerRuntime &runtime, const BatchTensor &desired_output){
    require_condition(
        runtime.y.batch_size == desired_output.batch_size &&
        runtime.y.height == desired_output.height &&
        runtime.y.width == desired_output.width &&
        runtime.y.channels == desired_output.channels,
        "compute_loss_batch: shape del batch incompatibile"
    );

    float total_loss = 0.0f;

    dispatch_loss_value_op(loss, [&](const auto &loss_value_op){
        for(int sample_index = 0; sample_index < runtime.y.batch_size; sample_index++){
            const std::size_t output_base = runtime.y.index(sample_index, 0);
            const std::size_t desired_base = desired_output.index(sample_index, 0);
            total_loss += reduce_loss_buffer(
                runtime.y.data.data() + output_base,
                desired_output.data.data() + desired_base,
                runtime.y.flat_size,
                loss_value_op,
                loss.reduction()
            );
        }
    });

    return total_loss;
}

inline float compute_derivative(const Loss &loss, float output_value, float desired_output, int output_size){
    float derivative = 0.0f;

    dispatch_derivative_op(loss, output_size, [&](const auto &derivative_op){
        derivative = derivative_op(output_value, desired_output);
    });

    return derivative;
}

} // namespace loss_kernels

extern Loss simple_loss;
extern Loss l1_loss;
extern Loss l2_loss;
extern Loss smooth_l1_loss;
extern Loss huber_loss;
extern Loss cross_entropy;
extern Loss ll_loss;
