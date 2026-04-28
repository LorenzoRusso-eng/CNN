#pragma once

// Questo file contiene le strategie di decay del learning rate.

#include <cmath>
#include <stdexcept>

#include "core/core_definitions.hpp"

enum class DecayKind {
    Constant,
    Exponential,
    TimeBased,
    Step,
    CosineAnnealing
};

class Decay {
    public:
        virtual ~Decay() = default;

        virtual float fn(int epoch) const = 0;
        virtual DecayKind kind() const = 0;

        virtual float decay_rate() const { return 0.0f; }
        virtual int step_size() const { return 0; }
        virtual float final_lr() const { return 0.0f; }
        virtual int max_epoch() const { return 0; }

        void set_initial_lr(float initial_lr){
            if(initial_lr <= 0.0f){
                throw std::invalid_argument("Decay: initial_lr deve essere maggiore di 0");
            }
            initial_lr_ = initial_lr;
        }

        float initial_lr() const {
            return initial_lr_;
        }

    protected:
        float initial_lr_ = 0.0f;
};

class Constant_decay final : public Decay {
    public:
        DecayKind kind() const override { return DecayKind::Constant; }

        float fn(int epoch) const override{
            (void)epoch;
            return initial_lr_;
        }
};

class Exponential_decay final : public Decay {
    public:
        DecayKind kind() const override { return DecayKind::Exponential; }

        void set_decay_rate(float decay_rate){
            if(decay_rate < 0.0f){
                throw std::invalid_argument("Exponential_decay: decay_rate non puo' essere negativo");
            }
            decay_rate_ = decay_rate;
        }

        float decay_rate() const override { return decay_rate_; }

        float fn(int epoch) const override{
            return initial_lr_ * std::exp(-decay_rate_ * static_cast<float>(epoch));
        }

    private:
        float decay_rate_ = 0.0f;
};

class Time_based_decay final : public Decay {
    public:
        DecayKind kind() const override { return DecayKind::TimeBased; }

        void set_decay_rate(float decay_rate){
            if(decay_rate < 0.0f){
                throw std::invalid_argument("Time_based_decay: decay_rate non puo' essere negativo");
            }
            decay_rate_ = decay_rate;
        }

        float decay_rate() const override { return decay_rate_; }

        float fn(int epoch) const override{
            return initial_lr_ / (1.0f + decay_rate_ * static_cast<float>(epoch));
        }

    private:
        float decay_rate_ = 0.0f;
};

class Step_decay final : public Decay {
    public:
        DecayKind kind() const override { return DecayKind::Step; }

        void set_decay_rate(float decay_rate){
            if(decay_rate <= 0.0f){
                throw std::invalid_argument("Step_decay: decay_rate deve essere maggiore di 0");
            }
            decay_rate_ = decay_rate;
        }

        float decay_rate() const override { return decay_rate_; }

        void set_step_size(int step_size){
            if(step_size <= 0){
                throw std::invalid_argument("Step_decay: step_size deve essere maggiore di 0");
            }
            step_size_ = step_size;
        }

        int step_size() const override { return step_size_; }

        float fn(int epoch) const override{
            return initial_lr_ * std::pow(decay_rate_, static_cast<float>(epoch / step_size_));
        }

    private:
        float decay_rate_ = 1.0f;
        int step_size_ = 1;
};

class Cosine_annealing final : public Decay {
    public:
        DecayKind kind() const override { return DecayKind::CosineAnnealing; }

        void set_final_lr(float final_lr){
            if(final_lr < 0.0f){
                throw std::invalid_argument("Cosine_annealing: final_lr non puo' essere negativo");
            }
            final_lr_ = final_lr;
        }

        float final_lr() const override { return final_lr_; }

        void set_max_epoch(int max_epoch){
            if(max_epoch <= 0){
                throw std::invalid_argument("Cosine_annealing: max_epoch deve essere maggiore di 0");
            }
            max_epoch_ = max_epoch;
        }

        int max_epoch() const override { return max_epoch_; }

        float fn(int epoch) const override{
            if(epoch >= max_epoch_){
                return final_lr_;
            }
            return final_lr_ + 0.5f * (initial_lr_ - final_lr_) * (1.0f + std::cos(PI_F * static_cast<float>(epoch) / static_cast<float>(max_epoch_)));
        }

    private:
        float final_lr_ = 0.0f;
        int max_epoch_ = 1;
};
