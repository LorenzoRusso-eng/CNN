#pragma once

// Questo file contiene le definizioni base condivise del progetto.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

// Costanti matematiche
constexpr long double Pi = 3.14159265358979323846264338327950288L;
constexpr float PI_F = static_cast<float>(Pi);

// Strutture tensoriali flat
struct Tensor{
    int height = 0;
    int width = 0;
    int channels = 0;
    int flat_size = 0;
    std::vector<float> data;

    Tensor() = default;

    Tensor(int height_in, int width_in, int channels_in, float value = 0.0f)
        : height(height_in),
          width(width_in),
          channels(channels_in),
          flat_size(height_in * width_in * channels_in),
          data(
              static_cast<std::size_t>(height_in * width_in * channels_in),
              value
          ) {}

    void resize(int height_in, int width_in, int channels_in, float value = 0.0f){
        height = height_in;
        width = width_in;
        channels = channels_in;
        flat_size = height_in * width_in * channels_in;
        data.assign(static_cast<std::size_t>(flat_size), value);
    }

    void clear(){
        height = width = channels = 0;
        flat_size = 0;
        data.clear();
    }

    int index(int i, int j, int k) const{
        return (i * width + j) * channels + k;
    }

};

struct DatasetSample{
    std::string image_path;
    int class_index = 0;
};

struct LazyDataset{
    int input_shape[3] = {0, 0, 0};
    int num_classes = 0;
    std::vector<std::string> class_names;
    std::vector<DatasetSample> samples;

    int size() const{
        return static_cast<int>(samples.size());
    }
};

struct BatchTensor{
    int batch_size = 0;
    int height = 0;
    int width = 0;
    int channels = 0;
    int flat_size = 0;
    std::vector<float> data;

    BatchTensor() = default;

    BatchTensor(int batch_size_in, int height_in, int width_in, int channels_in, float value = 0.0f)
        : batch_size(batch_size_in),
          height(height_in),
          width(width_in),
          channels(channels_in),
          flat_size(height_in * width_in * channels_in),
          data(
              static_cast<std::size_t>(batch_size_in) *
              static_cast<std::size_t>(height_in * width_in * channels_in),
              value
          ) {}

    void resize(int batch_size_in, int height_in, int width_in, int channels_in, float value = 0.0f){
        batch_size = batch_size_in;
        height = height_in;
        width = width_in;
        channels = channels_in;
        flat_size = height_in * width_in * channels_in;
        data.assign(static_cast<std::size_t>(batch_size_in) * static_cast<std::size_t>(flat_size), value);
    }

    void clear(){
        batch_size = 0;
        height = 0;
        width = 0;
        channels = 0;
        flat_size = 0;
        data.clear();
    }

    std::size_t index(int batch_index, int flat_index) const{
        return static_cast<std::size_t>(batch_index) *
               static_cast<std::size_t>(flat_size) +
               static_cast<std::size_t>(flat_index);
    }

};

// Buffer usati dal training e dall'ottimizzazione
struct ParameterBuffer{
    std::vector<std::vector<float>> dense_weights;
    std::vector<std::vector<float>> dense_biases;
    std::vector<std::vector<float>> conv_weights;
    std::vector<std::vector<float>> conv_biases;
};

// Tipi applicativi per metriche e report
struct ClassPerformance{
    int tp = 0;
    int fp = 0;
    int tn = 0;
    int fn = 0;
    float precision = 0.0f;
    float recall = 0.0f;
    float f1 = 0.0f;
};

struct TestPerformance{
    int num_classes = 0;
    int test_count = 0;
    int correct = 0;
    float accuracy = 0.0f;
    std::vector<std::vector<int>> confusion;
    std::vector<ClassPerformance> per_class;
    float macro_precision = 0.0f;
    float macro_recall = 0.0f;
    float macro_f1 = 0.0f;
};

struct TrainingSummary{
    int epochs_completed = 0;
    float final_loss = 0.0f;
    std::vector<double> epoch_times_seconds;
    double total_training_seconds = 0.0;
    bool stopped_early = false;
    bool stopped_by_loss = false;
    bool stopped_by_validation = false;
    bool used_validation = false;
    float best_validation_accuracy = 0.0f;
    int best_epoch = 0;
    int epochs_without_significant_improvement = 0;
};

struct TrainingRuntimeState{
    int completed_epochs = 0;
    std::int64_t optimizer_steps = 0;
    ParameterBuffer velocity;
    bool validation_observed = false;
    float best_validation_accuracy = 0.0f;
    int best_validation_epoch = 0;
    int epochs_without_significant_improvement = 0;
};

struct EarlyStoppingConfig{
    bool enabled = false;
    int validation_batch_size = 200;
    float relative_delta_threshold = 0.001f;
    int patience = 5;
};
