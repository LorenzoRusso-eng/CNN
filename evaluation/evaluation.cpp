#include "evaluation/evaluation.hpp"

// Questo file contiene l'inferenza di valutazione e il calcolo delle metriche.

#include "core/layer_validation.hpp"
#include "core/cuda_backend.hpp"
#include "engine/forward.hpp"
#include "evaluation/evaluation_metrics.hpp"

#include <algorithm>
#include <iostream>
#include <utility>
#include <vector>

namespace {
constexpr int kEvaluationBatchSize = 100;
} // namespace


TestPerformance run_test(LayerList &architecture, const cuda_backend::CudaParameterBuffer &parameters, const std::vector<int> &test_indices, const LazyDataset &dataset, const Activation &hidden_activation, const Activation &output_activation)
{
    TestPerformance perf;
    validate_architecture(architecture, "run_test");
    validate_dataset_indices_io_shapes(architecture, dataset, test_indices, "run_test");
    const int num_layers = static_cast<int>(architecture.size());
    const int test_count = static_cast<int>(test_indices.size());
    if (test_indices.empty())
    {
        std::cout << "Test saltato: nessun esempio selezionato." << std::endl;
        return perf;
    }

    const int num_classes = architecture[num_layers - 1].flat_output_size();
    std::vector<std::vector<int>> confusion(num_classes, std::vector<int>(num_classes, 0));
    int correct = 0;

    std::cout << "Inizio test..." << std::endl;
    cuda_backend::CudaBatchRuntimeList runtime;

    const int flat_size = num_classes;
    std::vector<float> predicted_output;

    for(int start = 0; start < test_count; start += kEvaluationBatchSize){
        const int end = std::min(test_count, start + kEvaluationBatchSize);
        const int batch_size = end - start;

        cuda_backend::init_cuda_forward_batch_runtime_buffers(architecture, runtime, batch_size);
        feed_input_batch(dataset, test_indices, start, end, architecture[0], runtime[0]);
        forwardprop_batch(architecture, runtime, parameters, hidden_activation, output_activation);

        predicted_output.resize(static_cast<std::size_t>(batch_size) * static_cast<std::size_t>(flat_size));
        runtime[num_layers - 1].y.copy_to_host(predicted_output.data(), predicted_output.size());

        for(int batch_index = 0; batch_index < batch_size; batch_index++){
            const int sample_index = test_indices[start + batch_index];
            const float *sample_output = predicted_output.data() + static_cast<std::size_t>(batch_index) * static_cast<std::size_t>(flat_size);
            int predicted = 0;
            float best_value = sample_output[0];
            for(int c = 1; c < flat_size; c++){
                const float value = sample_output[c];
                if(value > best_value){
                    best_value = value;
                    predicted = c;
                }
            }

            const int expected = dataset.samples[static_cast<std::size_t>(sample_index)].class_index;
            confusion[expected][predicted]++;
            if(predicted == expected){
                correct++;
            }
        }

        std::cout << "Testato esempio " << end << "/" << test_count << std::endl;
    }

    perf = evaluation_metrics::build_test_performance(num_classes, test_count, correct, std::move(confusion));
    std::cout << "Accuracy: " << perf.accuracy << " (" << correct << "/" << test_count << ")" << std::endl;

    std::cout << "Confusion matrix (righe=vero, colonne=predetto)" << std::endl;
    for (int r = 0; r < num_classes; r++)
    {
        for (int c = 0; c < num_classes; c++)
        {
            std::cout << perf.confusion[r][c];
            if (c < num_classes - 1)
            {
                std::cout << " ";
            }
        }
        std::cout << std::endl;
    }

    std::cout << "Metriche per classe" << std::endl;
    for (int c = 0; c < num_classes; c++)
    {
        const ClassPerformance &class_perf = perf.per_class[c];
        std::cout << "Classe " << c << ": TP=" << class_perf.tp << " FP=" << class_perf.fp << " TN=" << class_perf.tn << " FN=" << class_perf.fn
                  << " precision=" << class_perf.precision << " recall=" << class_perf.recall << " f1=" << class_perf.f1 << std::endl;
    }
    std::cout << "Macro precision: " << perf.macro_precision << std::endl;
    std::cout << "Macro recall: " << perf.macro_recall << std::endl;
    std::cout << "Macro f1: " << perf.macro_f1 << std::endl;
    return perf;
}

float run_validation_accuracy(LayerList &architecture, const cuda_backend::CudaParameterBuffer &parameters, cuda_backend::CudaBatchRuntimeList &runtime, const std::vector<int> &validation_indices, const LazyDataset &dataset, const Activation &hidden_activation, const Activation &output_activation, int batch_size)
{
    validate_architecture(architecture, "run_validation_accuracy");
    validate_dataset_indices_io_shapes(architecture, dataset, validation_indices, "run_validation_accuracy");
    const int num_layers = static_cast<int>(architecture.size());
    const int validation_count = static_cast<int>(validation_indices.size());
    if(validation_count == 0){
        return 0.0f;
    }

    const int effective_batch_size = std::max(1, batch_size);
    const int flat_size = architecture[num_layers - 1].flat_output_size();
    int correct = 0;
    std::vector<float> predicted_output;

    for(int start = 0; start < validation_count; start += effective_batch_size){
        const int end = std::min(validation_count, start + effective_batch_size);
        const int current_batch_size = end - start;

        cuda_backend::init_cuda_forward_batch_runtime_buffers(architecture, runtime, current_batch_size);
        feed_input_batch(dataset, validation_indices, start, end, architecture[0], runtime[0]);
        forwardprop_batch(architecture, runtime, parameters, hidden_activation, output_activation);

        predicted_output.resize(static_cast<std::size_t>(current_batch_size) * static_cast<std::size_t>(flat_size));
        runtime[num_layers - 1].y.copy_to_host(predicted_output.data(), predicted_output.size());

        for(int batch_index = 0; batch_index < current_batch_size; batch_index++){
            const int sample_index = validation_indices[start + batch_index];
            const float *sample_output = predicted_output.data() + static_cast<std::size_t>(batch_index) * static_cast<std::size_t>(flat_size);
            int predicted = 0;
            float best_value = sample_output[0];
            for(int c = 1; c < flat_size; c++){
                const float value = sample_output[c];
                if(value > best_value){
                    best_value = value;
                    predicted = c;
                }
            }

            const int expected = dataset.samples[static_cast<std::size_t>(sample_index)].class_index;
            if(predicted == expected){
                correct++;
            }
        }
    }

    return static_cast<float>(correct) / static_cast<float>(validation_count);
}
