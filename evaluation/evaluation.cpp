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


int argmax_target(const Tensor &target)
{
    int best_index = 0;
    float best_value = target.data[0];
    for (int c = 1; c < target.height; c++)
    {
        const float value = target.data[static_cast<std::size_t>(c)];
        if (value > best_value)
        {
            best_value = value;
            best_index = static_cast<int>(c);
        }
    }
    return best_index;
}

TestPerformance run_test(LayerList &architecture, int num_layers, const std::vector<int> &test_indices, const Dataset4D &input, const Dataset4D &output, const Activation &hidden_activation, const Activation &output_activation)
{
    TestPerformance perf;
    validate_architecture(architecture, num_layers, "run_test");
    validate_dataset_indices_io_shapes(architecture, num_layers, input, output, test_indices, "run_test");
    const int test_count = static_cast<int>(test_indices.size());
    if (test_indices.empty())
    {
        std::cout << "Test saltato: nessun esempio selezionato." << std::endl;
        return perf;
    }

    const int num_classes = architecture[num_layers - 1].dim_layer[0];
    std::vector<std::vector<int>> confusion(num_classes, std::vector<int>(num_classes, 0));
    int correct = 0;

    std::cout << "Inizio test..." << std::endl;
    cuda_backend::CudaParameterBuffer parameters;
    cuda_backend::CudaBatchRuntimeList runtime;
    cuda_backend::init_cuda_parameter_buffer(architecture, parameters);
    cuda_backend::sync_cuda_parameters_from_cpu(architecture, parameters);

    const int flat_size = architecture[num_layers - 1].flat_output_size();
    std::vector<float> predicted_output;

    for(int start = 0; start < test_count; start += kEvaluationBatchSize){
        const int end = std::min(test_count, start + kEvaluationBatchSize);
        const int batch_size = end - start;

        cuda_backend::init_cuda_batch_runtime_buffers(architecture, runtime, batch_size);
        feed_input_batch(input, test_indices, start, end, architecture[0], runtime[0]);
        forwardprop_batch(architecture, runtime, num_layers, parameters, hidden_activation, output_activation);

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

            const int expected = argmax_target(output[sample_index]);
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
