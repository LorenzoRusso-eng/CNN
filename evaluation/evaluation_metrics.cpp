#include "evaluation/evaluation_metrics.hpp"

#include <utility>

namespace evaluation_metrics {

TestPerformance build_test_performance(int num_classes, int test_count, int correct, std::vector<std::vector<int>> confusion){
    TestPerformance perf;
    const float accuracy = static_cast<float>(correct) / static_cast<float>(test_count);
    float macro_precision = 0.0f;
    float macro_recall = 0.0f;
    float macro_f1 = 0.0f;
    std::vector<ClassPerformance> per_class(num_classes);

    for(int c = 0; c < num_classes; c++){
        const int tp = confusion[c][c];
        int fp = 0;
        int fn = 0;
        for(int r = 0; r < num_classes; r++){
            if(r != c){
                fp += confusion[r][c];
                fn += confusion[c][r];
            }
        }
        const int tn = test_count - tp - fp - fn;

        const float precision = (tp + fp) > 0 ? static_cast<float>(tp) / static_cast<float>(tp + fp) : 0.0f;
        const float recall = (tp + fn) > 0 ? static_cast<float>(tp) / static_cast<float>(tp + fn) : 0.0f;
        const float f1 = (precision + recall) > 0.0f ? (2.0f * precision * recall) / (precision + recall) : 0.0f;
        macro_precision += precision;
        macro_recall += recall;
        macro_f1 += f1;

        per_class[c].tp = tp;
        per_class[c].fp = fp;
        per_class[c].tn = tn;
        per_class[c].fn = fn;
        per_class[c].precision = precision;
        per_class[c].recall = recall;
        per_class[c].f1 = f1;
    }

    macro_precision /= static_cast<float>(num_classes);
    macro_recall /= static_cast<float>(num_classes);
    macro_f1 /= static_cast<float>(num_classes);

    perf.num_classes = num_classes;
    perf.test_count = test_count;
    perf.correct = correct;
    perf.accuracy = accuracy;
    perf.confusion = std::move(confusion);
    perf.per_class = std::move(per_class);
    perf.macro_precision = macro_precision;
    perf.macro_recall = macro_recall;
    perf.macro_f1 = macro_f1;
    return perf;
}

} // namespace evaluation_metrics
