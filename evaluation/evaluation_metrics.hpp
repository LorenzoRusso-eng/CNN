#pragma once

#include "core/core_definitions.hpp"

#include <vector>

namespace evaluation_metrics {

TestPerformance build_test_performance(int num_classes, int test_count, int correct, std::vector<std::vector<int>> confusion);

} // namespace evaluation_metrics
