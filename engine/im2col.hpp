#pragma once

#include "core/layer.hpp"

#include <vector>

void im2col(const Layer &previous, const Layer &current, const LayerRuntime &previous_runtime, std::vector<float> &col);
void im2col_batch(const Layer &previous, const Layer &current, const BatchLayerRuntime &previous_runtime, std::vector<float> &col);
void col2im(const Layer &current, const Layer &next, const float *col_grad, std::vector<float> &input_grad, float alpha = 1.0f);
void col2im_batch(const Layer &current, const Layer &next, int batch_size, const float *col_grad, std::vector<float> &input_grad, float alpha = 1.0f);
