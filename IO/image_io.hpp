#pragma once

// Questo file contiene il caricamento di immagini in tensori normalizzati.

#include <filesystem>
#include <vector>

#include "core/core_definitions.hpp"

Tensor load_01scaled_image_tensor(const std::filesystem::path &image_path, int expected_height, int expected_width, int expected_channels);
void save_activation_channel_png(const std::filesystem::path &image_path, const std::vector<float> &values, int height, int width, int channels, int channel);
