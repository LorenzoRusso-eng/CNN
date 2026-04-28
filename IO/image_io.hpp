#pragma once

// Questo file contiene il caricamento di immagini in tensori normalizzati.

#include <filesystem>

#include "core/core_definitions.hpp"

Tensor3D load_01scaled_image_tensor(const std::filesystem::path &image_path, int expected_height, int expected_width, int expected_channels);
