#pragma once

// Questo file contiene il caricamento e salvataggio degli snapshot del modello.

#include <filesystem>
#include <string>
#include <vector>

#include "core/layer.hpp"
#include "IO/activation_codec.hpp"
#include "math/activations.hpp"

void load_model_snapshot(const std::filesystem::path &snapshot_path, LayerList &architecture, std::vector<std::string> &class_names, std::string &hidden_activation_name, std::string &output_activation_name);
void save_model_snapshot(const LayerList &architecture, const std::filesystem::path &file_path, const std::vector<std::string> &class_names, const Activation &hidden_activation, const Activation &output_activation);
