#pragma once

// Questo file contiene il caricamento del dataset organizzato per classi.

#include <filesystem>
#include <string>
#include <vector>

#include "core/core_definitions.hpp"
#include "IO/image_io.hpp"

std::vector<std::string> load_class_names_from_train_dir(const std::filesystem::path &train_dir);
void get_example(
    int (&input_shape)[3],
    int &num_classes,
    int &examples,
    Dataset4D &input,
    Dataset4D &output,
    std::vector<std::string> &class_names,
    std::vector<std::string> *dataset_manifest_paths = nullptr
);
