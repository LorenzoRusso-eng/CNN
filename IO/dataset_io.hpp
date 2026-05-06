#pragma once

// Questo file contiene il caricamento lazy del dataset organizzato per classi.

#include <filesystem>
#include <string>
#include <vector>

#include "core/core_definitions.hpp"
#include "IO/image_io.hpp"

std::vector<std::string> load_class_names_from_train_dir(const std::filesystem::path &train_dir);
void load_examples_from_manifest(
    const std::vector<std::string> &dataset_manifest_paths,
    const int (&input_shape)[3],
    const std::vector<std::string> &class_names,
    LazyDataset &dataset
);
void get_example(
    LazyDataset &dataset,
    std::vector<std::string> &class_names,
    std::vector<std::string> *dataset_manifest_paths = nullptr
);
