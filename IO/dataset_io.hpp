#pragma once

// Questo file contiene il caricamento lazy del dataset organizzato per classi.

#include <filesystem>
#include <string>
#include <vector>

#include "core/core_definitions.hpp"
#include "IO/image_io.hpp"

std::vector<std::string> load_class_names_from_train_dir(const std::filesystem::path &train_dir);
void get_example(
    LazyDataset &dataset,
    std::vector<std::string> &class_names
);
