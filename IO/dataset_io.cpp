#include "IO/dataset_io.hpp"

// Questo file contiene il caricamento lazy del dataset e il mapping classi -> indici.

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "stb_image.h"

namespace {

namespace fs = std::filesystem;

// Helper locali per enumerare file e riconoscere immagini supportate
bool has_supported_image_extension(const fs::path &path){
    std::string extension = path.extension().string();
    std::transform(extension.begin(), extension.end(), extension.begin(), [](unsigned char ch){
        return static_cast<char>(std::tolower(ch));
    });

    return extension == ".png" ||
           extension == ".jpg" ||
           extension == ".jpeg" ||
           extension == ".bmp" ||
           extension == ".tga";
}

std::vector<fs::path> collect_image_files(const fs::path &root_dir){
    std::vector<fs::path> image_files;
    if(!fs::exists(root_dir) || !fs::is_directory(root_dir)){
        throw std::invalid_argument("Directory non valida: " + root_dir.string());
    }

    for(const auto &class_entry : fs::directory_iterator(root_dir)){
        if(!class_entry.is_directory()){
            continue;
        }

        for(const auto &file_entry : fs::directory_iterator(class_entry.path())){
            if(file_entry.is_regular_file() && has_supported_image_extension(file_entry.path())){
                image_files.push_back(file_entry.path());
            }
        }
    }

    std::sort(image_files.begin(), image_files.end(), [](const fs::path &lhs, const fs::path &rhs){
        return lhs.lexically_normal().generic_string() < rhs.lexically_normal().generic_string();
    });

    return image_files;
}

} // namespace

std::vector<std::string> load_class_names_from_train_dir(const fs::path &train_dir){
    std::vector<std::string> class_names;
    if(!fs::exists(train_dir) || !fs::is_directory(train_dir)){
        return class_names;
    }

    for(const auto &entry : fs::directory_iterator(train_dir)){
        if(entry.is_directory()){
            class_names.push_back(entry.path().filename().string());
        }
    }
    std::sort(class_names.begin(), class_names.end());
    return class_names;
}

void load_examples_from_manifest(
    const std::vector<std::string> &dataset_manifest_paths,
    const int (&input_shape)[3],
    const std::vector<std::string> &class_names,
    LazyDataset &dataset
){
    if(dataset_manifest_paths.empty()){
        throw std::invalid_argument("Manifest dataset assente nello snapshot");
    }
    if(class_names.empty()){
        throw std::invalid_argument("Classi assenti nello snapshot");
    }

    dataset.input_shape[0] = input_shape[0];
    dataset.input_shape[1] = input_shape[1];
    dataset.input_shape[2] = input_shape[2];
    dataset.num_classes = static_cast<int>(class_names.size());
    dataset.class_names = class_names;
    dataset.samples.clear();
    dataset.samples.reserve(dataset_manifest_paths.size());

    for(int e = 0; e < static_cast<int>(dataset_manifest_paths.size()); e++){
        const fs::path image_path = fs::path(dataset_manifest_paths[static_cast<std::size_t>(e)]).lexically_normal();
        if(!fs::exists(image_path) || !fs::is_regular_file(image_path)){
            throw std::invalid_argument("File del manifest non trovato: " + image_path.string());
        }
        if(!has_supported_image_extension(image_path)){
            throw std::invalid_argument("File del manifest con estensione non supportata: " + image_path.string());
        }

        const std::string class_name = image_path.parent_path().filename().string();
        const auto class_it = std::find(class_names.begin(), class_names.end(), class_name);
        if(class_it == class_names.end()){
            throw std::invalid_argument("Classe del manifest non presente nello snapshot: " + class_name);
        }

        const int class_index = static_cast<int>(std::distance(class_names.begin(), class_it));
        int width = 0;
        int height = 0;
        int channels = 0;
        if(!stbi_info(image_path.string().c_str(), &width, &height, &channels)){
            throw std::invalid_argument("Impossibile leggere le dimensioni dell'immagine: " + image_path.string());
        }
        if(height != input_shape[0] || width != input_shape[1]){
            throw std::invalid_argument("Dimensioni immagine non valide per " + image_path.string());
        }
        dataset.samples.push_back(DatasetSample{image_path.generic_string(), class_index});
    }

    std::cout << "Dataset caricato dal manifest dello snapshot" << std::endl;
    std::cout << "Classi caricate: " << dataset.num_classes << std::endl;
    std::cout << "Esempi caricati: " << dataset.size() << std::endl;
    std::cout << "Dimensione immagini attesa: " << input_shape[0] << "x" << input_shape[1] << std::endl;
    std::cout << "Canali attesi: " << input_shape[2] << std::endl;
}

void get_example(
    LazyDataset &dataset,
    std::vector<std::string> &class_names,
    std::vector<std::string> *dataset_manifest_paths
){
    std::string path;

    std::cout
        << "Inserire il percorso del dataset (organizzarlo in maniera tale che "
        << "le immagini siano organizzate in sottocartelle per ogni classe)"
        << std::endl;
    std::getline(std::cin, path);
    const fs::path train_dir = path;

    if(!fs::exists(train_dir)){
        throw std::invalid_argument("Dataset non trovato");
    }

    class_names = load_class_names_from_train_dir(train_dir);

    if(class_names.empty()){
        throw std::invalid_argument("Nessuna classe trovata in " + train_dir.string());
    }

    std::vector<fs::path> train_files = collect_image_files(train_dir);

    dataset.num_classes = static_cast<int>(class_names.size());
    const int examples = static_cast<int>(train_files.size());

    if(examples <= 0){
        throw std::invalid_argument("Nessuna immagine trovata nel dataset");
    }

    int width = 0;
    int height = 0;
    int channels = 0;

    if(!stbi_info(train_files[0].string().c_str(), &width, &height, &channels)){
        throw std::invalid_argument("Impossibile leggere le dimensioni dell'immagine: " + train_files[0].string());
    }

    dataset.input_shape[0] = height;
    dataset.input_shape[1] = width;
    dataset.input_shape[2] = channels;
    dataset.class_names = class_names;
    dataset.samples.clear();
    dataset.samples.reserve(static_cast<std::size_t>(examples));

    for(int e=0; e<examples; e++){
        const fs::path &image_path = train_files[e];
        const std::string class_name = image_path.parent_path().filename().string();
        const auto class_it = std::find(class_names.begin(), class_names.end(), class_name);
        if(class_it == class_names.end()){
            throw std::invalid_argument("Classe non riconosciuta nel training set: " + class_name);
        }

        const int class_index = static_cast<int>(std::distance(class_names.begin(), class_it));
        int image_width = 0;
        int image_height = 0;
        int image_channels = 0;
        if(!stbi_info(image_path.string().c_str(), &image_width, &image_height, &image_channels)){
            throw std::invalid_argument("Impossibile leggere le dimensioni dell'immagine: " + image_path.string());
        }
        if(image_height != dataset.input_shape[0] || image_width != dataset.input_shape[1]){
            throw std::invalid_argument("Dimensioni immagine non valide per " + image_path.string());
        }
        const fs::path absolute_image_path = fs::absolute(image_path).lexically_normal();
        dataset.samples.push_back(DatasetSample{absolute_image_path.generic_string(), class_index});
    }

    if(dataset_manifest_paths != nullptr){
        dataset_manifest_paths->clear();
        dataset_manifest_paths->reserve(static_cast<std::size_t>(examples));
        for(const fs::path &image_path : train_files){
            dataset_manifest_paths->push_back(fs::absolute(image_path).lexically_normal().generic_string());
        }
    }

    std::cout << "Dataset rilevato in " << train_dir.string() << std::endl;
    std::cout << "Classi trovate: " << dataset.num_classes << std::endl;
    std::cout << "Esempi presenti: " << dataset.size() << std::endl;
    std::cout << "Canali rilevati: " << dataset.input_shape[2] << std::endl;
    std::cout << "Dimensione immagini rilevata: " << dataset.input_shape[0] << "x" << dataset.input_shape[1] << std::endl;
}
