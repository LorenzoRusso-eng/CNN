#include "IO/dataset_io.hpp"

// Questo file contiene il caricamento del dataset e il mapping classi -> target one-hot.

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

// Helper locali per enumerare file e costruire i target
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

    return image_files;
}

Tensor make_one_hot_target(int class_index, int num_classes){
    Tensor target(num_classes, 1, 1, 0.0f);
    target.data[static_cast<std::size_t>(target.index(class_index, 0, 0))] = 1.0f;
    return target;
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
    int &num_classes,
    int &examples,
    Dataset4D &input,
    Dataset4D &output
){
    if(dataset_manifest_paths.empty()){
        throw std::invalid_argument("Manifest dataset assente nello snapshot");
    }
    if(class_names.empty()){
        throw std::invalid_argument("Classi assenti nello snapshot");
    }

    num_classes = static_cast<int>(class_names.size());
    examples = static_cast<int>(dataset_manifest_paths.size());
    input.resize(examples);
    output.resize(examples);

    for(int e = 0; e < examples; e++){
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
        input[static_cast<std::size_t>(e)] = load_01scaled_image_tensor(image_path, input_shape[0], input_shape[1], input_shape[2]);
        output[static_cast<std::size_t>(e)] = make_one_hot_target(class_index, num_classes);
    }

    std::cout << "Dataset caricato dal manifest dello snapshot" << std::endl;
    std::cout << "Classi caricate: " << num_classes << std::endl;
    std::cout << "Esempi caricati: " << examples << std::endl;
    std::cout << "Dimensione immagini attesa: " << input_shape[0] << "x" << input_shape[1] << std::endl;
    std::cout << "Canali attesi: " << input_shape[2] << std::endl;
}

void get_example(
    int (&input_shape)[3],
    int &num_classes,
    int &examples,
    Dataset4D &input,
    Dataset4D &output,
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

    num_classes = static_cast<int>(class_names.size());
    examples = static_cast<int>(train_files.size());

    if(examples <= 0){
        throw std::invalid_argument("Nessuna immagine trovata nel dataset");
    }

    int width = 0;
    int height = 0;
    int channels = 0;

    if(!stbi_info(train_files[0].string().c_str(), &width, &height, &channels)){
        throw std::invalid_argument("Impossibile leggere le dimensioni dell'immagine: " + train_files[0].string());
    }

    input_shape[0] = height;
    input_shape[1] = width;
    input_shape[2] = channels;

    input.resize(examples);
    output.resize(examples);

    for(int e=0; e<examples; e++){
        const fs::path &image_path = train_files[e];
        const std::string class_name = image_path.parent_path().filename().string();
        const auto class_it = std::find(class_names.begin(), class_names.end(), class_name);
        if(class_it == class_names.end()){
            throw std::invalid_argument("Classe non riconosciuta nel training set: " + class_name);
        }

        const int class_index = static_cast<int>(std::distance(class_names.begin(), class_it));
        input[e] = load_01scaled_image_tensor(image_path, input_shape[0], input_shape[1], input_shape[2]);
        output[e] = make_one_hot_target(class_index, num_classes);
    }

    if(dataset_manifest_paths != nullptr){
        dataset_manifest_paths->clear();
        dataset_manifest_paths->reserve(static_cast<std::size_t>(examples));
        for(const fs::path &image_path : train_files){
            dataset_manifest_paths->push_back(image_path.lexically_normal().generic_string());
        }
    }

    std::cout << "Dataset rilevato in " << train_dir.string() << std::endl;
    std::cout << "Classi trovate: " << num_classes << std::endl;
    std::cout << "Esempi presenti: " << examples << std::endl;
    std::cout << "Canali rilevati: " << input_shape[2] << std::endl;
    std::cout << "Dimensione immagini rilevata: " << input_shape[0] << "x" << input_shape[1] << std::endl;
}
