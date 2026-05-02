#include "IO/image_io.hpp"

// Questo file contiene il decoding delle immagini e la conversione a tensore.

#include <stdexcept>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

namespace {

namespace fs = std::filesystem;

// Helper locali per leggere metadati dell'immagine

void read_image_dimensions(const fs::path &image_path, int &height, int &width, int &channels){
    if(!stbi_info(image_path.string().c_str(), &width, &height, &channels)){
        throw std::invalid_argument("Impossibile leggere le dimensioni dell'immagine: " + image_path.string());
    }
}

} // namespace

Tensor load_01scaled_image_tensor(const fs::path &image_path, int expected_height, int expected_width, int expected_channels){
    int width = 0;
    int height = 0;
    int channels = 0;
    read_image_dimensions(image_path, height, width, channels);
    unsigned char *data = stbi_load(image_path.string().c_str(), &width, &height, &channels, expected_channels);
    if(data == nullptr){
        throw std::invalid_argument("Impossibile leggere l'immagine: " + image_path.string());
    }

    if(height != expected_height || width != expected_width){
        stbi_image_free(data);
        throw std::invalid_argument("Dimensioni immagine non valide per " + image_path.string());
    }

    Tensor tensor(expected_height, expected_width, expected_channels, 0.0f);
    for(int i=0; i<expected_height; i++){
        for(int j=0; j<expected_width; j++){
            for(int k=0; k<expected_channels; k++){
                const int index = (i * expected_width + j) * expected_channels + k;
                tensor.data[static_cast<std::size_t>(tensor.index(i, j, k))] = static_cast<float>(data[index]) / 255.0f;
            }
        }
    }

    stbi_image_free(data);
    return tensor;
}
