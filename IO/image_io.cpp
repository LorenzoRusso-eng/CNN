#include "IO/image_io.hpp"

// Questo file contiene il decoding delle immagini e la conversione a tensore.

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

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

void save_activation_channel_png(
    const fs::path &image_path, const std::vector<float> &values,
    int height, int width, int channels, 
    int channel
){
    if(height <= 0 || width <= 0 || channels <= 0){
        throw std::invalid_argument("save_activation_channel_png: dimensioni non valide");
    }
    if(channel < 0 || channel >= channels){
        throw std::invalid_argument("save_activation_channel_png: canale fuori range");
    }

    const std::size_t expected_size =
        static_cast<std::size_t>(height) *
        static_cast<std::size_t>(width) *
        static_cast<std::size_t>(channels);
    if(values.size() < expected_size){
        throw std::invalid_argument("save_activation_channel_png: buffer valori troppo piccolo");
    }

    std::vector<float> channel_values(static_cast<std::size_t>(height) * static_cast<std::size_t>(width));
    for(int i = 0; i < height; i++){
        for(int j = 0; j < width; j++){
            const std::size_t image_index = static_cast<std::size_t>(i) * static_cast<std::size_t>(width) + static_cast<std::size_t>(j);
            const std::size_t tensor_index = image_index * static_cast<std::size_t>(channels) + static_cast<std::size_t>(channel);
            channel_values[image_index] = values[tensor_index];
        }
    }

    const auto [min_it, max_it] = std::minmax_element(channel_values.begin(), channel_values.end());
    const float min_value = *min_it;
    const float max_value = *max_it;
    const float range = max_value - min_value;

    std::vector<unsigned char> pixels(channel_values.size(), 0);
    for(std::size_t idx = 0; idx < channel_values.size(); idx++){
        float normalized = 0.0f;
        if(range > 0.0f && std::isfinite(range)){
            normalized = (channel_values[idx] - min_value) / range;
        }
        normalized = std::clamp(normalized, 0.0f, 1.0f);
        pixels[idx] = static_cast<unsigned char>(std::round(normalized * 255.0f));
    }

    const int ok = stbi_write_png(
        image_path.string().c_str(),
        width, height, 1,
        pixels.data(),
        width
    );
    if(ok == 0){
        throw std::runtime_error("Impossibile salvare l'immagine: " + image_path.string());
    }
}
