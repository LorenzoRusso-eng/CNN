#include "IO/model_io.hpp"

// Questo file contiene il parser e il serializer degli snapshot del modello.

#include "core/layer.hpp"
#include "IO/layer_text_codec.hpp"
#include "shared/network_globals.hpp"

#include <fstream>
#include <iomanip>
#include <limits>
#include <stdexcept>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace
{
    namespace fs = std::filesystem;

    void expect_token(std::istream &in, const std::string &expected)
    {
        std::string token;
        in >> token;
        if (!in.good() || token != expected)
        {
            throw std::invalid_argument("Formato snapshot non valido: atteso token '" + expected + "'");
        }
    }

    std::string read_rest_of_line(std::istream &in)
    {
        std::string value;
        if(in.peek() == ' '){
            in.get();
        }
        std::getline(in, value);
        if(!in && value.empty())
        {
            throw std::invalid_argument("Formato snapshot non valido durante la lettura di una riga");
        }
        return value;
    }

    void read_dense_params(std::istream &in, Layer &layer)
    {
        expect_token(in, "dense_params_begin");
        while (true)
        {
            std::string token;
            in >> token;
            if (!in.good())
            {
                throw std::invalid_argument("Fine file inattesa nei parametri Dense");
            }
            if (token == "dense_params_end")
            {
                break;
            }
            if (token == "neuron")
            {
                int i = 0, j = 0, k = 0;
                std::string bias_label;
                float bias_value = 0.0f;
                in >> i >> j >> k >> bias_label >> bias_value;
                if (!in.good() || bias_label != "bias")
                {
                    throw std::invalid_argument("Riga neuron non valida nello snapshot");
                }
                if (i < 0 || i >= layer.dim_layer[0] ||
                    j < 0 || j >= layer.dim_layer[1] ||
                    k < 0 || k >= layer.dim_layer[2])
                {
                    throw std::invalid_argument("Riga neuron fuori range nello snapshot");
                }
                layer.dense_params.bias[static_cast<std::size_t>(layer.flat_index(i, j, k))] = bias_value;
            }
            else if (token == "w")
            {
                int i = 0, j = 0, k = 0, m = 0, n = 0, p = 0;
                float weight = 0.0f;
                in >> i >> j >> k >> m >> n >> p >> weight;
                if (!in.good())
                {
                    throw std::invalid_argument("Riga w non valida nello snapshot");
                }
                if (i < 0 || i >= layer.dim_layer[0] ||
                    j < 0 || j >= layer.dim_layer[1] ||
                    k < 0 || k >= layer.dim_layer[2] ||
                    m < 0 || m >= layer.input_dim[0] ||
                    n < 0 || n >= layer.input_dim[1] ||
                    p < 0 || p >= layer.input_dim[2])
                {
                    throw std::invalid_argument("Riga w fuori range nello snapshot");
                }
                const int out_idx = layer.flat_index(i, j, k);
                const int in_idx = layer.input_flat_index(m, n, p);
                layer.dense_weight_at(out_idx, in_idx) = weight;
            }
            else
            {
                throw std::invalid_argument("Token inatteso nei parametri Dense: " + token);
            }
        }
    }

    void read_conv_params(std::istream &in, Layer &layer)
    {
        expect_token(in, "conv_params_begin");
        while (true)
        {
            std::string token;
            in >> token;
            if (!in.good())
            {
                throw std::invalid_argument("Fine file inattesa nei parametri Conv");
            }
            if (token == "conv_params_end")
            {
                break;
            }
            if (token == "filter_bias")
            {
                int oc = 0;
                float bias = 0.0f;
                in >> oc >> bias;
                if (!in.good())
                {
                    throw std::invalid_argument("Riga filter_bias non valida nello snapshot");
                }
                if (oc < 0 || oc >= layer.dim_layer[2])
                {
                    throw std::invalid_argument("Riga filter_bias fuori range nello snapshot");
                }
                layer.conv_params.bias[oc] = bias;
            }
            else if (token == "f")
            {
                int oc = 0, kh = 0, kw = 0, ic = 0;
                float value = 0.0f;
                in >> oc >> kh >> kw >> ic >> value;
                if (!in.good())
                {
                    throw std::invalid_argument("Riga f non valida nello snapshot");
                }
                if (oc < 0 || oc >= layer.dim_layer[2] ||
                    kh < 0 || kh >= layer.kernel_shape[0] ||
                    kw < 0 || kw >= layer.kernel_shape[1] ||
                    ic < 0 || ic >= layer.kernel_shape[2])
                {
                    throw std::invalid_argument("Riga f fuori range nello snapshot");
                }
                layer.conv_filter_at(oc, kh, kw, ic) = value;
            }
            else
            {
                throw std::invalid_argument("Token inatteso nei parametri Conv: " + token);
            }
        }
    }

    void write_dense_params(std::ostream &out, const Layer &layer)
    {
        out << "dense_params_begin" << std::endl;
        for (int i = 0; i < layer.dim_layer[0]; i++)
        {
            for (int j = 0; j < layer.dim_layer[1]; j++)
            {
                for (int k = 0; k < layer.dim_layer[2]; k++)
                {
                    const int out_idx = layer.flat_index(i, j, k);
                    out << "neuron " << i << " " << j << " " << k << " bias " << layer.dense_params.bias[out_idx] << std::endl;
                    for (int m = 0; m < layer.input_dim[0]; m++)
                    {
                        for (int n = 0; n < layer.input_dim[1]; n++)
                        {
                            for (int p = 0; p < layer.input_dim[2]; p++)
                            {
                                const int in_idx = layer.input_flat_index(m, n, p);
                                out << "w "
                                    << i << " " << j << " " << k << " "
                                    << m << " " << n << " " << p << " "
                                    << layer.dense_weight_at(out_idx, in_idx)
                                    << std::endl;
                            }
                        }
                    }
                }
            }
        }
        out << "dense_params_end" << std::endl;
    }

    void write_conv_params(std::ostream &out, const Layer &layer)
    {
        out << "conv_params_begin" << std::endl;
        for (int oc = 0; oc < layer.dim_layer[2]; oc++)
        {
            out << "filter_bias " << oc << " " << layer.conv_params.bias[oc] << std::endl;
            for (int kh = 0; kh < layer.kernel_shape[0]; kh++)
            {
                for (int kw = 0; kw < layer.kernel_shape[1]; kw++)
                {
                    for (int ic = 0; ic < layer.kernel_shape[2]; ic++)
                    {
                        out << "f " << oc << " " << kh << " " << kw << " " << ic << " " << layer.conv_filter_at(oc, kh, kw, ic) << std::endl;
                    }
                }
            }
        }
        out << "conv_params_end" << std::endl;
    }
    fs::path temporary_snapshot_path(const fs::path &file_path){
        fs::path temp_path = file_path;
        temp_path += ".tmp";
        return temp_path;
    }

    void replace_snapshot_file(const fs::path &temp_path, const fs::path &file_path){
#ifdef _WIN32
        if(!MoveFileExW(
               temp_path.wstring().c_str(),
               file_path.wstring().c_str(),
               MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH
           )){
            throw std::runtime_error(
                "Impossibile sostituire atomicamente il file snapshot: " +
                file_path.string() +
                " (Win32 error " + std::to_string(GetLastError()) + ")"
            );
        }
#else
        std::error_code rename_error;
        fs::rename(temp_path, file_path, rename_error);
        if(rename_error){
            throw std::runtime_error(
                "Impossibile sostituire atomicamente il file snapshot: " +
                file_path.string() +
                " (" + rename_error.message() + ")"
            );
        }
#endif
    }

    template <typename Writer>
    void write_snapshot_atomically(const fs::path &file_path, Writer &&writer){
        const fs::path temp_path = temporary_snapshot_path(file_path);

        {
            std::ofstream out(temp_path, std::ios::trunc);
            if(!out.good()){
                throw std::runtime_error("Impossibile creare il file temporaneo snapshot: " + temp_path.string());
            }

            out << std::fixed << std::setprecision(std::numeric_limits<float>::max_digits10);
            writer(out);
            out.flush();

            if(!out.good()){
                throw std::runtime_error("Errore durante la scrittura del file temporaneo snapshot: " + temp_path.string());
            }
        }

        try{
            replace_snapshot_file(temp_path, file_path);
        }
        catch(...){
            std::error_code remove_error;
            fs::remove(temp_path, remove_error);
            throw;
        }
    }

    void write_model_snapshot_contents(std::ostream &out, const LayerList &architecture, const std::vector<std::string> &class_names, const Activation &hidden_activation, const Activation &output_activation){
        const int num_layers = static_cast<int>(architecture.size());
        out << "MODEL_SNAPSHOT" << std::endl;
        out << "class_names " << class_names.size() << std::endl;
        for(size_t c = 0; c < class_names.size(); c++){
            out << "class_name " << c << " " << class_names[c] << std::endl;
        }
        out << "hidden_activation " << activation_to_snapshot_name(hidden_activation) << std::endl;
        out << "output_activation " << activation_to_snapshot_name(output_activation) << std::endl;
        out << "num_layers " << num_layers << std::endl;

        for(int l = 0; l < num_layers; l++){
            const Layer &layer = architecture[l];
            const Shape3D &serialized_kernel = (layer.type == Layer_type::Pooling) ? layer.pool_window_shape : layer.kernel_shape;
            out << "layer " << l << " type " << layer_text::layer_type_to_string(layer.type) << std::endl;
            out << "dim " << layer.dim_layer[0] << " " << layer.dim_layer[1] << " " << layer.dim_layer[2] << std::endl;
            out << "input_dim " << layer.input_dim[0] << " " << layer.input_dim[1] << " " << layer.input_dim[2] << std::endl;
            out << "kernel_shape " << serialized_kernel[0] << " " << serialized_kernel[1] << " " << serialized_kernel[2] << std::endl;
            out << "stride " << layer.stride[0] << " " << layer.stride[1] << std::endl;
            out << "padding " << layer.padding[0] << " " << layer.padding[1] << std::endl;
            out << "pooling_type " << layer_text::pooling_type_to_string(layer.pooling_type) << std::endl;

            if(layer.type == Layer_type::Dense){
                write_dense_params(out, layer);
            }
            else if(layer.type == Layer_type::Conv){
                write_conv_params(out, layer);
            }

            out << "layer_end" << std::endl;
        }
    }
}

void load_model_snapshot(const fs::path &snapshot_path, LayerList &architecture, std::vector<std::string> &class_names, std::string &hidden_activation_name, std::string &output_activation_name)
{
    std::ifstream in(snapshot_path);
    if (!in.good())
    {
        throw std::invalid_argument("Impossibile aprire il file snapshot: " + snapshot_path.string());
    }

    std::string header;
    in >> header;
    if (header != "MODEL_SNAPSHOT")
    {
        throw std::invalid_argument("Header snapshot non valido");
    }

    class_names.clear();
    hidden_activation_name.clear();
    output_activation_name.clear();

    expect_token(in, "class_names");
    int class_count = 0;
    in >> class_count;
    if (!in.good() || class_count < 0)
    {
        throw std::invalid_argument("Numero classi non valido nel file snapshot");
    }

    class_names.resize(class_count);
    for (int c = 0; c < class_count; c++)
    {
        expect_token(in, "class_name");
        int class_idx = 0;
        in >> class_idx;
        if (!in.good() || class_idx < 0 || class_idx >= class_count)
        {
            throw std::invalid_argument("Indice classe non valido nel file snapshot");
        }
        class_names[class_idx] = read_rest_of_line(in);
    }

    expect_token(in, "hidden_activation");
    hidden_activation_name = read_rest_of_line(in);
    expect_token(in, "output_activation");
    output_activation_name = read_rest_of_line(in);

    expect_token(in, "num_layers");
    int num_layers = 0;
    in >> num_layers;
    if (!in.good() || num_layers <= 0)
    {
        throw std::invalid_argument("Numero layer non valido nel file snapshot");
    }

    architecture.clear();
    architecture.resize(num_layers);

    for (int l = 0; l < num_layers; l++)
    {
        expect_token(in, "layer");
        int layer_idx = 0;
        std::string type_label;
        std::string layer_type_str;
        in >> layer_idx >> type_label >> layer_type_str;
        if (!in.good() || type_label != "type" || layer_idx < 0 || layer_idx >= num_layers)
        {
            throw std::invalid_argument("Metadati layer non validi nel file snapshot");
        }

        Layer &layer = architecture[layer_idx];
        const Layer_type layer_type = layer_text::parse_layer_type(layer_type_str);

        expect_token(in, "dim");
        int dim[3] = {0, 0, 0};
        in >> dim[0] >> dim[1] >> dim[2];

        expect_token(in, "input_dim");
        int input_dim[3] = {0, 0, 0};
        in >> input_dim[0] >> input_dim[1] >> input_dim[2];

        expect_token(in, "kernel_shape");
        int kernel_shape[3] = {0, 0, 0};
        in >> kernel_shape[0] >> kernel_shape[1] >> kernel_shape[2];

        expect_token(in, "stride");
        int stride[2] = {1, 1};
        in >> stride[0] >> stride[1];

        expect_token(in, "padding");
        int padding[2] = {0, 0};
        in >> padding[0] >> padding[1];

        expect_token(in, "pooling_type");
        std::string pooling_type_str;
        in >> pooling_type_str;

        switch (layer_type)
        {
        case Layer_type::Input:
            layer.init_input(dim);
            break;
        case Layer_type::Dense:
            layer.init_dense(dim, input_dim);
            break;
        case Layer_type::Conv:
            layer.init_conv(dim, input_dim, kernel_shape, stride, padding);
            break;
        case Layer_type::Pooling:
        {
            const Pooling_type pool_type = layer_text::parse_pooling_type(pooling_type_str);
            layer.init_pooling(dim, input_dim, kernel_shape, stride, padding, pool_type);
            break;
        }
        case Layer_type::Flatten:
            layer.init_flatten(dim, input_dim);
            break;
        case Layer_type::Softmax:
            layer.init_softmax(dim, input_dim);
            break;
        }

        if (layer_type == Layer_type::Dense)
        {
            read_dense_params(in, layer);
        }
        else if (layer_type == Layer_type::Conv)
        {
            read_conv_params(in, layer);
        }

        expect_token(in, "layer_end");
    }
}

void save_model_snapshot(const LayerList &architecture, const fs::path &file_path, const std::vector<std::string> &class_names, const Activation &hidden_activation, const Activation &output_activation)
{
    write_snapshot_atomically(file_path, [&](std::ostream &out){
        write_model_snapshot_contents(out, architecture, class_names, hidden_activation, output_activation);
    });
}

