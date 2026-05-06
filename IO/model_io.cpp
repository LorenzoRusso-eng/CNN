#include "IO/model_io.hpp"

// Questo file contiene il parser e il serializer degli snapshot del modello.

#include "core/layer.hpp"
#include "IO/layer_text_codec.hpp"
#include "shared/network_globals.hpp"

#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <algorithm>
#include <cctype>

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

    bool is_blank_string(const std::string &value){
        return std::all_of(value.begin(), value.end(), [](unsigned char ch){
            return std::isspace(ch) != 0;
        });
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
                    kh < 0 || kh >= layer.kernel_dim[0] ||
                    kw < 0 || kw >= layer.kernel_dim[1] ||
                    ic < 0 || ic >= layer.kernel_dim[2])
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
            for (int kh = 0; kh < layer.kernel_dim[0]; kh++)
            {
                for (int kw = 0; kw < layer.kernel_dim[1]; kw++)
                {
                    for (int ic = 0; ic < layer.kernel_dim[2]; ic++)
                    {
                        out << "f " << oc << " " << kh << " " << kw << " " << ic << " " << layer.conv_filter_at(oc, kh, kw, ic) << std::endl;
                    }
                }
            }
        }
        out << "conv_params_end" << std::endl;
    }

    std::string decay_kind_to_string(DecayKind kind){
        switch(kind){
            case DecayKind::Constant:
                return "Constant";
            case DecayKind::Exponential:
                return "Exponential";
            case DecayKind::TimeBased:
                return "TimeBased";
            case DecayKind::Step:
                return "Step";
            case DecayKind::CosineAnnealing:
                return "CosineAnnealing";
        }
        return "Constant";
    }

    DecayKind decay_kind_from_string(const std::string &value){
        if(value == "Constant") return DecayKind::Constant;
        if(value == "Exponential") return DecayKind::Exponential;
        if(value == "TimeBased") return DecayKind::TimeBased;
        if(value == "Step") return DecayKind::Step;
        if(value == "CosineAnnealing") return DecayKind::CosineAnnealing;
        throw std::invalid_argument("Decay kind non valido nel training snapshot: " + value);
    }

    void init_parameter_buffer_like_architecture(const LayerList &architecture, ParameterBuffer &buffer){
        const int num_layers = static_cast<int>(architecture.size());
        buffer.dense_weights.resize(num_layers);
        buffer.dense_biases.resize(num_layers);
        buffer.conv_weights.resize(num_layers);
        buffer.conv_biases.resize(num_layers);

        for(int l=0; l<num_layers; l++){
            const Layer &layer = architecture[static_cast<std::size_t>(l)];
            if(layer.type == Layer_type::Dense){
                buffer.dense_weights[static_cast<std::size_t>(l)].assign(
                    static_cast<std::size_t>(layer.dense_output_size) *
                    static_cast<std::size_t>(layer.dense_input_size),
                    0.0f
                );
                buffer.dense_biases[static_cast<std::size_t>(l)].assign(static_cast<std::size_t>(layer.dense_output_size), 0.0f);
            } else {
                buffer.dense_weights[static_cast<std::size_t>(l)].clear();
                buffer.dense_biases[static_cast<std::size_t>(l)].clear();
            }

            if(layer.type == Layer_type::Conv){
                buffer.conv_weights[static_cast<std::size_t>(l)].assign(static_cast<std::size_t>(layer.conv_filter_count()), 0.0f);
                buffer.conv_biases[static_cast<std::size_t>(l)].assign(static_cast<std::size_t>(layer.dim_layer[2]), 0.0f);
            } else {
                buffer.conv_weights[static_cast<std::size_t>(l)].clear();
                buffer.conv_biases[static_cast<std::size_t>(l)].clear();
            }
        }
    }

    void write_float_vector(std::ostream &out, const std::string &token, const std::vector<float> &values){
        out << token << " " << values.size();
        for(float value : values){
            out << " " << value;
        }
        out << std::endl;
    }

    void read_float_vector(std::istream &in, const std::string &token, std::vector<float> &values){
        expect_token(in, token);
        std::size_t count = 0;
        in >> count;
        if(!in.good()){
            throw std::invalid_argument("Formato training snapshot non valido durante la lettura di " + token);
        }
        values.assign(count, 0.0f);
        for(std::size_t i = 0; i < count; i++){
            in >> values[i];
            if(!in.good()){
                throw std::invalid_argument("Formato training snapshot non valido: valori insufficienti in " + token);
            }
        }
    }

    void write_int_vector(std::ostream &out, const std::string &token, const std::vector<int> &values){
        out << token << " " << values.size();
        for(int value : values){
            out << " " << value;
        }
        out << std::endl;
    }

    void read_int_vector(std::istream &in, const std::string &token, std::vector<int> &values){
        expect_token(in, token);
        std::size_t count = 0;
        in >> count;
        if(!in.good()){
            throw std::invalid_argument("Formato training snapshot non valido durante la lettura di " + token);
        }
        values.assign(count, 0);
        for(std::size_t i = 0; i < count; i++){
            in >> values[i];
            if(!in.good()){
                throw std::invalid_argument("Formato training snapshot non valido: valori insufficienti in " + token);
            }
        }
    }

    void write_string_vector(std::ostream &out, const std::string &header_token, const std::string &entry_token, const std::vector<std::string> &values){
        out << header_token << " " << values.size() << std::endl;
        for(std::size_t index = 0; index < values.size(); index++){
            out << entry_token << " " << index << " " << values[index] << std::endl;
        }
    }

    void write_velocity(std::ostream &out, const LayerList &architecture, const ParameterBuffer &velocity){
        out << "optimizer_velocity_begin" << std::endl;
        const int num_layers = static_cast<int>(architecture.size());
        out << "velocity_layers " << num_layers << std::endl;
        for(int l = 0; l < num_layers; l++){
            out << "velocity_layer " << l << std::endl;
            const std::size_t idx = static_cast<std::size_t>(l);
            write_float_vector(out, "dense_weights", velocity.dense_weights[idx]);
            write_float_vector(out, "dense_biases", velocity.dense_biases[idx]);
            write_float_vector(out, "conv_weights", velocity.conv_weights[idx]);
            write_float_vector(out, "conv_biases", velocity.conv_biases[idx]);
            out << "velocity_layer_end" << std::endl;
        }
        out << "optimizer_velocity_end" << std::endl;
    }

    void read_velocity(std::istream &in, const LayerList &architecture, ParameterBuffer &velocity){
        expect_token(in, "optimizer_velocity_begin");
        expect_token(in, "velocity_layers");
        int num_layers = 0;
        in >> num_layers;
        if(!in.good() || num_layers != static_cast<int>(architecture.size())){
            throw std::invalid_argument("Formato training snapshot non valido: numero layer velocity incoerente");
        }
        init_parameter_buffer_like_architecture(architecture, velocity);
        for(int l = 0; l < num_layers; l++){
            expect_token(in, "velocity_layer");
            int layer_idx = -1;
            in >> layer_idx;
            if(!in.good() || layer_idx != l){
                throw std::invalid_argument("Formato training snapshot non valido: indice velocity layer incoerente");
            }
            const std::size_t idx = static_cast<std::size_t>(l);
            read_float_vector(in, "dense_weights", velocity.dense_weights[idx]);
            read_float_vector(in, "dense_biases", velocity.dense_biases[idx]);
            read_float_vector(in, "conv_weights", velocity.conv_weights[idx]);
            read_float_vector(in, "conv_biases", velocity.conv_biases[idx]);

            const Layer &layer = architecture[idx];
            const std::size_t expected_dense_weights =
                (layer.type == Layer_type::Dense)
                    ? static_cast<std::size_t>(layer.dense_output_size) * static_cast<std::size_t>(layer.dense_input_size)
                    : 0u;
            const std::size_t expected_dense_biases =
                (layer.type == Layer_type::Dense)
                    ? static_cast<std::size_t>(layer.dense_output_size)
                    : 0u;
            const std::size_t expected_conv_weights =
                (layer.type == Layer_type::Conv)
                    ? static_cast<std::size_t>(layer.conv_filter_count())
                    : 0u;
            const std::size_t expected_conv_biases =
                (layer.type == Layer_type::Conv)
                    ? static_cast<std::size_t>(layer.dim_layer[2])
                    : 0u;

            if(velocity.dense_weights[idx].size() != expected_dense_weights){
                throw std::invalid_argument(
                    "Formato training snapshot non valido: dense_weights size incoerente al layer " + std::to_string(l)
                );
            }
            if(velocity.dense_biases[idx].size() != expected_dense_biases){
                throw std::invalid_argument(
                    "Formato training snapshot non valido: dense_biases size incoerente al layer " + std::to_string(l)
                );
            }
            if(velocity.conv_weights[idx].size() != expected_conv_weights){
                throw std::invalid_argument(
                    "Formato training snapshot non valido: conv_weights size incoerente al layer " + std::to_string(l)
                );
            }
            if(velocity.conv_biases[idx].size() != expected_conv_biases){
                throw std::invalid_argument(
                    "Formato training snapshot non valido: conv_biases size incoerente al layer " + std::to_string(l)
                );
            }

            expect_token(in, "velocity_layer_end");
        }
        expect_token(in, "optimizer_velocity_end");
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

    void write_model_snapshot_contents(std::ostream &out, const LayerList &architecture, int num_layers, const std::vector<std::string> &class_names, const Activation &hidden_activation, const Activation &output_activation){
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
            out << "layer " << l << " type " << layer_text::layer_type_to_string(layer.type) << std::endl;
            out << "dim " << layer.dim_layer[0] << " " << layer.dim_layer[1] << " " << layer.dim_layer[2] << std::endl;
            out << "input_dim " << layer.input_dim[0] << " " << layer.input_dim[1] << " " << layer.input_dim[2] << std::endl;
            out << "kernel_dim " << layer.kernel_dim[0] << " " << layer.kernel_dim[1] << " " << layer.kernel_dim[2] << std::endl;
            out << "stride " << layer.stride[0] << " " << layer.stride[1] << std::endl;
            out << "padding " << layer.padding[0] << " " << layer.padding[1] << std::endl;
            out << "pooling_type " << layer_text::pooling_type_to_string(layer.pooling_type) << std::endl;
            out << "lrn " << layer.lrn_local_size << " " << layer.lrn_alpha << " " << layer.lrn_beta << " " << layer.lrn_k << std::endl;

            if(layer.type == Layer_type::Dense){
                write_dense_params(out, layer);
            }
            else if(layer.type == Layer_type::Conv){
                write_conv_params(out, layer);
            }

            out << "layer_end" << std::endl;
        }
    }

    void write_training_snapshot_metadata(std::ostream &out, const LayerList &architecture, const TrainingSnapshotMetadata &metadata){
        out << "TRAINING_SNAPSHOT_V1_BEGIN" << std::endl;
        out << "model_name " << metadata.model_name << std::endl;
        out << "training_method " << metadata.training_method << std::endl;
        out << "training_type " << metadata.training_type << std::endl;
        out << "training_window " << metadata.training_window << std::endl;
        out << "use_nesterov " << (metadata.use_nesterov ? 1 : 0) << std::endl;
        out << "momentum " << metadata.momentum << std::endl;
        out << "target_loss " << metadata.target_loss << std::endl;
        out << "requested_epochs " << metadata.requested_epochs << std::endl;
        out << "completed_epochs " << metadata.completed_epochs << std::endl;
        out << "optimizer_steps " << metadata.optimizer_steps << std::endl;
        out << "training_finalized " << (metadata.training_finalized ? 1 : 0) << std::endl;
        out << "decay_kind " << decay_kind_to_string(metadata.decay_kind) << std::endl;
        out << "initial_learning_rate " << metadata.initial_learning_rate << std::endl;
        out << "decay_rate " << metadata.decay_rate << std::endl;
        out << "decay_step_size " << metadata.decay_step_size << std::endl;
        out << "cosine_final_lr " << metadata.cosine_final_lr << std::endl;
        out << "cosine_max_epoch " << metadata.cosine_max_epoch << std::endl;
        out << "loss_kind " << static_cast<int>(metadata.loss_kind) << std::endl;
        out << "loss_reduction " << static_cast<int>(metadata.loss_reduction) << std::endl;
        out << "loss_beta " << metadata.loss_beta << std::endl;
        out << "hold_out_ratio " << metadata.hold_out_ratio << std::endl;
        out << "k_folds " << metadata.k_folds << std::endl;
        write_int_vector(out, "train_indices", metadata.train_indices);
        write_int_vector(out, "test_indices", metadata.test_indices);
        write_string_vector(out, "dataset_manifest_paths", "dataset_manifest_path", metadata.dataset_manifest_paths);
        out << "shuffle_rng_state " << metadata.shuffle_rng_state << std::endl;
        write_velocity(out, architecture, metadata.optimizer_velocity);
        out << "TRAINING_SNAPSHOT_V1_END" << std::endl;
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

        expect_token(in, "kernel_dim");
        int kernel_dim[3] = {0, 0, 0};
        in >> kernel_dim[0] >> kernel_dim[1] >> kernel_dim[2];

        expect_token(in, "stride");
        int stride[2] = {1, 1};
        in >> stride[0] >> stride[1];

        expect_token(in, "padding");
        int padding[2] = {0, 0};
        in >> padding[0] >> padding[1];

        expect_token(in, "pooling_type");
        std::string pooling_type_str;
        in >> pooling_type_str;

        expect_token(in, "lrn");
        int lrn_local_size = 0;
        float lrn_alpha = 0.0f;
        float lrn_beta = 0.0f;
        float lrn_k = 1.0f;
        in >> lrn_local_size >> lrn_alpha >> lrn_beta >> lrn_k;

        switch (layer_type)
        {
        case Layer_type::Input:
            layer.init_input(dim);
            break;
        case Layer_type::Dense:
            layer.init_dense(dim, input_dim);
            break;
        case Layer_type::Conv:
            layer.init_conv(dim, input_dim, kernel_dim, stride, padding);
            break;
        case Layer_type::Pooling:
        {
            const Pooling_type pool_type = layer_text::parse_pooling_type(pooling_type_str);
            layer.init_pooling(dim, input_dim, kernel_dim, stride, padding, pool_type);
            break;
        }
        case Layer_type::Flatten:
            layer.init_flatten(dim, input_dim);
            break;
        case Layer_type::LRN:
            layer.init_lrn(dim, input_dim, lrn_local_size, lrn_alpha, lrn_beta, lrn_k);
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

void save_model_snapshot(const LayerList &architecture, int num_layers, const fs::path &file_path, const std::vector<std::string> &class_names, const Activation &hidden_activation, const Activation &output_activation)
{
    write_snapshot_atomically(file_path, [&](std::ostream &out){
        write_model_snapshot_contents(out, architecture, num_layers, class_names, hidden_activation, output_activation);
    });
}

void save_training_snapshot(const fs::path &file_path, const LayerList &architecture, int num_layers, const std::vector<std::string> &class_names, const Activation &hidden_activation, const Activation &output_activation, const TrainingSnapshotMetadata &metadata){
    write_snapshot_atomically(file_path, [&](std::ostream &out){
        write_model_snapshot_contents(out, architecture, num_layers, class_names, hidden_activation, output_activation);
        write_training_snapshot_metadata(out, architecture, metadata);
    });
}

void load_training_snapshot(const fs::path &file_path, LayerList &architecture, std::vector<std::string> &class_names, std::string &hidden_activation_name, std::string &output_activation_name, TrainingSnapshotMetadata &metadata){
    load_model_snapshot(file_path, architecture, class_names, hidden_activation_name, output_activation_name);

    std::ifstream in(file_path);
    if(!in.good()){
        throw std::invalid_argument("Impossibile aprire il file snapshot: " + file_path.string());
    }

    std::string token;
    bool found_training_section = false;
    while(in >> token){
        if(token == "TRAINING_SNAPSHOT_V1_BEGIN"){
            found_training_section = true;
            break;
        }
    }

    if(!found_training_section){
        throw std::invalid_argument("Il file non contiene una sezione TRAINING_SNAPSHOT_V1 (snapshot inference-only)");
    }

    expect_token(in, "model_name");
    metadata.model_name = read_rest_of_line(in);
    if(is_blank_string(metadata.model_name)){
        throw std::invalid_argument("Formato training snapshot non valido: model_name vuoto");
    }

    expect_token(in, "training_method");
    in >> metadata.training_method;
    expect_token(in, "training_type");
    in >> metadata.training_type;
    expect_token(in, "training_window");
    in >> metadata.training_window;

    expect_token(in, "use_nesterov");
    int use_nesterov_int = 0;
    in >> use_nesterov_int;
    metadata.use_nesterov = (use_nesterov_int != 0);

    expect_token(in, "momentum");
    in >> metadata.momentum;
    expect_token(in, "target_loss");
    in >> metadata.target_loss;
    expect_token(in, "requested_epochs");
    in >> metadata.requested_epochs;
    expect_token(in, "completed_epochs");
    in >> metadata.completed_epochs;
    expect_token(in, "optimizer_steps");
    in >> metadata.optimizer_steps;

    std::string next_token;
    in >> next_token;
    if(!in.good()){
        throw std::invalid_argument("Formato training snapshot non valido dopo optimizer_steps");
    }

    if(next_token == "training_finalized"){
        int training_finalized_int = 0;
        in >> training_finalized_int;
        if(!in.good()){
            throw std::invalid_argument("Formato training snapshot non valido durante la lettura di training_finalized");
        }
        metadata.training_finalized = (training_finalized_int != 0);
        in >> next_token;
        if(!in.good()){
            throw std::invalid_argument("Formato training snapshot non valido dopo training_finalized");
        }
    } else {
        metadata.training_finalized = false;
    }

    if(next_token != "decay_kind"){
        throw std::invalid_argument("Formato training snapshot non valido: atteso token 'decay_kind'");
    }
    std::string decay_kind_text;
    in >> decay_kind_text;
    metadata.decay_kind = decay_kind_from_string(decay_kind_text);

    expect_token(in, "initial_learning_rate");
    in >> metadata.initial_learning_rate;
    expect_token(in, "decay_rate");
    in >> metadata.decay_rate;
    expect_token(in, "decay_step_size");
    in >> metadata.decay_step_size;
    expect_token(in, "cosine_final_lr");
    in >> metadata.cosine_final_lr;
    expect_token(in, "cosine_max_epoch");
    in >> metadata.cosine_max_epoch;

    expect_token(in, "loss_kind");
    int loss_kind_int = 0;
    in >> loss_kind_int;
    if(loss_kind_int < static_cast<int>(LossKind::Simple) || loss_kind_int > static_cast<int>(LossKind::LL)){
        throw std::invalid_argument("loss_kind non valido nel training snapshot");
    }
    metadata.loss_kind = static_cast<LossKind>(loss_kind_int);

    expect_token(in, "loss_reduction");
    int loss_reduction_int = 0;
    in >> loss_reduction_int;
    if(loss_reduction_int < static_cast<int>(Reduction::Sum) || loss_reduction_int > static_cast<int>(Reduction::Mean)){
        throw std::invalid_argument("loss_reduction non valida nel training snapshot");
    }
    metadata.loss_reduction = static_cast<Reduction>(loss_reduction_int);

    expect_token(in, "loss_beta");
    in >> metadata.loss_beta;

    in >> next_token;
    if(!in.good()){
        throw std::invalid_argument("Formato training snapshot non valido dopo loss_beta");
    }
    if(next_token == "hold_out_ratio"){
        in >> metadata.hold_out_ratio;
        if(!in.good()){
            throw std::invalid_argument("Formato training snapshot non valido durante la lettura di hold_out_ratio");
        }
    } else {
        metadata.hold_out_ratio = 0.0f;
        if(next_token != "k_folds"){
            throw std::invalid_argument("Formato training snapshot non valido: atteso token 'hold_out_ratio' o 'k_folds'");
        }
    }

    if(next_token == "hold_out_ratio"){
        expect_token(in, "k_folds");
        in >> metadata.k_folds;
    } else {
        in >> metadata.k_folds;
    }

    read_int_vector(in, "train_indices", metadata.train_indices);
    read_int_vector(in, "test_indices", metadata.test_indices);

    in >> next_token;
    if(!in.good()){
        throw std::invalid_argument("Formato training snapshot non valido dopo test_indices");
    }

    if(next_token == "dataset_manifest_paths"){
        std::size_t count = 0;
        in >> count;
        if(!in.good()){
            throw std::invalid_argument("Formato training snapshot non valido durante la lettura di dataset_manifest_paths");
        }
        metadata.dataset_manifest_paths.assign(count, "");
        for(std::size_t i = 0; i < count; i++){
            expect_token(in, "dataset_manifest_path");
            std::size_t entry_index = 0;
            in >> entry_index;
            if(!in.good() || entry_index >= count){
                throw std::invalid_argument("Formato training snapshot non valido: indice non valido in dataset_manifest_path");
            }
            metadata.dataset_manifest_paths[entry_index] = read_rest_of_line(in);
        }

        in >> next_token;
        if(!in.good()){
            throw std::invalid_argument("Formato training snapshot non valido dopo dataset_manifest_paths");
        }
    } else {
        metadata.dataset_manifest_paths.clear();
    }

    if(next_token != "shuffle_rng_state"){
        throw std::invalid_argument("Formato training snapshot non valido: atteso token 'shuffle_rng_state'");
    }
    metadata.shuffle_rng_state = read_rest_of_line(in);
    read_velocity(in, architecture, metadata.optimizer_velocity);
    expect_token(in, "TRAINING_SNAPSHOT_V1_END");
}

