#include "cli/architecture_builder.hpp"
#include "cli/cli_utils.hpp"
#include "core/layer_validation.hpp"
#include "core/cuda_backend.hpp"
#include "engine/forward.hpp"
#include "evaluation/evaluation.hpp"
#include "training/training.hpp"
#include "IO/dataset_io.hpp"
#include "IO/image_io.hpp"
#include "IO/model_io.hpp"
#include "IO/report_io.hpp"
#include "shared/network_globals.hpp"
#include "training/training_method.hpp"
#include "IO/layer_text_codec.hpp"

#include <algorithm>
#include <cctype>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

bool is_blank_string(const std::string &value){
    return std::all_of(value.begin(), value.end(), [](unsigned char ch){
        return std::isspace(ch) != 0;
    });
}

void run_training_mode(){
    LayerList architecture;

    int num_layers = 0;
    int dim_input[3] = {0, 0, 0};
    int dim_output = 1;

    int num_examples = 0;
    float learning_rate = 0.0f;
    float target_loss = 0.0f;
    int num_epochs = 0;

    bool momentum_choice = false;
    bool use_nesterov = false;
    float momentum = 0.0f;
    int training_type = 1;
    int training_method = 1;
    float hold_out_ratio = 0.7f;
    float validation_ratio = 0.0f;
    int k_folds;

    LazyDataset dataset;
    std::vector<std::string> class_names;

    std::string model_name;

    std::cout << "Inserire il nome del modello" << std::endl;
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    std::getline(std::cin, model_name);
    require_condition(!is_blank_string(model_name), "Nome modello vuoto");

    get_example(dataset, class_names);
    dim_input[0] = dataset.input_shape[0];
    dim_input[1] = dataset.input_shape[1];
    dim_input[2] = dataset.input_shape[2];
    dim_output = dataset.num_classes;
    num_examples = dataset.size();

    create_architecture(num_layers, dim_input, dim_output, architecture);

    const bool final_is_softmax = architecture[num_layers - 1].type == Layer_type::Softmax;
    const Activation &hidden_activation = choose_hidden_activation();
    const Activation &output_activation = choose_output_activation(final_is_softmax);
    const Loss &loss = choose_loss_function(final_is_softmax);

    learning_rate = read_bounded_float(
        "Scegliere il learning rate iniziale per il training (maggiore di 0)",
        std::numeric_limits<float>::epsilon(),
        std::numeric_limits<float>::max(),
        "Inserire un valore maggiore di 0."
    );

    target_loss = read_bounded_float(
        "Scegliere il threshold di costo per il training (negativo se si vuole ignorare)",
        -std::numeric_limits<float>::max(),
        std::numeric_limits<float>::max(),
        "Inserire un valore numerico valido."
    );

    num_epochs = read_bounded_int(
        "Selezionare il numero massimo di epoche per il training (minimo: 1)",
        1,
        std::numeric_limits<int>::max(),
        "Inserire un intero maggiore o uguale a 1."
    );

    const Decay &learning_rate_decay = choose_learning_rate_decay(learning_rate, num_epochs);
    
    training_type = read_bounded_int("Scegliere il tipo di training: 1=Batch 2=Mini-batch SGD 3=Online SGD", 1, 3);

    momentum_choice = read_bounded_int("Usare il momento per accelerare il training? (0 = no, 1 = si)", 0, 1);
    if(momentum_choice){
        momentum = read_bounded_float("Scegliere il valore del momento (tra 0 e 1)", 0.0f, 1.0f, "Inserire un valore compreso tra 0 e 1.");
        if(momentum != 0.0f){
            use_nesterov = read_bounded_int("Usare Nesterov Accelerated Gradient? (0 = no, 1 = si)", 0, 1);
        }
    }


    training_method = read_bounded_int("Scegliere il metodo di training: 1=Hold-out 2=K-fold cross validation 3=Full_training", 1, 3);;
    switch(training_method){
    case 1:
        hold_out_ratio = read_bounded_float(
            "Scegliere il rapporto di training per l'hold-out in forma decimale (maggiore di 0 e minore di 1)",
            std::numeric_limits<float>::epsilon(),
            1.0f - std::numeric_limits<float>::epsilon(),
            "Inserire un valore decimale compreso tra 0 e 1, estremi esclusi."
        );
        validation_ratio = read_bounded_float(
            "Scegliere il rapporto di validation interno al training set in forma decimale (maggiore di 0 e minore di 1)",
            std::numeric_limits<float>::epsilon(),
            1.0f - std::numeric_limits<float>::epsilon(),
            "Inserire un valore decimale compreso tra 0 e 1, estremi esclusi."
        );
        hold_out(
            model_name, num_examples,
            training_type, use_nesterov,
            architecture, num_layers,
            learning_rate_decay, num_epochs, target_loss,
            dataset,
            loss, hidden_activation, output_activation,
            momentum, hold_out_ratio, validation_ratio,
            class_names
        );
        break;
    case 2:
        k_folds = read_bounded_int(
            "Scegliere il numero di fold K per la validazione incrociata (minimo: 2)",
            2,
            std::numeric_limits<int>::max(),
            "Inserire un intero maggiore o uguale a 2."
        );
        validation_ratio = read_bounded_float(
            "Scegliere il rapporto di validation interno al training set in forma decimale (maggiore di 0 e minore di 1)",
            std::numeric_limits<float>::epsilon(),
            1.0f - std::numeric_limits<float>::epsilon(),
            "Inserire un valore decimale compreso tra 0 e 1, estremi esclusi."
        );
        k_fold(
            model_name, k_folds, num_examples,
            training_type, use_nesterov,
            architecture, num_layers,
            learning_rate_decay, num_epochs, target_loss,
            dataset,
            loss, hidden_activation, output_activation,
            momentum,
            validation_ratio,
            class_names
        );
        break;
    case 3:
        full_training(
            model_name, num_examples,
            training_type, use_nesterov,
            architecture, num_layers,
            learning_rate_decay, num_epochs, target_loss,
            dataset,
            loss, hidden_activation, output_activation,
            momentum,
            class_names
        );
        break;
    default:
        throw std::invalid_argument("Tipo di training non valido");
    }
}

void run_inference_mode(){
    std::string snapshot_path_str;
    std::cout << "Inserire il percorso del file snapshot della rete" << std::endl;
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    std::getline(std::cin, snapshot_path_str);
    require_condition(!is_blank_string(snapshot_path_str), "Percorso snapshot vuoto");

    LayerList architecture;
    std::vector<std::string> class_names;
    std::string hidden_activation_name;
    std::string output_activation_name;
    load_model_snapshot(snapshot_path_str, architecture, class_names, hidden_activation_name, output_activation_name);
    const int num_layers = static_cast<int>(architecture.size());
    validate_architecture(architecture, num_layers, "Inference");
    const int input_h = architecture[0].dim_layer[0];
    const int input_w = architecture[0].dim_layer[1];
    const int input_c = architecture[0].dim_layer[2];
    int go = 1;

    std::cout << "La struttura caricata è la seguente:" << std::endl;
    for(int l = 0; l < num_layers; l++){
        std::cout << "layer " << l +1 << ":";
        std::cout << layer_text::layer_type_to_string(architecture[l].type) << "; ";
    }
    std::cout << std::endl;

    while(go){

        std::string image_path_str;
        std::cout << "Inserire il percorso dell'immagine da classificare" << std::endl;
        std::getline(std::cin, image_path_str);
        require_condition(!is_blank_string(image_path_str), "Percorso immagine vuoto");

        const Tensor image = load_01scaled_image_tensor(image_path_str, input_h, input_w, input_c);
        validate_tensor_shape(image, architecture[0].dim_layer, "Inference image");

        require_condition(!hidden_activation_name.empty(), "Snapshot senza attivazione hidden salvata");
        require_condition(!output_activation_name.empty(), "Snapshot senza attivazione output salvata");
        const Activation &hidden_activation = activation_from_snapshot_name(hidden_activation_name);
        const Activation &output_activation = activation_from_snapshot_name(output_activation_name);
        std::cout << "Attivazioni caricate dallo snapshot: hidden=" << hidden_activation_name << " output=" << output_activation_name << std::endl;

        cuda_backend::CudaParameterBuffer parameters;
        cuda_backend::CudaBatchRuntimeList runtime;
        cuda_backend::init_cuda_parameter_buffer(architecture, parameters);
        cuda_backend::sync_cuda_parameters_from_cpu(architecture, parameters);

        cuda_backend::init_cuda_forward_batch_runtime_buffers(architecture, runtime, 1);
        feed_input_tensor(image, architecture[0], runtime[0]);
        forwardprop_batch(architecture, runtime, num_layers, parameters, hidden_activation, output_activation);

        int obs = read_bounded_int("Osservare l'output di un layer? (0 = no, 1 = si)", 0, 1);

        while(obs){
            int l = read_bounded_int("Quale layer si vuole osservare?", 1, num_layers);

            std::vector<float> observed(static_cast<std::size_t>(architecture[l-1].flat_output_size()));

            runtime[l - 1].y.copy_to_host(observed.data(), observed.size());

            if(architecture[l - 1].type == Layer_type::Conv){
                std::cout << "Leggendo l'output prima dell'attivazione";
                runtime[l - 1].a.copy_to_host(observed.data(), observed.size());
                int filter_go = 1;
                while(filter_go){
                    int f = read_bounded_int("Quale canale si vuole osservare", 1, architecture[l - 1].dim_layer[2]);
                    for(int i = 0; i < architecture[l - 1].dim_layer[0]; i++){
                        for(int j = 0; j < architecture[l - 1].dim_layer[1]; j++){
                            std:: cout << observed[(i * architecture[l - 1].dim_layer[1] + j) * architecture[l - 1].dim_layer[2] + f -1] << "  ";
                        }
                        std::cout << std::endl;
                    }
                    int stamp = read_bounded_int("Salvare questo canale come immagine? (0 = no, 1 = si)", 0, 1);
                    if(stamp){
                        std::string output_path_str;
                        std::cout << "Inserire il percorso PNG di output" << std::endl;
                        std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
                        std::getline(std::cin, output_path_str);
                        require_condition(!is_blank_string(output_path_str), "Percorso output vuoto");
                        save_activation_channel_png(
                            output_path_str, observed,
                            architecture[l - 1].dim_layer[0], architecture[l - 1].dim_layer[1], architecture[l - 1].dim_layer[2],
                            f - 1
                        );
                        std::cout << "Immagine salvata in: " << output_path_str << std::endl;
                    }

                    filter_go = read_bounded_int("Osservare un altro canale? (0 = no, 1 = si)", 0, 1);
                }
            }
            else if(architecture[l - 1].type == Layer_type::Pooling){
                int channel_go = 1;
                while(channel_go){
                    int ch = read_bounded_int("Quale canale si vuole osservare", 1, architecture[l - 1].dim_layer[2]);
                    for(int i = 0; i < architecture[l - 1].dim_layer[0]; i++){
                        for(int j = 0; j < architecture[l - 1].dim_layer[1]; j++){
                            std:: cout << observed[(i * architecture[l - 1].dim_layer[1] + j) * architecture[l - 1].dim_layer[2] + ch - 1] << "  ";
                        }
                        std::cout << std::endl;
                    }

                    channel_go = read_bounded_int("Osservare un altro canale? (0 = no, 1 = si)", 0, 1);
                }
            }

            else{
                for(int i = 0; i < architecture[l -1].flat_output_size(); i++){
                    std:: cout << observed[i] << std::endl;
                }
            }

            obs = read_bounded_int("Continuare a osservare altri layer? (0 = no, 1 = si)", 0, 1);

        }

        const int flat_size = architecture[num_layers - 1].flat_output_size();
        std::vector<float> output_values(static_cast<std::size_t>(flat_size));
        runtime[num_layers - 1].y.copy_to_host(output_values.data(), output_values.size());

        int predicted_class = 0;
        float best_value = output_values[0];
        for(int c = 1; c < flat_size; c++){
            const float value = output_values[static_cast<std::size_t>(c)];
            if(value > best_value){
                best_value = value;
                predicted_class = c;
            }
        }
        const float confidence = output_values[static_cast<std::size_t>(predicted_class)];

        std::cout << "Classe predetta (indice): " << predicted_class << std::endl;
        std::cout << "Confidenza: " << confidence << std::endl;

        if(!class_names.empty() && static_cast<int>(class_names.size()) == flat_size){
            std::cout << "Classe predetta (nome): " << class_names[predicted_class] << std::endl;
        }

        go = read_bounded_int("Continuare? (0 = no, 1 = si)", 0, 1);
        if(go){
            std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
        }
}
}

} // namespace

int main(){
    try{
        const int mode = read_bounded_int("Selezionare modalita': 1=Training 2=Inference", 1, 2, "Scelta non valida");
        if(mode == 1){
            run_training_mode();
        }
        else{
            run_inference_mode();
        }
    }
    catch(const std::exception &ex){
        std::cerr << "Errore: " << ex.what() << std::endl;
        return 1;
    }

    return 0;
}
