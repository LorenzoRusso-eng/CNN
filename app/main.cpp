#include "cli/architecture_builder.hpp"
#include "cli/cli_utils.hpp"
#include "evaluation/evaluation.hpp"
#include "engine/forward.hpp"
#include "training/training.hpp"
#include "IO/dataset_io.hpp"
#include "IO/image_io.hpp"
#include "IO/model_io.hpp"
#include "IO/report_io.hpp"
#include "shared/network_globals.hpp"
#include "training/training_method.hpp"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

namespace fs = std::filesystem;

const Decay &configure_decay_from_training_snapshot(const TrainingSnapshotMetadata &snapshot){
    switch(snapshot.decay_kind){
        case DecayKind::Constant:
            constant_decay.set_initial_lr(snapshot.initial_learning_rate);
            return constant_decay;
        case DecayKind::Exponential:
            exponential_decay.set_initial_lr(snapshot.initial_learning_rate);
            exponential_decay.set_decay_rate(snapshot.decay_rate);
            return exponential_decay;
        case DecayKind::TimeBased:
            time_based_decay.set_initial_lr(snapshot.initial_learning_rate);
            time_based_decay.set_decay_rate(snapshot.decay_rate);
            return time_based_decay;
        case DecayKind::Step:
            step_decay.set_initial_lr(snapshot.initial_learning_rate);
            step_decay.set_decay_rate(snapshot.decay_rate);
            step_decay.set_step_size(snapshot.decay_step_size);
            return step_decay;
        case DecayKind::CosineAnnealing:
            cosine_annealing.set_initial_lr(snapshot.initial_learning_rate);
            cosine_annealing.set_final_lr(snapshot.cosine_final_lr);
            cosine_annealing.set_max_epoch(snapshot.cosine_max_epoch);
            return cosine_annealing;
    }
    throw std::invalid_argument("Decay snapshot non supportato");
}

const Loss &configure_loss_from_training_snapshot(const TrainingSnapshotMetadata &snapshot){
    switch(snapshot.loss_kind){
        case LossKind::Simple:
            simple_loss.beta = snapshot.loss_beta;
            simple_loss.set_reduction(snapshot.loss_reduction);
            return simple_loss;
        case LossKind::L1:
            l1_loss.beta = snapshot.loss_beta;
            l1_loss.set_reduction(snapshot.loss_reduction);
            return l1_loss;
        case LossKind::L2:
            l2_loss.beta = snapshot.loss_beta;
            l2_loss.set_reduction(snapshot.loss_reduction);
            return l2_loss;
        case LossKind::SmoothL1:
            smooth_l1_loss.beta = snapshot.loss_beta;
            smooth_l1_loss.set_reduction(snapshot.loss_reduction);
            return smooth_l1_loss;
        case LossKind::Huber:
            huber_loss.beta = snapshot.loss_beta;
            huber_loss.set_reduction(snapshot.loss_reduction);
            return huber_loss;
        case LossKind::CrossEntropy:
            cross_entropy.beta = snapshot.loss_beta;
            cross_entropy.set_reduction(snapshot.loss_reduction);
            return cross_entropy;
        case LossKind::LL:
            ll_loss.beta = snapshot.loss_beta;
            ll_loss.set_reduction(snapshot.loss_reduction);
            return ll_loss;
    }
    throw std::invalid_argument("Loss snapshot non supportata");
}

void validate_dataset_compatibility_with_architecture(const LayerList &architecture, const int dim_input[3], int dim_output){
    require_condition(!architecture.empty(), "Architettura snapshot vuota");
    require_condition(
        architecture[0].dim_layer[0] == dim_input[0] &&
        architecture[0].dim_layer[1] == dim_input[1] &&
        architecture[0].dim_layer[2] == dim_input[2],
        "Il dataset caricato non e' compatibile con la shape di input dello snapshot"
    );
    require_condition(
        architecture.back().dim_layer[0] == dim_output,
        "Il dataset caricato non e' compatibile con il numero classi dello snapshot"
    );
}

void validate_dataset_manifest_consistency(const std::vector<std::string> &expected_manifest, const std::vector<std::string> &runtime_manifest){
    if(expected_manifest.empty()){
        return;
    }

    require_condition(
        expected_manifest.size() == runtime_manifest.size(),
        "Il dataset caricato non corrisponde al manifest salvato nello snapshot (numero file differente)"
    );

    for(std::size_t index = 0; index < expected_manifest.size(); index++){
        if(expected_manifest[index] != runtime_manifest[index]){
            throw std::invalid_argument(
                "Il dataset caricato non corrisponde al manifest salvato nello snapshot (ordine o file differente all'indice " + std::to_string(index) + ")"
            );
        }
    }
}

void validate_class_names_consistency(const std::vector<std::string> &snapshot_class_names, const std::vector<std::string> &runtime_class_names){
    if(runtime_class_names.empty()){
        return;
    }

    require_condition(
        snapshot_class_names.size() == runtime_class_names.size(),
        "Il dataset caricato non corrisponde alle classi salvate nello snapshot (numero classi differente)"
    );

    for(std::size_t index = 0; index < snapshot_class_names.size(); index++){
        if(snapshot_class_names[index] != runtime_class_names[index]){
            throw std::invalid_argument(
                "Il dataset caricato non corrisponde alle classi salvate nello snapshot (classe differente all'indice " + std::to_string(index) + ")"
            );
        }
    }
}

std::string build_resume_model_name(const std::string &snapshot_model_name){
    const std::string marker = "_resume";
    std::string base_name = snapshot_model_name;
    int resume_round = 1;

    const std::size_t marker_pos = snapshot_model_name.rfind(marker);
    if(marker_pos != std::string::npos){
        const std::size_t suffix_pos = marker_pos + marker.size();
        const std::string suffix = snapshot_model_name.substr(suffix_pos);
        if(!suffix.empty() && std::all_of(suffix.begin(), suffix.end(), [](unsigned char ch){ return std::isdigit(ch) != 0; })){
            base_name = snapshot_model_name.substr(0, marker_pos);
            resume_round = std::stoi(suffix) + 1;
        }
    }

    return base_name + marker + std::to_string(resume_round);
}

int choose_mode(){
    int mode = 0;
    do{
        std::cout << "Selezionare modalita': 1=Training 2=Inference" << std::endl;
        std::cin >> mode;
        if(!std::cin.good() || (mode != 1 && mode != 2)){
            std::cout << "Scelta non valida." << std::endl;
            std::cin.clear();
            std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
        }
    } while(!std::cin.good() || (mode != 1 && mode != 2));
    return mode;
}

void run_training_mode(){
    const int training_mode_choice = read_bounded_int("Scegliere la modalita' di training: 1=Nuovo training 2=Resume da training snapshot", 1, 2);

    if(training_mode_choice == 2){
        std::string snapshot_path_str;
        std::cout << "Inserire il percorso del training snapshot completo" << std::endl;
        std::cin >> snapshot_path_str;

        LayerList architecture;
        std::vector<std::string> class_names;
        std::string hidden_activation_name;
        std::string output_activation_name;
        TrainingSnapshotMetadata resume_snapshot{};
        load_training_snapshot(snapshot_path_str, architecture, class_names, hidden_activation_name, output_activation_name, resume_snapshot);
        const int num_layers = static_cast<int>(architecture.size());
        validate_architecture(architecture, num_layers, "Resume training");
        require_condition(
            !resume_snapshot.training_finalized,
            "Resume non consentito: lo snapshot e' stato marcato come training finalizzato (stop intra-epoca)."
        );

        // Consuma il newline residuo lasciato da operator>> prima di usare getline in get_example.
        std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

        int dim_input[3] = {0, 0, 0};
        int dim_output = 0;
        int num_examples = 0;
        Dataset4D input;
        Dataset4D output;
        std::vector<std::string> runtime_class_names;
        std::vector<std::string> dataset_manifest_paths;
        get_example(dim_input, dim_output, num_examples, input, output, runtime_class_names, &dataset_manifest_paths);
        validate_dataset_compatibility_with_architecture(architecture, dim_input, dim_output);
        validate_dataset_manifest_consistency(resume_snapshot.dataset_manifest_paths, dataset_manifest_paths);
        validate_class_names_consistency(class_names, runtime_class_names);

        const std::string model_name = build_resume_model_name(resume_snapshot.model_name);
        std::cout << "Nome modello resume generato automaticamente: " << model_name << std::endl;

        const int additional_epochs = read_bounded_int(
            "Quante epoche aggiuntive vuoi eseguire nel resume? (minimo 1)",
            1,
            std::numeric_limits<int>::max(),
            "Inserire un intero >= 1."
        );
        const int target_total_epochs = resume_snapshot.completed_epochs + additional_epochs;

        if(resume_snapshot.decay_kind == DecayKind::CosineAnnealing && resume_snapshot.cosine_max_epoch < target_total_epochs){
            resume_snapshot.cosine_max_epoch = target_total_epochs;
        }
        resume_snapshot.requested_epochs = target_total_epochs;

        const Activation &hidden_activation = activation_from_snapshot_name(hidden_activation_name);
        const Activation &output_activation = activation_from_snapshot_name(output_activation_name);
        const Loss &loss = configure_loss_from_training_snapshot(resume_snapshot);
        const Decay &learning_rate_decay = configure_decay_from_training_snapshot(resume_snapshot);

        switch(resume_snapshot.training_method){
            case 1:
                hold_out_resume(
                    model_name, num_examples,
                    architecture, num_layers,
                    learning_rate_decay, target_total_epochs,
                    input, output,
                    loss, hidden_activation, output_activation,
                    class_names, resume_snapshot, dataset_manifest_paths
                );
                break;
            case 3:
                full_training_resume(
                    model_name, num_examples,
                    architecture, num_layers,
                    learning_rate_decay, target_total_epochs,
                    input, output,
                    loss, hidden_activation, output_activation,
                    class_names, resume_snapshot, dataset_manifest_paths
                );
                break;
            default:
                throw std::invalid_argument("Resume supportato solo per Hold-out e Full-training in questa versione.");
        }
        return;
    }

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
    int k_folds;

    Dataset4D input;
    Dataset4D output;
    std::vector<std::string> class_names;
    std::vector<std::string> dataset_manifest_paths;

    std::string model_name;

    std::cout << "Inserire il nome del modello" << std::endl;
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    std::getline(std::cin, model_name);

    get_example(dim_input, dim_output, num_examples, input, output, class_names, &dataset_manifest_paths);

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
    training_type = choose_training_type();
    training_method = choose_training_method();

    momentum_choice = read_bounded_int("Usare il momento per accelerare il training? (0 = no, 1 = si)", 0, 1);
    if(momentum_choice){
        momentum = read_bounded_float("Scegliere il valore del momento (tra 0 e 1)", 0.0f, 1.0f, "Inserire un valore compreso tra 0 e 1.");
        if(momentum != 0.0f){
            use_nesterov = read_bounded_int("Usare Nesterov Accelerated Gradient? (0 = no, 1 = si)", 0, 1);
        }
    }

    switch(training_method){
    case 1:
        hold_out_ratio = read_bounded_float(
            "Scegliere il rapporto di training per l'hold-out in forma decimale (maggiore di 0 e minore di 1)",
            std::numeric_limits<float>::epsilon(),
            1.0f - std::numeric_limits<float>::epsilon(),
            "Inserire un valore decimale compreso tra 0 e 1, estremi esclusi."
        );
        hold_out(
            model_name, num_examples,
            training_type, use_nesterov,
            architecture, num_layers,
            learning_rate_decay, num_epochs, target_loss,
            input, output,
            loss, hidden_activation, output_activation,
            momentum, hold_out_ratio,
            class_names, dataset_manifest_paths
        );
        break;
    case 2:
        k_folds = read_bounded_int(
            "Scegliere il numero di fold K per la validazione incrociata (minimo: 2)",
            2,
            std::numeric_limits<int>::max(),
            "Inserire un intero maggiore o uguale a 2."
        );
        k_fold(
            model_name, k_folds, num_examples,
            training_type, use_nesterov,
            architecture, num_layers,
            learning_rate_decay, num_epochs, target_loss,
            input, output,
            loss, hidden_activation, output_activation,
            momentum,
            class_names
        );
        break;
    case 3:
        full_training(
            model_name, num_examples,
            training_type, use_nesterov,
            architecture, num_layers,
            learning_rate_decay, num_epochs, target_loss,
            input, output,
            loss, hidden_activation, output_activation,
            momentum,
            class_names, dataset_manifest_paths
        );
        break;
    default:
        throw std::invalid_argument("Tipo di training non valido");
    }
}

void run_inference_mode(){
    std::string snapshot_path_str;
    std::cout << "Inserire il percorso del file snapshot della rete" << std::endl;
    std::cin >> snapshot_path_str;

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

    std::string image_path_str;
    std::cout << "Inserire il percorso dell'immagine da classificare" << std::endl;
    std::cin >> image_path_str;

    const Tensor3D image = load_01scaled_image_tensor(image_path_str, input_h, input_w, input_c);
    validate_tensor3d_shape(image, architecture[0].dim_layer, "Inference image");
    RuntimeList runtime;
    init_runtime_buffers(architecture, runtime);
    feed_input(image, architecture[0], runtime[0]);

    require_condition(!hidden_activation_name.empty(), "Snapshot senza attivazione hidden salvata");
    require_condition(!output_activation_name.empty(), "Snapshot senza attivazione output salvata");
    const Activation &hidden_activation = activation_from_snapshot_name(hidden_activation_name);
    const Activation &output_activation = activation_from_snapshot_name(output_activation_name);
    std::cout << "Attivazioni caricate dallo snapshot: hidden=" << hidden_activation_name << " output=" << output_activation_name << std::endl;

    forwardprop(architecture, runtime, num_layers, hidden_activation, output_activation);

    const int predicted_class = argmax_output(architecture[num_layers - 1], runtime[num_layers - 1]);
    const Layer &output_layer = architecture[num_layers - 1];
    const auto &output_values = runtime_output_buffer(output_layer, runtime[num_layers - 1]);
    const float confidence = output_values[static_cast<std::size_t>(predicted_class)];

    std::cout << "Classe predetta (indice): " << predicted_class << std::endl;
    std::cout << "Confidenza: " << confidence << std::endl;

    if(!class_names.empty() && static_cast<int>(class_names.size()) == architecture[num_layers - 1].dim_layer[0]){
        std::cout << "Classe predetta (nome): " << class_names[predicted_class] << std::endl;
    }
}

} // namespace

int main(){
    try{
        const int mode = choose_mode();
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
