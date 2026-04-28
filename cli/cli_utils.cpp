#include "cli/cli_utils.hpp"

// Questo file contiene l'implementazione delle utility CLI per prompt, selezioni e decay.

#include "shared/network_globals.hpp"

#include <iostream>
#include <limits>

namespace {

// Helper interni per la gestione dell'input e la risoluzione delle scelte

void reset_input_stream(){
    std::cin.clear();
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
}

const Activation& activation_from_choice(int choice){
    switch(choice){
        case 1: return identity;
        case 2: return sigmoid;
        case 3: return tanh_activation;
        case 4: return relu;
        case 5: return leaky_relu;
        case 6: return elu;
        case 7: return softplus;
        case 8: return swish;
        case 9: return mish;
        case 10: return gelu;
        default: return relu;
    }
}

const Decay& decay_from_choice(int choice){
    switch(choice){
        case 1: return constant_decay;
        case 2: return exponential_decay;
        case 3: return time_based_decay;
        case 4: return step_decay;
        case 5: return cosine_annealing;
        default: return constant_decay;
    }
}

int choose_activation_option(const std::string &title, bool show_cross_entropy_hint){
    std::cout << title << std::endl;
    std::cout << "1=Identity 2=Sigmoid 3=Tanh 4=ReLU 5=LeakyReLU 6=ELU 7=Softplus 8=Swish 9=Mish 10=GELU" << std::endl;
    if(show_cross_entropy_hint){
        std::cout << "Vincoli: per Cross_entropy serve un output nell'intervallo [0,1], quindi Sigmoid e' la scelta piu' sicura." << std::endl;
    }
    return read_bounded_int("", 1, 10);
}
 
} // namespace

// Input validato

int read_bounded_int(const std::string &prompt, int min_value, int max_value, const std::string &invalid_message){
    int value = 0;
    do{
        if(!prompt.empty()){
            std::cout << prompt << std::endl;
        }
        std::cin >> value;
        if(!std::cin.good() || value < min_value || value > max_value){
            std::cout << invalid_message << std::endl;
            reset_input_stream();
        }
    } while(!std::cin.good() || value < min_value || value > max_value);

    return value;
}

float read_bounded_float(const std::string &prompt, float min_value, float max_value, const std::string &invalid_message){
    float value = 0.0f;
    do{
        if(!prompt.empty()){
            std::cout << prompt << std::endl;
        }
        std::cin >> value;
        if(!std::cin.good() || value < min_value || value > max_value){
            std::cout << invalid_message << std::endl;
            reset_input_stream();
        }
    } while(!std::cin.good() || value < min_value || value > max_value);

    return value;
}

// Selettori di opzioni semplici

Pooling_type read_pooling_type(){
    const int pooling_choice = read_bounded_int("Scegliere il tipo di pooling: 1=Max 2=Average 3=L2", 1, 3);

    if(pooling_choice == 1){
        return Pooling_type::Max;
    }
    if(pooling_choice == 2){
        return Pooling_type::Average;
    }
    return Pooling_type::L2;
}

Reduction choose_loss_reduction(){
    const int choice = read_bounded_int("Scegliere la reduction della loss: 1=Sum 2=Mean", 1, 2);
    return (choice == 1) ? Reduction::Sum : Reduction::Mean;
}

// Selezione di attivazioni e loss

const Activation& choose_hidden_activation(){
    return activation_from_choice(choose_activation_option("Scegliere la funzione di attivazione dei layer nascosti:", false));
}

const Activation& choose_output_activation(bool final_is_softmax){
    if(final_is_softmax){
        std::cout << "Output finale con Softmax: il Dense precedente usa logits lineari, quindi la funzione di output e' fissata a Identity." << std::endl;
        return identity;
    }

    return activation_from_choice(choose_activation_option("Scegliere la funzione di attivazione del layer di output:", true));
}

const Loss& choose_loss_function(bool final_is_softmax){
    if(final_is_softmax){
        std::cout << "Output finale con Softmax: la loss compatibile e' fissata a LL_loss." << std::endl;
        ll_loss.set_reduction(choose_loss_reduction());
        return ll_loss;
    }

    std::cout << "Scegliere la funzione di loss:" << std::endl;
    std::cout << "1=Simple_loss 2=L1_loss 3=L2_loss 4=Smooth_L1_loss 5=Huber_Loss 6=Cross_entropy" << std::endl;
    std::cout << "Vincoli: Cross_entropy richiede un output nell'intervallo [0,1]." << std::endl;
    const int choice = read_bounded_int("", 1, 6);

    const Reduction reduction = choose_loss_reduction();

    switch(choice){
        case 1: simple_loss.set_reduction(reduction); return simple_loss;
        case 2: l1_loss.set_reduction(reduction); return l1_loss;
        case 3: l2_loss.set_reduction(reduction); return l2_loss;
        case 4: smooth_l1_loss.set_reduction(reduction); return smooth_l1_loss;
        case 5: huber_loss.set_reduction(reduction); return huber_loss;
        case 6: cross_entropy.set_reduction(reduction); return cross_entropy;
        default: simple_loss.set_reduction(reduction); return simple_loss;
    }
}

// Selezione della strategia di decay

const Decay& choose_learning_rate_decay(float initial_lr, int num_epochs){
    std::cout << "Scegliere la strategia di decay del learning rate:" << std::endl;
    std::cout << "1=None 2=Exponential 3=Time-based 4=Step 5=Cosine annealing" << std::endl;
    const int choice = read_bounded_int("", 1, 5);
    Decay &selected_decay = const_cast<Decay&>(decay_from_choice(choice));
    selected_decay.set_initial_lr(initial_lr);

    switch(choice){
        case 1:
            break;
        case 2: {
            const float decay_rate = read_bounded_float("Scegliere il decay rate esponenziale (maggiore o uguale a 0)", 0.0f);
            exponential_decay.set_decay_rate(decay_rate);
            break;
        }
        case 3: {
            const float decay_rate = read_bounded_float("Scegliere il decay rate time-based (maggiore o uguale a 0)", 0.0f);
            time_based_decay.set_decay_rate(decay_rate);
            break;
        }
        case 4: {
            const float decay_rate = read_bounded_float(
                "Scegliere il fattore di step decay (maggiore di 0)",
                std::numeric_limits<float>::epsilon(),
                std::numeric_limits<float>::max(),
                "Inserire un valore maggiore di 0."
            );
            const int step_size = read_bounded_int("Scegliere lo step size del decay (almeno 1 epoca)", 1);
            step_decay.set_decay_rate(decay_rate);
            step_decay.set_step_size(step_size);
            break;
        }
        case 5: {
            const float final_lr = read_bounded_float("Scegliere il learning rate finale del cosine annealing (maggiore o uguale a 0)", 0.0f);
            cosine_annealing.set_final_lr(final_lr);
            cosine_annealing.set_max_epoch(num_epochs);
            break;
        }
        default:
            break;
    }

    return selected_decay;
}
