#include "cli/training_prompts.hpp"

#include "cli/cli_utils.hpp"

#include <stdexcept>

int choose_training_window(int training_type, int max_window){
    int window = 0;
    switch(training_type){
        case 1:
            break;
        case 2:
            window = read_bounded_int("Scegliere la dimensione del batch per il training (tra 2 e il numero di esempi - sconsigliato)", 2, max_window);
            break;
        case 3:
            window = read_bounded_int("Scegliere il numero di step per la media della loss (tra 1 e il numero di esempi - sconsigliato)", 1, max_window);
            break;
        default:
            throw std::invalid_argument("Tipo di training non valido");
    }
    return window;
}
