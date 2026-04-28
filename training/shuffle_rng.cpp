#include "training/shuffle_rng.hpp"

#include <algorithm>
#include <random>
#include <sstream>
#include <stdexcept>

namespace {

std::mt19937 &shuffle_engine(){
    static std::mt19937 gen(std::random_device{}());
    return gen;
}

} // namespace

namespace training_shuffle {

void shuffle_vec(std::vector<int> &vec){
    std::shuffle(vec.begin(), vec.end(), shuffle_engine());
}

std::string serialize_rng_state(){
    std::ostringstream out;
    out << shuffle_engine();
    return out.str();
}

void restore_rng_state(const std::string &state){
    std::istringstream in(state);
    in >> shuffle_engine();
    if(!in.good() && !in.eof()){
        throw std::invalid_argument("restore_rng_state: stato RNG non valido");
    }
}

} // namespace training_shuffle
