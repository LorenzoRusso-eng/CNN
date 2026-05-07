#include "training/shuffle_rng.hpp"

#include <algorithm>
#include <random>

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

} // namespace training_shuffle
