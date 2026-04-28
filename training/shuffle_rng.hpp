#pragma once

#include <string>
#include <vector>

namespace training_shuffle {

void shuffle_vec(std::vector<int> &indices);
std::string serialize_rng_state();
void restore_rng_state(const std::string &state);

} // namespace training_shuffle
