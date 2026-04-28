#pragma once

// Questo file contiene la costruzione interattiva dell'architettura della rete.

#include "core/layer.hpp"

void create_architecture(int &num_layers, int input_shape[3], int num_classes, LayerList &architecture);
