#pragma once

// Questo file contiene la costruzione interattiva dell'architettura della rete.

#include "core/layer.hpp"

void create_architecture(const Shape3D &input_shape, int num_classes, LayerList &architecture);
