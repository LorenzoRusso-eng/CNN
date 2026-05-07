#include "cli/architecture_builder.hpp"

// Questo file contiene il builder CLI dei layer dell'architettura.

#include "cli/cli_utils.hpp"

#include <iostream>
#include <stdexcept>

void create_architecture(int &num_layers, int input_shape[3], int num_classes, LayerList &architecture){
    int i;
    std::cout << "Il layer di input ha dimensioni " << input_shape[0] << "x" << input_shape[1] << "x" << input_shape[2] << std::endl;
    std::cout << "Il layer di output ha dimensione " << num_classes << std::endl;
    num_layers = read_bounded_int("Scegliere il numero di layer dell'architettura (almeno 2 - il primo layer è di input e preinpostato)", 2);

    const bool use_softmax = (num_classes > 1) && read_bounded_int("Aggiungere un layer Softmax finale? (0 = no, 1 = sì)", 0, 1);
    if(use_softmax){
        num_layers += 1;
    }
    architecture.resize(num_layers);
    const int output_dense_index = use_softmax ? num_layers - 2 : num_layers - 1;

    architecture[0].dim_layer[0] = input_shape[0];
    architecture[0].dim_layer[1] = input_shape[1];
    architecture[0].dim_layer[2] = input_shape[2];
    architecture[0].init_input(architecture[0].dim_layer);

    for(i=1; i<output_dense_index; i++){
        const Layer &previous = architecture[i - 1];
        int layer_choice = 0;

        layer_choice = read_bounded_int("Scegliere il tipo del layer " + std::to_string(i + 1) + ": 1=Conv 2=Pooling 3=Flatten 4=Dense", 1, 4);

        switch(layer_choice){
            case 1: {
                if(previous.type == Layer_type::Flatten){
                    throw std::invalid_argument("Non e' possibile aggiungere un layer Conv dopo un Flatten");
                }

                int kernel[3] = {0, 0, previous.dim_layer[2]};
                int stride[2] = {1, 1};
                int padding[2] = {0, 0};
                int out_dim[3] = {0, 0, 0};

                kernel[0] = read_bounded_int("Altezza kernel convoluzionale", 1);
                kernel[1] = read_bounded_int("Larghezza kernel convoluzionale", 1);
                stride[0] = read_bounded_int("Stride verticale", 1);
                stride[1] = read_bounded_int("Stride orizzontale", 1);
                padding[0] = read_bounded_int("Padding verticale", 0);
                padding[1] = read_bounded_int("Padding orizzontale", 0);
                out_dim[2] = read_bounded_int("Numero di filtri in output", 1);

                out_dim[0] = compute_spatial_output_dim(previous.dim_layer[0], kernel[0], stride[0], padding[0], "builder conv", "height");
                out_dim[1] = compute_spatial_output_dim(previous.dim_layer[1], kernel[1], stride[1], padding[1], "builder conv", "width");

                architecture[i].init_conv(out_dim, architecture[i - 1].dim_layer, kernel, stride, padding);
                break;
            }

            case 2: {
                if(previous.type == Layer_type::Flatten){
                    throw std::invalid_argument("Non e' possibile aggiungere un layer Pooling dopo un Flatten");
                }

                int pool_window[3] = {0, 0, previous.dim_layer[2]};
                int stride[2] = {1, 1};
                int padding[2] = {0, 0};
                int out_dim[3] = {0, 0, previous.dim_layer[2]};
                const Pooling_type pooling_type = read_pooling_type();

                pool_window[0] = read_bounded_int("Altezza finestra di pooling", 1);
                pool_window[1] = read_bounded_int("Larghezza finestra di pooling", 1);
                stride[0] = read_bounded_int("Stride verticale pooling", 1);
                stride[1] = read_bounded_int("Stride orizzontale pooling", 1);
                padding[0] = read_bounded_int("Padding verticale pooling", 0);
                padding[1] = read_bounded_int("Padding orizzontale pooling", 0);

                out_dim[0] = compute_spatial_output_dim(
                    previous.dim_layer[0],
                    pool_window[0],
                    stride[0],
                    padding[0],
                    "builder pooling",
                    "height"
                );
                out_dim[1] = compute_spatial_output_dim(
                    previous.dim_layer[1],
                    pool_window[1],
                    stride[1],
                    padding[1],
                    "builder pooling",
                    "width"
                );

                architecture[i].init_pooling(
                    out_dim,
                    architecture[i - 1].dim_layer,
                    pool_window,
                    stride,
                    padding,
                    pooling_type
                );
                break;
            }

            case 3: {
                if(previous.type == Layer_type::Flatten){
                    throw std::invalid_argument("Non e' possibile applicare Flatten due volte di fila");
                }

                int out_dim[3] = {
                    previous.dim_layer[0] * previous.dim_layer[1] * previous.dim_layer[2],
                    1,
                    1
                };
                
                architecture[i].init_flatten(out_dim, architecture[i - 1].dim_layer);
                break;
            }

            case 4: {
                int out_dim[3] = {0, 1, 1};
                out_dim[0] = read_bounded_int("Numero di neuroni del layer Dense", 1);
                architecture[i].init_dense(out_dim, architecture[i - 1].dim_layer);
                break;
            }
        }
    }

    architecture[output_dense_index].dim_layer[0] = num_classes;
    architecture[output_dense_index].dim_layer[1] = 1;
    architecture[output_dense_index].dim_layer[2] = 1;
    architecture[output_dense_index].init_dense(
        architecture[output_dense_index].dim_layer,
        architecture[output_dense_index - 1].dim_layer
    );

    if(use_softmax){
        architecture[num_layers - 1].dim_layer[0] = num_classes;
        architecture[num_layers - 1].dim_layer[1] = 1;
        architecture[num_layers - 1].dim_layer[2] = 1;
        architecture[num_layers - 1].init_softmax(
            architecture[num_layers - 1].dim_layer,
            architecture[output_dense_index].dim_layer
        );
    }
}
