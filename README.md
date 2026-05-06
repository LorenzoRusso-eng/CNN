# CNN CUDA

Implementazione in C++20/CUDA di una rete neurale convoluzionale configurabile da CLI, con training, resume da snapshot e inference su immagini.

Il progetto usa CMake, CUDA Runtime e cuBLAS. L'eseguibile finale e' configurato di default con nome `NN_op`.

## Funzionalita'

- Creazione interattiva dell'architettura di rete.
- Layer supportati: Input, Convolution, LRN, Pooling, Flatten, Dense e Softmax.
- Training con:
  - batch;
  - mini-batch SGD;
  - online SGD.
- Strategie di valutazione:
  - hold-out;
  - k-fold cross validation;
  - full training.
- Momentum e Nesterov Accelerated Gradient.
- Learning rate decay: costante, esponenziale, time-based, step e cosine annealing.
- Salvataggio di snapshot del modello e snapshot completi di training.
- Resume del training da snapshot per hold-out e full training.
- Inference su singola immagine.
- Report prestazioni in Markdown con metriche e matrice di confusione.

## Requisiti

- CMake 3.20 o superiore.
- Compilatore C++ con supporto C++20.
- NVIDIA CUDA Toolkit.
- GPU NVIDIA compatibile con CUDA.
- cuBLAS, incluso nel CUDA Toolkit.

Su Windows e' consigliato usare Visual Studio Build Tools con toolchain MSVC e `nvcc` disponibili nel prompt di sviluppo.

## Struttura Del Progetto

```text
app/          Entry point dell'applicazione CLI
cli/          Prompt e builder interattivo dell'architettura
core/         Definizioni base, layer, validazione e backend CUDA
engine/       Forward, backward, gradienti e ottimizzazione
evaluation/   Metriche e valutazione del modello
IO/           Caricamento dataset/immagini, snapshot e report
kernels/      Kernel CUDA per operazioni CNN
math/         Attivazioni, loss e decadimento del learning rate
shared/       Stato globale condiviso della rete
training/     Loop di training, optimizer, progressi e metadata
```

## Build

### Windows Con Visual Studio Build Tools

Nel repository sono inclusi due script batch per configurare e compilare con MSVC/CUDA.

```bat
run_vs_and_cmake.bat
run_vs_and_build.bat
```

Gli script generano la build in `build_cuda_verify/` e compilano in configurazione `Release`.

In alternativa, da un Developer Command Prompt:

```bat
cmake -S . -B build_cuda_verify -DNN_ENABLE_LTO=OFF
cmake --build build_cuda_verify --config Release --parallel
```

L'eseguibile viene prodotto come:

```text
build_cuda_verify\Release\NN_op.exe
```

### Build CMake Generica

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
```

Opzioni CMake utili:

```text
NN_BINARY_NAME        Nome del binario prodotto, default: NN_op
NN_ENABLE_NATIVE_ARCH Abilita ottimizzazioni native CPU, default: ON
NN_ENABLE_LTO         Abilita link-time optimization, default: ON
NN_ENABLE_WARNINGS    Abilita warning del compilatore, default: ON
NN_ENABLE_MSVC_AVX2   Abilita /arch:AVX2 con MSVC, default: ON
```

## Uso

Avviare l'eseguibile:

```bat
build_cuda_verify\Release\NN_op.exe
```

All'avvio il programma chiede:

```text
Selezionare modalita': 1=Training 2=Inference
```

### Training

In modalita' training si puo' scegliere tra:

```text
1 = Nuovo training
2 = Resume da training snapshot
```

Per un nuovo training il programma richiede:

- nome del modello;
- percorso del dataset;
- architettura della rete;
- attivazioni;
- loss;
- learning rate e decay;
- tipo di training;
- eventuale momentum/Nesterov;
- metodo di training.

### Inference

In modalita' inference il programma richiede:

- percorso dello snapshot del modello;
- percorso dell'immagine da classificare.

L'immagine viene ridimensionata/caricata secondo la shape attesa dallo snapshot. Il programma stampa indice della classe predetta, confidenza e, se disponibili, nome della classe.

## Formato Del Dataset

Il dataset deve essere organizzato in sottocartelle, una per classe:

```text
dataset/
  classe_1/
    immagine_001.png
    immagine_002.jpg
  classe_2/
    immagine_003.png
    immagine_004.jpg
```

Estensioni supportate:

```text
.png, .jpg, .jpeg, .bmp, .tga
```

Tutte le immagini del dataset devono avere la stessa altezza e larghezza. Le classi vengono lette dai nomi delle sottocartelle e ordinate alfabeticamente.

## File Generati

Dopo il training vengono creati file nella directory di esecuzione:

```text
network_performance_report_<nome_modello>.md
trained_model_<nome_modello>_snapshot.txt
trained_model_<nome_modello>_training_snapshot.txt
```

Il file `trained_model_<nome_modello>_snapshot.txt` contiene architettura e pesi ed e' usato per l'inference.

Il file `trained_model_<nome_modello>_training_snapshot.txt` contiene anche i metadata necessari per riprendere il training.

## Resume Del Training

Il resume richiede uno snapshot completo di training:

```text
trained_model_<nome_modello>_training_snapshot.txt
```

Il dataset viene ricostruito dal manifest salvato nello snapshot, quando disponibile. In caso contrario, il programma richiede nuovamente il percorso del dataset e verifica compatibilita' di shape e classi.

Il resume e' supportato per:

- hold-out;
- full training.

## Note Per Git

Le directory di build e gli artefatti compilati sono esclusi da `.gitignore`. Prima di pubblicare il repository e' consigliato versionare solo sorgenti, configurazione CMake, script utili e questo README.
