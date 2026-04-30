# Performance Report - Ciao (Hold-out)

## General Information
| Field | Value |
|---|---|
| Model | Ciao |
| Run Type | Hold-out |
| Dataset | Dataset caricato da CLI |
| Total Examples | 15339 |
| Train Examples | 10738 |
| Test Examples | 4601 |
| Num Classes | 7 |
| Requested Epochs | 1 |
| Hold-out Ratio | 0.7000000 |

## Network Configuration
| Layer | Type | Input Dim | Output Dim | Details |
|---:|---|---|---|---|
| 0 | Input | 0x0x0 | 100x100x3 | - |
| 1 | Conv | 100x100x3 | 100x100x16 | kernel=3x3x3, stride=1x1, padding=1x1, params=448 |
| 2 | Conv | 100x100x16 | 100x100x16 | kernel=3x3x16, stride=1x1, padding=1x1, params=2320 |
| 3 | Pooling | 100x100x16 | 50x50x16 | type=Max, kernel=2x2x16, stride=2x2, padding=0x0 |
| 4 | Conv | 50x50x16 | 50x50x32 | kernel=3x3x16, stride=1x1, padding=1x1, params=4640 |
| 5 | Conv | 50x50x32 | 50x50x32 | kernel=3x3x32, stride=1x1, padding=1x1, params=9248 |
| 6 | Pooling | 50x50x32 | 25x25x32 | type=Max, kernel=2x2x32, stride=2x2, padding=0x0 |
| 7 | Conv | 25x25x32 | 25x25x64 | kernel=3x3x32, stride=1x1, padding=1x1, params=18496 |
| 8 | Pooling | 25x25x64 | 13x13x64 | type=Max, kernel=3x3x64, stride=2x2, padding=1x1 |
| 9 | Flatten | 13x13x64 | 10816x1x1 | - |
| 10 | Dense | 10816x1x1 | 128x1x1 | in=10816, out=128, params=1384576 |
| 11 | Dense | 128x1x1 | 64x1x1 | in=128, out=64, params=8256 |
| 12 | Dense | 64x1x1 | 7x1x1 | in=64, out=7, params=455 |
| 13 | Softmax | 7x1x1 | 7x1x1 | - |

## Training Configuration
| Parameter | Value |
|---|---|
| Training Variant | Mini-batch SGD |
| Batch Size | 200 |
| Nesterov | Yes |
| Momentum | 0.9000000 |
| Target Loss | -1.0000000 |
| Learning Rate Decay | Constant |
| Initial Learning Rate | 0.0050000 |
| Hidden Activation | LeakyReLU |
| Output Activation | Identity |
| Loss | LL |
| Loss Reduction | Mean |

## Training Results
| Metric | Value |
|---|---:|
| Executed Epochs | 1 |
| Final Loss | 1.6653239 |
| Total Training Seconds | 187.3856234 |
| Avg Epoch Seconds | 187.3830750 |

### Epoch Times
| Epoch | Seconds |
|---:|---:|
| 1 | 187.3830750 |

## Test Results
| Metric | Value |
|---|---:|
| Test Examples | 4601 |
| Correct Predictions | 1794 |
| Accuracy | 0.3899152 |

## Confusion Matrix
Rows: true class, Columns: predicted class

| True \ Pred | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 0 | 0 | 483 | 0 | 0 | 3 |
| 1 | 0 | 0 | 0 | 105 | 1 | 0 | 0 |
| 2 | 0 | 0 | 0 | 262 | 0 | 0 | 1 |
| 3 | 0 | 0 | 0 | 1786 | 0 | 0 | 1 |
| 4 | 0 | 0 | 0 | 731 | 1 | 0 | 6 |
| 5 | 0 | 0 | 0 | 257 | 1 | 0 | 2 |
| 6 | 0 | 0 | 0 | 954 | 0 | 0 | 7 |

## Metrics by Class
| Class | TP | FP | TN | FN | Precision | Recall | F1 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 0 | 4115 | 486 | 0.0000000 | 0.0000000 | 0.0000000 |
| 1 | 0 | 0 | 4495 | 106 | 0.0000000 | 0.0000000 | 0.0000000 |
| 2 | 0 | 0 | 4338 | 263 | 0.0000000 | 0.0000000 | 0.0000000 |
| 3 | 1786 | 2792 | 22 | 1 | 0.3901267 | 0.9994404 | 0.5611941 |
| 4 | 1 | 2 | 3861 | 737 | 0.3333333 | 0.0013550 | 0.0026991 |
| 5 | 0 | 0 | 4341 | 260 | 0.0000000 | 0.0000000 | 0.0000000 |
| 6 | 7 | 13 | 3627 | 954 | 0.3500000 | 0.0072841 | 0.0142712 |

## Aggregate Metrics
| Metric | Value |
|---|---:|
| Macro Precision | 0.1533514 |
| Macro Recall | 0.1440114 |
| Macro F1 | 0.0825949 |
| Accuracy | 0.3899152 |

