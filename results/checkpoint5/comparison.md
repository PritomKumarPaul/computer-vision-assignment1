# Checkpoint 5 multi-architecture transfer matrix

All checkpoint-5 runs use the same leakage-controlled split, 224x224 RGB pipeline, ImageNet normalization, augmentation, seed, batch size, regularization, scheduler, and maximum epoch budget. Learning rates are adapted to the optimization scope.

The ResNeXt-50 rows are the directly comparable checkpoint-4 results and were not retrained.

## Validation accuracy matrix

| Architecture | Scratch/all | Pretrained/all | Pretrained/last stage | Pretrained/head only |
|---|---:|---:|---:|---:|
| ResNet-18 | 82.92% | 94.58% | 93.75% | 90.62% |
| DenseNet-121 | 83.12% | 93.54% | 94.17% | 92.50% |
| EfficientNet-B0 | 81.25% | 93.54% | 93.54% | 93.33% |
| ConvNeXt-Tiny | 64.79% | 96.04% | 95.00% | 93.33% |
| ResNeXt-50 | 82.50% | 94.38% | 94.79% | 94.17% |

## Detailed results

| Architecture | Setup | Val. accuracy | Macro F1 | Trainable / total | Peak GPU | Train time |
|---|---|---:|---:|---:|---:|---:|
| ConvNeXt-Tiny | Pretrained/all | 96.04% | 96.04% | 27.832M / 27.832M | 2416.5 MiB | 191.2s |
| ConvNeXt-Tiny | Pretrained/last stage | 95.00% | 95.02% | 14.303M / 27.832M | 580.5 MiB | 102.3s |
| ResNeXt-50 | Pretrained/last stage | 94.79% | 94.76% | 14.578M / 23.013M | 501.1 MiB | 152.6s |
| ResNet-18 | Pretrained/all | 94.58% | 94.56% | 11.185M / 11.185M | 617.8 MiB | 170.3s |
| ResNeXt-50 | Pretrained/all | 94.38% | 94.38% | 23.013M / 23.013M | 2208.9 MiB | 128.2s |
| DenseNet-121 | Pretrained/last stage | 94.17% | 94.17% | 2.177M / 6.970M | 259.3 MiB | 141.4s |
| ResNeXt-50 | Pretrained/head only | 94.17% | 94.17% | 0.033M / 23.013M | 387.3 MiB | 138.4s |
| ResNet-18 | Pretrained/last stage | 93.75% | 93.68% | 8.402M / 11.185M | 306.9 MiB | 140.6s |
| DenseNet-121 | Pretrained/all | 93.54% | 93.54% | 6.970M / 6.970M | 2194.8 MiB | 142.1s |
| EfficientNet-B0 | Pretrained/all | 93.54% | 93.48% | 4.028M / 4.028M | 1477.9 MiB | 105.4s |
| EfficientNet-B0 | Pretrained/last stage | 93.54% | 93.45% | 1.150M / 4.028M | 235.3 MiB | 79.7s |
| EfficientNet-B0 | Pretrained/head only | 93.33% | 93.35% | 0.020M / 4.028M | 226.2 MiB | 102.7s |
| ConvNeXt-Tiny | Pretrained/head only | 93.33% | 93.32% | 0.012M / 27.832M | 471.6 MiB | 152.9s |
| DenseNet-121 | Pretrained/head only | 92.50% | 92.56% | 0.016M / 6.970M | 242.7 MiB | 199.0s |
| ResNet-18 | Pretrained/head only | 90.62% | 90.52% | 0.008M / 11.185M | 242.8 MiB | 85.2s |
| DenseNet-121 | Scratch/all | 83.12% | 82.83% | 6.970M / 6.970M | 2194.8 MiB | 328.3s |
| ResNet-18 | Scratch/all | 82.92% | 82.70% | 11.185M / 11.185M | 618.0 MiB | 287.9s |
| ResNeXt-50 | Scratch/all | 82.50% | 82.22% | 23.013M / 23.013M | 2206.6 MiB | 343.9s |
| EfficientNet-B0 | Scratch/all | 81.25% | 81.01% | 4.028M / 4.028M | 1477.9 MiB | 249.9s |
| ConvNeXt-Tiny | Scratch/all | 64.79% | 64.13% | 27.832M / 27.832M | 2416.7 MiB | 400.1s |

Freezing changes training cost, not inference MACs: every setup still executes its complete backbone at inference.

The labeled test set is intentionally excluded from model selection.
