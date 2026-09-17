# Checkpoint 3 architecture comparison

All checkpoint-3 variants use the same split, RGB augmentation, input size, training budget, optimizer, scheduler, and regularization. Checkpoint 2 is included as the ResNet-18 reference.

| Rank | Architecture | Val. accuracy | Macro F1 | Parameters | MACs/image | Train time | Peak GPU |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | resnext50_32x4d | 78.33% | 77.90% | 23.01M | 1380.7M | 250.9s | 1575.6 MiB |
| 2 | densenet121 | 76.88% | 76.61% | 6.97M | 925.1M | 212.5s | 1485.8 MiB |
| 3 | resnet18 (checkpoint2) | 76.46% | 76.05% | 11.18M | 592.2M | 196.0s | 488.0 MiB |
| 4 | efficientnet_b0 | 72.08% | 72.26% | 4.03M | 126.0M | 149.4s | 989.1 MiB |
| 5 | convnext_tiny | 61.88% | 61.47% | 27.83M | 1454.6M | 331.2s | 1814.3 MiB |
| 6 | mobilenet_v3_large | 6.25% | 0.74% | 4.22M | 72.2M | 41.0s | 590.4 MiB |

The comparison is a controlled recipe comparison, not an exhaustive hyperparameter search. A model may rank lower because the common optimizer or schedule is less suitable for that architecture.
