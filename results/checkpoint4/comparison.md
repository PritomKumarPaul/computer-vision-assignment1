# Checkpoint 4 ResNeXt transfer-learning comparison

All four runs use ResNeXt-50 32x4d, the same leakage-controlled split, 224x224 RGB inputs, ImageNet normalization, augmentation, seed, batch size, regularization, scheduler, and maximum epoch budget. Learning rates are selected for each optimization scope and are recorded below.

| Rank | Setup | LR | Val. accuracy | Macro F1 | Trainable / total parameters | Trainable | Train time | Peak GPU |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | Pretrained: layer4 + head | 3.0e-04 | 94.79% | 94.76% | 14.578M / 23.013M | 63.35% | 152.6s | 501.1 MiB |
| 2 | Pretrained: all layers | 1.0e-04 | 94.38% | 94.38% | 23.013M / 23.013M | 100.00% | 128.2s | 2208.9 MiB |
| 3 | Pretrained: head only | 1.0e-03 | 94.17% | 94.17% | 0.033M / 23.013M | 0.14% | 138.4s | 387.3 MiB |
| 4 | Scratch: all layers | 1.0e-03 | 82.50% | 82.22% | 23.013M / 23.013M | 100.00% | 343.9s | 2206.6 MiB |

The convolutional MAC count is nearly identical across setups at inference time. Freezing changes gradient computation, optimizer state, training memory, and training time; it does not remove backbone operations from inference.

The labeled test set is intentionally excluded from this comparison.
