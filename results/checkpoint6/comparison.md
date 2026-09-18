# Checkpoint 6 ConvNeXt capacity comparison

Both models use ImageNet-1K initialization, full fine-tuning, the same leakage-controlled split, 224x224 RGB inputs, augmentation, optimizer, learning rate, regularization, scheduler, batch size, and model-selection rule.

| Model | Val. accuracy | Macro F1 | Parameters | MACs/image | Peak GPU | Train time | Best epoch |
|---|---:|---:|---:|---:|---:|---:|---:|
| ConvNeXt-Tiny (checkpoint 5) | 96.04% | 96.04% | 27.83M | 4.45G | 2416.5 MiB | 191.2s | 21 |
| ConvNeXt-Large (checkpoint 6) | 96.25% | 96.28% | 196.25M | 34.36G | 9364.6 MiB | 202.3s | 2 |

The labeled test set is intentionally excluded. Select the final model using validation evidence before evaluating the test set once.
