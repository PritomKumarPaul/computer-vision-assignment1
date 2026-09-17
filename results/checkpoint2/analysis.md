# Checkpoint 2 analysis

## Hypothesis

Checkpoint 1 showed severe overfitting. Checkpoint 2 tested whether a combined
recipe—RGB input, a ResNet-18 trained from scratch, augmentation, weight decay,
dropout, label smoothing, cosine scheduling, and early stopping—would improve
generalization.

## Result

| Metric | Checkpoint 1 | Checkpoint 2 | Change |
|---|---:|---:|---:|
| Best validation accuracy | 46.04% | 76.46% | +30.42 pp |
| Validation macro F1 | 45.10% | 76.05% | +30.95 pp |
| Parameters | 57,776 | 11,184,720 | 193.6x |
| MACs per image | 0.611 M | 592.191 M | 969.0x |
| Parameter storage | 0.220 MiB | 42.703 MiB | 193.8x |
| Peak GPU memory | 66.36 MiB | 488.03 MiB | 7.4x |
| Total training time | 36.11 s | 195.99 s | 5.4x |

The best checkpoint occurred at epoch 49. Validation loss was noisy early in
training but stabilized around 1.18 late in the cosine schedule. Final training
accuracy was 99.53% and final validation accuracy was 76.25%, leaving a 23.28
percentage-point generalization gap. Overfitting therefore remains, but it is
substantially less damaging than checkpoint 1, whose final gap was about 58.65
points.

Label smoothing changes the numerical interpretation of training loss, so its
nonzero plateau should not be compared directly with checkpoint 1's unsmoothed
training loss.

## Class behavior

`Flower` recall increased from 36.67% to 96.67%. Retaining RGB is a plausible
contributor because `Flower` is the only class stored natively in color, but the
experiment changed several variables and cannot establish causality.

The weakest checkpoint-2 class is `Industrial` at 40.00% recall. Other notable
remaining errors include `OpenCountry` confused with `Coast`, and confusion
among semantically related indoor classes. These are better targets for the
next improvement than indiscriminately increasing model size.

## Limitation and next question

Checkpoint 2 demonstrates that the combined recipe works, not which component
caused the gain. A well-controlled next experiment should retain this data and
training pipeline while testing a focused change, such as ImageNet-pretrained
initialization, or should ablate RGB if the goal is to quantify the color cue.
