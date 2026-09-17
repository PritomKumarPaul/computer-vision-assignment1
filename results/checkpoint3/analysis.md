# Checkpoint 3 interpretation

## Experimental control

The five checkpoint-3 architectures were trained from scratch with the same
leakage-controlled train/validation manifests, 128x128 RGB input pipeline,
augmentations, batch size, optimizer, learning-rate schedule, regularization,
seed, and maximum epoch budget. The checkpoint-2 ResNet-18 result is included
as a reference because it used the same recipe and split. The labeled test set
was not used for architecture selection.

## Main results

| Architecture | Validation accuracy | Macro F1 | Parameters | MACs/image | Training time | Interpretation |
|---|---:|---:|---:|---:|---:|---|
| ResNeXt-50 32x4d | 78.33% | 77.90% | 23.01M | 1380.7M | 250.9s | Highest validation accuracy |
| DenseNet-121 | 76.88% | 76.61% | 6.97M | 925.1M | 212.5s | Similar accuracy with fewer parameters |
| ResNet-18 (checkpoint 2) | 76.46% | 76.05% | 11.18M | 592.2M | 196.0s | Strong balanced reference |
| EfficientNet-B0 | 72.08% | 72.26% | 4.03M | 126.0M | 149.4s | Best compute/accuracy compromise among successful new models |
| ConvNeXt-Tiny | 61.88% | 61.47% | 27.83M | 1454.6M | 331.2s | Expensive and poorly matched to this from-scratch recipe |
| MobileNetV3-Large | 6.25% | 0.74% | 4.22M | 72.2M | 41.0s | Failed to generalize under the shared recipe |

ResNeXt improved validation accuracy by 1.87 percentage points over ResNet-18,
but used about 2.1 times as many parameters, 2.3 times as many MACs per image,
and 3.2 times the peak training GPU memory. DenseNet improved by only 0.42
percentage points while using 37.7% fewer parameters than ResNet-18, although
its MAC count and observed peak GPU memory were higher. Thus ResNeXt is the
accuracy winner, while ResNet-18 remains the strongest overall balance in this
comparison. EfficientNet-B0 is attractive when inference compute is the main
constraint.

## MobileNet failure analysis

MobileNet did not encounter a runtime failure. It completed 11 epochs and then
stopped through the configured early-stopping rule. Its training accuracy rose
from 12.97% to 66.77%, and training loss fell from 2.69 to 1.34. In contrast,
validation accuracy stayed at exactly 6.25% (the chance rate for 16 balanced
classes), while validation loss rose from 2.78 to 3.20. The saved model
predicted `Suburb` for all 480 validation images, giving that class recall 1.0
and every other class recall 0.0.

This train/evaluation mismatch is consistent with unstable running statistics
in MobileNet's extensive batch-normalization layers when the architecture is
trained from scratch on this small dataset. Sensitivity to the common learning
rate and augmentation recipe may also contribute. This is a diagnosis
supported by the curves, not a proven causal mechanism. The data manifests,
class order, evaluation function, and RGB conversion were shared with the
models that validated normally, so a suite-wide data or label-order defect is
unlikely.

The failed run should remain in checkpoint 3. It is an informative negative
result demonstrating that parameter count and theoretical efficiency do not
guarantee stable training under one common recipe. A separate follow-up could
test pretrained weights, a smaller learning rate, or batch-normalization
recalibration, but such a run should be recorded as a new experiment rather
than replacing this result.

## Model-selection boundary

Checkpoint 3 compares architectures using validation data only. The labeled
test set should be evaluated once, after the final model and all hyperparameters
have been selected, to avoid adapting the project to test performance.
