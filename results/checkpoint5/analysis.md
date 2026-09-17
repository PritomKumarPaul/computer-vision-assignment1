# Checkpoint 5 interpretation

## Experimental question

Checkpoint 5 asks whether the transfer-learning behavior observed with
ResNeXt-50 generalizes across ResNet-18, DenseNet-121, EfficientNet-B0, and
ConvNeXt-Tiny. Each architecture uses the same duplicate-grouped split,
224x224 RGB pipeline, ImageNet normalization, augmentation, seed, batch size,
regularization, scheduler, and maximum epoch budget. Learning rates are
adapted to the trainable scope, so these are practical training recipes rather
than a strict single-variable learning-rate ablation. Checkpoint-4 ResNeXt
results are included because their experimental protocol is directly
comparable.

## Accuracy matrix

| Architecture | Scratch/all | Pretrained/all | Pretrained/last stage | Pretrained/head only |
|---|---:|---:|---:|---:|
| ResNet-18 | 82.92% | **94.58%** | 93.75% | 90.62% |
| DenseNet-121 | 83.12% | 93.54% | **94.17%** | 92.50% |
| EfficientNet-B0 | 81.25% | **93.54%** | **93.54%** | 93.33% |
| ConvNeXt-Tiny | 64.79% | **96.04%** | 95.00% | 93.33% |
| ResNeXt-50 (checkpoint 4) | 82.50% | 94.38% | **94.79%** | 94.17% |

The EfficientNet partial value is taken from its saved training history. Its
training completed normally, but the first suite pass did not run the separate
validation/finalization stages for that one variant. The resumable suite must
complete those stages before checkpoint 5 is committed.

## Main findings

ImageNet pretraining improved the best validation accuracy over scratch by
11.05--12.29 percentage points for ResNet, DenseNet, EfficientNet, and
ResNeXt. ConvNeXt benefited even more dramatically: full fine-tuning improved
it from 64.79% to 96.04%, a gain of 31.25 percentage points and 150 additional
correct predictions among 480 validation images. ConvNeXt's scratch curve
reached only about 82% training accuracy after 50 epochs, unlike the other
scratch models that nearly memorized the training data. Its poor scratch result
therefore reflects both optimization difficulty and a generalization gap, not
only conventional overfitting.

ConvNeXt-Tiny with full fine-tuning is the validation winner at 96.04% macro F1
and accuracy: 461 correct predictions and 19 errors. Its weakest class recall
is still 86.7% for `OpenCountry`; `Flower`, `Forest`, `Mountain`, `Office`,
`Store`, `Street`, and `Suburb` reach 100% recall. Accuracy and macro F1 match,
so the result is not caused by neglecting minority classes.

There is no universal best freezing rule:

- full fine-tuning beats partial fine-tuning for ResNet by 0.83 percentage
  points and ConvNeXt by 1.04 points;
- partial fine-tuning beats full fine-tuning for DenseNet by 0.63 points and
  ResNeXt by 0.42 points; and
- EfficientNet full and partial fine-tuning tie at 93.54%.

These differences correspond to only two to five validation images and come
from one split and one seed. They support architecture-specific behavior but
do not establish stable ordering without repeated seeds.

## Accuracy versus training cost

ConvNeXt full fine-tuning obtains the highest accuracy but is the most
expensive successful setup: 27.83M trainable parameters, 4.45G estimated MACs
per image, and 2416.5 MiB peak training GPU memory. ConvNeXt partial
fine-tuning retains 95.00% accuracy--five fewer correct images--while reducing
peak training memory to 580.5 MiB, approximately 76% lower.

DenseNet partial fine-tuning is another strong balance. It reaches 94.17% with
2.18M trainable parameters and 259.3 MiB peak memory, compared with 93.54%,
6.97M parameters, and 2194.8 MiB for full fine-tuning. In this run, freezing
the early DenseNet blocks improved accuracy while reducing peak memory by
about 88%.

EfficientNet head-only training is the strongest low-compute option. It reaches
93.33% with only 20,496 trainable parameters, 226.2 MiB peak training memory,
and 384.6M MACs per image. Full fine-tuning classifies only one additional
validation image correctly while requiring 4.03M trainable parameters and
1477.9 MiB peak memory. EfficientNet therefore provides the clearest evidence
that the pretrained representation is already linearly separable for this
dataset.

ResNet full fine-tuning is a useful middle ground: 94.58% accuracy, 11.18M
parameters, 1.81G MACs per image, and 617.8 MiB peak memory. It gives up seven
correct images relative to ConvNeXt full fine-tuning but requires much less
inference compute and training memory.

Freezing does not reduce inference MACs because the complete backbone still
runs for every image. It reduces backward computation, stored activations,
optimizer state, and training memory. Total suite time also depends on early
stopping and checkpoint-writing frequency, so it is not a pure per-step speed
measurement.

## Selection recommendation

If validation accuracy is the only objective, pretrained ConvNeXt-Tiny with
full fine-tuning is the current final-model candidate. If resource efficiency
matters, the preferred alternatives are:

- ConvNeXt partial fine-tuning for near-maximum accuracy;
- DenseNet partial fine-tuning for a smaller training footprint; or
- EfficientNet head-only training for minimum training and inference cost.

The strongest next scientific check would repeat ConvNeXt full, ConvNeXt
partial, and one efficient alternative over multiple seeds. If no further
development is planned, model selection should use validation only and the
labeled test set should be evaluated exactly once on the chosen ConvNeXt-full
checkpoint.
