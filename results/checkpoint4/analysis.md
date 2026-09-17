# Checkpoint 4 interpretation

## Question

Checkpoint 4 tests how ImageNet initialization and the amount of ResNeXt-50
fine-tuning affect scene classification. All four runs use the same fixed,
duplicate-grouped train/validation split. They also share 224x224 RGB inputs,
ImageNet normalization, augmentation, batch size, regularization, scheduler,
seed, and maximum epoch budget. Learning rates are intentionally adapted to
the optimization scope, so this is a comparison of practical transfer-learning
recipes rather than a strict single-variable freezing ablation.

## Results

| Setup | Best epoch | Correct / 480 | Validation accuracy | Macro F1 | Trainable parameters | Peak GPU memory | Run time |
|---|---:|---:|---:|---:|---:|---:|---:|
| Pretrained, `layer4` + head | 15 | 455 | **94.79%** | **94.76%** | 14.578M (63.35%) | 501.1 MiB | 152.6s |
| Pretrained, all layers | 8 | 453 | 94.38% | 94.38% | 23.013M (100%) | 2208.9 MiB | 128.2s |
| Pretrained, head only | 11 | 452 | 94.17% | 94.17% | 0.033M (0.14%) | **387.3 MiB** | 138.4s |
| Scratch, all layers | 44 | 396 | 82.50% | 82.22% | 23.013M (100%) | 2206.6 MiB | 343.9s |

The three pretrained setups made only 25, 27, and 28 validation errors. Their
rank therefore differs by just two or three images, which is too small to claim
that one transfer strategy is intrinsically superior from this single split
and seed. In contrast, each pretrained setup classified 56--59 more validation
images correctly than the scratch setup. The decisive conclusion is the value
of ImageNet pretraining, not the exact ordering of the pretrained variants.

Accuracy and macro F1 are nearly identical for every pretrained variant. This
indicates that their aggregate results are not being produced by sacrificing
most classes in favor of a few easy ones.

## Learning behavior

The scratch model learned slowly and continued improving until epoch 44. Its
training accuracy approached 100%, while its best validation accuracy stopped
at 82.50% and ended at 80.21%. This is the familiar high-capacity,
small-dataset overfitting pattern.

All pretrained models started above 86% validation accuracy in the first epoch
and reached their best results by epochs 8--15. Their validation curves remain
within a narrow band around 92--95%, even as training accuracy approaches
100%. The low full-fine-tuning learning rate avoided an obvious catastrophic
forgetting failure.

The partial model achieved the highest observed accuracy while freezing the
stem and `layer1`--`layer3`. Compared with full fine-tuning, it trained only
63.35% of the parameters and reduced peak GPU memory by about 77%, from
2208.9 MiB to 501.1 MiB. This suggests that preserving general low- and
mid-level ImageNet features while adapting the final residual stage is well
matched to this dataset.

The head-only result is the strongest efficiency finding. Updating only 32,784
parameters--0.14% of the network--still produced 94.17% validation accuracy,
only three correct predictions behind the partial model. Its peak training
memory was 387.3 MiB, about 82% lower than full fine-tuning. The pretrained
representation is therefore already highly linearly separable for these scene
classes.

Freezing does not reduce the reported inference MAC count: all four setups
still execute the same ResNeXt backbone at inference. It reduces backward-pass
work, optimizer state, and training memory. Total recorded run time also
depends on the number of epochs before early stopping and time spent saving
improved checkpoints, so it should not be interpreted as a pure per-step speed
benchmark.

## Class-level observations

Pretraining produced especially large recall gains for `Industrial` and
`LivingRoom`, the two weakest scratch classes. `LivingRoom` rose from 56.7%
recall with scratch training to 93.3--100% across the pretrained setups;
`Industrial` rose from 50.0% to 80.0--93.3%. The duplicate grouping remains
leakage-safe: identical LivingRoom files do not cross the train/validation
boundary.

The partial model was not best for every class. For example, its
`OpenCountry` recall was 80.0%, versus 93.3% for full fine-tuning, while the
head-only model had the best `Industrial` recall at 93.3%. Each class has only
30 validation images, so one image changes class recall by 3.33 percentage
points; these small differences should not be overinterpreted.

## Selection recommendation

If validation accuracy is the primary objective, the pretrained
`layer4`-plus-head checkpoint is the current candidate final model. If training
memory and update cost matter, the pretrained head-only model is the better
engineering choice for a loss of only 0.62 percentage points (three images).
Because the top pretrained results are so close, a stronger scientific choice
would repeat those variants over several seeds and compare mean and variation
before declaring a winner.

The labeled test set remains untouched. It should be evaluated only after the
final strategy is selected, rather than used to break this close validation
tie.
