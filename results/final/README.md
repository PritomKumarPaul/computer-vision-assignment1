# Final model selection and test evaluation

The final model was locked using validation evidence before the labeled test
set was evaluated. Checkpoint 6 ConvNeXt-Large achieved 96.25% validation
accuracy (462/480), narrowly exceeding checkpoint 5 ConvNeXt-Tiny at 96.04%
(461/480). No test results were available during this choice.

The guarded final evaluation was then run once and saved to
`test_metrics.json`:

- test accuracy: 98.50% (394/400);
- test macro F1: 98.48%;
- 12 of 16 classes: 100% recall;
- Kitchen, LivingRoom, and Mountain: 96% recall; and
- TallBuilding: 88% recall.

The selected model uses torchvision ConvNeXt-Large with ImageNet-1K V1
initialization and full-network fine-tuning. Its local checkpoint is
`artifacts/checkpoints/checkpoint6_convnext_large_pretrained_all.pt`.

- Release asset: https://github.com/PritomKumarPaul/computer-vision-assignment1/releases/download/checkpoint6/checkpoint6_convnext_large_pretrained_all.pt
- SHA-256: `d011ab54be56d0fbb0c643c1f624d935b1521b1c21f2f73f3df5b2f4649f8919`

The checkpoint is excluded from ordinary Git history because it is
approximately 749 MiB. The JSON metrics, validation evidence, configuration,
source code, and final report are tracked in the repository.
