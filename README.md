# Assignment 1: The CNN Challenge

This repository currently contains one experiment: the CNN supplied in the
starter notebook. Later experiments will be introduced only after its measured
behavior motivates a specific hypothesis.

## Data

The data remain outside Git and must have this layout:

```text
data/
  train/<class_name>/*.jpg
  test/<class_name>/*.jpg
  test2/*.jpg                 # unlabeled and unused
```

The fixed validation split contains 120 training and 30 validation images from
each of the 16 classes. Six exact duplicate pairs occur in the `LivingRoom`
training class; SHA-256 grouping keeps each pair entirely in training or
entirely in validation. The labeled test set is never used for splitting or
model selection.

The source files are mixed-mode: 2,250 training images are stored as grayscale
and all 150 `Flower` images are stored as RGB. This first experiment follows the
starter notebook and converts every input to one-channel grayscale. Whether to
retain RGB will be treated as a later controlled experiment, not assumed in
advance.

The audit and split can be reproduced with:

```bash
python scripts/audit_dataset.py --data-root data
python scripts/create_split.py --data-root data/train --output-dir splits --seed 0
```

## Environment

```bash
conda activate vlm_clean
cd "/home/ppaul11/computer vision/assignment1/assignment1"
python -m pip install -r requirements.txt
```

Confirm that PyTorch can see the GPU:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

## Train the starter CNN

```bash
python train_starter.py --device cuda
```

Use `--device cpu` for CPU execution. If worker processes are unavailable, add
`--num-workers 0`; this affects input-pipeline speed but not the model recipe.

The command displays live `tqdm` progress and records:

- training and validation loss/accuracy;
- validation macro F1 and per-class recall;
- epoch and total wall-clock time;
- training and validation images per second;
- peak allocated CUDA memory;
- total/trainable parameter counts and parameter storage;
- estimated Conv/Linear MACs and FLOPs per image; and
- the best validation epoch.

Outputs are written to:

```text
artifacts/runs/starter_cnn/best.pt
artifacts/runs/starter_cnn/history.json
artifacts/runs/starter_cnn/metadata.json
results/experiments.csv
```

The MAC/FLOP values are architecture estimates: one multiply-accumulate is
reported as two FLOPs, and pooling/activation costs are excluded. Measured wall
time, throughput, and GPU memory capture hardware-dependent behavior.

## Validate the selected checkpoint

```bash
python evaluate.py \
  --checkpoint artifacts/runs/starter_cnn/best.pt \
  --split val \
  --device cuda
```

## Finalize a numbered experimental milestone

After training and validation, generate the loss/accuracy diagrams and a
versioned result package:

```bash
python scripts/finalize_checkpoint.py \
  --run-dir artifacts/runs/starter_cnn \
  --name checkpoint1
```

This produces:

```text
artifacts/checkpoints/checkpoint1.pt       # local; ignored by Git
results/checkpoint1/history.csv
results/checkpoint1/summary.json
results/checkpoint1/validation_metrics.json
reports/figures/checkpoint1/loss_curve.png
reports/figures/checkpoint1/accuracy_curve.png
```

Then record the completed experiment:

```bash
git add results/experiments.csv results/checkpoint1 reports/figures/checkpoint1
git commit -m "checkpoint1: starter CNN experiment"
git tag checkpoint1
```

For the next experiment, use `--run-name checkpoint2` during training, evaluate
its best checkpoint, finalize it with `--name checkpoint2`, and then commit/tag
that complete evidence package. Continue the same numbering without reusing a
checkpoint name.

Do not use the labeled test set to choose architectures or hyperparameters.
After the final model has been selected using validation evidence, test it once:

```bash
python evaluate.py \
  --checkpoint artifacts/runs/<selected_run>/best.pt \
  --split test \
  --confirm-final-test \
  --device cuda
```

## Git workflow

Commit logical milestones rather than individual epochs: the starter system,
its completed result, each later hypothesis and implementation, and the final
report/reproducibility cleanup. The handout explicitly requires meaningful
history and warns against creating the entire repository as one last-minute
commit.

The supplied notebook is preserved unchanged at
`notebooks/starter_original.ipynb`.

## Checkpoint 2 hypothesis

Checkpoint 1 strongly overfit, so checkpoint 2 tests a deliberately combined
anti-overfitting recipe:

- RGB rather than forced grayscale input;
- ResNet-18 initialized from scratch;
- random resized crops, horizontal flips, mild brightness/contrast jitter, and
  random erasing;
- dropout, AdamW weight decay, and label smoothing;
- cosine learning-rate decay and validation-based early stopping; and
- 128x128 inputs with mixed precision on CUDA.

Because this changes several factors together, it tests whether the recipe as a
whole is useful; it does not establish which component caused any improvement.
A later ablation can isolate the most important factor.

Train and validate it with:

```bash
python train_checkpoint2.py --device cuda

python evaluate.py \
  --checkpoint artifacts/runs/checkpoint2/best.pt \
  --split val \
  --device cuda
```

After validation, package the evidence but do not commit until the diagrams
have been inspected:

```bash
python scripts/finalize_checkpoint.py \
  --run-dir artifacts/runs/checkpoint2 \
  --name checkpoint2
```

## Checkpoint 3 architecture suite

Checkpoint 3 compares five torchvision CNN families under the same 128x128
RGB pipeline, split, augmentations, optimizer, regularization, scheduler, and
50-epoch maximum budget:

- DenseNet-121;
- EfficientNet-B0;
- MobileNetV3-Large;
- ResNeXt-50 32x4d; and
- ConvNeXt-Tiny.

All are initialized from scratch so this checkpoint focuses on architecture.
The comparison also includes checkpoint 2's ResNet-18 result as a reference.
VGG is excluded because its roughly 130M parameters are far outside this group;
Inception uses a materially different input/training design; and Xception would
require an additional non-torchvision implementation.

Run the full suite sequentially on one GPU:

```bash
bash scripts/run_checkpoint3_suite.sh cuda
```

The script is resumable: completed training, validation, and finalization
stages are skipped. A failure in one architecture is logged and does not erase
other completed runs. Logs are saved under `artifacts/logs/checkpoint3/`.

After all runs, inspect:

```text
results/checkpoint3/comparison.csv
results/checkpoint3/comparison.md
reports/figures/checkpoint3/architecture_comparison.png
```

The suite deliberately does not commit, tag, or push. After reviewing the
comparison, checkpoint 3 can be recorded as one Git milestone.

## Checkpoint 4 ResNeXt transfer-learning suite

Checkpoint 4 holds the architecture fixed at ResNeXt-50 32x4d and compares
four initialization/freezing strategies:

1. random initialization with every layer trainable;
2. ImageNet-pretrained initialization with every layer fine-tuned;
3. ImageNet-pretrained initialization with only `layer4` and the classifier
   trainable; and
4. ImageNet-pretrained initialization with the complete convolutional
   backbone frozen and only the classifier trainable.

All variants use the same leakage-controlled split, RGB augmentation, 224x224
input size, ImageNet normalization, batch size, seed, regularization, scheduler,
and 50-epoch maximum budget. Learning rates differ by trainable scope: `1e-3`
for scratch and head-only training, `3e-4` for last-stage fine-tuning, and
`1e-4` for full fine-tuning. Frozen modules, including their batch-normalization
statistics, remain in evaluation mode during training.

The first pretrained run automatically downloads torchvision's
`IMAGENET1K_V2` weights if they are not already cached. Run all four setups
sequentially on one GPU with:

```bash
conda activate vlm_clean
cd "/home/ppaul11/computer vision/assignment1/assignment1"
bash scripts/run_checkpoint4_transfer_suite.sh cuda
```

The runner is resumable and writes separate logs beneath
`artifacts/logs/checkpoint4/`. It trains, validates, generates each setup's
loss/accuracy curves, and creates the final transfer-learning comparison. It
does not evaluate the test set or perform any Git operation.

After it finishes, inspect:

```text
results/checkpoint4/comparison.csv
results/checkpoint4/comparison.md
reports/figures/checkpoint4/transfer_comparison.png
reports/figures/checkpoint4/<variant>/loss_curve.png
reports/figures/checkpoint4/<variant>/accuracy_curve.png
```
