# Assignment 1: The CNN Challenge

Reproducible experiments for 16-class scene recognition on a small, balanced
dataset. The project follows a three-model progression:

1. the instructor's deliberately weak `TNet` sanity-check baseline;
2. a compact CNN trained from scratch; and
3. an ImageNet-pretrained ResNet-18 fine-tuned for the task.

The supplied data are not committed to Git. Expected layout:

```text
data/
  train/<class_name>/*.jpg
  test/<class_name>/*.jpg
  test2/*.jpg                 # unlabeled; not used unless its role is clarified
```

## Setup (`vlm_clean`)

Activate the existing environment and install only missing dependencies:

```bash
conda activate vlm_clean
cd "/home/ppaul11/computer vision/assignment1/assignment1"
python -m pip install -r requirements.txt
```

Every training and evaluation entry point accepts `--device auto`, `cuda`, or
`cpu`. `auto` uses CUDA when PyTorch can see a GPU and otherwise uses CPU.

## Reproducible data split

Create the fixed 80/20 training/validation manifests before training:

```bash
python scripts/audit_dataset.py --data-root data
python scripts/create_split.py --data-root data/train --output-dir splits --seed 0
```

The split is stratified to exactly 120 training and 30 validation images per
class. Exact duplicate files are grouped so they cannot cross the split. The
labeled test directory never participates in splitting or model selection.

The source files are mixed-mode: 2,250 training images are stored as grayscale
(`L`) and all 150 `Flower` images are stored as RGB. The baseline deliberately
converts every image to one grayscale channel, matching the starter notebook.
The second baseline and final candidate convert every input to RGB: native
grayscale images become three identical channels while native RGB images keep
their color. The `color_mode` config field makes a grayscale/RGB ablation
possible without changing model code.

## Train

```bash
python train_baseline1.py --device cuda
python train_baseline2.py --device cuda
python train_final.py --device cuda
```

Equivalent CPU commands use `--device cpu`. The three entry points share the
same tested training engine, while their YAML files independently record model,
preprocessing, optimizer, and schedule choices.

If worker processes are unavailable in a restricted shell, add
`--num-workers 0`. This changes input-loading performance, not the experiment.

Each run saves its best-validation checkpoint and full history beneath
`artifacts/runs/<run_name>/`. Compact experiment summaries are appended to
`results/experiments.csv`.

## Evaluate

Validation evaluation is safe to run repeatedly:

```bash
python evaluate.py \
  --checkpoint artifacts/runs/baseline_tnet/best.pt \
  --split val
```

For the other checkpoints, replace `baseline_tnet` with `small_cnn` or
`resnet18`.

The labeled test set must not guide model selection. Test evaluation therefore
requires an explicit acknowledgement:

```bash
python evaluate.py \
  --checkpoint artifacts/runs/resnet18/best.pt \
  --split test \
  --confirm-final-test
```

To produce predictions for an unlabeled image directory:

```bash
python predict.py \
  --checkpoint artifacts/runs/resnet18/best.pt \
  --input-dir data/test2 \
  --output predictions.csv
```

`test2` is not referenced by the assignment handout or starter notebook, so it
is excluded from the planned experiments unless the instructor clarifies its
purpose.

## Experimental protocol

- Use the same checked-in split for every model.
- Select checkpoints only by validation accuracy.
- Record accuracy, macro F1, per-class recall, confusion matrices, training
  time, parameter count, and the best epoch.
- Run the labeled test set only after the final recipe is selected.
- Keep actual AI-assisted decisions and verification notes in `AI_USAGE.md`.

The supplied notebook is preserved unchanged at
`notebooks/starter_original.ipynb`.

## Git workflow

Commit logical milestones rather than every epoch: repository scaffold,
baseline implementation, each meaningful completed experiment, final model,
and report/reproducibility cleanup. The handout explicitly requires meaningful
history and says not to create the entire repository as one last-minute commit,
so pushing one combined snapshot at the end is not recommended.
