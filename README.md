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

## Setup

Python 3.10+ is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The current cluster environment already provides the main dependencies. CUDA
is optional; every command also supports CPU execution.

## Reproducible data split

Create the fixed 80/20 training/validation manifests before training:

```bash
python scripts/create_split.py --data-root data/train --output-dir splits --seed 0
```

The split is stratified to exactly 120 training and 30 validation images per
class. Exact duplicate files are grouped so they cannot cross the split.

## Train

```bash
python train.py --config configs/baseline_tnet.yaml
python train.py --config configs/small_cnn.yaml
python train.py --config configs/resnet18.yaml
```

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
