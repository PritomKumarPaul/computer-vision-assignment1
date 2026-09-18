#!/usr/bin/env bash
set -uo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEVICE="${1:-cuda}"
RUN_NAME="checkpoint6_convnext_large_pretrained_all"
VARIANT="convnext_large_pretrained_all"
RUN_DIR="artifacts/runs/${RUN_NAME}"
RESULT_SUMMARY="results/checkpoint6/${VARIANT}/summary.json"
CONFIG="configs/checkpoint6/${VARIANT}.yaml"
LOG_DIR="${PROJECT_ROOT}/artifacts/logs/checkpoint6"

mkdir -p "${LOG_DIR}"
cd "${PROJECT_ROOT}"

if [[ ! -f "${RUN_DIR}/completed.json" ]]; then
  python -u train.py --config "${CONFIG}" --device "${DEVICE}" \
    2>&1 | tee "${LOG_DIR}/train.log"
  train_status=${PIPESTATUS[0]}
  if (( train_status != 0 )); then
    echo "Training failed with status ${train_status}."
    exit "${train_status}"
  fi
else
  echo "Training artifacts already exist; skipping training."
fi

if [[ ! -f "${RUN_DIR}/val_metrics.json" ]]; then
  python -u evaluate.py \
    --checkpoint "${RUN_DIR}/best.pt" \
    --split val \
    --device "${DEVICE}" \
    2>&1 | tee "${LOG_DIR}/validation.log"
  validation_status=${PIPESTATUS[0]}
  if (( validation_status != 0 )); then
    echo "Validation failed with status ${validation_status}."
    exit "${validation_status}"
  fi
else
  echo "Validation metrics already exist; skipping validation."
fi

if [[ ! -f "${RESULT_SUMMARY}" ]]; then
  if ! python scripts/finalize_checkpoint.py \
      --run-dir "${RUN_DIR}" \
      --name checkpoint6 \
      --variant "${VARIANT}"; then
    echo "Finalization failed."
    exit 1
  fi
else
  echo "Finalized result already exists; skipping finalization."
fi

if ! python scripts/compare_checkpoint6.py; then
  echo "Capacity comparison failed."
  exit 1
fi

echo
echo "Checkpoint 6 finished."
echo "No test-set evaluation, Git commit, tag, or push was performed."
echo "Inspect results/checkpoint6 and reports/figures/checkpoint6 first."
