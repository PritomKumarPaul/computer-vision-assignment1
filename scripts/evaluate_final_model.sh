#!/usr/bin/env bash
set -uo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEVICE="${1:-cuda}"
CHECKPOINT="artifacts/checkpoints/checkpoint6_convnext_large_pretrained_all.pt"
OUTPUT="results/final/test_metrics.json"
LOG_DIR="artifacts/logs/final"

cd "${PROJECT_ROOT}"

if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "Missing selected checkpoint: ${CHECKPOINT}"
  exit 1
fi

if [[ -f "${OUTPUT}" ]]; then
  echo "Refusing to evaluate the labeled test set again."
  echo "Existing final metrics: ${OUTPUT}"
  exit 2
fi

mkdir -p "${LOG_DIR}"
echo "Selected model: checkpoint 6 ConvNeXt-Large, chosen using validation only."
echo "This command performs the single final labeled-test evaluation."

python -u evaluate.py \
  --checkpoint "${CHECKPOINT}" \
  --split test \
  --confirm-final-test \
  --device "${DEVICE}" \
  --output "${OUTPUT}" \
  2>&1 | tee "${LOG_DIR}/test_evaluation.log"
evaluation_status=${PIPESTATUS[0]}
if (( evaluation_status != 0 )); then
  echo "Final test evaluation failed with status ${evaluation_status}."
  exit "${evaluation_status}"
fi

echo "Final test metrics saved to ${OUTPUT}. Do not use them for further tuning."
