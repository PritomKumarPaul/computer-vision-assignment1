#!/usr/bin/env bash
set -uo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEVICE="${1:-cuda}"
LOG_DIR="${PROJECT_ROOT}/artifacts/logs/checkpoint3"
mkdir -p "${LOG_DIR}"
cd "${PROJECT_ROOT}"

VARIANTS=(
  densenet121
  efficientnet_b0
  mobilenet_v3_large
  resnext50_32x4d
  convnext_tiny
)

FAILED=()

for variant in "${VARIANTS[@]}"; do
  run_name="checkpoint3_${variant}"
  run_dir="artifacts/runs/${run_name}"
  result_summary="results/checkpoint3/${variant}/summary.json"
  config="configs/checkpoint3/${variant}.yaml"

  echo
  echo "========== ${run_name} =========="

  if [[ ! -f "${run_dir}/completed.json" ]]; then
    if ! python -u train.py --config "${config}" --device "${DEVICE}" \
      2>&1 | tee "${LOG_DIR}/${variant}_train.log"; then
      echo "Training failed for ${variant}; continuing to the next architecture."
      FAILED+=("${variant}:train")
      continue
    fi
  else
    echo "Training artifacts already exist; skipping training."
  fi

  if [[ ! -f "${run_dir}/val_metrics.json" ]]; then
    if ! python -u evaluate.py \
      --checkpoint "${run_dir}/best.pt" \
      --split val \
      --device "${DEVICE}" \
      2>&1 | tee "${LOG_DIR}/${variant}_validation.log"; then
      echo "Validation failed for ${variant}; continuing to the next architecture."
      FAILED+=("${variant}:validation")
      continue
    fi
  else
    echo "Validation metrics already exist; skipping validation."
  fi

  if [[ ! -f "${result_summary}" ]]; then
    if ! python scripts/finalize_checkpoint.py \
      --run-dir "${run_dir}" \
      --name checkpoint3 \
      --variant "${variant}"; then
      echo "Finalization failed for ${variant}; continuing to the next architecture."
      FAILED+=("${variant}:finalize")
      continue
    fi
  else
    echo "Finalized result already exists; skipping finalization."
  fi
done

if ! python scripts/compare_checkpoint_suite.py --name checkpoint3; then
  echo "Comparison generation failed. Individual completed runs were preserved."
  exit 1
fi

echo
echo "Checkpoint 3 suite finished. No Git commit or push was performed."
echo "Inspect results/checkpoint3 and reports/figures/checkpoint3 first."
if (( ${#FAILED[@]} > 0 )); then
  echo "Failed stages: ${FAILED[*]}"
  exit 1
fi
