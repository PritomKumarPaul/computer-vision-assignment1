#!/usr/bin/env bash
set -uo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEVICE="${1:-cuda}"
LOG_DIR="${PROJECT_ROOT}/artifacts/logs/checkpoint5"
mkdir -p "${LOG_DIR}"
cd "${PROJECT_ROOT}"

ARCHITECTURES=(resnet18 densenet121 efficientnet_b0 convnext_tiny)
SETUPS=(scratch_all pretrained_all pretrained_last_stage pretrained_head_only)
FAILED=()

for architecture in "${ARCHITECTURES[@]}"; do
  for setup in "${SETUPS[@]}"; do
    variant="${architecture}_${setup}"
    run_name="checkpoint5_${variant}"
    run_dir="artifacts/runs/${run_name}"
    result_summary="results/checkpoint5/${variant}/summary.json"
    config="configs/checkpoint5/${variant}.yaml"

    echo
    echo "========== ${run_name} =========="

    if [[ ! -f "${run_dir}/completed.json" ]]; then
      if ! python -u train.py --config "${config}" --device "${DEVICE}" \
        2>&1 | tee "${LOG_DIR}/${variant}_train.log"; then
        echo "Training failed for ${variant}; continuing to the next setup."
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
        echo "Validation failed for ${variant}; continuing to the next setup."
        FAILED+=("${variant}:validation")
        continue
      fi
    else
      echo "Validation metrics already exist; skipping validation."
    fi

    if [[ ! -f "${result_summary}" ]]; then
      if ! python scripts/finalize_checkpoint.py \
        --run-dir "${run_dir}" \
        --name checkpoint5 \
        --variant "${variant}"; then
        echo "Finalization failed for ${variant}; continuing to the next setup."
        FAILED+=("${variant}:finalize")
        continue
      fi
    else
      echo "Finalized result already exists; skipping finalization."
    fi
  done
done

if ! python scripts/compare_transfer_matrix.py --name checkpoint5; then
  echo "Comparison generation failed. Individual completed runs were preserved."
  exit 1
fi

echo
echo "Checkpoint 5 transfer matrix finished."
echo "No test-set evaluation, Git commit, tag, or push was performed."
echo "Inspect results/checkpoint5 and reports/figures/checkpoint5 first."
if (( ${#FAILED[@]} > 0 )); then
  echo "Failed stages: ${FAILED[*]}"
  exit 1
fi
