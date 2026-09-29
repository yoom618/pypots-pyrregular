#!/usr/bin/env bash
set -euo pipefail

###############################
# ENVIRONMENT VARIABLES
###############################
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY_SCRIPT="${PROJECT_ROOT}/scripts/run_mascit_train_loss_only_single_dataset.py"
DATASET_DIR="${PROJECT_ROOT}/dataset"
OUTPUT_DIR="${PROJECT_ROOT}/testing_results/mascit_train_loss_only_single_dataset"
GPU_IDS="0"

################################
# CHECK PYTHON RUNNER
################################
if [[ ! -f "${PY_SCRIPT}" ]]; then
  echo "Error: Python runner not found: ${PY_SCRIPT}" >&2
  echo "Hint: cd to repository root and run again." >&2
  exit 1
fi

################################
# Python runner options
################################
# - use default hyperparameters from the Python script
# - override options by adding to EXTRA_ARGS below

EXTRA_ARGS=()
COMMON_ARGS=(
  --dataset-dir "${DATASET_DIR}"
  --output-dir "${OUTPUT_DIR}"
)
LOG_DIR="${OUTPUT_DIR}/logs"
mkdir -p "${LOG_DIR}"

################################
# GPU SETUP
################################
# - Set GPU IDs for CUDA_VISIBLE_DEVICES
# - If you have multiple GPUs, you can specify them as a comma-separated list (e.g., "0,1" for GPU 0 and GPU 1).

export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
echo "[GPU] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"


################################
# Single run mode (with dataset argument)
################################
# - If a dataset argument is provided, run the Python script for that single dataset.
# - ex) ./run_mascit_train_loss_only_single_dataset.sh ABF
# - ex) ./run_mascit_train_loss_only_single_dataset.sh --dataset P12 --max-trials 10
if [[ $# -gt 0 ]]; then
  DATASET="${1:-}"
  shift || true

  if [[ "${DATASET}" == "--dataset" ]]; then
    if [[ $# -eq 0 ]]; then
      echo "Error: --dataset requires a value." >&2
      exit 1
    fi
    DATASET="${1:-}"
    shift || true
  fi

  exec python "${PY_SCRIPT}" \
    --dataset "${DATASET}" \
    "${COMMON_ARGS[@]}" \
    "$@"
fi


################################
# Batch run mode (no dataset argument)
################################
# - If no dataset argument is provided, run the Python script for all datasets in the DATASETS array.
# - You can comment/uncomment datasets in the DATASETS array to control which datasets are run.
# - Datasets are grouped by prefix for easier viewing.
# - Note that running multiple datasets in parallel delays the total execution time. (3x-10x in personal experience)
#   Therefore, it is recommended to compute execution time separately for each dataset by running them one at a time, or in small batches.

DATASETS=(

  # A*
  "ABF" "AIS" "AN" "AOC" "APT" "ARC"
  # # C* - D*
  # "CT" "CTB" "DD" "DG" "DW"
  # # G*
  # "GM1" "GM2" "GM3" "GP1" "GP2"
  # # G*
  # "GX" "GY" "GZ"
  # # I* - M*
  # "IW" "JV" "LPA" "MI3" "MP"
  # # P*
  # "P12" "P19" "PGE" "PGZ" "PL"
  # # S* - V*
  # "SAD" "SE" "SGZ" "TA" "VE"

  # # Slow & High-memory
  # "PAM" "GSS"

  # # Slow & Low-memory
  # "P19" "IW" "TA"
)

# [Block] 반복 실행
# - 배열의 각 데이터셋을 병렬로 nohup 실행
for ds in "${DATASETS[@]}"; do
  echo "===== Running $ds ====="
  LOG_FILE="${LOG_DIR}/${ds}.out"
  echo "Log: ${LOG_FILE}"
  nohup python "${PY_SCRIPT}" \
    --dataset "$ds" \
    "${COMMON_ARGS[@]}" \
    "${EXTRA_ARGS[@]}" \
    > "${LOG_FILE}" 2>&1 &
done
