#!/usr/bin/env bash
set -euo pipefail

SCRIPT_PATH="$(readlink -f "$0")"
SWITCHING_DIR="$(dirname "$SCRIPT_PATH")"
REPO_ROOT="$(dirname "$SWITCHING_DIR")"

DEFAULT_KERNELS=(
  kernel_3mm
  kernel_2mm
  gsum
  bicg
  gemver
  cnn
  matvec
  stencil_2d
  iir
  fir
)

RUN_PASS=0
if [[ "${1:-}" == "--run-pass" ]]; then
  RUN_PASS=1
  shift
fi

if [[ "$#" -gt 0 ]]; then
  KERNELS=("$@")
else
  KERNELS=("${DEFAULT_KERNELS[@]}")
fi

for kernel in "${KERNELS[@]}"; do
  echo "[DEBUG] ===== kernel: ${kernel} ====="
  if [[ "$RUN_PASS" -eq 1 ]]; then
    if ! "${REPO_ROOT}/test_sa.sh" "${kernel}"; then
      echo "[WARN] test_sa.sh failed for ${kernel}; generating debug report from existing outputs."
    fi
  fi

  python3 "${SWITCHING_DIR}/debug_switching.py" \
    --repo-root "${REPO_ROOT}" \
    --kernel "${kernel}" \
    --window post-reset \
    --rel-error-threshold 0.10 \
    --zero-abs-threshold 5 \
    --ignore-both-below 50 || true
done
