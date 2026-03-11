#!/usr/bin/env bash

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DYNAMATIC_PATH="${DYNAMATIC_PATH:-$SCRIPT_DIR}"
BUFFER_ALGORITHM="${BUFFER_ALGORITHM:---buffer-algorithm fpga20}"
SHARING="${SHARING:-}"
GATING="${GATING:-}"
HDL="${HDL:-verilog}"
CLOCK_PERIOD="${CLOCK_PERIOD:-8}"
DISABLE_LSQ="--disable-lsq"

BENCHMARKS=(
  "kernel_3mm/kernel_3mm.c"
  "gsum/gsum.c"
  "bicg/bicg.c"
  "cnn/cnn.c"
  "matvec/matvec.c"
  "stencil_2d/stencil_2d.c"
  "matrix/matrix.c"
  "gcd/gcd.c"
  "fir/fir.c"
)

DYNAMATIC_BIN="$DYNAMATIC_PATH/bin/dynamatic"
if [[ ! -x "$DYNAMATIC_BIN" ]]; then
  echo "Dynamatic frontend not found or not executable: $DYNAMATIC_BIN"
  exit 1
fi

failed=()
passed=0
total=${#BENCHMARKS[@]}

for i in "${!BENCHMARKS[@]}"; do
  benchmark="${BENCHMARKS[$i]}"
  src="$DYNAMATIC_PATH/integration-test/$benchmark"

  echo "============================================================"
  echo "[$((i + 1))/$total] Running $benchmark with --disable-lsq"

  if [[ ! -f "$src" ]]; then
    echo "[FAIL] Source file does not exist: $src"
    failed+=("$benchmark (missing source)")
    continue
  fi

  cmd="set-dynamatic-path $DYNAMATIC_PATH; \
set-src $src; \
set-clock-period $CLOCK_PERIOD; \
compile $SHARING $BUFFER_ALGORITHM $GATING $DISABLE_LSQ; \
write-hdl --hdl $HDL; \
simulate; \
exit"

  if echo "$cmd" | "$DYNAMATIC_BIN" --exit-on-failure; then
    echo "[PASS] $benchmark"
    passed=$((passed + 1))
  else
    echo "[FAIL] $benchmark"
    failed+=("$benchmark")
  fi

done

echo "============================================================"
echo "Completed $total testcases: $passed passed, ${#failed[@]} failed"

if [[ ${#failed[@]} -ne 0 ]]; then
  echo "Failed testcases:"
  for item in "${failed[@]}"; do
    echo "  - $item"
  done
  exit 1
fi

echo "All testcases passed."
