#!/bin/bash
set -euo pipefail

########################################
# ANSI Color Codes
########################################
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
NC='\033[0m' # No color / reset

# ============================================================================ #
# Function Definitions
# ============================================================================ #

## Prints some information to stdout.
#   $1: the text to print
echo_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

# Prints a fatal error message to stdout.
#   $1: the text to print
echo_fatal() {
    echo -e "${RED}[FATAL]${NC} $1"
}

# Exits the script with a fatal error message if the last command that was
# called before this function failed, otherwise optionally prints an information
# message.
#   $1: fatal error message
#   $2: [optional] information message
exit_on_fail() {
    if [[ $? -ne 0 ]]; then
        if [[ ! -z $1 ]]; then
            echo_fatal "$1"
            exit 1
        fi
        echo_fatal "Failed!"
        exit 1
    else
        if [[ ! -z $2 ]]; then
            echo_info "$2"
        fi
    fi
}

echo_section() {
    echo ""
    echo -e "${GREEN}# ===----------------------------------------------------------------------=== #"
    echo -e "# $1"
    echo -e "# ===----------------------------------------------------------------------=== #${NC}"
    echo ""
}

# ============================================================================ #
# Variable definitions
# ============================================================================ #
# Absolute path to the script (resolving symlinks)
SCRIPT_PATH="$(readlink -f "$0")"
SCRIPT_DIR="$(dirname "$SCRIPT_PATH")"

# Script arguments
if [[ $# -ne 1 ]]; then
    echo_fatal "Usage: $0 <kernel-name>"
    exit 1
fi

KERNEL_NAME=$1
SRC_DIR="$SCRIPT_DIR/integration-test/$KERNEL_NAME"
OUTPUT_DIR="$SRC_DIR/out/comp"
DATA_LOG="$OUTPUT_DIR/profiling.log"
TRACE_LOG="$OUTPUT_DIR/data_trace.log"
SWITCHING_LOG="$OUTPUT_DIR/switching_estimation.csv"
COMPARE_LOG="$OUTPUT_DIR/switching_compare.log"
DEBUG_LOG="$OUTPUT_DIR/switching_debug.log"
F_FREQUENCIES="$OUTPUT_DIR/frequencies.csv"
DYNAMATIC_DIR="/home/jianliu/TCAD26/dynamatic-mlir"
VCD_GOLDEN="$SRC_DIR/out/sim/HLS_VERIFY/trace.vcd"

DYNAMATIC_PROFILER_BIN="$SCRIPT_DIR/bin/exp-frequency-profiler"
DYNAMATIC_OPT_BIN="$SCRIPT_DIR/bin/dynamatic-opt"
COMPARE_SCRIPT="$SCRIPT_DIR/switching_testing/test.py"
DEBUG_SCRIPT="$SCRIPT_DIR/switching_testing/debug_switching.py"

F_CF_DYN_TRANSFORMED="$OUTPUT_DIR/cf_transformed_mem_interface_marked.mlir"
F_HANDSHAKE_TRANSFORMED="$OUTPUT_DIR/handshake_transformed.mlir"
F_HANDSHAKE_EXPORT="$OUTPUT_DIR/handshake_export.mlir"
F_HANDSHAKE_SWITCH="$OUTPUT_DIR/handshake_switch_test.mlir"
TARGET_CP=8
MILP_SOLVER="gurobi"

# ---------------------------------------------------------------------------- #
# Switching-estimation debug flags (environment-overridable)
# ---------------------------------------------------------------------------- #
# SWITCH_DEBUG:
#   - false: disable structured debug stream (default)
#   - true : enable structured debug stream
# SWITCH_DEBUG_CATEGORIES:
#   - Comma-separated list of categories to print when SWITCH_DEBUG=true
#   - Available categories:
#       pipeline, cfdfc, profiling, data, mux, handshake, node, graph, dump, all
#   - Note: category "dump" enables both detailed dump files automatically.
# SWITCH_DUMP_DATA_CHANNELS:
#   - false: do not dump per-node data-channel detail file (default)
#   - true : dump "<switching_estimation>_data_channels.txt"
# SWITCH_DUMP_MG_HANDSHAKE:
#   - false: do not dump per-MG handshake detail file (default)
#   - true : dump "<switching_estimation>_mg_handshake.txt"
# Example:
#   SWITCH_DEBUG=true \
#   SWITCH_DEBUG_CATEGORIES=data,mux \
#   SWITCH_DUMP_DATA_CHANNELS=true \
#   SWITCH_DUMP_MG_HANDSHAKE=true \
#   ./test_sa.sh matvec
SWITCH_DEBUG="${SWITCH_DEBUG:-false}"
SWITCH_DEBUG_CATEGORIES="${SWITCH_DEBUG_CATEGORIES:-pipeline}"
SWITCH_DUMP_DATA_CHANNELS="${SWITCH_DUMP_DATA_CHANNELS:-false}"
SWITCH_DUMP_MG_HANDSHAKE="${SWITCH_DUMP_MG_HANDSHAKE:-false}"

# ---------------------------------------------------------------------------- #
# Comparison/report options (environment-overridable)
# ---------------------------------------------------------------------------- #
# WINDOW_MODE:
#   - full       : compare all simulated cycles (default in this script)
#   - post-reset : compare cycles after reset deassertion
# RESET_SIGNAL:
#   - empty string: auto-detect reset signal (default)
#   - explicit signal name: force a specific reset signal
# REL_ERROR_THRESHOLD:
#   - Relative error limit for channels with golden > 0
# ZERO_ABS_THRESHOLD:
#   - Absolute error limit for channels with golden = 0
# IGNORE_BOTH_BELOW:
#   - Ignore a channel if both est/golden are strictly below this value
# MAX_VIOLATION_REPORT:
#   - Max detailed violations printed by switching_testing/test.py
# DEBUG_TOP_K:
#   - Top failing entries printed by switching_testing/debug_switching.py
WINDOW_MODE="${WINDOW_MODE:-full}"
RESET_SIGNAL="${RESET_SIGNAL:-}"
REL_ERROR_THRESHOLD="${REL_ERROR_THRESHOLD:-0.10}"
ZERO_ABS_THRESHOLD="${ZERO_ABS_THRESHOLD:-5}"
IGNORE_BOTH_BELOW="${IGNORE_BOTH_BELOW:-50}"
MAX_VIOLATION_REPORT="${MAX_VIOLATION_REPORT:-40}"
DEBUG_TOP_K="${DEBUG_TOP_K:-20}"

# ============================================================================ #
# Switching Estimation Flow
# ============================================================================ #
# Check the existence of the handshake_transformed.mlir file
if [ -e $F_HANDSHAKE_TRANSFORMED ]; then
  echo "The handshake_transformed.mlir for $KERNEL_NAME exists"
else
  echo "[ERROR] The handshake_export.mlir file doesn't exist"
  exit 1
fi

# Run the profiler
echo_section "[Step 1] Running Profiler for ${KERNEL_NAME}"
"$DYNAMATIC_PROFILER_BIN" "$F_CF_DYN_TRANSFORMED" \
    --top-level-function="$KERNEL_NAME" \
    --input-args-file="$OUTPUT_DIR/profiler-inputs.txt" \
    --trace-log-file="$TRACE_LOG" \
    --mode=both > "$F_FREQUENCIES"
echo_info "Profiling Finished"

# Run the switching estimation pass
echo_section "[Step 2] Running Buffer Placement and Switching Estimation Pass for ${KERNEL_NAME}"
echo_info "Switching debug flags: debug=${SWITCH_DEBUG}, categories=${SWITCH_DEBUG_CATEGORIES}, dump-data-channels=${SWITCH_DUMP_DATA_CHANNELS}, dump-mg-handshake=${SWITCH_DUMP_MG_HANDSHAKE}"
cd "$OUTPUT_DIR"
export LSAN_OPTIONS=verbosity=1:log_threads=1
export UBSAN_OPTIONS=print_stacktrace=1:halt_on_error=1
if ! /usr/bin/time -v "$DYNAMATIC_OPT_BIN" "$F_HANDSHAKE_TRANSFORMED" \
    --handshake-mark-fpu-impl="impl=vivado" \
    --handshake-set-buffering-properties="version=fpga20" \
    --handshake-place-buffers="algorithm=fpga20 solver=$MILP_SOLVER frequencies=$F_FREQUENCIES timing-models=$SCRIPT_DIR/data/components.json target-period=$TARGET_CP timeout=300 dump-logs \
    blif-files=$DYNAMATIC_DIR/data/aig/ lut-delay=0.55 lut-size=6 acyclic-type" \
    --switching-estimation="data-trace=$TRACE_LOG timing-models=$DYNAMATIC_DIR/data/components.json target-period=$TARGET_CP dump-file=$SWITCHING_LOG debug=$SWITCH_DEBUG debug-categories=$SWITCH_DEBUG_CATEGORIES dump-data-channels=$SWITCH_DUMP_DATA_CHANNELS dump-mg-handshake=$SWITCH_DUMP_MG_HANDSHAKE" \
    2>&1 | tee "$F_HANDSHAKE_SWITCH"; then
    echo_fatal "Failed to place smart buffers and estimate switches"
    exit 1
fi
echo_info "Placed smart buffers and estimated switching"

echo_section "[Step 3] Comparing estimation and full-trace VCD for ${KERNEL_NAME}"
if [[ ! -f "$VCD_GOLDEN" ]]; then
    echo_fatal "Golden VCD not found at $VCD_GOLDEN"
    exit 1
fi

COMPARE_EXIT=0
if [[ -t 1 ]]; then
    if ! python3 "$COMPARE_SCRIPT" \
        --vcd "$VCD_GOLDEN" \
        --est-csv "$SWITCHING_LOG" \
        --window "$WINDOW_MODE" \
        --reset-signal "$RESET_SIGNAL" \
        --rel-error-threshold "$REL_ERROR_THRESHOLD" \
        --zero-abs-threshold "$ZERO_ABS_THRESHOLD" \
        --ignore-both-below "$IGNORE_BOTH_BELOW" \
        --max-violation-report "$MAX_VIOLATION_REPORT" \
        2>&1 | tee "$COMPARE_LOG" | awk \
            -v red="$(printf '%b' "$RED")" \
            -v nc="$(printf '%b' "$NC")" \
            '/\|[[:space:]]*FAIL[[:space:]]*\|/ { print red $0 nc; next } { print }'; then
        COMPARE_EXIT=1
    fi
else
    if ! python3 "$COMPARE_SCRIPT" \
        --vcd "$VCD_GOLDEN" \
        --est-csv "$SWITCHING_LOG" \
        --window "$WINDOW_MODE" \
        --reset-signal "$RESET_SIGNAL" \
        --rel-error-threshold "$REL_ERROR_THRESHOLD" \
        --zero-abs-threshold "$ZERO_ABS_THRESHOLD" \
        --ignore-both-below "$IGNORE_BOTH_BELOW" \
        --max-violation-report "$MAX_VIOLATION_REPORT" \
        2>&1 | tee "$COMPARE_LOG"; then
        COMPARE_EXIT=1
    fi
fi

if [[ -f "$DEBUG_SCRIPT" ]]; then
    python3 "$DEBUG_SCRIPT" \
        --repo-root "$SCRIPT_DIR" \
        --kernel "$KERNEL_NAME" \
        --window "$WINDOW_MODE" \
        --reset-signal "$RESET_SIGNAL" \
        --rel-error-threshold "$REL_ERROR_THRESHOLD" \
        --zero-abs-threshold "$ZERO_ABS_THRESHOLD" \
        --ignore-both-below "$IGNORE_BOTH_BELOW" \
        --top-k "$DEBUG_TOP_K" \
        2>&1 | tee "$DEBUG_LOG" || true
fi

if [[ $COMPARE_EXIT -ne 0 ]]; then
    echo_fatal "Switching comparison failed acceptance criteria (see $COMPARE_LOG)"
    echo_fatal "Debug report generated under $OUTPUT_DIR (switching_debug_report.md / switching_debug_violations.csv)"
    exit 1
fi
echo_info "Switching comparison passed acceptance criteria"

cd - > /dev/null
