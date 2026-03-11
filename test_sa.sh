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

# Prints a warning message to stdout.
#   $1: the text to print
echo_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
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

warn_if_exists() {
    if [[ -e "$1" ]]; then
        echo_warn "Existing artifact will be overwritten: $1"
    fi
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
BUFFER_PLACEMENT_LOG="$OUTPUT_DIR/buffer_placement.log"
SWITCHING_PASS_LOG="$OUTPUT_DIR/switching_estimation_pass.log"
F_FREQUENCIES="$OUTPUT_DIR/frequencies.csv"
VCD_GOLDEN="$SRC_DIR/out/sim/HLS_VERIFY/trace.vcd"

DYNAMATIC_PROFILER_BIN="$SCRIPT_DIR/bin/exp-frequency-profiler"
DYNAMATIC_OPT_BIN="$SCRIPT_DIR/bin/dynamatic-opt"
COMPARE_SCRIPT="$SCRIPT_DIR/switching_testing/test.py"
DEBUG_SCRIPT="$SCRIPT_DIR/switching_testing/debug_switching.py"
COMPONENTS_JSON="$SCRIPT_DIR/data/components.json"
BLIF_DIR="$SCRIPT_DIR/data/aig/"

F_CF_DYN_TRANSFORMED="$OUTPUT_DIR/cf_transformed_mem_interface_marked.mlir"
F_HANDSHAKE_TRANSFORMED="$OUTPUT_DIR/handshake_transformed.mlir"
F_HANDSHAKE_BUFFERED="$OUTPUT_DIR/handshake_buffered.mlir"
F_HANDSHAKE_EXPORT="$OUTPUT_DIR/handshake_export.mlir"
F_HANDSHAKE_SWITCH="$OUTPUT_DIR/handshake_switch_test.mlir"
CFDFC_CACHE_JSON="$OUTPUT_DIR/cfdfc_cache.json"
TARGET_CP=8
MILP_SOLVER="gurobi"
FORCE_BUFFER_PLACEMENT="${FORCE_BUFFER_PLACEMENT:-false}"

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
# COMPARISON_MODE:
#   - node      : print the legacy node-by-node report
#   - aggregate : print only the power-estimator aggregate proxy report
#   - both      : print both node and aggregate reports (default)
# ACCEPTANCE_MODE:
#   - node      : enforce thresholds on the node-by-node report
#   - aggregate : enforce thresholds on the aggregate proxy report (default)
# MAX_VIOLATION_REPORT:
#   - Max detailed violations printed by switching_testing/test.py
# DEBUG_TOP_K:
#   - Top failing entries printed by switching_testing/debug_switching.py
WINDOW_MODE="${WINDOW_MODE:-full}"
RESET_SIGNAL="${RESET_SIGNAL:-}"
REL_ERROR_THRESHOLD="${REL_ERROR_THRESHOLD:-0.10}"
ZERO_ABS_THRESHOLD="${ZERO_ABS_THRESHOLD:-5}"
IGNORE_BOTH_BELOW="${IGNORE_BOTH_BELOW:-50}"
COMPARISON_MODE="${COMPARISON_MODE:-both}"
ACCEPTANCE_MODE="${ACCEPTANCE_MODE:-aggregate}"
MAX_VIOLATION_REPORT="${MAX_VIOLATION_REPORT:-40}"
DEBUG_TOP_K="${DEBUG_TOP_K:-20}"

# ============================================================================ #
# Switching Estimation Flow
# ============================================================================ #
# Pre-flight checks
if [[ ! -f "$F_HANDSHAKE_TRANSFORMED" ]]; then
    echo_fatal "The handshake_transformed.mlir file doesn't exist: $F_HANDSHAKE_TRANSFORMED"
    exit 1
fi
echo_info "Found handshake IR: $F_HANDSHAKE_TRANSFORMED"

if [[ ! -f "$F_CF_DYN_TRANSFORMED" ]]; then
    echo_fatal "The profiled CF IR file doesn't exist: $F_CF_DYN_TRANSFORMED"
    exit 1
fi

if [[ ! -x "$DYNAMATIC_PROFILER_BIN" ]]; then
    echo_fatal "Profiler binary is missing or not executable: $DYNAMATIC_PROFILER_BIN"
    exit 1
fi

if [[ ! -x "$DYNAMATIC_OPT_BIN" ]]; then
    echo_fatal "dynamatic-opt is missing or not executable: $DYNAMATIC_OPT_BIN"
    exit 1
fi

if [[ ! -f "$COMPARE_SCRIPT" ]]; then
    echo_fatal "Comparison script not found: $COMPARE_SCRIPT"
    exit 1
fi

if [[ ! -f "$COMPONENTS_JSON" ]]; then
    echo_fatal "Timing-model database not found: $COMPONENTS_JSON"
    exit 1
fi

if [[ ! -d "$BLIF_DIR" ]]; then
    echo_fatal "AIG/BLIF directory not found: $BLIF_DIR"
    exit 1
fi

REUSE_BUFFER_PLACEMENT=false
if [[ "$FORCE_BUFFER_PLACEMENT" == "true" ]]; then
    echo_warn "FORCE_BUFFER_PLACEMENT=true, existing buffered IR and CFDFC cache will be ignored and regenerated"
elif [[ -s "$CFDFC_CACHE_JSON" || -s "$F_HANDSHAKE_BUFFERED" ]]; then
    if [[ -s "$CFDFC_CACHE_JSON" && -s "$F_HANDSHAKE_BUFFERED" ]]; then
        REUSE_BUFFER_PLACEMENT=true
        echo_info "Reusing existing buffered handshake IR and CFDFC cache; buffer placement will be skipped"
    elif [[ -s "$CFDFC_CACHE_JSON" ]]; then
        echo_fatal "CFDFC cache exists but buffered handshake IR is missing: $F_HANDSHAKE_BUFFERED"
        echo_fatal "Either restore the buffered IR or rerun with FORCE_BUFFER_PLACEMENT=true"
        exit 1
    else
        echo_fatal "Buffered handshake IR exists but CFDFC cache JSON is missing: $CFDFC_CACHE_JSON"
        echo_fatal "Either restore the cache JSON or rerun with FORCE_BUFFER_PLACEMENT=true"
        exit 1
    fi
fi

warn_if_exists "$F_FREQUENCIES"
warn_if_exists "$TRACE_LOG"
warn_if_exists "$F_HANDSHAKE_SWITCH"
warn_if_exists "$SWITCHING_LOG"
warn_if_exists "$COMPARE_LOG"
warn_if_exists "$DEBUG_LOG"
warn_if_exists "$SWITCHING_PASS_LOG"
if [[ "$REUSE_BUFFER_PLACEMENT" != "true" ]]; then
    warn_if_exists "$F_HANDSHAKE_BUFFERED"
    warn_if_exists "$CFDFC_CACHE_JSON"
    warn_if_exists "$BUFFER_PLACEMENT_LOG"
fi

# Run the profiler
echo_section "[Step 1] Running Profiler for ${KERNEL_NAME}"
"$DYNAMATIC_PROFILER_BIN" "$F_CF_DYN_TRANSFORMED" \
    --top-level-function="$KERNEL_NAME" \
    --input-args-file="$OUTPUT_DIR/profiler-inputs.txt" \
    --trace-log-file="$TRACE_LOG" \
    --mode=both > "$F_FREQUENCIES"
if [[ ! -s "$F_FREQUENCIES" ]]; then
    echo_fatal "Profiler finished but did not produce a non-empty frequency CSV: $F_FREQUENCIES"
    exit 1
fi
if [[ ! -s "$TRACE_LOG" ]]; then
    echo_fatal "Profiler finished but did not produce a non-empty trace log: $TRACE_LOG"
    exit 1
fi
echo_info "Profiling Finished"

cd "$OUTPUT_DIR"
export LSAN_OPTIONS=verbosity=1:log_threads=1
export UBSAN_OPTIONS=print_stacktrace=1:halt_on_error=1

# Run buffer placement and emit the CFDFC cache that switching-estimation will
# consume in a separate pass invocation, unless both artifacts already exist.
if [[ "$REUSE_BUFFER_PLACEMENT" == "true" ]]; then
    echo_section "[Step 2] Reusing Existing Buffer Placement Artifacts for ${KERNEL_NAME}"
    echo_info "Using buffered handshake IR: $F_HANDSHAKE_BUFFERED"
    echo_info "Using pre-extracted CFDFC cache JSON: $CFDFC_CACHE_JSON"
else
    echo_section "[Step 2] Running Buffer Placement for ${KERNEL_NAME}"
    echo_info "Emitting buffered handshake IR to: $F_HANDSHAKE_BUFFERED"
    echo_info "Emitting CFDFC cache JSON to: $CFDFC_CACHE_JSON"
    if ! /usr/bin/time -v "$DYNAMATIC_OPT_BIN" "$F_HANDSHAKE_TRANSFORMED" \
        --handshake-mark-fpu-impl="impl=vivado" \
        --handshake-set-buffering-properties="version=fpga20" \
        --handshake-place-buffers="algorithm=fpga20 solver=$MILP_SOLVER frequencies=$F_FREQUENCIES timing-models=$COMPONENTS_JSON target-period=$TARGET_CP timeout=300 dump-logs \
        blif-files=$BLIF_DIR lut-delay=0.55 lut-size=6 acyclic-type cfdfc-cache-out=$CFDFC_CACHE_JSON" \
        -o "$F_HANDSHAKE_BUFFERED" \
        2>&1 | tee "$BUFFER_PLACEMENT_LOG"; then
        echo_fatal "Failed to place smart buffers and emit the CFDFC cache JSON"
        exit 1
    fi
    if [[ ! -s "$F_HANDSHAKE_BUFFERED" ]]; then
        echo_fatal "Buffer placement completed but did not produce buffered handshake IR: $F_HANDSHAKE_BUFFERED"
        exit 1
    fi
    if [[ ! -s "$CFDFC_CACHE_JSON" ]]; then
        echo_fatal "Buffer placement completed but did not produce the CFDFC cache JSON: $CFDFC_CACHE_JSON"
        exit 1
    fi
    echo_info "Placed smart buffers and emitted the CFDFC cache"
fi

echo_section "[Step 3] Running Switching Estimation from CFDFC Cache for ${KERNEL_NAME}"
echo_info "Switching debug flags: debug=${SWITCH_DEBUG}, categories=${SWITCH_DEBUG_CATEGORIES}, dump-data-channels=${SWITCH_DUMP_DATA_CHANNELS}, dump-mg-handshake=${SWITCH_DUMP_MG_HANDSHAKE}"
echo_info "Using buffered handshake IR: $F_HANDSHAKE_BUFFERED"
echo_info "Using pre-extracted CFDFC cache: $CFDFC_CACHE_JSON"
if ! /usr/bin/time -v "$DYNAMATIC_OPT_BIN" "$F_HANDSHAKE_BUFFERED" \
    --switching-estimation="data-trace=$TRACE_LOG timing-models=$COMPONENTS_JSON target-period=$TARGET_CP cfdfc-cache-in=$CFDFC_CACHE_JSON dump-file=$SWITCHING_LOG debug=$SWITCH_DEBUG debug-categories=$SWITCH_DEBUG_CATEGORIES dump-data-channels=$SWITCH_DUMP_DATA_CHANNELS dump-mg-handshake=$SWITCH_DUMP_MG_HANDSHAKE" \
    -o "$F_HANDSHAKE_SWITCH" \
    2>&1 | tee "$SWITCHING_PASS_LOG"; then
    echo_fatal "Failed to estimate switching from the buffered IR and CFDFC cache"
    exit 1
fi
if [[ ! -s "$SWITCHING_LOG" ]]; then
    echo_fatal "Switching-estimation completed but did not produce a non-empty CSV: $SWITCHING_LOG"
    exit 1
fi
if [[ ! -s "$F_HANDSHAKE_SWITCH" ]]; then
    echo_warn "Switching-estimation did not leave behind the expected IR output: $F_HANDSHAKE_SWITCH"
fi
echo_info "Estimated switching using cached CFDFC data"

echo_section "[Step 4] Comparing estimation and full-trace VCD for ${KERNEL_NAME}"
echo_info "Comparison config: window=${WINDOW_MODE}, comparison-mode=${COMPARISON_MODE}, acceptance-mode=${ACCEPTANCE_MODE}, reset-signal=${RESET_SIGNAL:-auto}"
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
        --comparison-mode "$COMPARISON_MODE" \
        --acceptance-mode "$ACCEPTANCE_MODE" \
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
        --comparison-mode "$COMPARISON_MODE" \
        --acceptance-mode "$ACCEPTANCE_MODE" \
        --max-violation-report "$MAX_VIOLATION_REPORT" \
        2>&1 | tee "$COMPARE_LOG"; then
        COMPARE_EXIT=1
    fi
fi

if [[ -f "$DEBUG_SCRIPT" ]]; then
    if ! python3 "$DEBUG_SCRIPT" \
        --repo-root "$SCRIPT_DIR" \
        --kernel "$KERNEL_NAME" \
        --window "$WINDOW_MODE" \
        --reset-signal "$RESET_SIGNAL" \
        --rel-error-threshold "$REL_ERROR_THRESHOLD" \
        --zero-abs-threshold "$ZERO_ABS_THRESHOLD" \
        --ignore-both-below "$IGNORE_BOTH_BELOW" \
        --top-k "$DEBUG_TOP_K" \
        2>&1 | tee "$DEBUG_LOG"; then
        echo_warn "Supplemental debug report generation failed; continuing because the main comparison already finished"
    fi
else
    echo_warn "Debug helper script not found; skipping supplemental debug report generation: $DEBUG_SCRIPT"
fi

if [[ $COMPARE_EXIT -ne 0 ]]; then
    echo_fatal "Switching comparison failed acceptance criteria (see $COMPARE_LOG)"
    echo_fatal "Debug report generated under $OUTPUT_DIR (switching_debug_report.md / switching_debug_violations.csv)"
    exit 1
fi
echo_info "Switching comparison passed acceptance criteria"

cd - > /dev/null
