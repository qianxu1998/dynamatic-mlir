# Whole-Execution Handshake Aggregation Fix

This note documents the fix applied to whole-circuit handshake accumulation when SMT steady-state MG estimates are enabled.

## Scope

The change is in:

- `experimental/lib/Analysis/SwitchingEstimation/SwitchingEstimation.cpp`
  - `computeTotalHandshakeSwitching(...)`

No interface/CLI changes were required.

## Problem

Previous MG accumulation used a scalar model:

- `per-run contribution = numExec * per-MG steady-state toggles`
- plus a special `II=1` boundary boost (`+2`) for active channels.

This inflated totals for circuits with many MG phase runs and mixed run lengths, especially:

- repeated MG alternation (`matrix`, `matvec`, `bicg`)
- channels where run boundaries dominate behavior

## Implemented Fix

### 1) Run-level, edge-level accumulation

Instead of scaling node totals by a scalar, we now accumulate **per edge** using the MG waveform sets:

- valid waveform source: `mgSrc->setV[dst]`
- ready waveform source: `mgDst->setR[src]`

For each run (`segment`, `runLen=numExec`) and each edge:

1. Count entry boundary transition from previous global level:
   - `entry = (prev_level != first_bit)`
2. Count transitions inside the run by repeating the II-bit pattern:
   - non-cyclic transitions inside one execution
   - inter-execution boundary transitions (`last_bit -> first_bit`)
3. Update global current level to the run-ending level (`last_bit`).

This is implemented in `countRunSwitches(...)` in `computeTotalHandshakeSwitching`.

### 2) Global per-edge logic-state tracking

Added two maps for whole-execution boundary continuity:

- `curValidLevelByEdge["src->dst"]`
- `curReadyLevelByEdge["src->dst"]`

This prevents repeated artificial pulses when consecutive runs keep the same edge level.

### 3) Removed scalar II=1 boundary boost from MG aggregation

The old per-channel `+2` correction is no longer used in MG accumulation. Boundaries are now counted directly by level transitions.

### 4) S/T segment refinement (boundary-aware pulse)

S/T phases are still heuristic, but no longer blindly write fixed values without
boundary context.

For each active output edge in S/T:

- one execution contributes a pulse model (`0->1->0`)
- boundary-aware count:
  - if edge starts low: `2*N` transitions for `N` executions
  - if edge starts high: `2*N-1` (first rise already present)

Ready edges in S/T are settled to low, and any required boundary fall is counted
once.

### 5) New whole-execution diagnostics CSVs

When `smt-handshake-dump=<path>.json` is provided, two extra files are emitted:

- `<path>_phase_handshake.csv`
  - one row per execution phase
  - includes: segment label/type, mode, run length, II, touched nodes, added
    valid/ready transitions
- `<path>_segment_handshake_summary.csv`
  - aggregated totals per segment across all runs

Example paths:

- `integration-test/matrix/out/comp/smt_handshake_dump_phase_handshake.csv`
- `integration-test/matrix/out/comp/smt_handshake_dump_segment_handshake_summary.csv`

## Concrete Example

Given `II=2`, waveform bits `10`, run length `C=3`:

- one execution non-cyclic transitions: `1` (`1->0`)
- inter-execution boundary transition: `1` (`0->1`)
- run internal transitions: `C*1 + (C-1)*1 = 5`
- plus `entry` if prior level differs from the first bit.

This reflects actual concatenated waveform transitions, not a scalar multiply.

## What Was Tried and Reverted

Two experimental changes were tested and rejected:

1. **II=1 fast-path from inferred set/activity bits**
   - Result: severe MG-window regression on `fir/matvec/matrix/bicg` (many 1-cycle mismatches).
   - Reverted.

2. **Short-run (`count<3`) boundary-only suppression**
   - Result: severe full-trace regression on `matrix/matvec` valid/ready.
   - Reverted.

Current code keeps the stable run-level waveform model above.

## Verification

Validation flow rerun on:

- `fir`
- `matvec`
- `matrix`
- `histogram_transition`
- `bicg`

### MG-window validation (steady-state windows)

- `histogram_transition`: MG0/MG1 remain `OK` with low mismatch counts.
- `matrix`: MG0 remains `OK`; MG1/MG2/MG3/MG4 remain `SKIPPED` (`count<3` runs).
- `matvec`: MG0 `OK`, MG1 `SKIPPED`.
- `bicg`: MG0 `OK`, MG1 `SKIPPED`.
- `fir`: MG0 `OK`.

### Full-trace handshake comparison impact (post-reset)

Observed pass-rate deltas (legacy scalar model -> run-level waveform model):

- `matvec`
  - valid: `25.53% -> 76.62%`
  - ready: `15.62% -> 38.55%`
- `matrix`
  - valid: `42.37% -> 72.61%`
  - ready: `26.26% -> 36.99%`
- `histogram_transition`
  - valid: `90.32% -> 87.10%`
  - ready: `20.75% -> 20.75%` (unchanged)
- `bicg`
  - valid: `39.50% -> 30.00%`
  - ready: `19.66% -> 25.49%`

Interpretation:

- The fix significantly reduces over-scaling in high-run alternating cases (`matrix`, `matvec`).
- Remaining mismatch is primarily from:
  - S/E/T segment heuristics
  - MGs without steady runs (`count<3`) where no steady-window calibration exists
  - known ready-phase skew on specific edges (e.g., `histogram_transition` control-merge/lazy-fork path)

## Repro Commands

Build:

```bash
cmake --build build -j
```

Run profiling + switching + MG-window validation:

```bash
python3 switching_testing/run_smt_mg_validation.py \
  --bench fir --bench matvec --bench matrix --bench histogram_transition --bench bicg \
  --run-flow --use-smt
```

Compare full-trace CSV vs VCD:

```bash
python3 switching_testing/test.py \
  --vcd integration-test/<bench>/out/sim/HLS_VERIFY/trace.vcd \
  --est-csv integration-test/<bench>/out/comp/switching_estimation.csv \
  --window post-reset --no-enforce-thresholds
```
