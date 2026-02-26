# Data + Handshake Verification Notes (matrix/gsum/gemver/bicg/gcd)

This note documents the latest systematic fixes made in the switching-estimation
pipeline and the measured results against ModelSim VCD.

## Scope
- Handshake channel:
  - MG steady-state handshake is estimated with SMT (Z3) when enabled.
  - Whole-circuit handshake still uses segment/run accumulation.
- Data channel:
  - Compared estimator CSV vs VCD post-reset window.
  - Focused on robust modeling fixes (no benchmark-specific scaling constants).

## Concrete modeling fixes

### 1) Sparse profiler hole filling for ALU nodes
File: `experimental/lib/Analysis/SwitchingEstimation/DataChannelCal.cpp`

Problem:
- Some ALUs (for example in `gsum`) are sampled sparsely in software profile,
  while operands evolve at much higher cadence in hardware.
- This produced severe under-estimation (for example `addi0` at ~39 vs VCD ~2475).

Fix:
- Added `materializeMissingOriginalValue(...)` in glitch update flow.
- For a missing iteration `i`, ALU output is re-evaluated from operand timelines:
  - `addi = op0 + op1`
  - `subi = op0 - op1`
  - `muli = op0 * op1`
  - bitwise and shifts similarly
  - pass-through family (`extsi/extui/trunci/branch/merge`) forwards operand.
- Operand source resolution is anchored to the node’s home segment (MG owning
  the node’s BB), not the currently executed segment, to avoid wrong source
  lookups outside local transitions.

Example:
- If `addi0` has no explicit sample at iteration 487, but operands resolve to
  `fork6#3=101` and `const6=6`, we materialize `addi0[487]=107`.

### 2) Empty glitch waveform prevention + equal-arrival handling
File: `DataChannelCal.cpp`

Problem:
- Some glitch nodes could end with empty `glitchDataOutByIter[i]`, suppressing
  downstream propagation.
- Equal-arrival cases with glitching parents were not modeled.

Fix:
- Added guaranteed non-empty fallback:
  - if transient list is empty, push stable value at iteration `i`.
- Added equal-arrival interleaving model:
  - build source waveforms for both operands
  - interleave transitions to synthesize output transient sequence.
- Fixed two logic bugs in glitch computation:
  - removed accidental `op1` shadowing
  - fixed missing assignment in mux-source branch.

Example:
- If `srcA=[3,7]`, `srcB=[10,14]` in same cycle bucket, output now evaluates:
  - `f(3,10) -> f(7,10) -> f(7,14)`.

### 3) Invalid/sentinel data handling (`-1`) changed to hold-level semantics
Files:
- `experimental/lib/Analysis/SwitchingEstimation/GraphModel.cpp`
- `experimental/lib/Analysis/SwitchingEstimation/NodeModels.cpp`
- `DataChannelCal.cpp`

Problem:
- Treating `-1` as a real data value produced artificial XOR bursts.

Fix:
- `-1` is now interpreted as “no explicit data event this step -> hold last
  value”.
- Applied in:
  - `AdjNode::updateDataoutChannel(...)`
  - `AdjNode::totalDataSwitchingCounting(...)`
  - `CMergeNode::calDataout/updateDataout(...)`
  - control_merge data continuity in base-node update.

Example:
- Sequence `[42, -1, -1]` is treated as `[42, 42, 42]` (0 extra toggles), not
  `[42, -1, -1]`.

### 4) Mux transition heuristic cleanup
File: `DataChannelCal.cpp`

Problem:
- Legacy pre/next segment transition injection for muxes often over-counted
  data toggles.

Fix:
- Disabled injected pre/next transition value lists for mux data.
- Kept current-iteration selected-source waveform as the mux output basis.

### 5) Control-merge synthetic pulse cleanup
File: `DataChannelCal.cpp`

Problem:
- Injected synthetic `1->0` control pulses caused systematic over-estimation in
  mux/control paths.

Fix:
- Removed unconditional synthetic control pulse injection.
- Kept only profiled/derived control events.

### 6) II=1 whole-circuit handshake boundary correction
File: `experimental/lib/Analysis/SwitchingEstimation/SwitchingEstimation.cpp`

Problem:
- II=1 MGs have zero cyclic toggles in steady-state waveform, but whole-run
  traces still include run-entry/run-exit transitions.

Fix:
- In whole-circuit accumulation, add boundary correction (+2 toggles/channel per
  contiguous II=1 run) when channel activity exists (`setV/setR bit0 = 1`).

Example:
- For 100 consecutive II=1 executions of one MG, steady cyclic term is 0, but
  boundary correction contributes approximately one rise + one fall.

## Commands used

Build:
```bash
cmake --build build -j --target dynamatic-opt
```

Regenerate switching-estimation outputs:
```bash
python3 - <<'PY'
from pathlib import Path
from switching_testing.run_smt_mg_validation import BenchPaths, run_switching_estimation
repo=Path('/home/jianliu/TCAD25/dynamatic-mlir')
for b in ['matrix','gsum','gemver','bicg','gcd']:
    run_switching_estimation(BenchPaths(b,repo),target_period=8.0,milp_solver='gurobi',use_smt=True)
PY
```

Run MG-window SMT-vs-VCD comparison:
```bash
python3 switching_testing/run_smt_mg_validation.py \
  --bench matrix --bench gsum --bench gemver --bench bicg --bench gcd \
  --use-smt
```

## Current results

### Handshake MG-window validation
- `matrix`: MG0 `OK`; MG1/2 no steady run; MG3/4 UNSAT.
- `gsum`: MG0 `OK`; MG1 no steady run.
- `gemver`: MG0/MG1/MG2/MG6 `OK`; MG3/4 no steady run; MG5 UNSAT.
- `bicg`: MG0 `OK`; MG1 no steady run.
- `gcd`: MG1/MG2 `OK`; MG0 no steady run.

Interpretation:
- Supported steady-state MG windows align at transfer/toggle level for `OK`
  MGs.
- Some II>1 MGs remain UNSAT/unsupported in current model.

### Data-channel post-reset comparison (golden>50 focus)
- `matrix`: data pass<=3% `112/137` (81.75%), WMAPE `2.08%`.
- `gsum`: data pass<=3% `21/34` (61.76%), WMAPE `9.27%`.
- `gemver`: data pass<=3% `178/261` (68.20%), WMAPE `40.82%`.
- `bicg`: data pass<=3% `36/86` (41.86%), WMAPE `6.78%`.
- `gcd`: data pass<=3% `2/48` (4.17%), WMAPE `45.25%`.

## Remaining dominant mismatch patterns
- `mux` / `control_merge` data channels:
  - still the largest systematic source of large relative error.
- specific buffers:
  - some buffers remain strongly under/over due data-valid gating mismatch
    between profile-driven replay and RTL combinational visibility.
- whole-circuit handshake:
  - MG steady-state matches where comparable, but global S/T/E heuristics still
    dominate errors in full-trace node-level valid/ready totals.

## Next systematic targets
1. Replace current mux/control_merge data replay with explicit valid-gated
   cycle sampling using handshake `setV/setR` windows.
2. Derive buffer data updates from transfer (`valid&ready`) events rather than
   raw predecessor value changes.
3. For full-trace handshake totals, split S/T/E accumulation into explicit
   edge-level transition counting instead of fixed constants.
