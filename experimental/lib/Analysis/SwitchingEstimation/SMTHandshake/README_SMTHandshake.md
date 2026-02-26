# SMT RTL Handshake Estimation (MG Steady-State)

## What This Adds
This directory implements an optional Z3-based MG steady-state handshake estimator that runs in parallel with the legacy heuristic/event-driven path.

- Default behavior is unchanged.
- SMT path is enabled only when both:
  - build option `-DDYNAMATIC_ENABLE_SMT_HANDSHAKE=ON`
  - pass option `--switching-estimation=... use-smt-handshake=1`

If SMT is UNSAT/UNSUPPORTED/ERROR for an MG, the pass falls back to the legacy estimator for that MG only.

Detailed implementation notes for select-hint integration and UNSAT/crash fixes:
- `SMT_SELECT_HINTS_AND_UNSAT_FIXES.md`
Detailed data+handshake verification notes (with concrete examples and latest
benchmark numbers):
- `DATA_HANDSHAKE_VERIFICATION_NOTES.md`

Solver policy used by the SMT estimator:
- Attempt 1 (strict): heuristic buffer `SET_V/SET_R` pinning + strict select
  determinism + exact transfer-cardinality policy.
- If strict fails due selector don’t-cares: accept permissive-determinism SAT.
- If strict is UNSAT: progressively relax transfer cardinality while keeping
  strict buffer assumptions (`exact-one` -> `at-most-one` -> semi-unconstrained
  mandatory-backbone activity).
- If still UNSAT: switch to **RTL-stateful buffer modeling** and try in order:
  1. `RTLStatefulWithOccupancyHints` (exact occupancy counts from `SET_V/SET_R`)
  2. `RTLStatefulWithOccupancyRangeHints` (occupancy counts with `k±1` slack)
  3. `RTLStateful` (pure RTL-state model, no occupancy hints)
  each retried under `exact-one`, then `at-most-one`, then semi-unconstrained.
- Final fallback: fully unconstrained transfers (with diagnostics) and, if
  needed, occupancy-preserving relaxed buffer constraints.

II=1 optimization:
- For MGs with `II=1`, SMT solving is skipped. The estimator emits direct
  1-cycle waveforms and zero toggles for all channels.
- Example: `valid=[1]`, `ready=[0]`, `transfer=[0]` still yields `0` toggles
  for valid/ready/transfer because cyclic edge compares bit `0` to itself.

## Build
### 1) Build/install Z3 locally
Expected install prefix:
- `/home/jianliu/TCAD25/third_party/z3/install`

Expected artifacts:
- `include/z3.h`
- `include/z3++.h`
- `lib64/libz3.so` (or `lib/libz3.so`)

### 2) Configure/build Dynamatic with SMT option
```bash
cmake -S . -B build \
  -DDYNAMATIC_ENABLE_SMT_HANDSHAKE=ON \
  -DDYNAMATIC_SMT_Z3_ROOT=/home/jianliu/TCAD25/third_party/z3/install
cmake --build build -j
```

## Pass Options
Added in `switching-estimation`:
- `use-smt-handshake` (bool, default `false`)
- `smt-handshake-dump` (string, default empty)

Runtime env knob:
- none; legacy fallback is always applied per non-SAT MG.

Example:
```bash
bin/dynamatic-opt <handshake_transformed.mlir> \
  --handshake-mark-fpu-impl=impl=vivado \
  --handshake-set-buffering-properties=version=fpga20 \
  --handshake-place-buffers="..." \
  --switching-estimation="data-trace=<trace> timing-models=<components.json> target-period=8 \
                         dump-file=<switch.csv> use-smt-handshake=1 \
                         smt-handshake-dump=<smt_handshake_dump.json>"
```

## SMT Model
The model is built per MG over `t in [0, II-1]`.

For each edge `e`:
- `v_e[t]`: valid
- `r_e[t]`: ready
- `x_e[t]`: transfer
- hard constraint: `x_e[t] == v_e[t] & r_e[t]`

Core transfer-cardinality policy:
- `mandatory` edges (non-select backbone): exact-one per II in strict mode.
- `optional` edges (select/arbitration/memory-facing): at-most-one in strict mode.
- semi-unconstrained fallback keeps at-least-one on mandatory edges and leaves
  optional edges symbolic.

### Phase anchoring
To remove rotational symmetry:
- preferred anchor: a non-buffer backedge
- fallback: deterministic lexicographic non-buffer edge
- enforce anchor transfer onehot at `t=0`

If anchor cannot be enforced, extracted SAT waveforms are normalized by rotation so the first transfer of the selected base edge is at `t=0`.

### Buffer handling policy
Buffer modes are implemented and selected automatically by fallback:
1. `StrictPinning`: legacy `updateMGBufferSwitching(...)` provides per-cycle
   `setV/setR`, and SMT hard-pins those waveforms.
2. `RelaxedOccupancy`: keeps only activity counts from `setV/setR`
   (`sum v = K`, `sum r = K`) without phase pinning.
3. `RelaxedOccupancyRange`: keeps occupancy counts as bounded approximations
   (`K-1 <= sum v <= K+1`, `K-1 <= sum r <= K+1`, clamped to `[0, II]`).
4. `RTLStatefulWithOccupancyHints`: enforces exact RTL state equations and
   adds occupancy-count constraints from `setV/setR`.
5. `RTLStatefulWithOccupancyRangeHints`: same as (4) but with `k±1` count
   slack to absorb approximation error.
6. `RTLStateful`: ignores heuristic phase/count hints and encodes buffer handshake
   equations directly with periodic state:
   - `ONE_SLOT_BREAK_R` as TEHB
   - `ONE_SLOT_BREAK_DV` / `SHIFT_REG_BREAK_DV` as OEHB
   - `FIFO_BREAK_DV` / `ONE_SLOT_BREAK_DVR` as non-bypass elastic queue
   - `FIFO_BREAK_NONE` as bypass `tfifo_dataless` queue.

Concrete example (II=4):
- occupancy heuristic gives `setV(buffer->X) = {0,1}` and
  `setR(Y->buffer) = {1,2,3}`.
- strict mode pins:
  - `v_buffer_X = 1100`
  - `r_Y_buffer = 0111`
- relaxed mode enforces only counts:
  - `sum_t v_buffer_X[t] = 2`
  - `sum_t r_Y_buffer[t] = 3`
- this keeps occupancy-derived duty cycle while allowing phase shift to satisfy
  non-buffer RTL constraints.

Concrete RTL-stateful example (TEHB):
- `ins_ready[t] = !full[t]`
- `outs_valid[t] = ins_valid[t] | full[t]`
- `full[t+1] = (ins_valid[t] | full[t]) & !outs_ready[t]`
- periodicity: `full[0] = full[II]`

Concrete RTL-stateful example (multi-slot TFIFO/elastic queue):
- state: onehot occupancy `occ[k][t]` for `k in [0..NUM_SLOTS]`
- periodicity: `occ[k][0] = occ[k][II]`
- `nonEmpty[t] = OR_{k>=1} occ[k][t]`, `full[t] = occ[NUM_SLOTS][t]`
- `outs_valid[t] = ins_valid[t] | nonEmpty[t]` (bypass tfifo) or `nonEmpty[t]`
  (non-bypass queue)
- `ins_ready[t] = !full[t] | outs_ready[t]`
- transitions:
  - write-only: `occ(t+1)=occ(t)+1`
  - read-only: `occ(t+1)=occ(t)-1`
  - both/none: occupancy unchanged.

### Occupancy contradiction diagnostics
When occupancy-guided RTL solve is UNSAT, solver now computes a concrete
diagnostic by incrementally relaxing buffer occupancy hints and reporting the
first relaxed set that becomes SAT.

`histogram_transition` examples from
`integration-test/histogram_transition/out/comp/smt_handshake_dump.json`:
- MG0:
  - `RTLStatefulWithOccupancyHints` exact-one is UNSAT.
  - SAT appears only after relaxing occupancy hints for almost all buffers
    (`buffer0..buffer21` except `buffer9`), which indicates global
    contradiction between occupancy approximation and exact RTL coupling.
- MG1:
  - exact occupancy-hint mode is UNSAT.
  - `RTLStatefulWithOccupancyRangeHints` exact-one is SAT.

Interpretation:
- `SET_V/SET_R` occupancy approximations are useful as guidance, but not always
  globally compositional under exact RTL equations across forks/muxes/control
  merges/backpressure cycles.
- Solver therefore preserves occupancy hints when feasible, but falls back to
  less constrained RTL modes when diagnostics prove those hints contradictory.

## Detailed RTL Modeling (With Concrete Examples)

### Global channel constraints
- Variables for each channel `e` and cycle `t`: `v_e[t]`, `r_e[t]`, `x_e[t]`.
- Always enforced: `x_e[t] = v_e[t] & r_e[t]`.
- Non-buffer edges:
  - primary mode: `sum_t x_e[t] = 1`.
  - final fallback mode: `sum_t x_e[t] <= 1`.

Concrete example (II=3):
- if `v=101`, `r=011`, then `x=001`, which satisfies exactly-one.

### `join_type` (e.g. addi/subi/cmpi/andi/ori/shifts)
- `out_valid[t] = in0_valid[t] & in1_valid[t]`.
- `in0_ready[t] = out_ready[t] & in1_valid[t]`.
- `in1_ready[t] = out_ready[t] & in0_valid[t]`.

Concrete example:
- if `in0_valid=1`, `in1_valid=0`, `out_ready=1`, then
  `out_valid=0`, `in0_ready=0`, `in1_ready=1`.

### `fork_dataless` + eager fork register block (stateful)
- state per output: `transmitValue_i[t]`.
- `keep_i[t] = (!out_ready_i[t]) & transmitValue_i[t]`.
- `anyBlockStop[t] = OR_i keep_i[t]`.
- `in_ready[t] = !anyBlockStop[t]`.
- `backpressure[t] = in_valid[t] & anyBlockStop[t]`.
- `out_valid_i[t] = transmitValue_i[t] & in_valid[t]`.
- update:
  `transmitValue_i[t+1] = keep_i[t] | !backpressure[t]`.
- periodicity:
  `transmitValue_i[0] = transmitValue_i[II]`.

Concrete example (2 outputs):
- if output0 is stalled (`out_ready0=0`) and output1 is ready (`out_ready1=1`),
  fork keeps pending token for output0 while allowing output1 progress according
  to `transmitValue` state.

### `mux`
- selector represented as data-input index (`0..N-1`) per cycle.
- `selected_valid[t] = OR_i (sel[t]=i & index_valid[t] & in_valid_i[t])`.
- `out_valid[t] = selected_valid[t]`.
- `index_ready[t] = !index_valid[t] | (selected_valid[t] & out_ready[t])`.
- `in_ready_i[t] = ((sel[t]=i) & index_valid[t] & in_valid_i[t] & out_ready[t]) | !in_valid_i[t]`.
- selector can be fixed by pass hints per cycle; otherwise it remains symbolic
  with onehot/exclusivity constraints and determinism checks.

Concrete example:
- `sel=1`, `index_valid=1`, `in_valid1=1`, `out_ready=1`
  implies `out_valid=1`, `in_ready1=1`, other data inputs only get
  `in_ready_i = !in_valid_i`.

### `cond_br_dataless`
- `branch_valid[t] = data_valid[t] & condition_valid[t]`.
- `true_valid[t]  = condition[t]  & branch_valid[t]`.
- `false_valid[t] = !condition[t] & branch_valid[t]`.
- `branch_ready[t] = (false_ready[t] & !condition[t]) |
                     (true_ready[t]  &  condition[t])`.
- `data_ready[t] = condition_valid[t] & branch_ready[t]`.
- `condition_ready[t] = data_valid[t] & branch_ready[t]`.

Concrete example:
- `condition=0`, `data_valid=1`, `condition_valid=1`, `false_ready=1`
  gives `false_valid=1`, `true_valid=0`, `data_ready=1`.

### `control_merge_dataless`
- modeled as composition of:
  - merge-like selection,
  - `tehb_dataless` state (`fullReg[t]`),
  - fork behavior on the selected control/data.
- all introduced state has periodic constraints:
  `state[0] = state[II]`.

Concrete example:
- when `tehb` is full and downstream not ready, merge intake blocks until
  output transfer releases `fullReg`.

### `muli` (join + delay_buffer + oehb_dataless)
- input join constraints as above.
- delay line state (`LATENCY-1` stages) with enable tied to downstream
  readiness.
- `oehb_dataless` output state (`outputValid[t]`) modeled explicitly.
- all pipeline states enforce periodicity.

Concrete example (latency=3):
- two internal stage-valid bits rotate under enable; output transfer only when
  final stage valid and output ready.

### Pass-through families
- `extsi/extui/trunci/branch/simple merge/source/sink/constant`
  use handshake pass-through abstraction on MG-visible ports:
  - `out_valid[t] = in_valid[t]`
  - `in_ready[t] = out_ready[t]`

Concrete example:
- if downstream is always ready (`out_ready=111`), pass-through input ready is
  also `111`.

### Load/store abstractions
- `load`: modeled as two decoupled TEHB-style handshake interfaces
  (address path and response-data path). For validation flow alignment,
  address-side external ready is treated as always-ready.
- `store`: MG-visible inputs are wired to always-ready external adapters
  (matching non-blocking memory testbench behavior in validation).

### State periodicity rule (all stateful units)
- for every modeled state signal `s`:
  `s[0] = s[II]`.
- this closes the II-cycle unrolling into a steady-state loop.

Concrete example:
- if a fork pending flag is `1` at cycle 0, solver must also make it `1` at
  cycle II boundary, or adjust transitions to satisfy closure.

### RTL constraint builders implemented
- `join_type`
- `fork_dataless` + `eager_fork_register_block` (with periodic internal state)
- `mux`
- `cond_br_dataless`
- `control_merge_dataless` (merge + tehb + fork composition with periodic states)
- `muli` (`join + delay_buffer + oehb_dataless`, periodic states)
- pass-through adapters (`extsi/extui/trunci/pass/source/sink/constant/load/store`)

Unsupported examples:
- `lsq`, unknown/ambiguous classes
- non-deterministic select under steady-state constraints (strict mode)

`lazy_fork` nodes are currently routed through fork-style graph modeling so the
pipeline can run end-to-end; if semantics diverge for a specific MG, the SMT
path may still return `UNSUPPORTED` and fall back.

## Determinism check for select-dependent nodes
After SAT, SMT checks whether select symbols are uniquely determined.

If an alternative select assignment is also SAT:
- strict mode reports non-determinism,
- permissive mode may accept the SAT model and record a diagnostic reason
  (`accepted in permissive mode`) to avoid unnecessary fallback on selector
  don’t-cares.

Concrete example:
- suppose for one `mux` at `t=2`, both `sel=0` and `sel=1` satisfy all
  constraints because both data inputs are invalid (`in_valid0=in_valid1=0`).
- strict mode marks this as non-deterministic select assignment.
- permissive mode accepts the model and records this in `reason`.

## Selector hint sourcing from switching-estimation pass
- `cond_br` condition hints come from existing replay helper
  `getCondBrNodeCondValue(...)`.
- `mux/select` data-input hints come from existing replay helper
  `getMuxDataSrc(...)` and are mapped to RTL port indices.
- if a full II-cycle contiguous run is available, SMT uses per-cycle hint
  vectors; otherwise it falls back to a single-cycle hint and leaves unresolved
  cycles symbolic.

Concrete example:
- replay says `getMuxDataSrc(mux7, iter=120)="addi1"`.
- mux port map is `{cond:0, cmpi0:1, addi1:2}`.
- SMT converts this to selected data index `1` (port `2 - 1`), i.e. the second
  data input.

## Switching count extraction
After SAT model extraction:
- per-signal cyclic toggle count uses:
  `toggles(sig) = |{ t | sig[t] XOR sig[(t+1) mod II] }|`.

Concrete example (II=4):
- `valid = 1100` gives toggles `2` (at indices `1->2` and `3->0`).
- `ready = 1111` gives toggles `0`.

## JSON dump contents
When `smt-handshake-dump` is set, JSON includes:
- per-MG SAT status/reason
- II, anchor edge, rotation
- solver stats (bool vars, constraints, solve time)
- modeled/unsupported unit coverage
- per-edge valid/ready/transfer II-bit waveforms and toggles
- execution metadata:
  - `executed_segment_trace`
  - `execution_phase_runs` (phase, segment, count)

Additional readable reports (generated automatically when `smt-handshake-dump`
is provided):
- `<dump>_summary.csv`: one row per MG (status, II, solver stats, reason)
- `<dump>_mg_reports/mg_<label>_channels.csv`: per-edge waveforms and toggles

## MG-window VCD validation
Script:
- `experimental/lib/Analysis/SwitchingEstimation/SMTHandshake/VCDWindowExtract.py`

Driver for required benches:
- `switching_testing/run_smt_mg_validation.py`

Single command (required benches):
```bash
python3 switching_testing/run_smt_mg_validation.py \
  --bench fir --bench matvec --bench matrix --bench bicg \
  --run-flow --use-smt
```

Validation policy:
1. choose MG runs with `count >= 3`
2. pick interior occurrence with anti-warmup policy (`4th`/`3rd` preferred)
3. detect anchor transfer in VCD (`valid & ready`)
4. map MG occurrence to anchor-transfer index using **anchor-edge coverage
   across all MGs** (fixes shared-anchor misalignment)
5. evaluate small window offsets around the anchor (`[-(II-1), +(II-1)]`)
6. choose best `(offset, rotation)` by anchor-transfer distance, then total
   transfer mismatch
7. compare SMT vs VCD waveforms (valid/ready/transfer) and toggle counts
8. print top worst edges and aggregate mismatch-point totals

Note:
- SMT solving is attempted for all MGs (including non-steady runs), because
  those MG estimates are consumed by whole-circuit handshake accumulation.
- The `count>=3` criterion is applied in VCD window validation only.
- Validation JSON now records:
  - `selected_anchor_transfer_index`
  - `selected_start_cycle`
  - `selected_window_shift`
  - aggregate `valid/ready/transfer_mismatch_points`

## Interpreting common mismatch patterns
- Ready mismatches >> valid mismatches:
  likely backpressure-phase skew in stateful fork/control paths.
- Transfer matches but valid/ready mismatches remain:
  decomposition mismatch even though token-level throughput aligns.
- Uniform one-cycle skew across many edges:
  likely sampling/anchor alignment offset in VCD window extraction.
