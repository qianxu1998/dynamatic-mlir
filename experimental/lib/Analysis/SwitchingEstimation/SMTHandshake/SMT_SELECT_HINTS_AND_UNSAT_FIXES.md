# SMT Handshake: Select-Hint Integration and UNSAT/Crash Fixes

This note documents the concrete implementation changes made to stabilize the SMT-based MG steady-state handshake estimator and its VCD validation flow.

## 1) Select bits now come from switching-estimation replay data

### What was added
- `collectSelectHints(...)` now extracts selector information from existing pass state (`SwitchingInfo`) for:
  - `cond_br` nodes (condition bits)
  - `mux`/`select` nodes (selected data-input index)
- Preferred source for mux selection:
  - `getMuxDataSrc(...)` from `DataChannelCal` (existing replay helper)
  - then mapped to RTL data-input index using `preNameToPortIdxMap`
- Fallback source:
  - sampled control value from data-state / edge-history maps.

### Per-cycle support over II
- Select hints support both:
  - `*_ByCycle` waveforms (preferred)
  - single-cycle constants (fallback)
- New fields in `SelectHints`:
  - `muxSelectedDataInputByCycle`
  - `condBrCondBitByCycle`
  - plus legacy scalar maps kept for fallback.

### Why this matters
- Previously, selectors were effectively constant symbols; this could over-constrain II>1 MGs.
- Now, constraints can consume per-cycle selector hints when a valid contiguous II window is available.

## 2) RTL constraints fixed to match MG-local topology

### Mux hint mapping fix
- Mux hints are now interpreted in **RTL index space** and then mapped to MG-local visible inputs.
- This fixes prior failures like:
  - `mux select hint out of range: idx=1 but num_inputs=1`
- MG-local single-input mux views are handled as structurally deterministic.

### Cycle-varying selector constraints
- `addMux(...)` now creates selector choice booleans per cycle `t`.
- `addCondBr(...)` now creates `cond_true[t]` / `cond_false[t]` per cycle.
- For unresolved cycles, onehot constraints are added and determinism is checked post-SAT.

### Constant-hint safety for II>1
- Single-sample constant hints are only forced for `II==1`.
- For `II>1`, if no full per-cycle hint window exists, selectors stay symbolic (with onehot), avoiding false UNSAT from over-constraining control.

### UNSAT fallback for non-steady/choice-heavy MGs
- Strict mode keeps exact buffer phase pinning from `SET_V/SET_R`.
- Relaxed buffer mode now still respects buffer occupancy results by enforcing
  the same active-cycle counts (`popcount(SET_V)`, `popcount(SET_R)`) while
  allowing phase relocation.
- Added a final fallback solve mode that relaxes transfer cardinality from:
  - exactly-one transfer per II
  to
  - at-most-one transfer per II
- This is only used after strict and relaxed-buffer exact-one attempts fail.
- It resolves cases like `gcd` MG0 where exact-one assumptions are too
  restrictive.

## 3) Steady-state policy split: solving vs validation

### Current behavior
- SMT solve is attempted for all MGs, including MGs that are not observed in a
  `count>=3` consecutive run.
- The `count>=3` criterion is enforced by the VCD comparison script when
  selecting steady-state windows.

### Practical impact
- MGs that appear only in non-steady execution can still contribute SMT-based
  estimates to whole-circuit handshake accumulation.
- Validation remains focused on steady-state windows, consistent with the paper
  assumptions.

## 4) Graph-construction/runtime crash fixes needed for new benchmarks

### `select` support in graph build
- `handshake.select` is now modeled through mux-style node construction in `GraphModel`.
- `MuxNode` constructor now supports both `handshake.mux` and `handshake.select` ops.

### `cond_br` operand-name robustness
- `CBrNode` constructor now handles block-argument operands safely.
- Prevents null-defining-op dereference crashes seen in `gcd`.

### `lazy_fork` graph support
- `handshake.lazy_fork` now creates a fork node in graph construction (instead of null), allowing full graph formation.
- SMT builder no longer hard-rejects `lazy_fork` names outright.

### negative-cycle normalization fix
- `normalizeCycleIndex(...)` now properly wraps negative cycle indices into `[0, II)`.
- Fixes `SmallBitVector::set` assertion crashes in fallback heuristic paths.

## 5) Validation status after fixes

Validation command used:
```bash
python3 switching_testing/run_smt_mg_validation.py \
  --bench fir --bench matvec --bench matrix --bench bicg \
  --bench gcd --bench gsum --bench gemver \
  --use-smt
```

Observed steady-state comparison outcomes:
- `fir`: MG0 `OK`
- `matvec`: MG0 `OK`, MG1 skipped (no run>=3)
- `matrix`: MG0 `OK`, MG1..MG4 skipped (no run>=3)
- `bicg`: MG0 `OK`, MG1 skipped (no run>=3)
- `gcd`: MG1 `OK`, MG2 `OK`, MG0 skipped (unsupported: no run>=3)
- `gsum`: MG0 `OK`, MG1 skipped (unsupported)
- `gemver`: MG0/MG1/MG2/MG6 `OK`, MG3/MG4/MG5 skipped (unsupported)

## 7) New report outputs

When `smt-handshake-dump` is set, sidecar CSVs are emitted:
- `<dump>_summary.csv`: per-MG status/stats/reason
- `<dump>_mg_reports/mg_<label>_channels.csv`: per-edge valid/ready/transfer
  bit-vectors and toggle counts

Also, `II=1` MGs now use a fast path (no SMT solve) that directly emits
zero-toggle channel estimates.

## 6) Notes on remaining mismatch diagnostics

Some `OK` MG reports still show per-edge bit mismatches (typically II=1 windows).
- Common pattern: mismatch concentrated on buffer/control-adjacent channels.
- Transfer/toggle mismatches are logged for diagnosis; they do not currently force `SKIPPED` status when the MG-level compare succeeds.

## 8) 2026-02 update: shared-anchor and buffer-RTL fixes

### Shared-anchor window mapping fix
- `VCDWindowExtract.py` previously mapped MG occurrence -> anchor transfer using
  only occurrences of the current MG label.
- When the same anchor edge exists in multiple MGs, this selected the wrong VCD
  window (major source of false mismatches).
- New behavior:
  - build anchor coverage from all `mg_results`
  - map by occurrence rank over all executed segments that contain the anchor
  - record `selected_anchor_transfer_index`, `selected_start_cycle`,
    `selected_window_shift` in validation JSON.

### Offset+rotation co-optimization
- VCD compare now evaluates start offsets in `[-(II-1), +(II-1)]` around the
  mapped anchor transfer and chooses best `(offset, rotation)` alignment.
- Objective order:
  1. anchor transfer Hamming distance
  2. total transfer mismatch
  3. total valid+ready mismatch.

### RTL-stateful buffer solver mode
- Added buffer mode `RTLStateful` in SMT solver fallback.
- Encodes periodic handshake state directly for:
  - `ONE_SLOT_BREAK_R` (TEHB)
  - `ONE_SLOT_BREAK_DV`/`FIFO_BREAK_DV` (OEHB-like)
  - `FIFO_BREAK_NONE`/`ONE_SLOT_BREAK_DVR` (TFIFO-like one-slot occupancy abstraction)
- This mode is now tried before strict heuristic `at-most-one` fallback.
- Practical effect: `histogram_transition` MG0 and MG1 both became SAT in
  RTL-stateful exact-one mode instead of degrading to fully unconstrained
  transfer models.

### Load/store adapter assumptions for validation flow
- Load address-side external ready is fixed to `1`.
- Store external ready is fixed to `1`.
- These match the non-blocking memory interface behavior seen in the
  HLS_VERIFY simulation setup and prevent artificial stall-heavy SAT models.

### Additional validation metrics
- `VCDWindowExtract.py` now reports aggregate mismatch points per MG:
  - `valid_mismatch_points`
  - `ready_mismatch_points`
  - `transfer_mismatch_points`
- Printed summary line format:
  - `mismatch_points: valid=A/T, ready=B/T, transfer=C/T`.

These diagnostics remain available in each benchmark JSON:
- `integration-test/<bench>/out/comp/smt_mg_window_validation.json`

## 9) 2026-02 occupancy-approximation contradiction analysis and fix

### What was contradictory
The old occupancy-guided SMT assumption treated occupancy-derived `SET_V/SET_R`
counts as exact equalities inside exact RTL equations:
- `sum_t valid_buffer_out[t] == popcount(SET_V)`
- `sum_t ready_buffer_in[t] == popcount(SET_R)`

This can be contradictory because `SET_V/SET_R` are heuristic approximations
derived per-buffer, while RTL equations couple all buffers through
fork/mux/cond_br/control-merge backpressure in one global fixed-point.

Concrete contradiction observed in `histogram_transition`:
- MG0 (`II=3`): occupancy-guided RTL exact-one is UNSAT.
- Diagnostic relaxation shows SAT only after relaxing occupancy hints for
  almost every buffer (`buffer0, buffer1, buffer10, ..., buffer21` except
  `buffer9`), proving the approximation is globally inconsistent for this MG.

### Code-level fixes
1. Added explicit occupancy approximation modes:
- `RelaxedOccupancyRange`: bounded count constraints (`k-1 <= sum <= k+1`).
- `RTLStatefulWithOccupancyRangeHints`: exact RTL buffer state + bounded
  occupancy hints.

2. Added generic PB utilities in `Z3Util`:
- `addAtMostK(...)`
- `addAtLeastK(...)`
Used to encode bounded occupancy counts cleanly.

3. Added contradiction diagnostics in `SMTHandshake.cpp`:
- For occupancy-guided exact-one UNSAT, solver runs an internal diagnostic that
  incrementally relaxes buffer hints and records the first SAT relaxation set.
- This diagnostic reason is appended into per-MG `reason` in
  `smt_handshake_dump.json`.

### Resulting behavior
- Occupancy hints are now treated as guidance with explicit approximation
  semantics, not hidden hard assumptions.
- When they contradict exact RTL semantics, the flow logs concrete evidence and
  falls back to less constrained RTL modes instead of silently producing UNSAT.
