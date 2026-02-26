# Handshake Simulator Plan (Resume Point)

## 1) Ultimate Goal
Build a maintainable C++ handshake IR simulator that:
- Runs generated handshake circuits from `handshake_export.mlir`.
- Consumes real profiler-style inputs (same execution window as `frequency-profiler` / `profiler-inputs.txt`).
- Produces `test_sa`-style per-channel switching reports (handshake + data).
- Matches VCD-derived switching counts per channel without ad-hoc tweaks.
- Passes selected integration benchmarks in `run.sh` (excluding `cnn`, `kernel_2mm`, `kernel_3mm`).

## 2) Scope / Validation Definition
For selected benchmarks, success is **test_sa-style switching validation** (not strict cycle-by-cycle waveform identity):
- Compare per-channel switching numbers between simulator output and VCD-derived reference.
- Include handshake signals (`valid`, `ready`, transfer activity) and data-channel switching.
- Report mean relative error and violating channels like `test_sa.sh`.

## 3) Progress So Far (Latest Baseline)
Latest rerun artifacts:
- Summary: `/tmp/hsim/systematic_current21_selected/summary.json`
- Logs:
  - `/tmp/hsim/systematic_current21_selected/fir_testsa.log`
  - `/tmp/hsim/systematic_current21_selected/matvec_testsa.log`
  - `/tmp/hsim/systematic_current21_selected/matrix_testsa.log`
  - `/tmp/hsim/systematic_current21_selected/gcd_testsa.log`
  - `/tmp/hsim/systematic_current21_selected/gsum_testsa.log`
  - `/tmp/hsim/systematic_current21_selected/gemver_testsa.log`
  - `/tmp/hsim/systematic_current21_selected/bicg_testsa.log`
  - `/tmp/hsim/systematic_current21_selected/stencil_2d_testsa.log`
  - `/tmp/hsim/systematic_current21_selected/histogram_transition_testsa.log`

Current status:
- PASS: `fir`, `matvec`, `matrix`, `gsum`, `stencil_2d`
- FAIL: `gcd` (2 violating channels), `gemver` (22), `bicg` (178), `histogram_transition` (20)

Observed failure patterns:
- `gcd`: localized data mismatches (`mux0.data`, `mux1.data`).
- `gemver`: mixed handshake + data issues around load/lsq/store/fork paths.
- `bicg`, `histogram_transition`: dominant handshake mismatch (ready/valid toggle inflation), especially in memory/control-heavy regions.

## 4) What Was Tried (and Reverted)
These experiments were tested and reverted due to regressions or no net gain:
- High-level load passthrough change (introduced deadlocks, including on stable kernels).
- Fork synchronization variant (stalls / excessive runtime).
- LSQ invalid-output data-hold variant (no meaningful fail-count improvement).

Takeaway:
- Keep baseline behavior for stable kernels; avoid global behavioral tweaks without per-benchmark evidence.

## 5) Code Areas to Continue From
Primary files:
- `experimental/include/experimental/Support/HandshakeSimulator.h`
- `experimental/lib/Support/HandshakeSimulator.cpp`
- `experimental/tools/handshake-simulator/handshake-simulator.cpp`

Focus subsystems:
- Memory path modeling (`mem_controller`, `load`, `store`, `lsq`)
- Event sampling / switching extraction
- Data-channel value tracking alignment with bit-vector semantics

## 6) Next Steps (Prioritized, Systematic)
1. Fix `gcd` first (smallest failing surface).
- Trace `mux0/mux1` data event sequence vs VCD-derived sequence at transfer points.
- Ensure data value update/hold policy is consistent with handshake transfer semantics.
- Re-run `gcd` and lock expected behavior.

2. Fix `histogram_transition` next (explicit user priority chain started from this set).
- Instrument ready/valid transitions around LSQ/control groups.
- Identify inflation source (extra re-assertion / premature de-assertion / repeated pending-token exposure).
- Correct local model behavior and re-run only this benchmark.

3. Fix `bicg` after histogram transition.
- Use same instrumentation path, prioritize channels with largest mismatch contribution.
- Validate no regressions on already passing kernels.

4. Fix `gemver` last in failing set.
- Triage by top violating units (`fork`, `load`, `lsq`, `store`, `trunci` path).

5. Full selected-suite rerun after each milestone.
- Run all cases in `run.sh` except `cnn`, `kernel_2mm`, `kernel_3mm`.
- Regenerate summary + per-benchmark logs + per-unit reports.

## 7) Guardrails for Future Changes
- No ratio scaling, window tuning, or post-hoc correction factors.
- No acceptance based only on “pass/fail”; always report numeric switching deltas/MRE.
- Preserve deterministic simulation and failure diagnostics.
- Keep refactors modular: model-specific logic should remain separated and well-commented.

## 8) Definition of Done (Selected Suite)
Done when all selected benchmarks pass `test_sa`-style switching validation:
- `fir`, `matvec`, `matrix`, `gcd`, `gsum`, `gemver`, `bicg`, `stencil_2d`, `histogram_transition`
- Excluding `cnn`, `kernel_2mm`, `kernel_3mm` per current scope.

## 9) Quick Resume Checklist (Next Session)
1. Start from current baseline (`systematic_current21_selected` results).
2. Reproduce `gcd` mismatch with focused logs.
3. Patch minimally and re-run `gcd` + passing sanity set (`fir`, `matvec`).
4. Move to `histogram_transition`, then `bicg`, then `gemver`.
5. Re-run full selected suite and refresh summary report.
