# Switching Estimation Debug Toolkit

This folder now includes scripts to debug switching-estimation mismatches quickly and point to likely pass hotspots.

## 1) Single-kernel debug report

Run after `test_sa.sh <kernel>` has generated:
- `integration-test/<kernel>/out/comp/switching_estimation.csv`
- `integration-test/<kernel>/out/sim/HLS_VERIFY/trace.vcd`

```bash
python3 switching_testing/debug_switching.py \
  --repo-root . \
  --kernel fir \
  --window post-reset \
  --rel-error-threshold 0.10 \
  --zero-abs-threshold 5 \
  --ignore-both-below 50
```

Artifacts generated:
- `integration-test/<kernel>/out/comp/switching_debug_report.md`
- `integration-test/<kernel>/out/comp/switching_debug_violations.csv`

## 2) Batch debug across benchmark kernels

Analyze existing outputs:

```bash
switching_testing/debug_batch.sh
```

Analyze specific kernels:

```bash
switching_testing/debug_batch.sh fir gsum iir
```

Re-run pass flow then analyze:

```bash
switching_testing/debug_batch.sh --run-pass fir gsum
```

## 3) What the report gives you

The markdown report highlights:
- failing channel groups by `(channel, node_type, over/under)`
- top per-node channel violations
- likely source files to inspect first

The CSV includes machine-readable violation rows with:
- node/channel error values
- pass/fail reason
- file hint and short note

## 4) Practical triage loop

1. Run `test_sa.sh <kernel>` (or `debug_batch.sh --run-pass ...`).
2. Open `switching_debug_report.md` and fix the top failing group first.
3. Use `switching_debug_violations.csv` to verify the exact channels that changed.
4. Re-run and repeat until violations converge.

## 5) Dump data-channel values in decimal

Use this helper to extract per-channel value changes from `trace.vcd` and print
decimal values (instead of raw binary vectors):

```bash
python3 switching_testing/dump_data_channels.py \
  --vcd integration-test/gsum/out/sim/HLS_VERIFY/trace.vcd \
  --node load0 \
  --window post-reset \
  --show-binary \
  --output integration-test/gsum/out/comp/load0_data_values.txt \
  --csv integration-test/gsum/out/comp/load0_data_values.csv
```

Useful options:
- `--node <name>`: only dump vectors for a specific node (repeatable).
- `--vector-regex <regex>`: extra name filter on full vector names.
- `--signed`: print signed two's-complement decimal values.
- `--start-time/--end-time`: explicit time window (overrides `--window`).
- `--view propagation`: global propagation view for all channels.
- `--view channel-sequential`: group values per channel in channel-local sequence order.

Example: dump all data channels in propagation order (good for tracing value flow):

```bash
python3 switching_testing/dump_data_channels.py \
  --vcd integration-test/gsum/out/sim/HLS_VERIFY/trace.vcd \
  --window post-reset \
  --view propagation \
  --show-binary \
  --output integration-test/gsum/out/comp/all_data_propagation.txt \
  --csv integration-test/gsum/out/comp/all_data_propagation.csv
```

Example: same data, but grouped channel-by-channel in sequence order:

```bash
python3 switching_testing/dump_data_channels.py \
  --vcd integration-test/gsum/out/sim/HLS_VERIFY/trace.vcd \
  --window post-reset \
  --view channel-sequential \
  --show-binary \
  --output integration-test/gsum/out/comp/all_data_channel_sequential.txt \
  --csv integration-test/gsum/out/comp/all_data_channel_sequential.csv
```
