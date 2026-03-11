// REQUIRES: gurobi

// RUN: dynamatic-opt %S/../../../integration-test/fir/out/comp/handshake_transformed.mlir \
// RUN:   --handshake-mark-fpu-impl="impl=vivado" \
// RUN:   --handshake-set-buffering-properties="version=fpga20" \
// RUN:   --handshake-place-buffers="algorithm=fpga20 solver=gurobi frequencies=%S/../../../integration-test/fir/out/comp/frequencies.csv timing-models=%S/../../../data/components.json target-period=8 timeout=300 blif-files=%S/../../../data/aig/ lut-delay=0.55 lut-size=6 acyclic-type cfdfc-cache-out=%t-cache.json" \
// RUN:   --switching-estimation="data-trace=%S/../../../integration-test/fir/out/comp/data_trace.log timing-models=%S/../../../data/components.json target-period=8 dump-file=%t-live.csv" \
// RUN:   -o %t-buffered.mlir
// RUN: dynamatic-opt %t-buffered.mlir \
// RUN:   --switching-estimation="data-trace=%S/../../../integration-test/fir/out/comp/data_trace.log timing-models=%S/../../../data/components.json target-period=8 cfdfc-cache-in=%t-cache.json dump-file=%t-cached.csv" \
// RUN:   -o /dev/null
// RUN: diff -u %t-live.csv %t-cached.csv

module {
  // The RUN lines exercise the cache round-trip on the checked-in `fir`
  // artifacts. No IR checks are needed here because the test asserts identical
  // switching CSV output between the live-analysis and cache-only paths.
}
