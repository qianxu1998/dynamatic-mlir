#!/bin/bash

dynamatic_path="."

# BUFFER_ALGORITHM="--buffer-algorithm fpl22"
BUFFER_ALGORITHM="--buffer-algorithm fpga20"
# BUFFER_ALGORITHM="--buffer-algorithm on-merges"
SHARING="--sharing"
SHARING=""
# HDL="vhdl"
HDL="verilog"
STAGE="synth"
# STAGE="pre"
FLATTEN_HIERARCHY="1"

# f_benchmark_src="kernel_3mm/kernel_3mm.c"
# f_benchmark_src="kernel_2mm/kernel_2mm.c"
# f_benchmark_src="gsum/gsum.c"
# f_benchmark_src="bicg/bicg.c"
# f_benchmark_src="gemver/gemver.c"
# f_benchmark_src="cnn/cnn.c"
f_benchmark_src="matvec/matvec.c"
# f_benchmark_src="stencil_2d/stencil_2d.c"
# f_benchmark_src="iir/iir.c"
# f_benchmark_src="fir/fir.c"

# Power estimation
# s_source_file="$dynamatic_path/integration-test/$f_benchmark_src"
# [ -f "$s_source_file" ] || \
#   { echo "Source file $s_source_file does not exist!"; exit 1; }

# echo "set-dynamatic-path $dynamatic_path; \
#   set-src $dynamatic_path/integration-test/$f_benchmark_src; \
#   set-clock-period 8; \
#   compile $SHARING $BUFFER_ALGORITHM; \
#   write-hdl --hdl $HDL; \
#   simulate; \
#   estimate-power --hdl $HDL --stage $STAGE; \
#   exit" \
#   | "$dynamatic_path/bin/dynamatic" --exit-on-failure --debug

# exit

# Power evaluation
s_source_file="$dynamatic_path/integration-test/$f_benchmark_src"
[ -f "$s_source_file" ] || \
  { echo "Source file $s_source_file does not exist!"; exit 1; }

echo "set-dynamatic-path $dynamatic_path; \
  set-src $dynamatic_path/integration-test/$f_benchmark_src; \
  set-clock-period 8; \
  compile $SHARING $BUFFER_ALGORITHM; \
  write-hdl --hdl $HDL; \
  simulate; \
  evaluate-power --hdl $HDL --stage $STAGE --flatten-hierarchy $FLATTEN_HIERARCHY; \
  exit" \
  | "$dynamatic_path/bin/dynamatic" --exit-on-failure

exit
