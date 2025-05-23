#!/bin/bash

dynamatic_path="."

BUFFER_ALGORITHM="--buffer-algorithm fpga20"
SHARING="--sharing"
SHARING=""
HDL="verilog"
HDL="vhdl"

# f_benchmark_src="mvt_float/mvt_float.c"
# f_benchmark_src="sharing/share_test_2/share_test_2.c"
# f_benchmark_src="fir/fir.c"
# f_benchmark_src="atax/atax.c"
# f_benchmark_src="dct/dct.c"
# f_benchmark_src="cnn/cnn.c"
# f_benchmark_src="gemm/gemm.c"
# f_benchmark_src="matching/matching.c"
# f_benchmark_src="matching_2/matching_2.c"
# f_benchmark_src="sharing/share_test_1/share_test_1.c"
# f_benchmark_src="while_loop_1/while_loop_1.c"
# f_benchmark_src="while_loop_3/while_loop_3.c"
# f_benchmark_src="while_loop_4/while_loop_4.c"
# f_benchmark_src="complexdiv/complexdiv.c"
# f_benchmark_src="covariance_float/covariance_float.c"
# f_benchmark_src="kernel_3mm_float/kernel_3mm_float.c"
# f_benchmark_src="if_loop_add/if_loop_add.c"
# f_benchmark_src="if_loop_mul/if_loop_mul.c"
# f_benchmark_src="kernel_3mm/kernel_3mm.c"
# f_benchmark_src="atax/atax.c"
# f_benchmark_src="atax_float/atax_float.c"
# f_benchmark_src="syr2k_float/syr2k_float.c"
# f_benchmark_src="bicg_float/bicg_float.c"
# f_benchmark_src="gesummv_float/gesummv_float.c"
# f_benchmark_src="gemm_float/gemm_float.c"
f_benchmark_src="gemm/gemm.c"
# f_benchmark_src="memory/test_memory_11/test_memory_11.c"
# f_benchmark_src="memory/test_memory_8/test_memory_8.c"
# f_benchmark_src="symm_float/symm_float.c"
# f_benchmark_src="kmp/kmp.c"
# f_benchmark_src="gsumif/gsumif.c"
# f_benchmark_src="cordic/cordic.c"
# f_benchmark_src="atax/atax.c"
# f_benchmark_src="correlation_float/correlation_float.c"
# f_benchmark_src="memory/test_memory_18/test_memory_18.c"
# f_benchmark_src="covariance/covariance.c"
# f_benchmark_src="spmv/spmv.c"
# f_benchmark_src="float_basic/float_basic.c"
# f_benchmark_src="triangular/triangular.c"
# f_benchmark_src="gaussian/gaussian.c"
# f_benchmark_src="get_tanh/get_tanh.c"
# f_benchmark_src="jacobi_1d_imper/jacobi_1d_imper.c"
# f_benchmark_src="trisolv/trisolv.c"
# f_benchmark_src="covariance_float/covariance_float.c"
# f_benchmark_src="gemm/gemm.c"
# f_benchmark_src="while_loop_2/while_loop_2.c"
# f_benchmark_src="simple_example_1/simple_example_1.c"
# f_benchmark_src="lu/lu.c"
# f_benchmark_src="matvec/matvec.c"
# f_benchmark_src="histogram/histogram.c"
# f_benchmark_src="gemver_float/gemver_float.c"
# f_benchmark_src="gemver/gemver.c"
# f_benchmark_src="polyn_mult/polyn_mult.c"
# f_benchmark_src="gsum/gsum.c"
# f_benchmark_src="admm/admm.c"
# f_benchmark_src="sobel/sobel.c"
# f_benchmark_src="gcd/gcd.c"
# f_benchmark_src="matrix_power/matrix_power.c"
# f_benchmark_src="video_filter/video_filter.c"
# f_benchmark_src="vector_rescale/vector_rescale.c"

s_source_file="$dynamatic_path/integration-test/$f_benchmark_src"
[ -f "$s_source_file" ] || \
  { echo "Source file $s_source_file does not exist!"; exit 1; }

echo "set-dynamatic-path .; \
  set-src ./integration-test/$f_benchmark_src; \
  set-clock-period 8; \
  compile $SHARING $BUFFER_ALGORITHM; \
  # write-hdl --hdl $HDL; \
  # simulate; \
  # visualize; \
  exit" \
  | bin/dynamatic --exit-on-failure --debug

exit
