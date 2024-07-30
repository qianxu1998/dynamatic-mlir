// NOTE: Assertions have been autogenerated by utils/generate-test-checks.py
// RUN: dynamatic-opt --handshake-optimize-bitwidths --remove-operation-names %s --split-input-file | FileCheck %s

// CHECK-LABEL:   handshake.func @forkFW(
// CHECK-SAME:                           %[[VAL_0:.*]]: !handshake.channel<i16>,
// CHECK-SAME:                           %[[VAL_1:.*]]: !handshake.control<>, ...) -> (!handshake.channel<i32>, !handshake.channel<i32>) attributes {argNames = ["arg0", "start"], resNames = ["out0", "out1"]} {
// CHECK:           %[[VAL_2:.*]]:2 = fork  [2] %[[VAL_0]] : <i16>
// CHECK:           %[[VAL_3:.*]] = extsi %[[VAL_2]]#1 : <i16> to <i32>
// CHECK:           %[[VAL_4:.*]] = extsi %[[VAL_2]]#0 : <i16> to <i32>
// CHECK:           %[[VAL_5:.*]]:2 = return %[[VAL_4]], %[[VAL_3]] : <i32>, <i32>
// CHECK:           end %[[VAL_5]]#0, %[[VAL_5]]#1 : <i32>, <i32>
// CHECK:         }
handshake.func @forkFW(%arg0: !handshake.channel<i16>, %start: !handshake.control<>) -> (!handshake.channel<i32>, !handshake.channel<i32>) {
  %ext0 = extsi %arg0 : <i16> to <i32>
  %results:2 = fork [2] %ext0 : <i32>
  %returnVals:2 = return %results#0, %results#1 : <i32>, <i32>
  end %returnVals#0, %returnVals#1 : <i32>, <i32>
}

// -----

// CHECK-LABEL:   handshake.func @lazyForkFW(
// CHECK-SAME:                               %[[VAL_0:.*]]: !handshake.channel<i16>,
// CHECK-SAME:                               %[[VAL_1:.*]]: !handshake.control<>, ...) -> (!handshake.channel<i32>, !handshake.channel<i32>) attributes {argNames = ["arg0", "start"], resNames = ["out0", "out1"]} {
// CHECK:           %[[VAL_2:.*]]:2 = lazy_fork  [2] %[[VAL_0]] : <i16>
// CHECK:           %[[VAL_3:.*]] = extsi %[[VAL_2]]#1 : <i16> to <i32>
// CHECK:           %[[VAL_4:.*]] = extsi %[[VAL_2]]#0 : <i16> to <i32>
// CHECK:           %[[VAL_5:.*]]:2 = return %[[VAL_4]], %[[VAL_3]] : <i32>, <i32>
// CHECK:           end %[[VAL_5]]#0, %[[VAL_5]]#1 : <i32>, <i32>
// CHECK:         }
handshake.func @lazyForkFW(%arg0: !handshake.channel<i16>, %start: !handshake.control<>) -> (!handshake.channel<i32>, !handshake.channel<i32>) {
  %ext0 = extsi %arg0 : <i16> to <i32>
  %results:2 = lazy_fork [2] %ext0 : <i32>
  %returnVals:2 = return %results#0, %results#1 : <i32>, <i32>
  end %returnVals#0, %returnVals#1 : <i32>, <i32>
}

// -----

// CHECK-LABEL:   handshake.func @mergeFW(
// CHECK-SAME:                            %[[VAL_0:.*]]: !handshake.channel<i8>, %[[VAL_1:.*]]: !handshake.channel<i16>,
// CHECK-SAME:                            %[[VAL_2:.*]]: !handshake.control<>, ...) -> !handshake.channel<i32> attributes {argNames = ["arg0", "arg1", "start"], resNames = ["out0"]} {
// CHECK:           %[[VAL_3:.*]] = extsi %[[VAL_0]] {handshake.bb = 0 : ui32} : <i8> to <i16>
// CHECK:           %[[VAL_4:.*]] = merge %[[VAL_3]], %[[VAL_1]] : <i16>
// CHECK:           %[[VAL_5:.*]] = extsi %[[VAL_4]] : <i16> to <i32>
// CHECK:           %[[VAL_6:.*]] = return %[[VAL_5]] : <i32>
// CHECK:           end %[[VAL_6]] : <i32>
// CHECK:         }
handshake.func @mergeFW(%arg0: !handshake.channel<i8>, %arg1: !handshake.channel<i16>, %start: !handshake.control<>) -> !handshake.channel<i32> {
  %ext0 = extsi %arg0 : <i8> to <i32>
  %ext1 = extsi %arg1 : <i16> to <i32>
  %merge = merge %ext0, %ext1 : <i32>
  %returnVal = return %merge : <i32>
  end %returnVal : <i32>
}

// -----

// CHECK-LABEL:   handshake.func @branchFW(
// CHECK-SAME:                             %[[VAL_0:.*]]: !handshake.channel<i16>,
// CHECK-SAME:                             %[[VAL_1:.*]]: !handshake.control<>, ...) -> !handshake.channel<i32> attributes {argNames = ["arg0", "start"], resNames = ["out0"]} {
// CHECK:           %[[VAL_2:.*]] = br %[[VAL_0]] : <i16>
// CHECK:           %[[VAL_3:.*]] = extsi %[[VAL_2]] : <i16> to <i32>
// CHECK:           %[[VAL_4:.*]] = return %[[VAL_3]] : <i32>
// CHECK:           end %[[VAL_4]] : <i32>
// CHECK:         }
handshake.func @branchFW(%arg0: !handshake.channel<i16>, %start: !handshake.control<>) -> !handshake.channel<i32> {
  %ext0 = extsi %arg0 : <i16> to <i32>
  %branch = br %ext0 : <i32>
  %returnVal = return %branch : <i32>
  end %returnVal : <i32>
}

// -----

// CHECK-LABEL:   handshake.func @cmergeFW(
// CHECK-SAME:                             %[[VAL_0:.*]]: !handshake.channel<i8>, %[[VAL_1:.*]]: !handshake.channel<i16>,
// CHECK-SAME:                             %[[VAL_2:.*]]: !handshake.control<>, ...) -> (!handshake.channel<i32>, !handshake.channel<i8>) attributes {argNames = ["arg0", "arg1", "start"], resNames = ["out0", "out1"]} {
// CHECK:           %[[VAL_3:.*]] = extsi %[[VAL_0]] {handshake.bb = 0 : ui32} : <i8> to <i16>
// CHECK:           %[[VAL_4:.*]], %[[VAL_5:.*]] = control_merge %[[VAL_3]], %[[VAL_1]]  : <i16>, <i1>
// CHECK:           %[[VAL_6:.*]] = extsi %[[VAL_4]] : <i16> to <i32>
// CHECK:           %[[VAL_7:.*]] = extui %[[VAL_5]] : <i1> to <i8>
// CHECK:           %[[VAL_8:.*]]:2 = return %[[VAL_6]], %[[VAL_7]] : <i32>, <i8>
// CHECK:           end %[[VAL_8]]#0, %[[VAL_8]]#1 : <i32>, <i8>
// CHECK:         }
handshake.func @cmergeFW(%arg0: !handshake.channel<i8>, %arg1: !handshake.channel<i16>, %start: !handshake.control<>) -> (!handshake.channel<i32>, !handshake.channel<i8>) {
  %ext0 = extsi %arg0 : <i8> to <i32>
  %ext1 = extsi %arg1 : <i16> to <i32>
  %merge, %index = control_merge %ext0, %ext1 : <i32>, <i8>
  %returnVals:2 = return %merge, %index : <i32>, <i8>
  end %returnVals#0, %returnVals#1 : <i32>, <i8>
}

// -----

// CHECK-LABEL:   handshake.func @muxFW(
// CHECK-SAME:                          %[[VAL_0:.*]]: !handshake.channel<i8>, %[[VAL_1:.*]]: !handshake.channel<i16>, %[[VAL_2:.*]]: !handshake.channel<i8>,
// CHECK-SAME:                          %[[VAL_3:.*]]: !handshake.control<>, ...) -> !handshake.channel<i32> attributes {argNames = ["arg0", "arg1", "index", "start"], resNames = ["out0"]} {
// CHECK:           %[[VAL_4:.*]] = extsi %[[VAL_0]] {handshake.bb = 0 : ui32} : <i8> to <i16>
// CHECK:           %[[VAL_5:.*]] = trunci %[[VAL_2]] {handshake.bb = 0 : ui32} : <i8> to <i1>
// CHECK:           %[[VAL_6:.*]] = mux %[[VAL_5]] {{\[}}%[[VAL_4]], %[[VAL_1]]] : <i1>, <i16>
// CHECK:           %[[VAL_7:.*]] = extsi %[[VAL_6]] : <i16> to <i32>
// CHECK:           %[[VAL_8:.*]] = return %[[VAL_7]] : <i32>
// CHECK:           end %[[VAL_8]] : <i32>
// CHECK:         }
handshake.func @muxFW(%arg0: !handshake.channel<i8>, %arg1: !handshake.channel<i16>, %index: !handshake.channel<i8>, %start: !handshake.control<>) -> !handshake.channel<i32> {
  %ext0 = extsi %arg0 : <i8> to <i32>
  %ext1 = extsi %arg1 : <i16> to <i32>
  %mux = mux %index [%ext0, %ext1] : <i8>, <i32>
  %returnVal = return %mux : <i32>
  end %returnVal : <i32>
}

// -----

// CHECK-LABEL:   handshake.func @condBrFw(
// CHECK-SAME:                             %[[VAL_0:.*]]: !handshake.channel<i16>, %[[VAL_1:.*]]: !handshake.channel<i1>,
// CHECK-SAME:                             %[[VAL_2:.*]]: !handshake.control<>, ...) -> (!handshake.channel<i32>, !handshake.channel<i32>) attributes {argNames = ["arg0", "cond", "start"], resNames = ["out0", "out1"]} {
// CHECK:           %[[VAL_3:.*]], %[[VAL_4:.*]] = cond_br %[[VAL_1]], %[[VAL_0]] : <i1>, <i16>
// CHECK:           %[[VAL_5:.*]] = extsi %[[VAL_4]] : <i16> to <i32>
// CHECK:           %[[VAL_6:.*]] = extsi %[[VAL_3]] : <i16> to <i32>
// CHECK:           %[[VAL_7:.*]]:2 = return %[[VAL_6]], %[[VAL_5]] : <i32>, <i32>
// CHECK:           end %[[VAL_7]]#0, %[[VAL_7]]#1 : <i32>, <i32>
// CHECK:         }
handshake.func @condBrFw(%arg0: !handshake.channel<i16>, %cond: !handshake.channel<i1>, %start: !handshake.control<>) -> (!handshake.channel<i32>, !handshake.channel<i32>) {
  %ext0 = extsi %arg0 : <i16> to <i32>
  %true, %false = cond_br %cond, %ext0 : <i1>, <i32>
  %returnVals:2 = return %true, %false : <i32>, <i32>
  end %returnVals#0, %returnVals#1 : <i32>, <i32>
}

// -----

// CHECK-LABEL:   handshake.func @bufferFW(
// CHECK-SAME:                             %[[VAL_0:.*]]: !handshake.channel<i16>,
// CHECK-SAME:                             %[[VAL_1:.*]]: !handshake.control<>, ...) -> !handshake.channel<i32> attributes {argNames = ["arg0", "start"], resNames = ["out0"]} {
// CHECK:           %[[VAL_2:.*]] = oehb [2] %[[VAL_0]] : !handshake.channel<i16>
// CHECK:           %[[VAL_3:.*]] = extsi %[[VAL_2]] : <i16> to <i32>
// CHECK:           %[[VAL_4:.*]] = return %[[VAL_3]] : <i32>
// CHECK:           end %[[VAL_4]] : <i32>
// CHECK:         }
handshake.func @bufferFW(%arg0: !handshake.channel<i16>, %start: !handshake.control<>) -> !handshake.channel<i32> {
  %ext0 = extsi %arg0 : <i16> to <i32>
  %buf = oehb [2] %ext0 : <i32>
  %returnVal = return %buf : <i32>
  end %returnVal : <i32>
}
