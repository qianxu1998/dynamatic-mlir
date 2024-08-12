// NOTE: Assertions have been autogenerated by utils/generate-test-checks.py
// RUN: dynamatic-opt --handshake-minimize-cst-width="opt-negatives" --remove-operation-names %s --split-input-file | FileCheck %s

// CHECK-LABEL:   handshake.func @doNothing(
// CHECK-SAME:                              %[[VAL_0:.*]]: !handshake.control<>, ...) -> !handshake.channel<i6> attributes {argNames = ["start"], resNames = ["out0"]} {
// CHECK:           %[[VAL_1:.*]] = constant %[[VAL_0]] {value = 31 : i6} : <i6>
// CHECK:           end %[[VAL_1]] : <i6>
// CHECK:         }
handshake.func @doNothing(%start: !handshake.control<>) -> !handshake.channel<i6> {
  %cst = constant %start {value = 31 : i6} : !handshake.channel<i6>
  end %cst : !handshake.channel<i6>
}

// -----

// CHECK-LABEL:   handshake.func @zeroCst(
// CHECK-SAME:                            %[[VAL_0:.*]]: !handshake.control<>, ...) -> !handshake.channel<i32> attributes {argNames = ["start"], resNames = ["out0"]} {
// CHECK:           %[[VAL_1:.*]] = constant %[[VAL_0]] {value = false} : <i1>
// CHECK:           %[[VAL_2:.*]] = extsi %[[VAL_1]] : <i1> to <i32>
// CHECK:           end %[[VAL_2]] : <i32>
// CHECK:         }
handshake.func @zeroCst(%start: !handshake.control<>) -> !handshake.channel<i32> {
  %cst = constant %start {value = 0 : i32} : <i32>
  end %cst : <i32>
}

// -----

// CHECK-LABEL:   handshake.func @oneCst(
// CHECK-SAME:                           %[[VAL_0:.*]]: !handshake.control<>, ...) -> !handshake.channel<i32> attributes {argNames = ["start"], resNames = ["out0"]} {
// CHECK:           %[[VAL_1:.*]] = constant %[[VAL_0]] {value = 1 : i2} : <i2>
// CHECK:           %[[VAL_2:.*]] = extsi %[[VAL_1]] : <i2> to <i32>
// CHECK:           end %[[VAL_2]] : <i32>
// CHECK:         }
handshake.func @oneCst(%start: !handshake.control<>) -> !handshake.channel<i32> {
  %cst = constant %start {value = 1 : i32} : <i32>
  end %cst : <i32>
}

// -----

// CHECK-LABEL:   handshake.func @powerOfTwoMinusOne(
// CHECK-SAME:                                       %[[VAL_0:.*]]: !handshake.control<>, ...) -> !handshake.channel<i32> attributes {argNames = ["start"], resNames = ["out0"]} {
// CHECK:           %[[VAL_1:.*]] = constant %[[VAL_0]] {value = 31 : i6} : <i6>
// CHECK:           %[[VAL_2:.*]] = extsi %[[VAL_1]] : <i6> to <i32>
// CHECK:           end %[[VAL_2]] : <i32>
// CHECK:         }
handshake.func @powerOfTwoMinusOne(%start: !handshake.control<>) -> !handshake.channel<i32> {
  %cst = constant %start {value = 31 : i32} : <i32>
  end %cst : <i32>
}

// -----

// CHECK-LABEL:   handshake.func @powerOfTwo(
// CHECK-SAME:                               %[[VAL_0:.*]]: !handshake.control<>, ...) -> !handshake.channel<i32> attributes {argNames = ["start"], resNames = ["out0"]} {
// CHECK:           %[[VAL_1:.*]] = constant %[[VAL_0]] {value = 32 : i7} : <i7>
// CHECK:           %[[VAL_2:.*]] = extsi %[[VAL_1]] : <i7> to <i32>
// CHECK:           end %[[VAL_2]] : <i32>
// CHECK:         }
handshake.func @powerOfTwo(%start: !handshake.control<>) -> !handshake.channel<i32> {
  %cst = constant %start {value = 32 : i32} : <i32>
  end %cst : <i32>
}

// -----

// CHECK-LABEL:   handshake.func @maxPosVal(
// CHECK-SAME:                              %[[VAL_0:.*]]: !handshake.control<>, ...) -> !handshake.channel<i64> attributes {argNames = ["start"], resNames = ["out0"]} {
// CHECK:           %[[VAL_1:.*]] = constant %[[VAL_0]] {value = 9223372036854775807 : i64} : <i64>
// CHECK:           end %[[VAL_1]] : <i64>
// CHECK:         }
handshake.func @maxPosVal(%start: !handshake.control<>) -> !handshake.channel<i64> {
  %cst = constant %start {value = 9223372036854775807 : i64} : !handshake.channel<i64>
  end %cst : !handshake.channel<i64>
}

// -----

// CHECK-LABEL:   handshake.func @negPowerOfMinusOne(
// CHECK-SAME:                                       %[[VAL_0:.*]]: !handshake.control<>, ...) -> !handshake.channel<i32> attributes {argNames = ["start"], resNames = ["out0"]} {
// CHECK:           %[[VAL_1:.*]] = constant %[[VAL_0]] {value = -33 : i7} : <i7>
// CHECK:           %[[VAL_2:.*]] = extsi %[[VAL_1]] : <i7> to <i32>
// CHECK:           end %[[VAL_2]] : <i32>
// CHECK:         }
handshake.func @negPowerOfMinusOne(%start: !handshake.control<>) -> !handshake.channel<i32> {
  %cst = constant %start {value = -33 : i32} : <i32>
  end %cst : <i32>
}

// -----

// CHECK-LABEL:   handshake.func @negPowerOfTwo(
// CHECK-SAME:                                  %[[VAL_0:.*]]: !handshake.control<>, ...) -> !handshake.channel<i32> attributes {argNames = ["start"], resNames = ["out0"]} {
// CHECK:           %[[VAL_1:.*]] = constant %[[VAL_0]] {value = -32 : i6} : <i6>
// CHECK:           %[[VAL_2:.*]] = extsi %[[VAL_1]] : <i6> to <i32>
// CHECK:           end %[[VAL_2]] : <i32>
// CHECK:         }
handshake.func @negPowerOfTwo(%start: !handshake.control<>) -> !handshake.channel<i32> {
  %cst = constant %start {value = -32 : i32} : <i32>
  end %cst : <i32>
}

// -----

// CHECK-LABEL:   handshake.func @minNegVal(
// CHECK-SAME:                              %[[VAL_0:.*]]: !handshake.control<>, ...) -> !handshake.channel<i64> attributes {argNames = ["start"], resNames = ["out0"]} {
// CHECK:           %[[VAL_1:.*]] = constant %[[VAL_0]] {value = -9223372036854775808 : i64} : <i64>
// CHECK:           end %[[VAL_1]] : <i64>
// CHECK:         }
handshake.func @minNegVal(%start: !handshake.control<>) -> !handshake.channel<i64> {
  %cst = constant %start {value = -9223372036854775808 : i64} : !handshake.channel<i64>
  end %cst : !handshake.channel<i64>
}


// -----

// CHECK-LABEL:   handshake.func @multipleUsers(
// CHECK-SAME:                                  %[[VAL_0:.*]]: !handshake.control<>, ...) -> !handshake.channel<i32> attributes {argNames = ["start"], resNames = ["out0"]} {
// CHECK:           %[[VAL_1:.*]] = constant %[[VAL_0]] {value = 32 : i7} : <i7>
// CHECK:           %[[VAL_2:.*]] = extsi %[[VAL_1]] : <i7> to <i32>
// CHECK:           %[[VAL_3:.*]] = addi %[[VAL_2]], %[[VAL_2]] : <i32>
// CHECK:           %[[VAL_4:.*]] = addi %[[VAL_3]], %[[VAL_2]] : <i32>
// CHECK:           end %[[VAL_4]] : <i32>
// CHECK:         }
handshake.func @multipleUsers(%start: !handshake.control<>) -> !handshake.channel<i32> {
  %cst = constant %start {value = 32 : i32} : <i32>
  %add = addi %cst, %cst : <i32>
  %add2 = addi %add, %cst : <i32>
  end %add2 : <i32>
}

// -----

// CHECK-LABEL:   handshake.func @inheritBB(
// CHECK-SAME:                              %[[VAL_0:.*]]: !handshake.control<>, ...) -> !handshake.channel<i32> attributes {argNames = ["start"], resNames = ["out0"]} {
// CHECK:           %[[VAL_1:.*]] = constant %[[VAL_0]] {value = 32 : i7} : <i7>
// CHECK:           %[[VAL_2:.*]] = extsi %[[VAL_1]] : <i7> to <i32>
// CHECK:           end %[[VAL_2]] : <i32>
// CHECK:         }
handshake.func @inheritBB(%start: !handshake.control<>) -> !handshake.channel<i32> {
  %cst = constant %start {value = 32 : i32, handshake.bb = 0 : ui32} : <i32>
  end %cst : <i32>
}

// -----

// CHECK-LABEL:   handshake.func @duplicateDoNothingDiff(
// CHECK-SAME:                                           %[[VAL_0:.*]]: !handshake.control<>, ...) -> !handshake.channel<i32> attributes {argNames = ["start"], resNames = ["out0"]} {
// CHECK:           %[[VAL_1:.*]] = merge %[[VAL_0]] : <>
// CHECK:           %[[VAL_2:.*]] = constant %[[VAL_0]] {value = 3 : i3} : <i3>
// CHECK:           %[[VAL_3:.*]] = extsi %[[VAL_2]] : <i3> to <i32>
// CHECK:           %[[VAL_4:.*]] = constant %[[VAL_1]] {value = 3 : i3} : <i3>
// CHECK:           %[[VAL_5:.*]] = extsi %[[VAL_4]] : <i3> to <i32>
// CHECK:           %[[VAL_6:.*]] = constant %[[VAL_0]] {value = 2 : i3} : <i3>
// CHECK:           %[[VAL_7:.*]] = extsi %[[VAL_6]] : <i3> to <i32>
// CHECK:           %[[VAL_8:.*]] = addi %[[VAL_3]], %[[VAL_5]] : <i32>
// CHECK:           %[[VAL_9:.*]] = addi %[[VAL_8]], %[[VAL_7]] : <i32>
// CHECK:           end %[[VAL_9]] : <i32>
// CHECK:         }
handshake.func @duplicateDoNothingDiff(%start: !handshake.control<>) -> !handshake.channel<i32> {
  %mergeStart = merge %start : <>
  %cst1 = constant %start {value = 3 : i32} : <i32>
  %cst2 = constant %mergeStart {value = 3 : i32} : <i32>
  %cst3 = constant %start {value = 2 : i32} : <i32>
  %add1 = addi %cst1, %cst2 : <i32>
  %add2 = addi %add1, %cst3 : <i32>
  end %add2 : <i32>
}

// -----

// CHECK-LABEL:   handshake.func @duplicateDoNothingPrevious(
// CHECK-SAME:                                               %[[VAL_0:.*]]: !handshake.control<>, ...) -> !handshake.channel<i7> attributes {argNames = ["start"], resNames = ["out0"]} {
// CHECK:           %[[VAL_1:.*]] = constant %[[VAL_0]] {value = 32 : i7} : <i7>
// CHECK:           %[[VAL_2:.*]] = constant %[[VAL_0]] {value = 32 : i7} : <i7>
// CHECK:           %[[VAL_3:.*]] = addi %[[VAL_1]], %[[VAL_2]] : <i7>
// CHECK:           end %[[VAL_3]] : <i7>
// CHECK:         }
handshake.func @duplicateDoNothingPrevious(%start: !handshake.control<>) -> !handshake.channel<i7> {
  %cst1 = constant %start {value = 32 : i7} : !handshake.channel<i7>
  %cst2 = constant %start {value = 32 : i7} : !handshake.channel<i7>
  %add = addi %cst1, %cst2 : !handshake.channel<i7>
  end %add : !handshake.channel<i7>
}

// -----

// CHECK-LABEL:   handshake.func @deleteDuplicate(
// CHECK-SAME:                                    %[[VAL_0:.*]]: !handshake.control<>, ...) -> !handshake.channel<i32> attributes {argNames = ["start"], resNames = ["out0"]} {
// CHECK:           %[[VAL_1:.*]] = constant %[[VAL_0]] {value = 32 : i7} : <i7>
// CHECK:           %[[VAL_2:.*]] = extsi %[[VAL_1]] : <i7> to <i32>
// CHECK:           %[[VAL_3:.*]] = extsi %[[VAL_1]] : <i7> to <i32>
// CHECK:           %[[VAL_4:.*]] = addi %[[VAL_3]], %[[VAL_2]] : <i32>
// CHECK:           end %[[VAL_4]] : <i32>
// CHECK:         }
handshake.func @deleteDuplicate(%start: !handshake.control<>) -> !handshake.channel<i32> {
  %cst1 = constant %start {value = 32 : i32} : <i32>
  %cst2 = constant %start {value = 32 : i32} : <i32>
  %add = addi %cst1, %cst2 : <i32>
  end %add : <i32>
}


// -----

// CHECK-LABEL:   handshake.func @deleteDuplicateMatchExists(
// CHECK-SAME:                                               %[[VAL_0:.*]]: !handshake.control<>, ...) -> !handshake.channel<i32> attributes {argNames = ["start"], resNames = ["out0"]} {
// CHECK:           %[[VAL_1:.*]] = constant %[[VAL_0]] {value = 32 : i7} : <i7>
// CHECK:           %[[VAL_2:.*]] = extsi %[[VAL_1]] : <i7> to <i32>
// CHECK:           %[[VAL_3:.*]] = extsi %[[VAL_1]] : <i7> to <i32>
// CHECK:           %[[VAL_4:.*]] = addi %[[VAL_3]], %[[VAL_2]] : <i32>
// CHECK:           end %[[VAL_4]] : <i32>
// CHECK:         }
handshake.func @deleteDuplicateMatchExists(%start: !handshake.control<>) -> !handshake.channel<i32> {
  %cst1 = constant %start {value = 32 : i7} : !handshake.channel<i7>
  %cst2 = constant %start {value = 32 : i32} : <i32>
  %cst1ext = extsi %cst1 : !handshake.channel<i7> to <i32>
  %add = addi %cst1ext, %cst2 : <i32>
  end %add : <i32>
}
