//===- SwitchingEstimation.h - Switching estimation -------------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the --switching-estimation pass.
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_EXPERIMENTAL_ANALYSIS_SWITCHINGESTIMATION_PASS_H
#define DYNAMATIC_EXPERIMENTAL_ANALYSIS_SWITCHINGESTIMATION_PASS_H

#include "dynamatic/Support/DynamaticPass.h"
#include "dynamatic/Support/LLVM.h"

namespace dynamatic {
namespace experimental {

#define GEN_PASS_DECL_SWITCHINGESTIMATION
#include "experimental/Analysis/Passes.h.inc"

} // namespace dynamatic
} // namespace experimental

#endif // DYNAMATIC_EXPERIMENTAL_ANALYSIS_SWITCHINGESTIMATION_PASS_H
