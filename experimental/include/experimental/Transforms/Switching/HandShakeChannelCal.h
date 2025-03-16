//===- HandShakeChannelCal.h - Switching Estimation -----*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===------------------------------------------------------------------------------===//
//
// This file declares all functions used for handshake channel switching calculation
//
//===------------------------------------------------------------------------------===//

#ifndef EXPERIMENTAL_TRANSFORMS_HANDSHAKE_SWITCHING_H
#define EXPERIMENTAL_TRANSFORMS_HANDSHAKE_SWITCHING_H

#include "experimental/Transforms/Switching/SwitchingSupport.h"
#include "experimental/Transforms/Switching/ProfilingAnalyzer.h"
#include "experimental/Transforms/Switching/ExecModel.h"
#include "dynamatic/Transforms/BufferPlacement/CFDFC.h"
#include "mlir/IR/Attributes.h"
#include "dynamatic/Dialect/Handshake/HandshakeAttributes.h"
#include "experimental/Support/StdProfiler.h"
#include "dynamatic/Support/TimingModels.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"

#include <unordered_set>
#include <algorithm>
#include <cmath>
#include <vector>
#include <set>
#include <regex>
#include <string>
#include <cctype>
#include <typeinfo>
#include <optional>

using namespace mlir;
using namespace dynamatic;
using namespace dynamatic::handshake;

// This function calculates the valid start time of different buffers in the specified cfdfc
// With this information, we can determine the Active time range of the ready and valid signal
// of different buffers.
// All info will be stored in the corresponding structure in the corresponding buffer node
void extractBufferInfo(SwitchingInfo &switchInfo, std::string selMG, bool debug);


#endif // EXPERIMENTAL_TRANSFORMS_HANDSHAKE_SWITCHING_H
