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
#include "experimental/Transforms/Switching/DataChannelCal.h"
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

// This function iterativly counts the number of handshake switches of all nodes in the selected cfdfc
void mgHandshakeSwitchingCounting(SwitchingInfo &switchInfo, std::string selMG, bool debug);

// This function updates the status of the selected node's handshake signals
void nodeHandshakeUpdate(SwitchingInfo &switchInfo, std::string &selNode, std::string &selMG, unsigned selMGII, bool debug);

//===---------------------------------------------------------------------------------===//
//
// Functions for DFS in the graph, should be merged with the other functions if possible
//
//===---------------------------------------------------------------------------------===//
std::vector<std::string> findInfluencedLoadNodes(SwitchingInfo &switchInfo, std::string selMG, std::vector<std::string> bufferList);

// This function will return the steady state starting time for the selected node in the specified mg
int mgGetNodeStartingPoint(SwitchingInfo &switchInfo, std::string &selNode, std::string &selMG);

#endif // EXPERIMENTAL_TRANSFORMS_HANDSHAKE_SWITCHING_H
