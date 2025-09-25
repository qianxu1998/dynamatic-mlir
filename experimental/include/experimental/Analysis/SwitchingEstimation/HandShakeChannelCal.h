//===- HandShakeChannelCal.h - Switching Estimation -----*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===------------------------------------------------------------------------------===//
//
// This file declares all functions used for handshake channel switching
// calculation
//
//===------------------------------------------------------------------------------===//

#ifndef EXPERIMENTAL_ANALYSIS_SWITCHINGESTIMATION_HANDSHAKECHANNELCAL_H
#define EXPERIMENTAL_ANALYSIS_SWITCHINGESTIMATION_HANDSHAKECHANNELCAL_H

#include "dynamatic/Dialect/Handshake/HandshakeAttributes.h"
#include "dynamatic/Support/TimingModels.h"
#include "dynamatic/Transforms/BufferPlacement/CFDFC.h"
#include "experimental/Analysis/SwitchingEstimation/NodeModels.h"
#include "experimental/Analysis/SwitchingEstimation/ProfilingAnalyzer.h"
#include "experimental/Analysis/SwitchingEstimation/SwitchingSupport.h"
#include "mlir/IR/Attributes.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <optional>
#include <regex>
#include <set>
#include <string>
#include <typeinfo>
#include <unordered_set>
#include <vector>

using namespace llvm;
using namespace mlir;
using namespace dynamatic;
using namespace dynamatic::handshake;

using IISet = llvm::SmallBitVector;

// This function calculates the the valid start time of different buffers in the
// specified cfdfc with this information, we can determine the active time of
// the ready and valid signal of different buffers. All info will be stored in
// the corresponding buffer node
void updateMGBufferSwitching(SwitchingInfo &switchInfo, std::string selMG,
                             bool debug);

// This function iterativly counts the number of handshake switches of all nodes
// in the selected cfdfc
void mgHandshakeSwitchingCounting(SwitchingInfo &switchInfo, std::string selMG,
                                  bool debug);

// This function updates the status of the selected node's handshake signals
void nodeHandshakeUpdate(SwitchingInfo &switchInfo, std::string &selNode,
                         std::string &selMG, unsigned selMGII, bool debug);

// This function update information for all Join type node in the pending list
// to faciliatate Handshake signal updates *Assumption: we assume at this stage
// all Join type node's valid signal is resolved *The update process shall be
// simplified
std::vector<std::string>
breakHandshakeUpdateDeadlock(SwitchingInfo &switchInfo,
                             const std::vector<std::string> &pendingNodeList,
                             std::string selMG, unsigned selMGII);

//===---------------------------------------------------------------------------------===//
//
// Functions for DFS in the graph,
// TODO: Merge with other DFS kernels
//===---------------------------------------------------------------------------------===//
std::vector<std::string>
getLoadsInfluencedByBuffers(SwitchingInfo &switchInfo, std::string selMG,
                            std::vector<std::string> bufferList);

// This function will return steady state starting time for the selected node in
// the specified CFDFC
int mgGetNodeStartingPoint(SwitchingInfo &switchInfo, std::string &selNode,
                           std::string &selMG);

// This function calculates the switching activities of all handshake channels
// in the dataflow graph
#endif // EXPERIMENTAL_ANALYSIS_SWITCHINGESTIMATION_HANDSHAKECHANNELCAL_H
