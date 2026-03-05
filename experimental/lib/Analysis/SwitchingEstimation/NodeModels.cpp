//===- NodeModels.cpp - Switching estimation ------------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Implements switching models for different nodes.
//===----------------------------------------------------------------------===//

#include "experimental/Analysis/SwitchingEstimation/NodeModels.h"
#include "experimental/Analysis/SwitchingEstimation/Debug.h"

using namespace dynamatic::experimental;

namespace {
inline unsigned switchingFromSet(const IISet &set) {
  if (set.none() || set.all())
    return 0U;
  return 2U;
}
} // namespace

//===----------------------------------------------------------------------===//
//
// Buffer Node
//
//===----------------------------------------------------------------------===//

BufferNode::BufferNode(mlir::Operation *op,
                       const std::vector<std::string> &predecessors,
                       const std::vector<std::string> &successors,
                       const std::map<std::string, unsigned> &sucDataWidthMap,
                       const unsigned &latency, const unsigned &bbIndex)
    : AdjNode(op, predecessors, successors, sucDataWidthMap, latency, bbIndex) {
}

void BufferNode::calValidSwitching(const std::string &sucNodeName,
                                   unsigned II) {
  if (setV.find(sucNodeName) != setV.end()) {
    const IISet &set = setV[sucNodeName];
    // Buffer sets are explicitly constructed by the buffer model. An empty set
    // means "always low" in steady state and should not generate edge toggles.
    const unsigned sw = (set.none() || set.all()) ? 0U : 2U;
    setValid(this, sucNodeName, sw);
  } else {
    llvm::errs() << "Warning: BufferNode::calValidSwitching - No valid set "
                    "found for successor node "
                 << sucNodeName << "\n";
  }
}

void BufferNode::calReadySwitching(const std::string &preNodeName,
                                   unsigned II) {
  if (setR.find(preNodeName) != setR.end()) {
    const IISet &set = setR[preNodeName];
    const unsigned sw = (set.none() || set.all()) ? 0U : 2U;
    setReady(this, preNodeName, sw);
  }
}

void BufferNode::calValidSet(const std::string &sucNodeName,
                             const IISet &inSetV) {
  setV[sucNodeName] = inSetV;
}

void BufferNode::calReadySet(const std::string &preNodeName,
                             const IISet &inSetR) {
  setR[preNodeName] = inSetR;
}

void BufferNode::printNodeDetails() {
  // Call base class's printNodeDetails()
  AdjNode::printNodeDetails();

  switchingDebugStream(SwitchingDebugCategory::Node) << "[DEBUG] \t\tSTART: " << START << ";\n";
  switchingDebugStream(SwitchingDebugCategory::Node) << "[DEBUG] \t\tOccupancy: " << occupancy << ";\n";
  switchingDebugStream(SwitchingDebugCategory::Node) << "[DEBUG] \t\tNumSlots: " << numSlots << ";\n";
  switchingDebugStream(SwitchingDebugCategory::Node) << "[DEBUG] \t\ttransparent: " << transparent
                          << ";\n";
  switchingDebugStream(SwitchingDebugCategory::Node) << "[DEBUG] \t\tBufferType: "
                          << static_cast<int>(buffType) << ";\n";
}

//===----------------------------------------------------------------------===//
//
// Join Node
//
//===----------------------------------------------------------------------===//
void JoinNode::calValidSwitching(const std::string &sucNodeName, int &numValid0,
                                 int &numValid1, unsigned &II) {
  unsigned val = 0;

  if (numValid0 > 0 || numValid1 > 0) {
    val = 2;
  } else if (numValid0 == 0 && numValid1 == 0) {
    val = 0;
  } else if (II == 1) {
    val = 0;
  } else {
    val = 2;
  }

  setValid(this, sucNodeName, val);
}

void JoinNode::calValidSet(const std::string &sucNodeName,
                           unsigned &nodeStartTime, unsigned &II) {
  // Check if validSignal contains sucNodeName.
  if (validSignal.find(sucNodeName) != validSignal.end()) {
    if (validSignal[sucNodeName] == 0) {
      setVSet(this, sucNodeName, fullSet(II));
    } else {
      setVSet(this, sucNodeName, singleton(nodeStartTime, II));
    }
  }
}

void JoinNode::calReadySwitching(const std::string &preNodeName, int &numValid,
                                 int &numReady) {
  // RTL `join_type`: each input ready is gated by both the other input-valid
  // and the output ready. If either side is classed as inactive in steady
  // state, input ready stays level-stable.
  unsigned val = (numValid > 0 && numReady > 0) ? 2 : 0;
  setReady(this, preNodeName, val);
}

void JoinNode::calReadySet(const std::string &preNodeName,
                           const IISet *setValid, const IISet *setReady,
                           unsigned &nodeStartTime, unsigned &II) {
  if (setValid != nullptr && setReady != nullptr) {
    setRSet(this, preNodeName, intersection_(*setValid, *setReady));
  } else if (readySignal.find(preNodeName) != readySignal.end()) {
    if (readySignal[preNodeName] == 0) {
      setR[preNodeName] = fullSet(II);
    } else if (readySignal[preNodeName] > 0) {
      setRSet(this, preNodeName, singleton(nodeStartTime, II));
    }
  }
}

void JoinNode::setReadySet(const std::string &preNodeName) {
  // If setVMap is not empty, take the first available set and copy it to
  // setRMap.
  if (!setV.empty()) {
    // setRMap.
    auto it = setV.begin();
    setR[preNodeName] = it->second;
  }

  auto it = setR.find(preNodeName);
  if (it != setR.end())
    setReady(this, preNodeName, switchingFromSet(it->second));
}

//===----------------------------------------------------------------------===//
//
// Pass Node
//
//===----------------------------------------------------------------------===//

void PassNode::calValidSwitching(const std::string &sucNodeName,
                                 int &numValid) {
  // Treat unresolved predecessor activity as active to keep fixed-point
  // propagation moving in cyclic MGs.
  setValid(this, sucNodeName, numValid == 0 ? 0U : 2U);
}

void PassNode::calValidSet(const std::string &sucNodeName,
                           unsigned &nodeStartTime, unsigned &II) {
  if (validSignal.find(sucNodeName) != validSignal.end()) {
    if (validSignal[sucNodeName] == 0) {
      setV[sucNodeName] = IISet(II, true);
    } else {
      IISet tmp(II, false);
      tmp.set(normalizeCycleIndex(static_cast<int>(nodeStartTime), II));
      setV[sucNodeName] = std::move(tmp);
    }
  }
}

void PassNode::calReadySwitching(const std::string &preNodeName,
                                 int &numReady) {
  // Treat unresolved successor readiness as active to avoid freezing pass-like
  // nodes at constant-ready during early fixed-point iterations.
  setReady(this, preNodeName, numReady == 0 ? 0U : 2U);
}

void PassNode::calReadySet(const std::string &preNodeName,
                           const IISet *setReady, unsigned &nodeStartTime,
                           unsigned &II) {
  if (setReady != nullptr) {
    setR[preNodeName] = *setReady;
  } else if (readySignal.find(preNodeName) != readySignal.end()) {
    if (readySignal[preNodeName] == 0) {
      setR[preNodeName] = IISet(II, true);
    } else if (readySignal[preNodeName] > 0) {
      IISet tmp(II, false);
      tmp.set(normalizeCycleIndex(static_cast<int>(nodeStartTime), II));
      setR[preNodeName] = std::move(tmp);
    } else if (!setV.empty()) {
      // TODO: Verify the logic here, 21/09/2025
      // Fallback: copy the set from the first key in setVMap if it is not the
      // full set.
      auto it = setV.begin();
      IISet fullSet(II, true);
      if (it->second != fullSet)
        setR[preNodeName] = it->second;
    }
  }

  auto it = setR.find(preNodeName);
  if (it != setR.end())
    ::setReady(this, preNodeName, switchingFromSet(it->second));
}

void PassNode::setReadySwitching(const std::string &preNodeName,
                                 unsigned &numReadySwitches) {
  readySignal[preNodeName] = numReadySwitches;
}

void PassNode::setReadySet(const std::string &preNodeName,
                           const IISet &setReady) {
  setR[preNodeName] = setReady;
}

//===----------------------------------------------------------------------===//
//
// MUX Node
//
//===----------------------------------------------------------------------===//

MuxNode::MuxNode(mlir::Operation *op,
                 const std::vector<std::string> &predecessors,
                 const std::vector<std::string> &successors,
                 const std::map<std::string, unsigned> &sucDataWidthMap,
                 unsigned latency, const unsigned &bbIndex)
    : AdjNode(op, predecessors, successors, sucDataWidthMap, latency, bbIndex) {
  // Get the operator
  auto muxOp = dyn_cast<handshake::MuxOp>(op);
  auto condPreValue = muxOp.getSelectOperand();

  conPreNodeName = condPreValue.getDefiningOp()
                       ->getAttrOfType<mlir::StringAttr>("handshake.name")
                       .getValue()
                       .str();

  // Construct the port map
  auto opOperands = muxOp.getOperands();
  for (unsigned i = 0; i < opOperands.size(); i++) {
    std::string nodeName = "inArgument";
    if (opOperands[i].getDefiningOp()) {
      nodeName = opOperands[i]
                     .getDefiningOp()
                     ->getAttrOfType<mlir::StringAttr>("handshake.name")
                     .getValue()
                     .str();
    } else if (auto blockArg = opOperands[i].dyn_cast<mlir::BlockArgument>()) {
      // This operand is a block argument
      unsigned argNumber = blockArg.getArgNumber();

      auto funcOp = blockArg.getOwner()->getParentOp();
      if (auto handshakeFunc = dyn_cast<handshake::FuncOp>(funcOp)) {
        // Check whether the argName attributes present
        auto argNameAttr =
            handshakeFunc->getAttrOfType<mlir::ArrayAttr>("argNames");
        if (!argNameAttr) {
          llvm::errs()
              << "ERROR: No argNames attributes found on the function!\n";
        } else {
          if (argNumber < argNameAttr.size()) {
            auto strAttr =
                dyn_cast<mlir::StringAttr>(argNameAttr.getValue()[argNumber]);
            nodeName = strAttr.getValue().str();
          }
        }
      }
    }
    preNameToPortIdxMap[nodeName] = i;
  }
}

void MuxNode::calValidSwitching(const std::string &sucNodeName, unsigned II) {
  if (II == 1)
    validSignal[sucNodeName] = 0;
  else
    validSignal[sucNodeName] = 2;
}

void MuxNode::calValidSet(const std::string &sucNodeName,
                          unsigned nodeStartTime, unsigned II) {
  if (validSignal.find(sucNodeName) != validSignal.end() &&
      validSignal[sucNodeName] == 0) {
    setV[sucNodeName] = fullSet(II);
  } else {
    IISet tmp(II, false);
    tmp.set(normalizeCycleIndex(static_cast<int>(nodeStartTime), II));
    setV[sucNodeName] = std::move(tmp);
  }
}

void MuxNode::calReadySwitching(const std::string &preNodeName,
                                unsigned condValue, int num_v,
                                const IISet *setV0, const IISet *setVSelect,
                                const IISet *setReady, unsigned II) {
  IISet uSet = IISet(II, true);
  const unsigned condBit = condValue ? 1U : 0U;
  if (preNodeName == conPreNodeName) {
    // Case 1: this channel is used for the condition signal.
    if (setVSelect != nullptr && setReady != nullptr) {
      if (*setVSelect == uSet && *setReady == uSet) {
        readySignal[preNodeName] = 0;
        return;
      }
    }
    if (setV0 != nullptr && setVSelect != nullptr && setReady != nullptr) {
      // Compute intersection: (setVSelect ∩ setReady)
      IISet inter = intersection_(*setVSelect, *setReady);
      // Check if setV0 is a subset of the intersection.
      IISet tmp_setV0 = *setV0;
      tmp_setV0 &= inter;
      bool subset = (tmp_setV0 == *setV0);
      readySignal[preNodeName] = subset ? 0 : 2;
    }
  } else {
    // Case 2: data channel
    auto prePortIt = preNameToPortIdxMap.find(preNodeName);
    if (prePortIt == preNameToPortIdxMap.end() || prePortIt->second == 0) {
      readySignal[preNodeName] = 0;
      return;
    }
    unsigned prePort = prePortIt->second;
    if (condBit != (prePort - 1)) {
      // Unselected data inputs stay deasserted on ready in steady state.
      readySignal[preNodeName] = 0;
    } else {
      if (setV0 != nullptr && setReady != nullptr) {
        if (*setReady == uSet && *setV0 == uSet) {
          readySignal[preNodeName] = 0;
          return;
        }
      }
      if (setV0 != nullptr && setVSelect != nullptr && setReady != nullptr) {
        IISet inter = intersection_(*setV0, *setReady);
        // Check if setVSelect is a subset of setV0.
        IISet tmp_setVSelect = *setVSelect;
        tmp_setVSelect &= inter;
        bool subset = (tmp_setVSelect == *setVSelect);
        readySignal[preNodeName] = subset ? 0 : 2;
      }
    }
  }
}

void MuxNode::calReadySet(const std::string &preNodeName, const IISet *setCond,
                          const IISet *setValid, const IISet *setReady,
                          unsigned II) {
  if (readySignal.find(preNodeName) != readySignal.end()) {
    if (readySignal[preNodeName] == 0) {
      setR[preNodeName] = fullSet(II);
    } else {
      if (setCond != nullptr && setValid != nullptr && setReady != nullptr) {
        // Compute setCond ∩ setV.
        IISet inter = intersection_(*setCond, *setValid);
        IISet finalSet = intersection_(inter, *setReady);

        setR[preNodeName] = std::move(finalSet);
      }
    }
  }
}

void MuxNode::printNodeDetails() {
  AdjNode::printNodeDetails();

  switchingDebugStream(SwitchingDebugCategory::Node) << "[DEBUG] \t\tCon_pre_node_name: " << conPreNodeName
                          << "\n";
  switchingDebugStream(SwitchingDebugCategory::Node) << "[DEBUG] \t\tInput Port Mapping: \n";
  for (const auto &[selName, portIdx] : preNameToPortIdxMap) {
    switchingDebugStream(SwitchingDebugCategory::Node) << "[DEBUG] \t\t\tNode Name: " << selName
                            << "; Port Index: " << portIdx << ";\n";
  }
}

//===----------------------------------------------------------------------===//
//
// Lazy Fork Node
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
//
// Fork Node
//
//===----------------------------------------------------------------------===//

void ForkNode::calValidSwitching(
    const std::string &sucNodeName, int numValid,
    std::unordered_map<std::string, IISet *> &setRDict,
    const std::unordered_map<std::string, int> &numReadyDict,
    unsigned sucNodeStart, unsigned nodeSteadyStart, unsigned II) {
  // If any valid switching occurs at the input, the fork must toggle.
  if (numValid > 0) {
    validSignal[sucNodeName] = 2;
    return;
  }

  // Case 1: check if all ready counts are 0
  bool allReadyZero = true;
  for (const auto &entry : numReadyDict) {
    if (entry.second != 0) {
      allReadyZero = false;
      break;
    }
  }

  if (allReadyZero) {
    validSignal[sucNodeName] = 0;
    return;
  }

  // Case 2: Check the active ready set for sucNodeName
  unsigned selStartPoint = 0;
  auto it = setRDict.find(sucNodeName);
  if (it != setRDict.end()) {
    IISet *s = it->second;
    if (s && s->any()) {
      if (s->count() == 1)
        selStartPoint = s->find_first();
      else {
        validSignal[sucNodeName] = 2;
        return;
      }
    } else {
      selStartPoint = 0;
    }
  } else {
    selStartPoint = 0;
  }

  bool flag = true;
  bool existFlag = true;
  // For each entry in setRDict (other than sucNodeName), check if selStartPoint
  // is contained.
  for (auto &entry : setRDict) {
    if (entry.first != sucNodeName) {
      //! Testing
      switchingDebugStream(SwitchingDebugCategory::Node) << "selSuc " << nodeSteadyStart << "\n";
      IISet *s = entry.second;
      // In our design, an empty set is equivalent to None.
      if (!s) {
        // If the corresponding setR is not empty
        flag = false;
      } else if (!s->test(selStartPoint)) {
        flag = false;
      }
    }
  }

  //! Testing
  switchingDebugStream(SwitchingDebugCategory::Node) << "Node Steady Start: " << nodeSteadyStart << "\n";
  switchingDebugStream(SwitchingDebugCategory::Node) << "Sel Start Point: " << selStartPoint << "\n";
  switchingDebugStream(SwitchingDebugCategory::Node) << "Exist Flag: " << existFlag << "\n";

  if (flag && existFlag) {
    validSignal[sucNodeName] = 0;
    return;
  } else {
    //! Testing
    switchingDebugStream(SwitchingDebugCategory::Node) << "Hit Valid Switching Calculation Case III\n";
    if (existFlag) {
      // Case 3: Compute desired cycle time
      // TODO: Verify the following desired criteria
      unsigned desiredCycleTime =
          (nodeSteadyStart != 0) ? (nodeSteadyStart - 1) : (II - 1);
      if (selStartPoint == 0) {
        validSignal[sucNodeName] = 0;
      } else {
        validSignal[sucNodeName] = 2;
      }
    }
  }
}

void ForkNode::calValidSet(const std::string &sucNodeName,
                           unsigned nodeStartTime, int numValid,
                           std::unordered_map<std::string, IISet *> &setRDict,
                           unsigned II) {
  if (validSignal.find(sucNodeName) != validSignal.end()) {
    if (validSignal[sucNodeName] == 0) {
      setV[sucNodeName] = IISet(II, true);
    } else {
      if (numValid > 0) {
        IISet tmp(II, false);
        tmp.set(normalizeCycleIndex(static_cast<int>(nodeStartTime), II));
        setV[sucNodeName] = std::move(tmp);
      } else {
        auto it = setRDict.find(sucNodeName);
        if (it != setRDict.end() && it->second)
          setV[sucNodeName] = *(it->second);
      }
    }
  }
}

void ForkNode::calReadySwitching(const std::string &preNodeName,
                                 const std::vector<int> &numReadyList) {
  // If any value in numReadyList is greater than 0, set ready signal to 2 and
  // return.
  bool tmpFlag = true;
  for (int value : numReadyList) {
    if (value > 0) {
      readySignal[preNodeName] = 2;
      return;
    }

    if (value != 0)
      tmpFlag = false;
  }
  // Otherwise, if all values are 0, set ready signal to 0.
  if (tmpFlag)
    readySignal[preNodeName] = 0;
}

void ForkNode::calReadySet(
    const std::string &preNodeName,
    const std::unordered_map<std::string, IISet *> &setRDict, unsigned II) {
  if (readySignal.find(preNodeName) != readySignal.end()) {
    if (readySignal[preNodeName] == 0) {
      setR[preNodeName] = fullSet(II);
    } else {
      // Check existence of all nReady sets for successors.
      bool exist = true;
      IISet unionSet(II, false);
      // Start with an empty union.
      for (const std::string &key : sucs) {
        auto it = setRDict.find(key);
        if (it != setRDict.end()) {
          IISet *s = it->second;
          if (!s) {
            exist = false;
            unionSet.reset(); // intersection with an empty set remains empty
          } else {
            // Compute the union: unionSet = unionSet ∪ s.
            unionSet |= *s;
          }
        }
      }
      if (exist) {
        IISet fullMask(II, true);

        if (unionSet == fullMask) {
          IISet tmp(II, false);
          tmp.set(0);
          setR[preNodeName] = std::move(tmp);
        } else {
          // Retrieve the last element of unionSet.
          unsigned last = 0;
          for (unsigned bitidx : unionSet.set_bits()) {
            // if (!unionSet.empty())
            last = bitidx;
          }
          IISet tmp(II, false);
          tmp.set(last);
          setR[preNodeName] = std::move(tmp);
        }
      }
    }
  }

  auto it = setR.find(preNodeName);
  if (it != setR.end())
    ::setReady(this, preNodeName, switchingFromSet(it->second));
}

//===----------------------------------------------------------------------===//
//
// CMergeNode Node
//
//===----------------------------------------------------------------------===//

CMergeNode::CMergeNode(mlir::Operation *op,
                       const std::vector<std::string> &predecessors,
                       const std::vector<std::string> &successors,
                       const std::map<std::string, unsigned> &sucDataWidthMap,
                       unsigned latency, const unsigned &bbIndex)
    : AdjNode(op, predecessors, successors, sucDataWidthMap, latency, bbIndex) {
  // Determine the control and data channel successor names.
  // We assume that the mapping sucDataWidthMap associates each successor name
  // with a port number.
  conSucNodeName = "";
  dataSucNodeName = "";
  for (const auto &pair : sucDataWidthMap) {
    if (pair.second == 1) {
      conSucNodeName = pair.first;
      break;
    }
  }

  // For control merge node, we don't have actual data output
  for (const auto &selNode : successors) {
    if (selNode != conSucNodeName)
      dataSucNodeName = selNode;
  }
}

void CMergeNode::calValidSwitching(const std::string &sucNodeName,
                                   unsigned II) {
  if (II == 1)
    validSignal[sucNodeName] = 0;
  else
    validSignal[sucNodeName] = 2;
}

void CMergeNode::calValidSet(const std::string &sucNodeName,
                             unsigned nodeStartTime, unsigned II) {
  if (validSignal[sucNodeName] == 0) {
    setV[sucNodeName] = fullSet(II);
  } else {
    IISet tmp(II, false);
    tmp.set(normalizeCycleIndex(static_cast<int>(nodeStartTime), II));
    setV[sucNodeName] = std::move(tmp);
  }
}

void CMergeNode::calReadySwitching(const std::string &preNodeName) {
  readySignal[preNodeName] = 0;
}

void CMergeNode::calReadySet(const std::string &preNodeName, unsigned II) {
  setR[preNodeName] = fullSet(II);
}

void CMergeNode::calDataout(int inputValue) {
  // Update the control channel:
  if (dataOut.find(conSucNodeName) == dataOut.end() ||
      dataOut[conSucNodeName].empty()) {
    dataOut[conSucNodeName] = {inputValue};
  } else {
    dataOut[conSucNodeName].push_back(inputValue);
  }
  // For the data channel, simply set its output to [-1]
  dataOut[dataSucNodeName] = {-1};
}

void CMergeNode::updateDataout(int inputData) {
  int valueDiff = 0;
  // Iterate over the successor names (we assume 'sucs' is available from
  // AdjNode as a vector<string>)
  for (const std::string &suc : sucs) {
    // Check if there is already data for this successor.
    if (dataOut.find(suc) != dataOut.end() && !dataOut[suc].empty()) {
      if (suc == conSucNodeName) {
        valueDiff = dataOut[suc].back() ^ inputData;
        dataOut[suc].push_back(inputData);
        // Get positions of set bits in the difference (using the inherited
        // getPositionList)
        std::vector<unsigned> posList = getPositionList(valueDiff);
        for (unsigned pos : posList) {
          if (perChannelToggle[suc].find(pos) != perChannelToggle[suc].end())
            perChannelToggle[suc][pos]++;
        }
      } else {
        dataOut[suc].push_back(-1);
      }
    } else {
      if (suc == conSucNodeName) {
        valueDiff = inputData;
        dataOut[suc] = {inputData};
        std::vector<unsigned> posList = getPositionList(valueDiff);
        for (unsigned pos : posList) {
          if (perChannelToggle[suc].find(pos) != perChannelToggle[suc].end())
            perChannelToggle[suc][pos]++;
        }
      } else {
        dataOut[suc] = {-1};
      }
    }
  }
}

void CMergeNode::printNodeDetails() {
  AdjNode::printNodeDetails();

  switchingDebugStream(SwitchingDebugCategory::Node) << "[DEBUG] \t\tCon_suc_node_name: " << conSucNodeName
                          << "\n";
  switchingDebugStream(SwitchingDebugCategory::Node) << "[DEBUG] \t\tData_suc_node_name: "
                          << dataSucNodeName << "\n";
}

//===----------------------------------------------------------------------===//
//
// CBrNode Node
//
//===----------------------------------------------------------------------===//

CBrNode::CBrNode(mlir::Operation *op,
                 const std::vector<std::string> &predecessors,
                 const std::vector<std::string> &successors,
                 const std::map<std::string, unsigned> &sucDataWidthMap,
                 unsigned latency, const unsigned &bbIndex)
    : AdjNode(op, predecessors, successors, sucDataWidthMap, latency, bbIndex),
      lastValidDataValue(-1) {
  // Get the ConditionalBranch node
  auto CBrOp = dyn_cast<handshake::ConditionalBranchOp>(op);
  auto condPreOperand = CBrOp.getConditionOperand();
  auto dataPreOperand = CBrOp.getDataOperand();
  auto trueSucOperand = CBrOp.getTrueResult();
  auto falseSucOperand = CBrOp.getFalseResult();

  condPreNodeName = condPreOperand.getDefiningOp()
                        ->getAttrOfType<mlir::StringAttr>("handshake.name")
                        .getValue()
                        .str();
  dataPreNodeName = dataPreOperand.getDefiningOp()
                        ->getAttrOfType<mlir::StringAttr>("handshake.name")
                        .getValue()
                        .str();
  trueSucNodeName = getHandshakeNodeName(trueSucOperand);
  falseSucNodeName = getHandshakeNodeName(falseSucOperand);

  // Initialize per_channel_dataout for channels 0 and 1.
  // In MLIR implementation
  // True result port : 0
  // False result port : 1
  outChannelNameToIndexMap[trueSucNodeName] = 0;
  outChannelNameToIndexMap[falseSucNodeName] = 1;
  per_channel_dataout[0] = std::vector<int>();
  per_channel_dataout[1] = std::vector<int>();
}

void CBrNode::calValidSwitching(const std::string &sucNodeName,
                                unsigned condValue, int numValid0,
                                int numValid1) {
  const unsigned condBit = condValue ? 1U : 0U;
  auto idxIt = outChannelNameToIndexMap.find(sucNodeName);
  if (idxIt == outChannelNameToIndexMap.end()) {
    validSignal[sucNodeName] = 0;
    return;
  }
  // In RTL, cond=1 selects true output(port 0), cond=0 selects false(port 1).
  // outChannelNameToIndexMap stores this as selectedIndex = 1 - condBit.
  if ((1U - condBit) == idxIt->second) {
    if (numValid0 > 0 || numValid1 > 0)
      validSignal[sucNodeName] = 2;
    else
      validSignal[sucNodeName] = 0;
  } else {
    validSignal[sucNodeName] = 0;
  }
}

void CBrNode::calValidSet(const std::string &sucNodeName,
                          unsigned nodeStartTime, unsigned condValue,
                          unsigned II) {
  if (validSignal.find(sucNodeName) != validSignal.end()) {
    if (validSignal[sucNodeName] == 0 || II == 1) {
      setV[sucNodeName] = fullSet(II);
    } else {
      const unsigned condBit = condValue ? 1U : 0U;
      auto idxIt = outChannelNameToIndexMap.find(sucNodeName);
      if (idxIt == outChannelNameToIndexMap.end() ||
          (1U - condBit) != idxIt->second)
        setV[sucNodeName] = IISet(II, false);
      else {
        IISet tmp(II, false);
        tmp.set(normalizeCycleIndex(static_cast<int>(nodeStartTime), II));
        setV[sucNodeName] = std::move(tmp);
      }
    }
  }
}

void CBrNode::calReadySwitching(const std::string &preNodeName, int numValid,
                                int numReady) {
  if (numValid == 0 && numReady == 0)
    readySignal[preNodeName] = 0;
  else if (numValid > 0 || numReady > 0)
    readySignal[preNodeName] = 2;
}

void CBrNode::calReadySet(const std::string &preNodeName, const IISet *setValid,
                          const IISet *setReady, unsigned II) {
  if (setReady != nullptr && setValid != nullptr) {
    IISet finalSet = intersection_(*setValid, *setReady);
    setR[preNodeName] = std::move(finalSet);
  } else if (readySignal.find(preNodeName) != readySignal.end() &&
             readySignal[preNodeName] == 0) {
    setR[preNodeName] = fullSet(II);
  }

  auto it = setR.find(preNodeName);
  if (it != setR.end())
    ::setReady(this, preNodeName, switchingFromSet(it->second));
}

void CBrNode::setReadySet(const std::string &preNodeName, unsigned II) {
  std::string selSucName = "";
  for (const auto &entry : validSignal) {
    if (entry.second != 0) {
      selSucName = entry.first;
      break;
    }
  }
  if (!selSucName.empty())
    setR[preNodeName] = setV[selSucName];
  else {
    setR[preNodeName] = fullSet(II);
  }
}

void CBrNode::updateDataout(int inputData, unsigned condValue) {
  const unsigned condBit = condValue ? 1U : 0U;
  if (inputData == -1 && lastValidDataValue != -1)
    inputData = lastValidDataValue;
  else
    lastValidDataValue = inputData;

  int diffValue = 0;
  for (const std::string &suc : sucs) {
    if (dataOut.find(suc) != dataOut.end() && !dataOut[suc].empty()) {
      diffValue = dataOut[suc].back() ^ inputData;
      dataOut[suc].push_back(inputData);
    } else {
      diffValue = inputData;
      dataOut[suc] = {inputData};
    }
    std::vector<unsigned> posList = getPositionList(diffValue);
    for (unsigned pos : posList) {
      if (perChannelToggle[suc].find(pos) != perChannelToggle[suc].end())
        perChannelToggle[suc][pos]++;
    }
  }
  // TODO: Check this assignment strategy
  // Update per_channel_dataout for channel (1 - condValue)
  int targetChannel = 1 - static_cast<int>(condBit);
  per_channel_dataout[targetChannel].push_back(inputData);
}

void CBrNode::printNodeDetails() {
  AdjNode::printNodeDetails();

  switchingDebugStream(SwitchingDebugCategory::Node) << "[DEBUG] \t\tCond_pre_node_name: "
                          << condPreNodeName << "\n";
  switchingDebugStream(SwitchingDebugCategory::Node) << "[DEBUG] \t\tData_pre_node_name: "
                          << dataPreNodeName << "\n";
  switchingDebugStream(SwitchingDebugCategory::Node) << "[DEBUG] \t\tTrue_suc_node_name: "
                          << trueSucNodeName << "\n";
  switchingDebugStream(SwitchingDebugCategory::Node) << "[DEBUG] \t\tFalse_suc_node_name: "
                          << falseSucNodeName << "\n";
  switchingDebugStream(SwitchingDebugCategory::Node) << "[DEBUG] \t\tOut Channel Name to Index Map:\n";
  for (const auto &[outName, portIdx] : outChannelNameToIndexMap) {
    switchingDebugStream(SwitchingDebugCategory::Node) << "[DEBUG] \t\t\tOutput Node: " << outName
                            << "; Port Idx: " << portIdx << "\n";
  }
  switchingDebugStream(SwitchingDebugCategory::Node) << "[DEBUG] \t\tPer_channel_dataout:\n";
  for (const auto &entry : per_channel_dataout) {
    switchingDebugStream(SwitchingDebugCategory::Node) << "[DEBUG] \t\t\tChannel " << entry.first << ": ";
    for (int v : entry.second)
      switchingDebugStream(SwitchingDebugCategory::Node) << v << " ";
    switchingDebugStream(SwitchingDebugCategory::Node) << "\n";
  }
}

//===----------------------------------------------------------------------===//
//
// DLoad Node
//
//===----------------------------------------------------------------------===//

DLoadNode::DLoadNode(mlir::Operation *op,
                     const std::vector<std::string> &predecessors,
                     const std::vector<std::string> &successors,
                     const std::map<std::string, unsigned> &sucDataWidthMap,
                     const unsigned &latency, const unsigned &bbIndex)
    : AdjNode(op, predecessors, successors, sucDataWidthMap, latency, bbIndex) {
  // Get the LoadOp
  auto loadOp = dyn_cast<handshake::LoadOp>(op);
  auto dataOutRes = loadOp.getDataResult();
  auto addrOutRes = loadOp.getAddressResult();
  auto addrInRes = loadOp.getAddressInput();

  dataOutNodeName = getHandshakeNodeName(dataOutRes);
  addressOutNodeName = getHandshakeNodeName(addrOutRes);
  addressInNodeName = addrInRes.getDefiningOp()
                          ->getAttrOfType<mlir::StringAttr>("handshake.name")
                          .getValue()
                          .str();
}

void DLoadNode::calValidSwitching(const std::string &sucNodeName,
                                  int &numValid) {
  if (numValid == 0)
    validSignal[sucNodeName] = 0;
  else if (numValid > 0)
    validSignal[sucNodeName] = 2;
}

void DLoadNode::calValidSet(const std::string &sucNodeName, const IISet *inSetV,
                            unsigned &II) {
  if (validSignal.find(sucNodeName) != validSignal.end()) {
    if (validSignal[sucNodeName] == 0) {
      setV[sucNodeName] = fullSet(II);
    } else if (inSetV != nullptr) {
      IISet finalSet(II, false);
      for (unsigned idx : inSetV->set_bits()) {
        finalSet.set((idx + nodeLatency) % II);
      }
      setV[sucNodeName] = finalSet;
    }
  }
}

void DLoadNode::calReadySwitching(const std::string &preNodeName,
                                  int &numReady) {
  if (numReady == 0)
    readySignal[preNodeName] = 0;
  else if (numReady > 0)
    readySignal[preNodeName] = 2;
}

void DLoadNode::calReadySet(const std::string &preNodeName, const IISet *inSetR,
                            unsigned &nodeStartTime, unsigned &II) {
  if (inSetR != nullptr) {
    setR[preNodeName] = *inSetR;
  } else if (readySignal.find(preNodeName) != readySignal.end()) {
    if (readySignal[preNodeName] == 0) {
      setR[preNodeName] = fullSet(II);
    } else if (readySignal[preNodeName] > 0) {
      IISet tmp(II, false);
      tmp.set(normalizeCycleIndex(static_cast<int>(nodeStartTime), II));
      setR[preNodeName] = std::move(tmp);
    }
  }

  auto it = setR.find(preNodeName);
  if (it != setR.end())
    ::setReady(this, preNodeName, switchingFromSet(it->second));
}

void DLoadNode::updateDataout(int &inputData, int addressValue) {
  // --- Update data output channel ---
  int dataDiff = 0;
  if (dataOut.find(dataOutNodeName) != dataOut.end() &&
      !dataOut[dataOutNodeName].empty()) {
    dataDiff = dataOut[dataOutNodeName].back() ^ inputData;
    dataOut[dataOutNodeName].push_back(inputData);
  } else {
    dataDiff = inputData;
    dataOut[dataOutNodeName] = {inputData};
  }
  // Update per-channel toggle counts (using getPositionList() from AdjNode)
  auto dataPositions = getPositionList(dataDiff);
  for (unsigned pos : dataPositions) {
    if (perChannelToggle[dataOutNodeName].find(pos) !=
        perChannelToggle[dataOutNodeName].end())
      perChannelToggle[dataOutNodeName][pos]++;
  }

  // --- Update address output channel ---
  int addressDiff = 0;
  if (dataOut.find(addressOutNodeName) != dataOut.end() &&
      !dataOut[addressOutNodeName].empty()) {
    if (addressValue == -10)
      addressValue = dataOut[addressOutNodeName].back();
    addressDiff = dataOut[addressOutNodeName].back() ^ addressValue;
    dataOut[addressOutNodeName].push_back(addressValue);
  } else {
    addressDiff = addressValue;
    dataOut[addressOutNodeName] = {addressValue};
  }
  auto addrPositions = getPositionList(addressDiff);
  for (unsigned pos : addrPositions) {
    if (perChannelToggle[addressOutNodeName].find(pos) !=
        perChannelToggle[addressOutNodeName].end())
      perChannelToggle[addressOutNodeName][pos]++;
  }
}

void DLoadNode::printNodeDetails() {
  // Call base class's printNodeDetails()
  AdjNode::printNodeDetails();

  switchingDebugStream(SwitchingDebugCategory::Node) << "[DEBUG] \t\tAddress input node: "
                          << addressInNodeName << ";\n";
  switchingDebugStream(SwitchingDebugCategory::Node) << "[DEBUG] \t\tAddress output node: "
                          << addressOutNodeName << ";\n";
  switchingDebugStream(SwitchingDebugCategory::Node) << "[DEBUG] \t\tData output node: " << dataOutNodeName
                          << ";\n";
}

//===----------------------------------------------------------------------===//
//
// DStoreNode Node
//
//===----------------------------------------------------------------------===//
// TODO: Seperate MC_store and LSQ_store
DStoreNode::DStoreNode(mlir::Operation *op,
                       const std::vector<std::string> &predecessors,
                       const std::vector<std::string> &successors,
                       const std::map<std::string, unsigned> &sucDataWidthMap,
                       unsigned latency, const unsigned &bbIndex)
    : AdjNode(op, predecessors, successors, sucDataWidthMap, latency, bbIndex) {
  // Add the data channel to the mem_controller
  unsigned dataWidth = 0;
  for (const auto &selPair : sucDataWidthMap) {
    dataWidth = selPair.second;
    break;
  }
  sucsDataWidthMap["mem_data"] = dataWidth;

  // Get the StoreOp
  auto storeOp = dyn_cast<handshake::StoreOp>(op);
  auto dataInValue = storeOp.getDataInput();
  auto addrInNodeValue = storeOp.getAddressInput();

  dataInNode = dataInValue.getDefiningOp()
                   ->getAttrOfType<mlir::StringAttr>("handshake.name")
                   .getValue()
                   .str();
  addressInNode = addrInNodeValue.getDefiningOp()
                      ->getAttrOfType<mlir::StringAttr>("handshake.name")
                      .getValue()
                      .str();

  // The "source" nodes for these inputs are to be set externally.
  dataInSrcNode = "";
  addressInSrcNode = "";
}

void DStoreNode::calValidSwitching(int numValid1, int numValid2) {
  // `store.v` has two independent output-valid channels:
  //   - addrOut_valid   (address path)
  //   - dataToMem_valid (data path)
  // Keep both channels explicit so total valid switching matches the RTL
  // interface shape.
  validSignal["mem_addr"] = (numValid1 > 0) ? 2U : 0U;
  validSignal["mem_data"] = (numValid2 > 0) ? 2U : 0U;
}

void DStoreNode::calValidSet(const IISet *setV0, const IISet *setV1,
                             unsigned II) {
  auto shiftSet = [&](const IISet &inSet) {
    IISet outSet(II, false);
    for (unsigned bitidx : inSet.set_bits())
      outSet.set((bitidx + nodeLatency) % II);
    return outSet;
  };

  if (setV0 != nullptr)
    setV["mem_addr"] = shiftSet(*setV0);
  else if (validSignal.find("mem_addr") != validSignal.end() &&
           validSignal["mem_addr"] == 0)
    setV["mem_addr"] = fullSet(II);

  if (setV1 != nullptr)
    setV["mem_data"] = shiftSet(*setV1);
  else if (validSignal.find("mem_data") != validSignal.end() &&
           validSignal["mem_data"] == 0)
    setV["mem_data"] = fullSet(II);
}

void DStoreNode::calReadySwitching(const std::string &preNodeName,
                                   int numValid1, int numValid2) {
  int localValid = 0;
  if (preNodeName == addressInNode)
    localValid = numValid1;
  else if (preNodeName == dataInNode)
    localValid = numValid2;
  else
    localValid = (numValid1 > 0 || numValid2 > 0) ? 2 : 0;

  if (localValid > 0)
    readySignal[preNodeName] = 2;
  else if (localValid == 0)
    readySignal[preNodeName] = 0;
  else
    readySignal[preNodeName] = 2;
}

void DStoreNode::calReadySet(const std::string &preNodeName, const IISet *setV0,
                             const IISet *setV1, unsigned II) {
  // `store.v` models address/data handshakes independently. Mirror that by
  // mapping each predecessor to its own valid window instead of intersecting
  // both channels.
  const IISet *selected = nullptr;
  if (preNodeName == addressInNode)
    selected = setV0;
  else if (preNodeName == dataInNode)
    selected = setV1;

  if (selected != nullptr) {
    IISet finalSet(II, false);
    for (unsigned i : selected->set_bits())
      finalSet.set((i + nodeLatency) % II);
    setR[preNodeName] = finalSet;
  } else if (readySignal.find(preNodeName) != readySignal.end() &&
             readySignal[preNodeName] == 0) {
    setR[preNodeName] = fullSet(II);
  }

  auto it = setR.find(preNodeName);
  if (it != setR.end())
    ::setReady(this, preNodeName, switchingFromSet(it->second));
}

void DStoreNode::updateDataout(int inputData, const std::string &srcInputNode) {
  // The method assumes that dataInSrcNode and addressInSrcNode have been set.
  assert(!dataInSrcNode.empty());
  assert(!addressInSrcNode.empty());

  int valueDiff = 0;
  if (srcInputNode == addressInSrcNode) {
    // Update the address output channel (use key "address_out")
    if (dataOut.find("address_out") != dataOut.end() &&
        !dataOut["address_out"].empty()) {
      valueDiff = dataOut["address_out"].back() ^ inputData;
      dataOut["address_out"].push_back(inputData);
    } else {
      valueDiff = inputData;
      dataOut["address_out"] = {inputData};
    }
  } else if (srcInputNode == dataInSrcNode) {
    // Update the actual data output channel (use key "data_out")
    if (dataOut.find("data_out") != dataOut.end() &&
        !dataOut["data_out"].empty()) {
      valueDiff = dataOut["data_out"].back() ^ inputData;
      dataOut["data_out"].push_back(inputData);
    } else {
      valueDiff = inputData;
      dataOut["data_out"] = {inputData};
    }
  }
  // (Optional: update perChannelToggle accordingly.)
}

void DStoreNode::printNodeDetails() {
  // Call the base class printNodeDetails for common node info.
  AdjNode::printNodeDetails();

  switchingDebugStream(SwitchingDebugCategory::Node) << "[DEBUG] \t\tAddress input node: " << addressInNode
                          << "\n";
  switchingDebugStream(SwitchingDebugCategory::Node) << "[DEBUG] \t\tSrc address input node: "
                          << addressInSrcNode << "\n";
  switchingDebugStream(SwitchingDebugCategory::Node) << "[DEBUG] \t\tData input node: " << dataInNode
                          << "\n";
  switchingDebugStream(SwitchingDebugCategory::Node) << "[DEBUG] \t\tSrc data input node: "
                          << dataInSrcNode << "\n";
}

bool DStoreNode::handshakeSwitchingChecking() {
  // For example, if the number of ready channels equals the number of
  // predecessors.
  return (readySignal.size() == pres.size());
}

//===----------------------------------------------------------------------===//
//
// Start Node
//
//===----------------------------------------------------------------------===//

StartNode::StartNode(mlir::Operation *op,
                     const std::vector<std::string> &predecessors,
                     const std::vector<std::string> &successors,
                     const std::map<std::string, unsigned> &sucDataWidthDict,
                     unsigned latency, const unsigned &bbIndex)
    : AdjNode(op, predecessors, successors, sucDataWidthDict, latency,
              bbIndex) {
  // No extra initialization needed.
}

void StartNode::calValidSwitching(const std::string &sucNodeName) {
  // In the Start node, set the valid signal for the given successor to 2.
  validSignal[sucNodeName] = 2;
}

void StartNode::calValidSet() {
  // No additional valid-set computation for Start node.
}

void StartNode::calReadySwitching() {
  // No ready switching computation for Start node.
}

void StartNode::calReadySet() {
  // No ready set computation for Start node.
}
