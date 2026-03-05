//===- GraphModel.cpp - Switching estimation ------------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Implements the supporting datastructures for the switching estimation pass.
//===----------------------------------------------------------------------===//

#include "experimental/Analysis/SwitchingEstimation/GraphModel.h"
#include "experimental/Analysis/SwitchingEstimation/Debug.h"
#include "dynamatic/Dialect/Handshake/HandshakeAttributes.h"
#include "dynamatic/Support/CFG.h"
#include "dynamatic/Transforms/BufferPlacement/CFDFC.h"
#include "experimental/Analysis/SwitchingEstimation/NodeModels.h"
#include "experimental/Analysis/SwitchingEstimation/utils.h"
#include "experimental/Support/StdProfiler.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;
using namespace mlir;
using namespace dynamatic;
using namespace dynamatic::experimental;
using namespace dynamatic::handshake;
using namespace dynamatic::buffer;

//===----------------------------------------------------------------------===//
//
// Definitions of AdjNode
//
//===----------------------------------------------------------------------===//

AdjNode::AdjNode(Operation *selOp, const std::vector<std::string> &predecessors,
                 const std::vector<std::string> &successors,
                 const std::map<std::string, unsigned> &sucDataWidthMap,
                 const unsigned &latency, const unsigned &bbIndex) {
  // Initialize all variables
  this->op = selOp;
  this->pres = predecessors;
  this->sucs = successors;
  // Note: We may not need the datawidth map for power estimation
  this->sucsDataWidthMap = sucDataWidthMap;
  this->nodeLatency = latency;
  this->bbindex = bbIndex;

  // Initialize the toggle count map
  for (auto &[key, value] : sucsDataWidthMap) {
    std::map<unsigned, unsigned> tmpBitMap;

    for (unsigned i = 0; i < value; i++) {
      tmpBitMap[i] = 0;
    }
    perChannelToggle[key] = tmpBitMap;
  }
}

// This function checks whether the handshake channel switching information
// updating is finished or not for the node itself and all its predecessors
// NOTE: Relaxed conditions
bool AdjNode::handshakeUpdateFinished() {
  if (isa<DStoreNode>(this)) {
    return !validSignal.empty() && readySignal.size() >= pres.size() &&
           !setV.empty() && setR.size() >= pres.size();
  }

  if (validSignal.size() >= sucs.size()) {
    if (readySignal.size() >= pres.size()) {
      if (setR.size() >= pres.size()) {
        if (setV.size() >= sucs.size()) {
          return true;
        }
      }
    }
  }

  return false;
}

bool AdjNode::handshakeSwitchingChecking() {
  if (isa<DStoreNode>(this))
    return !validSignal.empty() && readySignal.size() >= pres.size();

  if (validSignal.size() >= sucs.size()) {
    if (readySignal.size() >= pres.size()) {
      return true;
    }
  }

  return false;
}

void AdjNode::totalHandshakeSwitchingCounting() {
  // Recompute from the current per-channel maps. This function can be called
  // multiple times during fixed-point solving, so totals must be reset to
  // avoid artificial accumulation across iterations.
  totalValidSwitching = 0;
  totalReadySwitching = 0;

  for (const auto &[s, selValue] : validSignal) {
    totalValidSwitching += selValue;
  }

  for (const auto &[s, selValue] : readySignal) {
    totalReadySwitching += selValue;
  }
}

void AdjNode::updateDataoutChannel(int inputData) {
  // For each successor, we do xor for the data to calculate the toggle number
  for (const auto &suc : sucs) {
    int diff = 0;

    // Check whether the key is in dataOut
    if (dataOut.find(suc) != dataOut.end()) {
      auto &hist = dataOut[suc];
      diff = hist.back() ^ inputData;
      hist.push_back(inputData);
    } else {
      diff = inputData;
      std::vector<int> tmpVec = {inputData};
      dataOut[suc] = tmpVec;
    }

    // Update the per_channel count
    auto posList = getPositionList(diff);

    for (const auto &selPos : posList) {
      if (perChannelToggle[suc].find(selPos) != perChannelToggle[suc].end()) {
        perChannelToggle[suc][selPos] += 1;
      }
    }
  }
}

std::vector<unsigned> AdjNode::getPositionList(int number) {
  std::vector<unsigned> posList;

  unsigned idx = 0;
  unsigned tmp = static_cast<unsigned>(number);
  while (tmp) {
    if (tmp & 1U) {
      posList.push_back(idx);
    }
    tmp >>= 1U;
    idx++;
  }
  return posList;
}

void AdjNode::updateHandshakeChannelSwitching(unsigned validChannelSwitching,
                                              unsigned readyChannelSwitching) {
  totalValidSwitching += validChannelSwitching;
  totalReadySwitching += readyChannelSwitching;

  // Update the handshake status as well
  handshakeUpdateFlag = true;
}

void AdjNode::totalDataSwitchingCounting(bool mapped) {
  // Reset the total data switching
  totalDataSwitching = 0;

  for (const auto &[suc, valueVec] : dataOut) {
    // Define tmp storing structure
    unsigned numSwitches = 0;

    // If this suc is one of the units at scf level
    int lastInput = mapped && valueVec[0] == -1 ? 1 : valueVec[0];

    // The corresponding output channel maynot have dataout
    // Check the validity of the data channel value
    for (unsigned i = 0; i < valueVec.size(); i++) {
      int curVal = (valueVec[i] == -1) ? 1 : valueVec[i];
      unsigned diff = static_cast<unsigned>(lastInput ^ curVal);

      // Count bits
      unsigned bitCount = 0;
      while (diff) {
        bitCount += (diff & 1U);
        diff >>= 1U;
      }
      numSwitches += bitCount;
      lastInput = curVal;
    }

    // Store the total number of channel switches
    dataSwitches[suc] = numSwitches;
    totalDataSwitching += numSwitches;
  }
}

//===----------------------------------------------------------------------===//
//
// AdjNode Debug Printing Functions
//
//===----------------------------------------------------------------------===//

// Define all printing functions to facilitate debugging
void AdjNode::printNodeDetails() {

  switchingDebugStream(SwitchingDebugCategory::Graph)
      << "[DEBUG] "
         "\t=============================================================\n";
  switchingDebugStream(SwitchingDebugCategory::Graph) << "[DEBUG] \t[Node Info Start]\n";

  //
  switchingDebugStream(SwitchingDebugCategory::Graph) << "[DEBUG] \t\tNode Name: "
               << op->getAttrOfType<StringAttr>("handshake.name").str() << "\n";
  switchingDebugStream(SwitchingDebugCategory::Graph) << "[DEBUG] \t\tLatency: " << nodeLatency << ";\n";
  switchingDebugStream(SwitchingDebugCategory::Graph) << "[DEBUG] \t\tPredecessors: [";
  for (const auto &p : pres) {
    switchingDebugStream(SwitchingDebugCategory::Graph) << p << " ";
  }
  switchingDebugStream(SwitchingDebugCategory::Graph) << "]\n[DEBUG] \t\tSuccessors: [";
  for (const auto &s : sucs) {
    switchingDebugStream(SwitchingDebugCategory::Graph) << s << " ";
  }
  switchingDebugStream(SwitchingDebugCategory::Graph) << "]\n";

  switchingDebugStream(SwitchingDebugCategory::Graph) << "[DEBUG] \t\tSuccessor Channel DataWidth: \n";
  for (const auto &[s, width] : sucsDataWidthMap) {
    switchingDebugStream(SwitchingDebugCategory::Graph) << "[DEBUG] \t\t\tNode: " << s << ", Data_width: " << width
                 << "\n";
  }

  switchingDebugStream(SwitchingDebugCategory::Graph) << "[DEBUG] \t\tSuccessor Channel Data Value\n";
  for (const auto &[s, valueVec] : dataOut) {
    switchingDebugStream(SwitchingDebugCategory::Graph) << "[DEBUG] \t\t\tNode: " << s << ", Output Value Vector: \n";
    int counter = 0;
    for (const auto &val : valueVec) {
      switchingDebugStream(SwitchingDebugCategory::Graph) << "[" << counter++ << "] : " << val << "; ";
    }
    switchingDebugStream(SwitchingDebugCategory::Graph) << "\n";
  }

  switchingDebugStream(SwitchingDebugCategory::Graph) << "[DEBUG] \t\tValid Channel Switching: \n";
  for (const auto &[s, numSwitches] : validSignal) {
    switchingDebugStream(SwitchingDebugCategory::Graph) << "[DEBUG] \t\t\tNode: " << s
                 << ", Data_width: " << numSwitches << "\n";
  }

  switchingDebugStream(SwitchingDebugCategory::Graph) << "[DEBUG] \t\tReady Channel Switching: \n";
  for (const auto &[s, numSwitches] : readySignal) {
    switchingDebugStream(SwitchingDebugCategory::Graph) << "[DEBUG] \t\t\tNode: " << s
                 << ", Data_width: " << numSwitches << "\n";
  }

  switchingDebugStream(SwitchingDebugCategory::Graph) << "[DEBUG] \t\tValid Active Range: \n";
  for (const auto &[s, valueVec] : setV) {
    switchingDebugStream(SwitchingDebugCategory::Graph) << "[DEBUG] \t\t\tNode: " << s << " ; Active Range: [";
    for (unsigned i = 0, e = valueVec.size(); i < e; ++i)
      switchingDebugStream(SwitchingDebugCategory::Graph) << (valueVec.test(i) ? 1 : 0) << " ";
    switchingDebugStream(SwitchingDebugCategory::Graph) << "]\n";
  }

  switchingDebugStream(SwitchingDebugCategory::Graph) << "[DEBUG] \t\tReady Active Range: \n";
  for (const auto &[s, valueVec] : setR) {
    switchingDebugStream(SwitchingDebugCategory::Graph) << "[DEBUG] \t\t\tNode: " << s << " ; Active Range: [";
    for (unsigned i = 0, e = valueVec.size(); i < e; ++i)
      switchingDebugStream(SwitchingDebugCategory::Graph) << (valueVec.test(i) ? 1 : 0) << " ";
    switchingDebugStream(SwitchingDebugCategory::Graph) << "]\n";
  }
}

void AdjNode::printHandshakeSwitching() {
  switchingDebugStream(SwitchingDebugCategory::Graph) << "[DEBUG] \t\tTotal Valid Switching Number: "
               << totalValidSwitching << "\n";
  switchingDebugStream(SwitchingDebugCategory::Graph) << "[DEBUG] \t\tTotal Ready Switching Number: "
               << totalReadySwitching << "\n";

  switchingDebugStream(SwitchingDebugCategory::Graph) << "[DEBUG] \t\tValid Channel Switching: \n";
  for (const auto &[s, numSwitches] : validSignal) {
    switchingDebugStream(SwitchingDebugCategory::Graph) << "[DEBUG] \t\t\tNode: " << s
                 << ", Data_width: " << numSwitches << "\n";
  }

  switchingDebugStream(SwitchingDebugCategory::Graph) << "[DEBUG] \t\tReady Channel Switching: \n";
  for (const auto &[s, numSwitches] : readySignal) {
    switchingDebugStream(SwitchingDebugCategory::Graph) << "[DEBUG] \t\t\tNode: " << s
                 << ", Data_width: " << numSwitches << "\n";
  }

  switchingDebugStream(SwitchingDebugCategory::Graph) << "[DEBUG] \t\tValid Active Range: \n";
  for (const auto &[s, valueVec] : setV) {
    switchingDebugStream(SwitchingDebugCategory::Graph) << "[DEBUG] \t\t\tNode: " << s << " ; Active Range: [";
    for (unsigned i = 0, e = valueVec.size(); i < e; ++i)
      switchingDebugStream(SwitchingDebugCategory::Graph) << (valueVec.test(i) ? 1 : 0) << " ";
    switchingDebugStream(SwitchingDebugCategory::Graph) << "]\n";
  }

  switchingDebugStream(SwitchingDebugCategory::Graph) << "[DEBUG] \t\tReady Active Range: \n";
  for (const auto &[s, valueVec] : setR) {
    switchingDebugStream(SwitchingDebugCategory::Graph) << "[DEBUG] \t\t\tNode: " << s << " ; Active Range: [";
    for (unsigned i = 0, e = valueVec.size(); i < e; ++i)
      switchingDebugStream(SwitchingDebugCategory::Graph) << (valueVec.test(i) ? 1 : 0) << " ";
    switchingDebugStream(SwitchingDebugCategory::Graph) << "]\n";
  }
}

void AdjNode::printDataChannelSwitching() {

  switchingDebugStream(SwitchingDebugCategory::Graph) << "[DEBUG] \t\tTotal Number of Data Channel Switches: "
               << totalDataSwitching << "\n";

  // Print per channel data switches
  for (const auto &[suc, value] : dataSwitches) {
    switchingDebugStream(SwitchingDebugCategory::Graph) << "[DEBUG] \t\t\tChannel: " << suc
                 << ", Num Switches: " << value << "\n";
  }
}

void AdjNode::printPerDataChannelPerBitToggleNumber() {

  switchingDebugStream(SwitchingDebugCategory::Graph) << "[DEBUG] \t\tPer Data Channel Per Bit Toggle Number: \n";

  for (const auto &[suc, valueVec] : perChannelToggle) {
    switchingDebugStream(SwitchingDebugCategory::Graph) << "[DEBUG] \t\t Node: " << suc << "\n";

    for (const auto &[selBit, value] : valueVec) {
      switchingDebugStream(SwitchingDebugCategory::Graph) << "[DEBUG] \t\t\tBit " << selBit << ": " << value << "\n";
    }
  }
}

void AdjNode::printPerHandshakeChannelToggleNumber() {
  switchingDebugStream(SwitchingDebugCategory::Graph) << "[DEBUG] \t\t[VALID CHANNEL]\n";
  for (const auto &[suc, validSwitch] : validSignal) {
    switchingDebugStream(SwitchingDebugCategory::Graph) << "[DEBUG] \t\t\tNode: " << suc
                 << ", Valid Switching: " << validSwitch << "\n";
  }

  switchingDebugStream(SwitchingDebugCategory::Graph) << "[DEBUG] \t\t[READY CHANNEL]\n";
  for (const auto &[pre, readySwitching] : readySignal) {
    switchingDebugStream(SwitchingDebugCategory::Graph) << "[DEBUG] \t\t\tNode: " << pre
                 << ", Ready Switching: " << readySwitching << "\n";
  }
}

//===----------------------------------------------------------------------===//
//
// Definitions of AdjGraph
//
//===----------------------------------------------------------------------===//
// Function declaration
static unsigned extractNodeLatency(Operation *op, TimingDatabase timingDB,
                                   const double &targetPeriod);

// Initialize the whole adjacency graph for the selected segment
AdjGraph::AdjGraph(CFDFC *cfdfc, const TimingDatabase &timingDB,
                   const unsigned &II, const unsigned &mgIndex,
                   const double &targetPeriod) {
  // Initialize the internal variables
  this->cfdfcPrt = cfdfc;
  this->cfdfcIndex = mgIndex;
  this->cfdfcII = II;

  std::map<std::string, std::vector<std::string>> nodeToPresMap;
  std::map<std::string, std::vector<std::string>> nodeToSucsMap;

  // Step 1: Construct the pres and sucs map for all units in the selected CFDFC
  for (const auto &selChannel : cfdfc->channels) {
    // Source operation
    Operation *srcOp = selChannel.getDefiningOp();
    std::string srcName =
        srcOp->getAttrOfType<StringAttr>("handshake.name").str();

    for (const auto &dstOp : selChannel.getUsers()) {
      std::string dstName =
          dstOp->getAttrOfType<StringAttr>("handshake.name").str();

      // Get the BB index of the dstOp
      unsigned dstBB;
      if (std::optional<unsigned> optBB = getLogicBB(dstOp); !optBB.has_value())
        continue;
      else
        dstBB = *optBB;

      // Check whether this node is in the cfdfc
      if (!cfdfc->cycle.contains(dstBB))
        continue;

      // Check whether this is a backedge
      if (cfdfc->isCFDFCBackedge(selChannel)) {
        backedges.push_back(std::make_pair(srcName, dstName));

        // Add the name of the dstNode to the start node vector
        segStartNodes.push_back(dstName);
      }

      // Insert to sucs
      insertToSurroundingList(nodeToSucsMap, srcName, dstName);

      // Insert to pres
      insertToSurroundingList(nodeToPresMap, dstName, srcName);
    }
  }

  // Step 2: Traverse all nodes in the CFDFC
  for (auto &selNode : cfdfc->units) {
    std::string unitType = selNode->getName().getStringRef().str();
    std::string unitName =
        selNode->getAttrOfType<StringAttr>("handshake.name").str();

    // Preserve the relative order of nodes
    orderedNodeName.push_back(unitName);

    // Get the unit latency
    unsigned nodeLatency = extractNodeLatency(selNode, timingDB, targetPeriod);

    // Get the BB index
    auto nodeBBIndexAttr = selNode->getAttrOfType<IntegerAttr>("handshake.bb");
    unsigned nodeBBIndex = 0;
    if (!nodeBBIndexAttr) {
      switchingDebugStream(SwitchingDebugCategory::Graph)
          << "[WARNING] \t[AdjGraph] Cannot find the BB index for node: "
          << unitName << "\n";
    } else {
      nodeBBIndex = nodeBBIndexAttr.getUInt();
    }

    //! Testing
    //
    //     switchingDebugStream(SwitchingDebugCategory::Graph)
    //     << "[DEBUG] "
    //        "\t=============================================================\n";
    // switchingDebugStream(SwitchingDebugCategory::Graph) << "[DEBUG] \tNode Name: " << unitName << "\n";
    // switchingDebugStream(SwitchingDebugCategory::Graph) << "[DEBUG] \tNode Latency From DataBase: "
    //                         << nodeLatency << "\n";
    // switchingDebugStream(SwitchingDebugCategory::Graph)
    //            << "[DEBUG] \tNode BB Index: " << nodeBBIndex << "\n";

    // Step 2.1: Construct the node storing structure
    auto newNode = createNodeFromOperation(selNode, nodeToPresMap[unitName],
                                           nodeToSucsMap[unitName], nodeLatency,
                                           nodeBBIndex);

    if (!newNode)
      continue;

    //! Testing
    // newNode->printNodeDetails();

    // Store the new node
    nodes[unitName] = newNode;
  }

  // TODO: Need to check the completeness of the graph. 14/09/2025
}

AdjGraph::AdjGraph(
    const TimingDatabase &timingDB, const unsigned &II,
    handshake::FuncOp funcOp,
    std::vector<std::pair<std::string, std::string>> &allBackedges,
    const double &targetPeriod)
    : backedges(allBackedges) {

  cfdfcII = II;
  // Iterate over all ops in the funcOp
  for (Operation &op : funcOp.getOps()) {
    std::string unitName = op.getAttrOfType<StringAttr>("handshake.name").str();
    //! For now, we exclude lsq and mem_controller
    // TODO: Add support for these two ops
    if (contains(unitName, "lsq") || contains(unitName, "mem_controller"))
      continue;

    // Preserve the relative order of nodes
    orderedNodeName.push_back(unitName);

    //! Testing
    //
    //     switchingDebugStream(SwitchingDebugCategory::Graph)
    //     << "[DEBUG] "
    //        "\t=============================================================\n";
    // switchingDebugStream(SwitchingDebugCategory::Graph) << "[DEBUG] \tNode Name: " << unitName << "\n";

    std::vector<std::string> pres;
    std::vector<std::string> sucs;
    // Construct the sucs
    for (OpResult res : op.getResults()) {
      assert(std::distance(res.getUsers().begin(), res.getUsers().end()) == 1 &&
             "value must have unique user");

      Operation *user = *res.getUsers().begin();
      std::string dstName =
          user->getAttrOfType<StringAttr>("handshake.name").str();

      // Excluding lsq and mem_controller
      if ((!contains(dstName, "lsq")) && (!contains(dstName, "mem_controller")))
        sucs.push_back(dstName);
    }

    // Construct pres
    auto getBlockArgName = [&](BlockArgument blockArg) -> std::string {
      auto *ownerBlock = blockArg.getOwner();
      if (!ownerBlock)
        return "";
      auto *funcOp = ownerBlock->getParentOp();
      auto handshakeFunc = dyn_cast_or_null<handshake::FuncOp>(funcOp);
      if (!handshakeFunc)
        return "";

      auto argNamesAttr =
          handshakeFunc->getAttrOfType<mlir::ArrayAttr>("argNames");
      if (!argNamesAttr)
        return "";

      unsigned argIdx = blockArg.getArgNumber();
      if (argIdx >= argNamesAttr.size())
        return "";

      auto strAttr =
          dyn_cast<mlir::StringAttr>(argNamesAttr.getValue()[argIdx]);
      if (!strAttr)
        return "";

      return strAttr.getValue().str();
    };

    for (auto operand : op.getOperands()) {
      // TODO: Need to check do we need to include the block argument in the
      // graph or not
      // TODO: Add support for LSQ and mem_controller
      if (operand.getDefiningOp()) {
        std::string preName = operand.getDefiningOp()
                                  ->getAttrOfType<StringAttr>("handshake.name")
                                  .str();
        if ((!contains(preName, "lsq")) &&
            (!contains(preName, "mem_controller")))
          pres.push_back(preName);
      } else if (auto blockArg = operand.dyn_cast<mlir::BlockArgument>()) {
        std::string preName = getBlockArgName(blockArg);
        if (!preName.empty())
          pres.push_back(preName);
      }
    }

    unsigned nodeLatency = extractNodeLatency(&op, timingDB, targetPeriod);

    // Get the BB index
    auto nodeBBIndexAttr = op.getAttrOfType<IntegerAttr>("handshake.bb");
    // We assign a large number to those nodes without a bbIndex
    unsigned nodeBBIndex = 100;
    if (!nodeBBIndexAttr) {
      // For now we just print a warning message and keep a large number for the
      // bbIndex
      switchingDebugStream(SwitchingDebugCategory::Graph) << "[DEBUG] \tCan't get the BB index of the op: " << unitName
                   << "\n";
    } else {
      nodeBBIndex = nodeBBIndexAttr.getUInt();
    }

    auto newNode =
        createNodeFromOperation(&op, pres, sucs, nodeLatency, nodeBBIndex);

    if (!newNode)
      continue;

    //! Testing
    // newNode->printNodeDetails();

    // Store the new node
    nodes[unitName] = newNode;
  }

  // TODO: Need to check the completeness of the graph. 14/09/2025
}

void AdjGraph::insertToSurroundingList(
    std::map<std::string, std::vector<std::string>> &selMap, std::string &key,
    std::string &value) {
  if (contains(selMap, key)) {
    selMap[key].push_back(value);
  } else {
    std::vector<std::string> tmpVec{value};
    selMap[key] = tmpVec;
  }
}

//===----------------------------------------------------------------------===//
//
// Definitions of the node latency extraction function
//
//===----------------------------------------------------------------------===//

/// Extracts the latency for each operation
/// This is done in 3 ways:
/// 1. If the operation is in the timingDB, the latency is extracted from the
/// timingDB
/// 2. If the operation is a buffer operation, the latency is extracted from the
/// buffer type with getLatencyDV()
/// 3. If the operation is neither, then its latency is set to 0
static unsigned extractNodeLatency(Operation *op, TimingDatabase timingDB,
                                   const double &targetPeriod) {

  double latency = 0;

  // Case 1: The operation is in the timingDB
  if (!failed(
          timingDB.getLatency(op, SignalType::DATA, latency, targetPeriod))) {
    return latency;
  }

  // Case 2: The operation is a buffer operation
  if (auto bufferOp = dyn_cast<handshake::BufferOp>(op)) {
    return bufferOp.getLatencyDV();
  }

  return 0;
}

//===----------------------------------------------------------------------===//
//
// Create per node storing structure based on unit type
//
//===----------------------------------------------------------------------===//
std::shared_ptr<AdjNode>
AdjGraph::createNodeFromOperation(Operation *op, std::vector<std::string> &pres,
                                  std::vector<std::string> &sucs,
                                  unsigned &nodeLatency, unsigned &bbIndex) {
  // NOTE: The datawidth map may be not necessary for power estimation
  std::map<std::string, unsigned> nodeSucsDataWidthMap;

  // Get the datawidth of each output channel
  for (unsigned resIndex = 0, e = op->getNumResults(); resIndex < e;
       ++resIndex) {
    Value selRes = op->getResult(resIndex);

    // The result is of type !handshake.channel<...>
    if (auto chanType = selRes.getType().dyn_cast<handshake::ChannelType>()) {
      // Extract the datawidth
      unsigned dataWidth = chanType.getDataBitWidth();

      // Iterate over all users of this result
      for (Operation *user : selRes.getUsers()) {
        if (auto nameAttr = user->getAttrOfType<StringAttr>("handshake.name")) {
          nodeSucsDataWidthMap[nameAttr.getValue().str()] = dataWidth;
        }
      }
    }
  }

  // Return a unique pointer to the created node
  return llvm::TypeSwitch<Operation *, std::shared_ptr<AdjNode>>(op)
      // handshake::AddIOp operator
      .Case<handshake::AddIOp>([&](auto selNode) {
        auto node = std::make_shared<AddiNode>(
            op, pres, sucs, nodeSucsDataWidthMap, nodeLatency, bbIndex);
        return node;
      })
      // handshake::SubIOp operator
      .Case<handshake::SubIOp>([&](auto selNode) {
        auto node = std::make_shared<SubiNode>(
            op, pres, sucs, nodeSucsDataWidthMap, nodeLatency, bbIndex);
        return node;
      })
      // handshake::MulIOp operator
      .Case<handshake::MulIOp>([&](auto selNode) {
        auto node = std::make_shared<MuliNode>(
            op, pres, sucs, nodeSucsDataWidthMap, nodeLatency, bbIndex);
        return node;
      })
      // handshake::CmpIOp operator
      .Case<handshake::CmpIOp>([&](handshake::CmpIOp selNode) {
        auto node = std::make_shared<CmpiNode>(
            op, pres, sucs, nodeSucsDataWidthMap, nodeLatency, bbIndex);
        return node;
      })
      // handshake::BufferOp operator
      .Case<handshake::BufferOp>([&](handshake::BufferOp selNode) {
        // For the buffer node, we need to get the corresponding attributes
        auto selBuffUnit = dyn_cast<handshake::BufferOp>(op);
        auto buffSlots = selBuffUnit.getNumSlots();
        auto buffType = selBuffUnit.getBufferType();
        auto transType = selBuffUnit.isBypassDV();

        auto node = std::make_shared<BufferNode>(
            op, pres, sucs, nodeSucsDataWidthMap, nodeLatency, bbIndex);

        node->numSlots = buffSlots;
        node->buffType = buffType;
        node->transparent = transType;
        // Get the buffer occupancy from the cfdfc analysis
        float_t occ = 0.0;
        if (cfdfcPrt) {
          if (auto it = cfdfcPrt->unitOccupancy.find(op);
              it != cfdfcPrt->unitOccupancy.end())
            occ = it->second;
        }
        node->occupancy = occ;

        return node;
      })
      // handshake::MuxOp operator
      .Case<handshake::MuxOp>([&](handshake::MuxOp selNode) {
        auto node = std::make_shared<MuxNode>(
            op, pres, sucs, nodeSucsDataWidthMap, nodeLatency, bbIndex);
        return node;
      })
      .Case<handshake::SelectOp>([&](handshake::SelectOp selNode) {
        llvm::errs() << "[ERROR] \t\t Missing Implementation for SELECT NODE\n";
        return std::shared_ptr<AdjNode>(nullptr);
      })
      // handshake::OrIOp operator
      .Case<handshake::OrIOp>([&](handshake::OrIOp selNode) {
        auto node = std::make_shared<OriNode>(
            op, pres, sucs, nodeSucsDataWidthMap, nodeLatency, bbIndex);
        return node;
      })
      // handshake::AndIOp operator
      .Case<handshake::AndIOp>([&](handshake::AndIOp selNode) {
        auto node = std::make_shared<AndiNode>(
            op, pres, sucs, nodeSucsDataWidthMap, nodeLatency, bbIndex);
        return node;
      })
      // handshake::ForkOp operator
      .Case<handshake::ForkOp>([&](handshake::ForkOp selNode) {
        auto node = std::make_shared<ForkNode>(
            op, pres, sucs, nodeSucsDataWidthMap, nodeLatency, bbIndex);
        return node;
      })
      // handshake::LazyForkOp operator
      .Case<handshake::LazyForkOp>([&](handshake::LazyForkOp selNode) {
        llvm::errs()
            << "[ERROR] \t\t Missing Implementation for LAZY FORK NODE\n";
        return std::shared_ptr<AdjNode>(nullptr);
      })
      // handshake::TruncIOp operator
      .Case<handshake::TruncIOp>([&](auto selNode) {
        auto node = std::make_shared<TrunciNode>(
            op, pres, sucs, nodeSucsDataWidthMap, nodeLatency, bbIndex);
        return node;
      })
      // handshake::ExtSIOp operator
      .Case<handshake::ExtSIOp>([&](auto selNode) {
        auto node = std::make_shared<ExtsiNode>(
            op, pres, sucs, nodeSucsDataWidthMap, nodeLatency, bbIndex);
        return node;
      })
      // handshake::ExtUIOp operator
      .Case<handshake::ExtUIOp>([&](auto selNode) {
        auto node = std::make_shared<ExtuiNode>(
            op, pres, sucs, nodeSucsDataWidthMap, nodeLatency, bbIndex);
        return node;
      })
      // handshake::ControlMergeOp operator
      .Case<handshake::ControlMergeOp>([&](handshake::ControlMergeOp selNode) {
        auto node = std::make_shared<CMergeNode>(
            op, pres, sucs, nodeSucsDataWidthMap, nodeLatency, bbIndex);
        return node;
      })
      // handshake::ConditionalBranchOp operator
      .Case<handshake::ConditionalBranchOp>(
          [&](handshake::ConditionalBranchOp selNode) {
            auto node = std::make_shared<CBrNode>(
                op, pres, sucs, nodeSucsDataWidthMap, nodeLatency, bbIndex);
            return node;
          })
      // handshake::SourceOp operator
      .Case<handshake::SourceOp>([&](auto selNode) {
        auto node = std::make_shared<SourceNode>(
            op, pres, sucs, nodeSucsDataWidthMap, nodeLatency, bbIndex);
        return node;
      })
      // handshake::ConstantOp operator
      .Case<handshake::ConstantOp>([&](auto selNode) {
        auto node = std::make_shared<ConstantNode>(
            op, pres, sucs, nodeSucsDataWidthMap, nodeLatency, bbIndex);
        return node;
      })
      // handshake::LoadOp operator
      .Case<handshake::LoadOp>([&](handshake::LoadOp selNode) {
        auto memOp = findMemInterface(selNode.getAddressResult());
        if (isa_and_present<handshake::LSQOp>(memOp)) {
          // TODO: Need to change the latency obtaining method for lsq load op
          auto node = std::make_shared<DLoadNode>(
              op, pres, sucs, nodeSucsDataWidthMap, nodeLatency, bbIndex);
          return node;
        } else {
          auto node = std::make_shared<DLoadNode>(
              op, pres, sucs, nodeSucsDataWidthMap, nodeLatency, bbIndex);
          return node;
        }
      })
      // handshake::StoreOp operator
      .Case<handshake::StoreOp>([&](handshake::StoreOp selNode) {
        auto node = std::make_shared<DStoreNode>(
            op, pres, sucs, nodeSucsDataWidthMap, nodeLatency, bbIndex);
        return node;
      })
      // handshake::ShLIOp operator
      .Case<handshake::ShLIOp>([&](auto selNode) {
        auto node = std::make_shared<ShliNode>(
            op, pres, sucs, nodeSucsDataWidthMap, nodeLatency, bbIndex);
        return node;
      })
      // handshake::ShRSIOp operator
      .Case<handshake::ShRSIOp>([&](auto selNode) {
        auto node = std::make_shared<ShrsiNode>(
            op, pres, sucs, nodeSucsDataWidthMap, nodeLatency, bbIndex);
        return node;
      })
      // handshake::ShRUIOp operator
      .Case<handshake::ShRUIOp>([&](auto selNode) {
        auto node = std::make_shared<ShruiNode>(
            op, pres, sucs, nodeSucsDataWidthMap, nodeLatency, bbIndex);
        return node;
      })
      // handshake::SinkOp operator
      .Case<handshake::SinkOp>([&](auto selNode) {
        auto node = std::make_shared<SinkNode>(
            op, pres, sucs, nodeSucsDataWidthMap, nodeLatency, bbIndex);
        return node;
      })
      // handshake::EndOp operator
      .Case<handshake::EndOp>([&](auto selNode) {
        auto node = std::make_shared<EndNode>(
            op, pres, sucs, nodeSucsDataWidthMap, nodeLatency, bbIndex);
        return node;
      })
      // handshake::BranchOp operator
      // In case the pass is called before the canonicalization pass
      .Case<handshake::BranchOp>([&](auto selNode) {
        // Canonicalized branch is single-input pass-through on handshake
        // channels. We model it as PassNode to reuse ready/valid equations.
        auto node = std::make_shared<PassNode>(
            op, pres, sucs, nodeSucsDataWidthMap, nodeLatency, bbIndex);
        return node;
      })
      // handshake::MergeOp operator
      .Case<handshake::MergeOp>([&](auto selNode) {
        // Non-control merge is also propagated as a pass-through node in the
        // current handshake model.
        auto node = std::make_shared<PassNode>(
            op, pres, sucs, nodeSucsDataWidthMap, nodeLatency, bbIndex);
        return node;
      })
      // Default case: unknown operation
      .Default([](auto selNode) {
        selNode->emitOpError() << "Unknown operation!";

        return std::shared_ptr<AdjNode>(nullptr);
      });
}

unsigned AdjGraph::calPathLatency(const Path &selPath, bool useGlobalOrder) {
  unsigned latencySum = 0;

  for (const auto &selNode : selPath.nodeList) {
    latencySum += nodes[selNode]->nodeLatency;

    if (useGlobalOrder) {
      if ((!containsValue(segStartNodes, selNode)) &&
          latencySum < graphGlobalOrder[selNode].second) {
        latencySum = graphGlobalOrder[selNode].second;
      }
    }
  }

  // Check backedges
  if (selPath.contain_backedge) {
    latencySum -= (selPath.backedges.size() * cfdfcII);
  }

  return latencySum;
}

std::vector<Path> AdjGraph::findPaths(const std::string &srcNode,
                                      const std::string &dstNode,
                                      bool noStartingNode,
                                      bool useGlobalOrder) {
  // Build excluding list
  // TODO: replace noStartingNode with an actual excluding list
  std::set<std::string> excludingSet;
  if (noStartingNode) {
    for (const auto &sn : segStartNodes) {
      if (sn != dstNode)
        excludingSet.insert(sn);
    }
  }

  // Define storing structure
  std::vector<std::string> mainStack;
  std::vector<std::vector<std::string>> adjStack;
  std::vector<Path> foundPaths;

  // Initialization
  mainStack.push_back(srcNode);
  adjStack.push_back(nodes[srcNode]->sucs);

  //
  while (!mainStack.empty()) {
    // avoid out of bounds access eg if adjstack empty
    if (adjStack.empty()) {
      llvm::errs() << "[ERROR] AdjStack is empty while MainStack is not!\n";
      exit(1);
    }
    std::vector<std::string> curAdjList = adjStack.back();
    adjStack.pop_back();

    if (!curAdjList.empty()) {
      std::string curNode = curAdjList.back();
      curAdjList.pop_back();

      mainStack.push_back(curNode);
      adjStack.push_back(curAdjList);

      if (!contains(nodes, curNode)) {
        llvm::errs() << "[ERROR] Node " << curNode << " not in nodes map!\n";
        mainStack.pop_back();
        exit(1);
      }
      // Insert new adj_list
      std::vector<std::string> tmpAdjList;
      std::vector<std::string> newAdjList = nodes[curNode]->sucs;

      // If node not in the mainStack and the excluding list
      for (const auto &n : newAdjList) {
        bool inStack = containsValue(mainStack, n);
        bool isExcluded = contains(excludingSet, n);
        if (!inStack && !isExcluded) { // if not in stack and not excluded
          tmpAdjList.push_back(n);
        }
      }

      adjStack.push_back(tmpAdjList);
    } else {
      mainStack.pop_back();
    }

    //! Testing
    // printMainStack(mainStack);
    // printAdjStack(adjStack);

    // Found a path
    if (!mainStack.empty() && mainStack.back() == dstNode) {
      std::vector<std::string> pathList = mainStack;

      // Build edge list
      std::vector<std::pair<std::string, std::string>> edgeList;
      edgeList.reserve(pathList.size() > 1 ? pathList.size() - 1 : 0);
      for (size_t i = 0; i < pathList.size() - 1; ++i) {
        edgeList.push_back({pathList[i], pathList[i + 1]});
      }

      Path selPath(pathList);

      // Retrieve all the backedges in the path
      for (auto &edge : edgeList) {
        if (containsValue(backedges, edge))
          selPath.add_backedge(edge);
      }

      // store the path
      foundPaths.push_back(selPath);
      unsigned tmpPathLatency = calPathLatency(selPath, useGlobalOrder);
      foundPaths.back().set_latency(tmpPathLatency);

      //
      mainStack.pop_back();
      if (!adjStack.empty())
        adjStack.pop_back();
    }
  }

  return foundPaths;
}

std::string
AdjGraph::graphBacktrack(std::string srcNode,
                         std::unordered_set<std::string> &baseNodeSet) {
  if (srcNode.empty())
    return "";

  // If srcNode is in baseNodes => return it
  if (contains(baseNodeSet, srcNode)) {
    return srcNode;
  }

  // Non-graph names (e.g., function arguments) may still be valid data-base
  // anchors; if they are not tracked in this graph we fall back to the raw
  // name.
  if (!contains(nodes, srcNode) || !nodes[srcNode]) {
    llvm::errs() << "[WARNING] graphBacktrack source node " << srcNode
                 << " is not tracked in graph nodes; fallback to raw name\n";
    return srcNode;
  }

  // Define storing structure
  std::vector<std::string> mainStack;
  std::vector<std::vector<std::string>> adjStack;

  // Initialization
  mainStack.push_back(srcNode);
  if (contains(srcNode, "cond_br")) {
    if (auto *cbrNode = dyn_cast<CBrNode>(nodes[srcNode].get())) {
      std::string tmpDataPreNode = cbrNode->dataPreNodeName;
      if (tmpDataPreNode.empty())
        tmpDataPreNode = cbrNode->condPreNodeName;
      adjStack.push_back({tmpDataPreNode});
    }
  } else {
    adjStack.push_back(nodes[srcNode]->pres);
  }

  //
  while (!mainStack.empty()) {
    std::vector<std::string> curAdjList = adjStack.back();
    adjStack.pop_back();

    if (!curAdjList.empty()) {
      //
      std::string curNode = curAdjList.back();
      curAdjList.pop_back();
      adjStack.push_back(curAdjList);

      if (contains(baseNodeSet, curNode)) {
        return curNode;
      } else {
        if (!contains(nodes, curNode) || !nodes[curNode]) {
          mainStack.push_back(curNode);
          continue;
        }
        mainStack.push_back(curNode);

        if (contains(curNode, "cond_br")) {
          if (auto *cbrNode = dyn_cast<CBrNode>(nodes[curNode].get())) {
            std::string tmpDataPreNode = cbrNode->dataPreNodeName;
            if (tmpDataPreNode.empty())
              tmpDataPreNode = cbrNode->condPreNodeName;
            if (std::find(mainStack.begin(), mainStack.end(), tmpDataPreNode) ==
                mainStack.end()) {
              adjStack.push_back({tmpDataPreNode});
            }
          }
        } else {
          std::vector<std::string> tmpAdjList;
          std::vector<std::string> newAdjList = nodes[curNode]->pres;

          for (const auto &n : newAdjList) {
            bool inStack{containsValue(mainStack, n)};
            if (!inStack)
              tmpAdjList.push_back(n);
          }

          adjStack.push_back(tmpAdjList);
        }
      }
    } else {
      mainStack.pop_back();
    }
  }

  llvm::errs() << "[WARNING] Could not find base node for " << srcNode
               << ", trying direct predecessor fallback\n";

  // Try one-hop predecessor fallback before returning the original source.
  for (const auto &pre : nodes[srcNode]->pres) {
    if (pre.empty())
      continue;
    if (pre == srcNode)
      continue;
    if (contains(baseNodeSet, pre))
      return pre;
    if (contains(nodes, pre) && nodes[pre]) {
      std::string src = graphBacktrack(pre, baseNodeSet);
      if (!src.empty())
        return src;
    }
  }

  return srcNode;
}

// std::vector<std::string> mgBacktrackBuffer(std::string srcNode) {

// }

void AdjGraph::computeStartNodeShifts() {
  // Find the largest value in the global order map
  unsigned tmpMaxValue = 0;
  for (const auto &[nodeName, delayPair] : graphGlobalOrder) {
    if (delayPair.second > tmpMaxValue) {
      tmpMaxValue = delayPair.second;
      baseNode = delayPair.first;
    }
  }

  // Update the cycle time of the MG
  for (const auto &selBackedge : backedges) {
    auto [tmpPaths, _pathname_unused] =
        getMaxLatency(selBackedge.second, selBackedge.first, false, true);
    cycleTimeMap[selBackedge.second] = tmpPaths;
  }

  // Analyze the shifting between different start ndoes and the base node
  for (const auto &selStart : segStartNodes) {
    auto nodeVal = cycleTimeMap[selStart];
    auto baseVal = cycleTimeMap[baseNode];

    if (selStart == baseNode) {
      startBaseNodeShiftMap[selStart] = baseVal % cfdfcII;
      continue;
    }

    if ((nodeVal % cfdfcII) == (baseVal % cfdfcII)) {
      startBaseNodeShiftMap[selStart] = 0;
    } else {
      if (nodeVal > baseVal) {
        int tmpDiff = nodeVal - baseVal;
        startBaseNodeShiftMap[selStart] = tmpDiff % cfdfcII;
      } else {
        int tmpDiff = baseVal - nodeVal;
        startBaseNodeShiftMap[selStart] = -1 * (tmpDiff % cfdfcII);
      }
    }
  }
}

void AdjGraph::obtainNodeGlobalOrder() {
  // Iterate over all nodes in the AdjGraph
  for (const auto &name : orderedNodeName) {
    if (containsValue(segStartNodes, name)) {
      continue;
    } else {
      unsigned maxLatency = 0;
      std::string finalStartNode = "";

      for (const auto &selStartNode : segStartNodes) {
        //! Testing
        // switchingDebugStream(SwitchingDebugCategory::Graph) << "[DEBUG] \t\tNode: " << selStartNode << "\n";

        auto [tmpPathLat, _skip] =
            getMaxLatency(selStartNode, name, true, false);
        if (tmpPathLat >= maxLatency) {
          maxLatency = tmpPathLat;
          finalStartNode = selStartNode;
        }
      }
      // Store the global order
      graphGlobalOrder[name] = std::make_pair(finalStartNode, maxLatency);

      //! Testing
      // switchingDebugStream(SwitchingDebugCategory::Graph) << "[DEBUG] \tNode: " << name << "; Global Order: ("
      //              << finalStartNode << ", " << maxLatency << ");\n";
    }
  }
}

void AdjGraph::buildSrcMaps() {
  // Traverse all nodes in the dataflow graph
  for (const auto &selNode : orderedNodeName) {
    switch (nodes[selNode]->getKind()) {
    // If this is a mux node
    case AdjNode::NodeKind::MuxNodeKind: {
      auto *selMuxNode = dyn_cast<MuxNode>(nodes[selNode].get());
      std::string ctrlPreNodeName = selMuxNode->conPreNodeName;
      std::string dataPre0NodeName = "";
      std::string dataPre1NodeName = "";
      for (const auto &[nodeName, portIdx] : selMuxNode->preNameToPortIdxMap) {
        if (portIdx == 1)
          dataPre0NodeName = nodeName;
        else if (portIdx == 2)
          dataPre1NodeName = nodeName;
      }

      auto fail = [&](std::string const &what) {
        if (what.empty())
          llvm::errs() << "[ERROR] graphBacktrack returned empty for "
                       << selNode << '\n';
      };

      auto resolveSrc = [&](const std::string &preNode) -> std::string {
        if (preNode.empty())
          return "";
        std::string src = graphBacktrack(preNode, allDataBaseNode);
        return src.empty() ? preNode : src;
      };

      // Get the source node for all three ports
      // TODO: Need to check the following logic for backnode tracking
      std::string ctrlSrc = resolveSrc(ctrlPreNodeName);
      std::string dataSrc0 = resolveSrc(
          dataPre0NodeName.empty() ? ctrlPreNodeName : dataPre0NodeName);

      std::string dataSrc1 = resolveSrc(
          dataPre1NodeName.empty() ? dataPre0NodeName : dataPre1NodeName);
      fail(ctrlSrc);
      fail(dataSrc0);
      fail(dataSrc1);

      if (ctrlSrc.empty()) {
        llvm::errs() << "[WARNING] Mux node " << selNode
                     << " has empty control source; fallback to data source 0 ("
                     << dataSrc0 << ")\n";
        ctrlSrc = dataSrc0;
      }
      if (dataSrc0.empty()) {
        llvm::errs() << "[WARNING] Mux node " << selNode
                     << " has empty data source 0; fallback to control source ("
                     << ctrlSrc << ")\n";
        dataSrc0 = ctrlSrc;
      }
      if (dataSrc1.empty()) {
        llvm::errs() << "[WARNING] Mux node " << selNode
                     << " has empty data source 1; fallback to data source 0 ("
                     << dataSrc0 << ")\n";
        dataSrc1 = dataSrc0;
      }
      if (ctrlSrc.empty() || dataSrc0.empty() || dataSrc1.empty()) {
        llvm::errs() << "[WARNING] Mux node " << selNode
                     << " still has unresolved sources after fallback: control="
                     << ctrlSrc << ", data0=" << dataSrc0
                     << ", data1=" << dataSrc1 << "\n";
      }

      // Update the control_merge to mux map
      if (!ctrlSrc.empty() && contains(cmToMuxMap, ctrlSrc)) {
        cmToMuxMap[ctrlSrc].push_back(selNode);
      } else if (!ctrlSrc.empty()) {
        cmToMuxMap[ctrlSrc] = {selNode};
      }

      // Build mux src map
      std::map<std::string, std::string> tmpMuxPortMap{// structured bingings
                                                       {"control", ctrlSrc},
                                                       {"0", dataSrc0},
                                                       {"1", dataSrc1}};
      // Update the global map
      muxToSrcNodeMap[selNode] = tmpMuxPortMap;

      // Update the src to mux map
      if (!dataSrc0.empty() && contains(srcNodeToMuxMap, dataSrc0)) {
        srcNodeToMuxMap[dataSrc0].push_back(std::make_pair(selNode, 0));
      } else if (!dataSrc0.empty()) {
        srcNodeToMuxMap[dataSrc0] = {std::make_pair(selNode, 0)};
      }

      if (!dataSrc1.empty() && contains(srcNodeToMuxMap, dataSrc1)) {
        srcNodeToMuxMap[dataSrc1].push_back(std::make_pair(selNode, 1));
      } else if (!dataSrc1.empty()) {
        srcNodeToMuxMap[dataSrc1] = {std::make_pair(selNode, 1)};
      }

      break;
    }
    case AdjNode::NodeKind::CBrNodeKind: {
      // Ger the cond_br node storing structure
      auto *selCBrNode = dyn_cast<CBrNode>(nodes[selNode].get());

      std::string controlSrcNode =
          graphBacktrack(selCBrNode->condPreNodeName, allDataBaseNode);
      if (controlSrcNode.empty())
        controlSrcNode = selCBrNode->condPreNodeName;
      condBrToConSrcMap[selNode] = controlSrcNode;

      // Updated the connected buffers as well
      for (const auto &selSucNode : selCBrNode->sucs) {
        if (nodes[selSucNode]->getKind() == AdjNode::NodeKind::BufferNodeKind) {
          // Get the port index
          unsigned selPortIdx =
              selCBrNode->outChannelNameToIndexMap[selSucNode];
          if (contains(condBrToBufferMap, selNode)) {
            condBrToBufferMap[selNode].push_back(
                std::make_pair(selSucNode, selPortIdx));
          } else {
            condBrToBufferMap[selNode] = {
                std::make_pair(selSucNode, selPortIdx)};
          }
        }
      }

      break;
    }
    case AdjNode::NodeKind::DStoreNodeKind: {
      auto *selStoreNode = dyn_cast<DStoreNode>(nodes[selNode].get());
      std::string addrPreNode = selStoreNode->addressInNode;
      std::string dataPreNode = selStoreNode->dataInNode;
      selStoreNode->addressInSrcNode =
          graphBacktrack(addrPreNode, allDataBaseNode);
      selStoreNode->dataInSrcNode =
          graphBacktrack(dataPreNode, allDataBaseNode);
      if (selStoreNode->addressInSrcNode.empty())
        selStoreNode->addressInSrcNode = addrPreNode;
      if (selStoreNode->dataInSrcNode.empty())
        selStoreNode->dataInSrcNode = dataPreNode;
      break;
    }
    case AdjNode::NodeKind::BufferNodeKind: {
      auto *selBufferNode = dyn_cast<BufferNode>(nodes[selNode].get());
      if (selBufferNode->transparent) {
        break;
      } else {
        // We only backtrack the opaque buffers
        if (selBufferNode->pres.size() > 1) {
          llvm::errs() << "[WARNING] Buffer node " << selNode
                       << " has multiple predecessors; only backtracking the "
                          "first one\n";
        }
        std::string preNode =
            selBufferNode->pres.empty() ? "" : selBufferNode->pres[0];
        selBufferNode->dataInSrcNode = graphBacktrack(preNode, allDataBaseNode);
      }

      // Update the data source to buffer map
      if (!selBufferNode->dataInSrcNode.empty())
        dataSrcToOpaqueBufferMap[selBufferNode->dataInSrcNode] = selNode;
      //! Testing
      switchingDebugStream(SwitchingDebugCategory::Graph) << "[DEBUG] \t\tBuffer Node: " << selNode
                   << "; DataInSrcNode: " << selBufferNode->dataInSrcNode
                   << "\n";

      break;
    }
    }
  }
}
