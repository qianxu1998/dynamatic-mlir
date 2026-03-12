//===- DataChannelCal.cpp - Estimate Swithicng Activities ------*- C++ -*-===//
//
// This file declares all functions used for data channel switching calculation
//
//===----------------------------------------------------------------------===//

#include "experimental/Analysis/SwitchingEstimation/DataChannelCal.h"
#include "experimental/Analysis/SwitchingEstimation/Debug.h"
#include "experimental/Analysis/SwitchingEstimation/DFSKernel.h"
#include "experimental/Analysis/SwitchingEstimation/SwitchingSupport.h"
#include "experimental/Analysis/SwitchingEstimation/utils.h"

#include "dynamatic/Dialect/Handshake/HandshakeAttributes.h"
#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Support/Attribute.h"
#include "dynamatic/Support/Backedge.h"
#include "dynamatic/Support/CFG.h"
#include "dynamatic/Support/DynamaticPass.h"
#include "dynamatic/Support/LLVM.h"
#include "dynamatic/Support/Logging.h"
#include "dynamatic/Support/TimingModels.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Path.h"

#include <algorithm>
#include <cassert>
#include <memory>
#include <set>

using namespace dynamatic::experimental;

namespace {
// Keep meaningful toggles while removing adjacent duplicates
// (e.g. [0,0,1,1,0] -> [0,1,0]).
void dedupConsecutiveValues(std::vector<int> &values) {
  values.erase(std::unique(values.begin(), values.end()), values.end());
}

// Render integer vectors as "[a, b, c]" for readable debug traces.
std::string formatIntVector(const std::vector<int> &values) {
  std::string rendered;
  llvm::raw_string_ostream os(rendered);
  os << "[";
  for (size_t i = 0; i < values.size(); ++i) {
    if (i != 0)
      os << ", ";
    os << values[i];
  }
  os << "]";
  os.flush();
  return rendered;
}

// Safely read a node's profiled output value at `iter`, falling back to the
// latest known value not newer than `iter` (or `fallback` if unavailable).
int getDataOutValueAtOrBefore(const std::shared_ptr<DataBase> &nodeData,
                              unsigned iter, int fallback = 0) {
  if (!nodeData) {
    switchingDebugStream(SwitchingDebugCategory::Data) << "[WARNING] Node data is null, return fallback " << fallback
                 << "\n";
    return fallback;
  }

  const auto &outMap = nodeData->originalDataOut;
  if (outMap.empty())
    return fallback;

  auto exactIt = outMap.find(iter);
  if (exactIt != outMap.end())
    return exactIt->second.value;

  auto upper = outMap.upper_bound(iter);
  if (upper == outMap.begin())
    return upper->second.value;

  --upper;
  return upper->second.value;
}

// For mux token semantics we must distinguish "a source produced this value on
// this iteration" from "this was the latest value seen earlier".
std::optional<int> getDataOutValueIfPresent(const std::shared_ptr<DataBase> &nodeData,
                                            unsigned iter) {
  if (!nodeData)
    return std::nullopt;

  auto it = nodeData->originalDataOut.find(iter);
  if (it == nodeData->originalDataOut.end())
    return std::nullopt;
  return it->second.value;
}

// Read the estimator's current driven value for `iter`. If a glitch waveform
// has already been materialized for this iteration, it represents the bus
// value seen by downstream nodes and must win over the token payload stored in
// `originalDataOut`.
std::optional<int>
getCurrentDataOutValueIfPresent(const std::shared_ptr<DataBase> &nodeData,
                                unsigned iter) {
  if (!nodeData)
    return std::nullopt;

  auto glitchIt = nodeData->glitchDataOutByIter.find(iter);
  if (glitchIt != nodeData->glitchDataOutByIter.end() &&
      !glitchIt->second.empty())
    return glitchIt->second.back();

  return getDataOutValueIfPresent(nodeData, iter);
}

// Read the latest value the estimator believes is being driven at or before
// `iter`. Unlike getCurrentDataOutValueAtOrBefore(), this never pulls in a
// future value when the source has not produced anything yet.
std::optional<int>
getCurrentDataOutValueAtOrBeforeStrict(
    const std::shared_ptr<DataBase> &nodeData, unsigned iter) {
  if (!nodeData)
    return std::nullopt;

  std::optional<std::pair<unsigned, int>> bestValue;

  auto originalUpper = nodeData->originalDataOut.upper_bound(iter);
  if (originalUpper != nodeData->originalDataOut.begin()) {
    --originalUpper;
    bestValue = std::make_pair(originalUpper->first,
                               originalUpper->second.value);
  }

  auto glitchUpper = nodeData->glitchDataOutByIter.upper_bound(iter);
  while (glitchUpper != nodeData->glitchDataOutByIter.begin()) {
    --glitchUpper;
    if (glitchUpper->second.empty())
      continue;

    if (!bestValue.has_value() || glitchUpper->first >= bestValue->first)
      bestValue =
          std::make_pair(glitchUpper->first, glitchUpper->second.back());
    break;
  }

  if (!bestValue.has_value())
    return std::nullopt;
  return bestValue->second;
}

int getCurrentDataOutValueAtOrBefore(const std::shared_ptr<DataBase> &nodeData,
                                     unsigned iter, int fallback = 0) {
  if (!nodeData)
    return fallback;

  if (std::optional<int> exact = getCurrentDataOutValueIfPresent(nodeData, iter))
    return *exact;

  const auto &outMap = nodeData->originalDataOut;
  if (outMap.empty())
    return fallback;

  auto upper = outMap.upper_bound(iter);
  if (upper == outMap.begin())
    return upper->second.value;

  --upper;
  return upper->second.value;
}

unsigned getSegmentII(const SwitchingInfo &switchInfo,
                      const std::string &segLabel) {
  if (segLabel.empty())
    return 1;
  if (!std::all_of(segLabel.begin(), segLabel.end(),
                   [](unsigned char c) { return std::isdigit(c); }))
    return 1;

  unsigned mgIndex = static_cast<unsigned>(std::stoul(segLabel));
  auto it = switchInfo.staticInfo.cfdfcIIs.find(mgIndex);
  if (it == switchInfo.staticInfo.cfdfcIIs.end())
    return 1;

  unsigned ii = static_cast<unsigned>(std::llround(it->second));
  return std::max(ii, 1U);
}

// In data-channel glitch propagation, only opaque buffers block glitches.
// Transparent buffers are treated as pass-through for glitch events.
bool isOpaqueBuffer(const SwitchingInfo &switchInfo, const std::string &node) {
  if (!contains(node, "buffer"))
    return false;

  auto nodeIt = switchInfo.staticInfo.dataflowGraph->nodes.find(node);
  if (nodeIt == switchInfo.staticInfo.dataflowGraph->nodes.end() ||
      !nodeIt->second)
    return false;

  if (auto *bufferNode = dyn_cast<BufferNode>(nodeIt->second.get()))
    return !bufferNode->transparent;

  // Keep legacy conservative behavior if we cannot classify this node.
  return false;
}

std::optional<std::string>
findNearestMuxReplaySource(const SwitchingInfo &switchInfo,
                           std::string current) {
  std::set<std::string> visited;

  while (!current.empty() && visited.insert(current).second) {
    auto stateIt = switchInfo.dataInfo.nodeToDataState.find(current);
    if (stateIt != switchInfo.dataInfo.nodeToDataState.end() && stateIt->second)
      return current;

    auto graphIt = switchInfo.staticInfo.dataflowGraph->nodes.find(current);
    if (graphIt == switchInfo.staticInfo.dataflowGraph->nodes.end() ||
        !graphIt->second)
      break;

    if (auto *bufferNode = dyn_cast<BufferNode>(graphIt->second.get())) {
      current = bufferNode->pres.empty() ? "" : bufferNode->pres.front();
      continue;
    }

    if (auto *cbrNode = dyn_cast<CBrNode>(graphIt->second.get())) {
      current = cbrNode->dataPreNodeName.empty() ? cbrNode->condPreNodeName
                                                 : cbrNode->dataPreNodeName;
      continue;
    }

    if (graphIt->second->pres.size() == 1) {
      current = graphIt->second->pres.front();
      continue;
    }

    break;
  }

  return std::nullopt;
}

std::string getMuxReplayDataSrcNode(const SwitchingInfo &switchInfo,
                                    const std::string &selMuxNode,
                                    int resolvedCond) {
  auto immediateMuxIt =
      switchInfo.staticInfo.dataflowGraph->muxToImmediateInputMap.find(
          selMuxNode);
  if (immediateMuxIt !=
      switchInfo.staticInfo.dataflowGraph->muxToImmediateInputMap.end()) {
    auto preIt = immediateMuxIt->second.find(std::to_string(resolvedCond));
    if (preIt != immediateMuxIt->second.end() && !preIt->second.empty()) {
      if (std::optional<std::string> replaySrc =
              findNearestMuxReplaySource(switchInfo, preIt->second))
        return *replaySrc;
      return preIt->second;
    }
  }

  auto muxIt =
      switchInfo.staticInfo.dataflowGraph->muxToSrcNodeMap.find(selMuxNode);
  if (muxIt == switchInfo.staticInfo.dataflowGraph->muxToSrcNodeMap.end())
    return "";

  auto srcIt = muxIt->second.find(std::to_string(resolvedCond));
  if (srcIt == muxIt->second.end())
    return "";

  return srcIt->second;
}

std::string getMuxImmediateInputNode(const SwitchingInfo &switchInfo,
                                     const std::string &selMuxNode,
                                     int resolvedCond) {
  auto immediateMuxIt =
      switchInfo.staticInfo.dataflowGraph->muxToImmediateInputMap.find(
          selMuxNode);
  if (immediateMuxIt ==
      switchInfo.staticInfo.dataflowGraph->muxToImmediateInputMap.end())
    return "";

  auto preIt = immediateMuxIt->second.find(std::to_string(resolvedCond));
  if (preIt == immediateMuxIt->second.end())
    return "";
  return preIt->second;
}

static bool bufferAllowsHeldMuxReplay(const BufferNode &bufferNode) {
  switch (bufferNode.buffType) {
  case BufferType::ONE_SLOT_BREAK_R:
    return false;
  case BufferType::ONE_SLOT_BREAK_DV:
  case BufferType::ONE_SLOT_BREAK_DVR:
  case BufferType::FIFO_BREAK_DV:
  case BufferType::FIFO_BREAK_NONE:
  case BufferType::SHIFT_REG_BREAK_DV:
    return true;
  }
  return false;
}

static bool immediateInputPathHasCurrentEventImpl(
    const SwitchingInfo &switchInfo, const std::string &nodeName, unsigned iter,
    std::set<std::string> &visited) {
  if (nodeName.empty() || !visited.insert(nodeName).second)
    return false;

  auto dataIt = switchInfo.dataInfo.nodeToDataState.find(nodeName);
  if (dataIt != switchInfo.dataInfo.nodeToDataState.end() && dataIt->second) {
    if (contains(dataIt->second->originalDataOut, iter))
      return true;
    auto glitchIt = dataIt->second->glitchDataOutByIter.find(iter);
    if (glitchIt != dataIt->second->glitchDataOutByIter.end() &&
        !glitchIt->second.empty())
      return true;
  }

  auto nodeIt = switchInfo.staticInfo.dataflowGraph->nodes.find(nodeName);
  if (nodeIt == switchInfo.staticInfo.dataflowGraph->nodes.end() ||
      !nodeIt->second)
    return false;

  if (auto *cbrNode = dyn_cast<CBrNode>(nodeIt->second.get()))
    return immediateInputPathHasCurrentEventImpl(
        switchInfo, cbrNode->dataPreNodeName, iter, visited);

  if (auto *bufferNode = dyn_cast<BufferNode>(nodeIt->second.get())) {
    if (bufferNode->transparent && !bufferNode->pres.empty())
      return immediateInputPathHasCurrentEventImpl(
          switchInfo, bufferNode->pres.front(), iter, visited);
    return false;
  }

  if (nodeIt->second->pres.size() == 1)
    return immediateInputPathHasCurrentEventImpl(
        switchInfo, nodeIt->second->pres.front(), iter, visited);

  return false;
}

static bool immediateInputPathAllowsHeldTokenImpl(
    const SwitchingInfo &switchInfo, const std::string &nodeName,
    std::set<std::string> &visited) {
  if (nodeName.empty() || !visited.insert(nodeName).second)
    return false;

  if (contains(nodeName, "constant") || contains(nodeName, "source") ||
      contains(nodeName, "start"))
    return true;

  auto nodeIt = switchInfo.staticInfo.dataflowGraph->nodes.find(nodeName);
  if (nodeIt == switchInfo.staticInfo.dataflowGraph->nodes.end() ||
      !nodeIt->second)
    return false;

  if (auto *cbrNode = dyn_cast<CBrNode>(nodeIt->second.get()))
    return immediateInputPathAllowsHeldTokenImpl(
        switchInfo, cbrNode->dataPreNodeName, visited);

  if (auto *bufferNode = dyn_cast<BufferNode>(nodeIt->second.get()))
    return bufferAllowsHeldMuxReplay(*bufferNode);

  if (nodeIt->second->pres.size() == 1)
    return immediateInputPathAllowsHeldTokenImpl(
        switchInfo, nodeIt->second->pres.front(), visited);

  return false;
}

bool muxImmediateInputHasCurrentEvent(const SwitchingInfo &switchInfo,
                                      const std::string &selMuxNode,
                                      int resolvedCond, unsigned iter) {
  std::set<std::string> visited;
  return immediateInputPathHasCurrentEventImpl(
      switchInfo, getMuxImmediateInputNode(switchInfo, selMuxNode, resolvedCond),
      iter, visited);
}

bool muxImmediateInputAllowsHeldToken(const SwitchingInfo &switchInfo,
                                      const std::string &selMuxNode,
                                      int resolvedCond) {
  std::set<std::string> visited;
  return immediateInputPathAllowsHeldTokenImpl(
      switchInfo, getMuxImmediateInputNode(switchInfo, selMuxNode, resolvedCond),
      visited);
}

bool muxSourceAllowsHeldFallback(const SwitchingInfo &switchInfo,
                                 const std::string &selMuxNode,
                                 int resolvedCond,
                                 const std::string &selDataSrcNode) {
  if (selDataSrcNode.empty())
    return false;

  return muxImmediateInputAllowsHeldToken(switchInfo, selMuxNode,
                                          resolvedCond);
}

bool muxSelectedInputIsValid(SwitchingInfo &switchInfo,
                             const std::string &selMuxNode,
                             int resolvedCond, unsigned iter) {
  auto immediateMuxIt =
      switchInfo.staticInfo.dataflowGraph->muxToImmediateInputMap.find(
          selMuxNode);
  if (immediateMuxIt ==
      switchInfo.staticInfo.dataflowGraph->muxToImmediateInputMap.end())
    return true;

  const std::string portKey = std::to_string(resolvedCond);
  auto preIt = immediateMuxIt->second.find(portKey);
  if (preIt == immediateMuxIt->second.end())
    return true;

  const std::string &immediatePreNode = preIt->second;
  if (immediatePreNode.empty() || !contains(immediatePreNode, "cond_br"))
    return true;

  auto condBrIt =
      switchInfo.staticInfo.dataflowGraph->nodes.find(immediatePreNode);
  if (condBrIt == switchInfo.staticInfo.dataflowGraph->nodes.end() ||
      !condBrIt->second)
    return true;

  auto *condBrNode = dyn_cast<CBrNode>(condBrIt->second.get());
  if (!condBrNode)
    return true;

  auto sucIt = condBrNode->outChannelNameToIndexMap.find(selMuxNode);
  if (sucIt == condBrNode->outChannelNameToIndexMap.end())
    return true;

  if (!contains(switchInfo.staticInfo.dataflowGraph->condBrToConSrcMap,
                immediatePreNode))
    return true;

  const std::string &condSrcNode =
      switchInfo.staticInfo.dataflowGraph->condBrToConSrcMap[immediatePreNode];
  auto condStateIt = switchInfo.dataInfo.nodeToDataState.find(condSrcNode);
  if (condStateIt == switchInfo.dataInfo.nodeToDataState.end() ||
      !condStateIt->second)
    return false;

  std::optional<int> condValue =
      getCurrentDataOutValueIfPresent(condStateIt->second, iter);
  if (!condValue.has_value())
    return false;

  const unsigned selectedBranch = sucIt->second;
  const unsigned activeBranch = (*condValue != 0) ? 0U : 1U;
  return selectedBranch == activeBranch;
}
} // namespace

void mapBBPairToControlMerge(SwitchingInfo &switchInfo) {
  // *Some control merge nodes with only 1 input is also included in the map
  for (const auto &nodeName :
       switchInfo.staticInfo.dataflowGraph->orderedNodeName) {
    auto *op = switchInfo.staticInfo.dataflowGraph->nodes[nodeName]->op;
    if (!op) { // first check if the pointer is null pointer or not (some op's
               // irrelevat )
      llvm::errs() << "[WARNING] The op pointer for node " << nodeName
                   << " is null, skip it.\n";
      continue;
    }

    if (auto CMOp = dyn_cast<handshake::ControlMergeOp>(op)) {
      // Get the BB of the control_merge node
      unsigned CMBB;
      if (std::optional<unsigned> optBB = getLogicBB(CMOp); !optBB.has_value())
        continue;
      else
        CMBB = *optBB;

      // Get the BB of the two inputs of the control_merge node
      // The input of a control_merge node will not be from a block argument,
      // so we don't check it
      auto opOperands = CMOp.getOperands();
      for (size_t i = 0; i < opOperands.size(); i++) {
        auto inputSrcOp = opOperands[i].getDefiningOp();
        unsigned preBB;
        if (std::optional<unsigned> optBB = getLogicBB(inputSrcOp);
            !optBB.has_value())
          continue;
        else
          preBB = *optBB;

        // Update the storing dict
        if (contains(switchInfo.dataInfo.bbPairToControlMergeOutputs,
                     std::make_pair(preBB, CMBB))) {
          switchInfo.dataInfo
              .bbPairToControlMergeOutputs[std::make_pair(preBB, CMBB)]
              .push_back(std::make_pair(nodeName, i));
        } else {
          switchInfo.dataInfo
              .bbPairToControlMergeOutputs[std::make_pair(preBB, CMBB)] = {
              std::make_pair(nodeName, i)};
        }
      }
    }
  }
}

void getDataBaseNodes(SwitchingInfo &switchInfo,
                      SCFProfilingResult &profileResults) {
  // Instantiate all SegmentDataSourceNodes for all segments
  for (const auto &[segLabel, BBVec] : switchInfo.staticInfo.segToBBs) {
    SegmentDataSourceNodes tmpDataBaseNodes;
    switchInfo.dataInfo.segmentToDataSourceNodes[segLabel] = tmpDataBaseNodes;

    // Update the ordered mapped list
    switchInfo.dataInfo.segmentToOrderedDataSourceNodes[segLabel] = {};
    switchInfo.dataInfo.segmentToOrderedAluNodes[segLabel] = {};
    SegmentControlNodes tmpList;
    switchInfo.dataInfo.segmentControlNodes[segLabel] = tmpList;
  }

  // Traverse all the nodes in the dataflow graph
  for (const auto &nodeName :
       switchInfo.staticInfo.dataflowGraph->orderedNodeName) {
    unsigned nodeBB;
    auto op = switchInfo.staticInfo.dataflowGraph->nodes[nodeName]->op;
    if (!op) {
      llvm::errs() << "[WARNING] The op pointer for node " << nodeName
                   << " is null, skip it.\n";
      continue;
    }
    if (std::optional<unsigned> optBB = getLogicBB(op); !optBB.has_value())
      continue;
    else
      nodeBB = *optBB;

    auto dataflowGraph = switchInfo.staticInfo.dataflowGraph;
    // Check the types of the nodes
    // [TYPE 1]: DATA nodes from scf profiling
    if ((contains(profileResults.nodeToValueTrace, nodeName)) ||
        (contains(nodeName, "constant")) || (contains(nodeName, "source"))) {
      // Check each segment
      for (const auto &[segLabel, BBVec] : switchInfo.staticInfo.segToBBs) {
        if (std::find(BBVec.begin(), BBVec.end(), nodeBB) != BBVec.end()) {
          // If this is a CFDFC
          if (contains(switchInfo.staticInfo.segToGraph, segLabel)) {
            switchInfo.staticInfo.segToGraph[segLabel]->profileBaseNodes.insert(
                nodeName);
            switchInfo.staticInfo.segToGraph[segLabel]->allDataBaseNode.insert(
                nodeName);
          }

          switchInfo.dataInfo.segmentToDataSourceNodes[segLabel].data.push_back(
              nodeName);
          switchInfo.dataInfo.segmentToDataSourceNodes[segLabel].all.push_back(
              nodeName);
          switchInfo.dataInfo.segmentToOrderedDataSourceNodes[segLabel]
              .push_back(nodeName);

          if ((!contains(nodeName, "constant")) &&
              (!contains(nodeName, "load")) && (!contains(nodeName, "source")))
            switchInfo.dataInfo.segmentToOrderedAluNodes[segLabel].push_back(
                nodeName);
        }
      }
      switchInfo.staticInfo.dataflowGraph->profileBaseNodes.insert(nodeName);
      switchInfo.staticInfo.dataflowGraph->allDataBaseNode.insert(nodeName);
      // [TYPE 2]: Control Nodes
    } else if (contains(nodeName, "control_merge") ||
               contains(nodeName, "mux")) {
      // Check each segment
      for (const auto &[segLabel, BBVec] : switchInfo.staticInfo.segToBBs) {
        if (containsValue(BBVec, nodeBB)) {
          // If this is a CFDFC
          if (contains(switchInfo.staticInfo.segToGraph, segLabel)) {
            switchInfo.staticInfo.segToGraph[segLabel]->allDataBaseNode.insert(
                nodeName);
          }

          switchInfo.dataInfo.segmentToDataSourceNodes[segLabel]
              .control.push_back(nodeName);
          switchInfo.dataInfo.segmentToDataSourceNodes[segLabel].all.push_back(
              nodeName);

          if (contains(nodeName, "control_merge")) {
            switchInfo.dataInfo.segmentControlNodes[segLabel]
                .controlMergeNodes.push_back(nodeName);
          } else {
            switchInfo.dataInfo.segmentControlNodes[segLabel]
                .muxNodes.push_back(nodeName);
          }
        }
      }
      switchInfo.staticInfo.dataflowGraph->allDataBaseNode.insert(nodeName);
    } else if (isOpaqueBuffer(switchInfo, nodeName)) {
      // [TYPE 3]: Buffer Nodes (opaque)
      // Check each segment
      for (const auto &[segLabel, BBVec] : switchInfo.staticInfo.segToBBs) {
        if (std::find(BBVec.begin(), BBVec.end(), nodeBB) != BBVec.end()) {
          // If this is a CFDFC
          if (contains(switchInfo.staticInfo.segToGraph, segLabel)) {
            switchInfo.staticInfo.segToGraph[segLabel]->allDataBaseNode.insert(
                nodeName);
          }

          switchInfo.dataInfo.segmentToDataSourceNodes[segLabel]
              .opaque_buffers.push_back(nodeName);
          switchInfo.dataInfo.segmentToDataSourceNodes[segLabel].all.push_back(
              nodeName);
          switchInfo.dataInfo.segmentToOrderedDataSourceNodes[segLabel]
              .push_back(nodeName);
        }
      }
      switchInfo.staticInfo.dataflowGraph->allDataBaseNode.insert(nodeName);
    }

    // Add all arguments into data base nodes
    for (const auto &selArg : profileResults.argumentNames) {
      switchInfo.staticInfo.dataflowGraph->profileBaseNodes.insert(selArg);
      switchInfo.staticInfo.dataflowGraph->allDataBaseNode.insert(selArg);
    }
  }
}

void dataChannelBaseNodesValueUpdate(SwitchingInfo &switchInfo,
                                     SCFProfilingResult &profileResults) {
  auto getNodeDataBase =
      [&](const std::string &nodeName) -> std::shared_ptr<DataBase> {
    auto it = switchInfo.dataInfo.nodeToDataState.find(nodeName);
    if (it == switchInfo.dataInfo.nodeToDataState.end() || !it->second)
      return nullptr;
    return it->second;
  };
  auto setOriginalDataOut = [&](const std::string &nodeName, unsigned iterIndex,
                                int value, bool overwrite) -> bool {
    auto nodeDb = getNodeDataBase(nodeName);
    if (!nodeDb)
      return false;
    if (!overwrite && contains(nodeDb->originalDataOut, iterIndex))
      return false;
    nodeDb->originalDataOut[iterIndex] = {value, iterIndex};
    nodeDb->originalDataOutVec.push_back(value);
    return true;
  };
  auto updateBufferFromSource = [&](const std::string &srcNode, int value,
                                    unsigned iterIndex,
                                    bool tokenPresent = true) {
    auto writeSingleBuffer = [&](const std::string &bufferNode,
                                 unsigned targetIter, int newValue,
                                 bool bufferTokenPresent) {
      auto nodeIt = switchInfo.staticInfo.dataflowGraph->nodes.find(bufferNode);
      if (nodeIt != switchInfo.staticInfo.dataflowGraph->nodes.end() &&
          nodeIt->second) {
        if (auto *bufferNodeModel =
                dyn_cast<BufferNode>(nodeIt->second.get())) {
          const bool holdsLastValidData =
              bufferNodeModel->buffType == BufferType::ONE_SLOT_BREAK_DV ||
              bufferNodeModel->buffType == BufferType::FIFO_BREAK_DV ||
              bufferNodeModel->buffType == BufferType::SHIFT_REG_BREAK_DV ||
              bufferNodeModel->buffType == BufferType::ONE_SLOT_BREAK_DVR;
          if (!bufferTokenPresent && holdsLastValidData)
            return;
        }
      }
      setOriginalDataOut(bufferNode, targetIter, newValue, /*overwrite=*/false);
    };

    auto getNodeBB = [&](const std::string &nodeName, unsigned &bbIdx) -> bool {
      auto nodeIt = switchInfo.staticInfo.dataflowGraph->nodes.find(nodeName);
      if (nodeIt == switchInfo.staticInfo.dataflowGraph->nodes.end() ||
          !nodeIt->second) {
        return false;
      }
      bbIdx = nodeIt->second->bbindex;
      return true;
    };

    std::set<std::string> visited;
    std::string currentSrcNode = srcNode;
    unsigned targetIter = iterIndex;
    while (true) {
      auto bufferIt =
          switchInfo.staticInfo.dataflowGraph->dataSrcToOpaqueBufferMap.find(
              currentSrcNode);
      if (bufferIt ==
          switchInfo.staticInfo.dataflowGraph->dataSrcToOpaqueBufferMap.end()) {
        break;
      }

      const std::string &bufferNode = bufferIt->second;
      if (visited.find(bufferNode) != visited.end()) {
        break;
      }
      visited.insert(bufferNode);

      writeSingleBuffer(bufferNode, targetIter, value, tokenPresent);

      auto nextIt =
          switchInfo.staticInfo.dataflowGraph->dataSrcToOpaqueBufferMap.find(
              bufferNode);
      if (nextIt ==
          switchInfo.staticInfo.dataflowGraph->dataSrcToOpaqueBufferMap.end()) {
        break;
      }

      unsigned currentBufBB = 0;
      unsigned nextBufBB = 0;
      const bool hasCurBB = getNodeBB(bufferNode, currentBufBB);
      const bool hasNextBB = getNodeBB(nextIt->second, nextBufBB);
      if (hasCurBB && hasNextBB && currentBufBB != nextBufBB)
        targetIter += 1;

      currentSrcNode = bufferNode;
    }
  };

  // [STEP 1] Iterate over all CMerge, MUX nodes and Opaque buffer nodes
  for (const auto &selNode :
       switchInfo.staticInfo.dataflowGraph->allDataBaseNode) {
    // Check the type of the nodes
    if (contains(selNode, "control_merge")) {
      switchInfo.dataInfo.nodeToDataState[selNode] =
          std::make_shared<CMergeData>(selNode);
    } else if (contains(selNode, "mux")) {
      switchInfo.dataInfo.nodeToDataState[selNode] =
          std::make_shared<DataBase>(selNode);
    } else if (contains(selNode, "buffer")) {
      switchInfo.dataInfo.nodeToDataState[selNode] =
          std::make_shared<DataBase>(selNode);
    }
  }

  // [STEP 2] Iterate all profile base node in the dataflow graph
  for (const auto &selNode :
       switchInfo.staticInfo.dataflowGraph->profileBaseNodes) {
    // Define tmp storing structure
    switchInfo.dataInfo.nodeToDataState[selNode] =
        std::make_shared<DataBase>(selNode);

    // Check the type of the node
    if (contains(selNode, "cmp")) {
      for (const auto &[value, iterIdx] :
           profileResults.nodeToValueTrace[selNode]) {
        const int cmpValue = std::abs(value);
        setOriginalDataOut(selNode, iterIdx, cmpValue, /*overwrite=*/false);
        updateBufferFromSource(selNode, cmpValue, iterIdx);
      }

      //! Testing
      // switchInfo.dataInfo.nodeToDataState[selNode]->printDetail();
    } else if (contains(selNode, "constant")) {
      // We need to get the constant value from the attribute
      // Get the mlir op
      auto constantOp = dyn_cast<handshake::ConstantOp>(
          switchInfo.staticInfo.dataflowGraph->nodes[selNode]->op);
      if (!constantOp) {
        llvm::errs() << "[ERROR] The op pointer for node " << selNode << "\n";
        continue;
      }
      auto valueAttr = constantOp->getAttrOfType<IntegerAttr>("value");
      if (!valueAttr) {
        llvm::errs() << "[ERROR] The value attribute for node " << selNode
                     << " is null\n";
        continue;
      }
      int constantValue = valueAttr.getInt();

      setOriginalDataOut(selNode, 0, constantValue, /*overwrite=*/false);
      updateBufferFromSource(selNode, constantValue, 0);

      //! Testing
      // switchInfo.dataInfo.nodeToDataState[selNode]->printDetail();
    } else if (contains(selNode, "source")) {
      setOriginalDataOut(selNode, 0, 0, /*overwrite=*/false);
      updateBufferFromSource(selNode, 0, 0);
    } else {
      for (const auto &[value, iterIdx] :
           profileResults.nodeToValueTrace[selNode]) {
        setOriginalDataOut(selNode, iterIdx, value, /*overwrite=*/false);
        updateBufferFromSource(selNode, value, iterIdx);
      }

      //! Testing
      // switchInfo.dataInfo.nodeToDataState[selNode]->printDetail();
    }
  }

  // Define temporary storing structure
  std::map<std::string, int> tmpMuxOutputMap;
  std::map<std::string, bool> tmpMuxTokenPresentMap;
  std::vector<std::string> allMuxNodes;
  allMuxNodes.reserve(
      switchInfo.staticInfo.dataflowGraph->muxToSrcNodeMap.size());
  for (const auto &muxEntry :
       switchInfo.staticInfo.dataflowGraph->muxToSrcNodeMap) {
    allMuxNodes.push_back(muxEntry.first().str());
  }
  switchingDebugStream(SwitchingDebugCategory::Mux) << "[MUX_TRACE][BASE] total_mux_nodes=" << allMuxNodes.size()
               << "\n";

  // [STEP 3] Traverse the executed BB trace
  // Skip the first BB, as it will always be BB 0
  for (unsigned i = 1; i < profileResults.executedBasicBlockTrace.size(); i++) {
    unsigned preBB = profileResults.executedBasicBlockTrace[i - 1];
    unsigned curBB = profileResults.executedBasicBlockTrace[i];
    const unsigned edgeIterIndex =
        getExecutionIter(i - 1, curBB, profileResults);

    std::pair<unsigned, unsigned> key_pair(preBB, curBB);

    // Get the corresponding control merge value
    // [Step 3.1] We first update the value of all influenced control_merge
    // node in the circuit
    auto itCM = switchInfo.dataInfo.bbPairToControlMergeOutputs.find(key_pair);
    if (itCM != switchInfo.dataInfo.bbPairToControlMergeOutputs.end()) {
      std::set<std::string> activeMuxNodesAtEdge;
      switchingDebugStream(SwitchingDebugCategory::Mux) << "[MUX_TRACE][BASE] edge=" << preBB << "->" << curBB
                   << " iter=" << edgeIterIndex
                   << " control_merges=" << itCM->second.size() << "\n";

      // Iterate over all (CMNode, output value) pairs
      for (auto selValuePair : itCM->second) {
        auto nodeName = selValuePair.first;
        int outValue = selValuePair.second;
        // `executedBasicBlockTrace[i]` corresponds to edge-index `(i - 1)` in
        // the profiler log. Align the query index with
        // `edgeIndexToIterationMap` keys.
        unsigned iterIndex = edgeIterIndex;

        //! Testing
        // switchingDebugStream(SwitchingDebugCategory::Data) << "[DEBUG] \t\tCtrlMerge Node: " << nodeName << "\n";
        // switchingDebugStream(SwitchingDebugCategory::Data) << "[DEBUG] \t\t\tCon output Value: " << outValue <<
        // "\n"; switchingDebugStream(SwitchingDebugCategory::Data) << "[DEBUG] \t\t\tIter Index: " << iterIndex <<
        // "\n";

        // Update the value
        auto cmDb = getNodeDataBase(nodeName);
        auto selCtrlMergeNode =
            cmDb ? dyn_cast<CMergeData>(cmDb.get()) : nullptr;
        if (!selCtrlMergeNode) {
          llvm::errs() << "[ERROR] Control-merge data node " << nodeName
                       << " is missing in nodeToDataState\n";
          continue;
        }

        // Data output for the control merge node, we assign -1 to it.
        // Keep the first event mapped to this iteration to avoid clobbering
        // control values when multiple BB edges are observed in the same
        // iteration bucket.
        if (setOriginalDataOut(nodeName, iterIndex, -1, /*overwrite=*/false)) {
          // We don't update the data output for control merge node.
          // updateBufferFromSource(nodeName, tmpDataValuePair.value,
          // iterIndex);
        }

        // Control output
        if (!contains(selCtrlMergeNode->controlDataOut, iterIndex)) {
          IterationValue tmpConValuePair = {outValue, iterIndex};
          selCtrlMergeNode->controlDataOut[iterIndex] = tmpConValuePair;
        }

        // Define the Vector for unupdated mux
        std::vector<std::string> tmpUnUpdatedMux;
        // [Step 3.2] Update all related MUX node
        // Get the list of influenced mux node
        std::vector<std::string> selMuxVec =
            switchInfo.staticInfo.dataflowGraph->cmToMuxMap[nodeName];
        for (const auto &selMuxNode : selMuxVec)
          activeMuxNodesAtEdge.insert(selMuxNode);
        switchingDebugStream(SwitchingDebugCategory::Mux) << "[MUX_TRACE][BASE]   cm=" << nodeName
                     << " ctrl_out=" << outValue
                     << " influenced_muxes=" << selMuxVec.size() << "\n";

        // Build a dependency graph for the multiplexers
        // We may encounter different situation when updating the value for
        // MUX nodes
        for (const auto &selMuxNode : selMuxVec) {
          // Only update muxes this control-merge actually influences
          if (!containsValue(selMuxVec, selMuxNode))
            continue;
          std::string selMuxDataSrcNode =
              getMuxReplayDataSrcNode(switchInfo, selMuxNode, outValue);
          int tmpMuxOutput = 0;
          bool muxOutputHasToken = false;
          std::string resolveCase = "unresolved";

          //! Testing
          // switchingDebugStream(SwitchingDebugCategory::Data) << "[DEBUG] \t\t\t\tMux Node: " << selMuxNode <<
          // "\n"; switchingDebugStream(SwitchingDebugCategory::Data) << "[DEBUG] \t\t\t\tSel Data Src Node: " <<
          // selMuxDataSrcNode << "\n";

          if (!muxSelectedInputIsValid(switchInfo, selMuxNode, outValue,
                                       iterIndex)) {
            tmpMuxOutput = 0;
            resolveCase = "selected_input_invalid_default_0";
          } else if (contains(selMuxDataSrcNode, "constant") ||
              contains(selMuxDataSrcNode, "source") ||
              contains(selMuxDataSrcNode, "start")) {
            auto srcDb = getNodeDataBase(selMuxDataSrcNode);
            if (srcDb && !srcDb->originalDataOut.empty())
              tmpMuxOutput = getDataOutValueAtOrBefore(srcDb, 0, 0);
            else
              tmpMuxOutput = 0;
            muxOutputHasToken = true;
            resolveCase = "const_or_source";
          } else {
            auto srcDb = getNodeDataBase(selMuxDataSrcNode);
            if (!srcDb) {
              llvm::errs() << "[ERROR] Data source node " << selMuxDataSrcNode
                           << " not found for mux node " << selMuxNode << "\n";
              tmpMuxOutput = 0;
              resolveCase = "missing_src_node_default_0";
              bool wroteOriginal = setOriginalDataOut(
                  selMuxNode, iterIndex, tmpMuxOutput, /*overwrite=*/false);
              switchingDebugStream(SwitchingDebugCategory::Mux) << "[MUX_TRACE][BASE]     mux=" << selMuxNode
                           << " iter=" << iterIndex << " ctrl=" << outValue
                           << " src="
                           << (selMuxDataSrcNode.empty() ? "<empty>"
                                                         : selMuxDataSrcNode)
                           << " resolve=" << resolveCase
                           << " value=" << tmpMuxOutput << " write_original="
                           << (wroteOriginal ? "yes" : "no") << "\n";
              tmpMuxOutputMap[selMuxNode] = tmpMuxOutput;
              continue;
            }
            // If the src node is a mux node
            if (contains(selMuxDataSrcNode, "mux")) {
              // Case 1: the source node is the mux node itself

              if (selMuxDataSrcNode == selMuxNode) {
                //! Testing
                // switchingDebugStream(SwitchingDebugCategory::Data)
                //     << "[MUX_DEBUG][BASE] resolve_case=SELF_FEEDBACK\n";
                // Use the previous value
                if (contains(tmpMuxOutputMap, selMuxDataSrcNode)) {
                  tmpMuxOutput = tmpMuxOutputMap[selMuxDataSrcNode];
                  muxOutputHasToken = tmpMuxTokenPresentMap[selMuxDataSrcNode];
                  resolveCase = "self_feedback_tmp_map";
                } else {
                  llvm::errs()
                      << "[ERROR] Data src(" << selMuxDataSrcNode
                      << ") of Mux Node: " << selMuxNode << ", not found.\n";
                  resolveCase = "self_feedback_missing_tmp_map_default_0";
                }
              } else if (contains(srcDb->originalDataOut, iterIndex)) {
                //! Testing
                // switchingDebugStream(SwitchingDebugCategory::Data) << "[MUX_DEBUG][BASE]     "
                //                 "resolve_case=SRC_HAS_ITER_VALUE\n";
                tmpMuxOutput = srcDb->originalDataOut[iterIndex].value;
                muxOutputHasToken = true;
                resolveCase = "mux_src_has_iter_value";
              } else if (contains(tmpMuxOutputMap, selMuxDataSrcNode)) {
                //! Testing
                // switchingDebugStream(SwitchingDebugCategory::Data)
                //     << "[MUX_DEBUG][BASE] resolve_case=SRC_FROM_TMP_MAP\n";
                tmpMuxOutput = tmpMuxOutputMap[selMuxDataSrcNode];
                muxOutputHasToken = tmpMuxTokenPresentMap[selMuxDataSrcNode];
                resolveCase = "mux_src_tmp_map";
              } else {
                tmpUnUpdatedMux.push_back(selMuxNode);
                resolveCase = "mux_src_deferred_same_iter";
                switchingDebugStream(SwitchingDebugCategory::Mux)
                    << "[MUX_TRACE][BASE]     mux=" << selMuxNode
                    << " iter=" << iterIndex << " ctrl=" << outValue << " src="
                    << (selMuxDataSrcNode.empty() ? "<empty>"
                                                  : selMuxDataSrcNode)
                    << " resolve=" << resolveCase
                    << " action=defer_to_second_pass\n";
                continue;
              }
            } else {
              if (selMuxDataSrcNode.empty()) {
                llvm::errs()
                    << "[ERROR] The data src node for Mux Node: " << selMuxNode
                    << " is empty\n";
                // Unconnected , use default vlue
                tmpMuxOutput = 0;
                resolveCase = "empty_src_default_0";
              } else {
                auto &omap = srcDb->originalDataOut;
                if (omap.find(iterIndex) != omap.end()) {
                  tmpMuxOutput = omap[iterIndex].value;
                  muxOutputHasToken = true;
                  resolveCase = "src_has_iter_value";
                } else if (std::optional<int> heldValue =
                               muxSourceAllowsHeldFallback(
                                   switchInfo, selMuxNode, outValue,
                                   selMuxDataSrcNode)
                                   ? getCurrentDataOutValueAtOrBeforeStrict(
                                         srcDb, iterIndex)
                                   : std::nullopt) {
                  tmpMuxOutput = *heldValue;
                  muxOutputHasToken = true;
                  resolveCase = "src_held_value";
                } else if (contains(tmpMuxOutputMap, selMuxDataSrcNode)) {
                  tmpMuxOutput = tmpMuxOutputMap[selMuxDataSrcNode];
                  muxOutputHasToken = tmpMuxTokenPresentMap[selMuxDataSrcNode];
                  resolveCase = "src_tmp_map";
                } else {
                  tmpMuxOutput = 0;
                  resolveCase = "src_no_event_default_0";
                }
              }
            }
          }

          // Update the storing structure and the tmp dict
          //! Testing
          // switchingDebugStream(SwitchingDebugCategory::Data) << "[DEBUG] \t\t\t\t[OutValue] " << tmpMuxOutput <<
          // "\n";

          bool wroteOriginal = setOriginalDataOut(
              selMuxNode, iterIndex, tmpMuxOutput, /*overwrite=*/false);
          tmpMuxOutputMap[selMuxNode] = tmpMuxOutput;
          tmpMuxTokenPresentMap[selMuxNode] = muxOutputHasToken;
          updateBufferFromSource(selMuxNode, tmpMuxOutput, iterIndex,
                                 muxOutputHasToken);
          switchingDebugStream(SwitchingDebugCategory::Mux) << "[MUX_TRACE][BASE]     mux=" << selMuxNode
                       << " iter=" << iterIndex << " ctrl=" << outValue
                       << " src="
                       << (selMuxDataSrcNode.empty() ? "<empty>"
                                                     : selMuxDataSrcNode)
                       << " resolve=" << resolveCase
                       << " value=" << tmpMuxOutput
                       << " write_original=" << (wroteOriginal ? "yes" : "no")
                       << "\n";
        }

        // Update all remaining mux nodes
        // TODO: This maybe useless, check it out
        for (const auto &leftMuxNode : tmpUnUpdatedMux) {
          std::string selMuxDataSrcNode =
              getMuxReplayDataSrcNode(switchInfo, leftMuxNode, outValue);
          if (selMuxDataSrcNode.empty())
            llvm::errs() << "[ERROR] Mux node " << leftMuxNode
                         << " has no replay data source for control value "
                         << outValue << "\n";
          int tmpMuxOutput = 0;
          bool muxOutputHasToken = false;
          std::string resolveCase = "deferred_unresolved";

          if (contains(tmpMuxOutputMap, selMuxDataSrcNode)) {
            tmpMuxOutput = tmpMuxOutputMap[selMuxDataSrcNode];
            muxOutputHasToken = tmpMuxTokenPresentMap[selMuxDataSrcNode];
            resolveCase = "deferred_tmp_map";
          } else {
            llvm::errs() << "[ERROR] Data src(" << selMuxDataSrcNode
                         << ") of Mux Node: " << leftMuxNode
                         << ", not found in tmpMuxOutputMap.\n";
            resolveCase = "deferred_tmp_map_miss_default_0";
          }

          bool wroteOriginal = setOriginalDataOut(
              leftMuxNode, iterIndex, tmpMuxOutput, /*overwrite=*/false);
          tmpMuxTokenPresentMap[leftMuxNode] = muxOutputHasToken;
          updateBufferFromSource(leftMuxNode, tmpMuxOutput, iterIndex,
                                 muxOutputHasToken);
          switchingDebugStream(SwitchingDebugCategory::Mux) << "[MUX_TRACE][BASE]     mux=" << leftMuxNode
                       << " iter=" << iterIndex << " ctrl=" << outValue
                       << " src="
                       << (selMuxDataSrcNode.empty() ? "<empty>"
                                                     : selMuxDataSrcNode)
                       << " resolve=" << resolveCase
                       << " value=" << tmpMuxOutput
                       << " write_original=" << (wroteOriginal ? "yes" : "no")
                       << " deferred=yes\n";
        }
      }

      // Update all inactive mux nodes at this BB edge iteration.
      // If a mux isn't influenced by the active control-merge nodes on this
      // edge, default its payload to zero.
      for (const auto &muxNode : allMuxNodes) {
        if (activeMuxNodesAtEdge.find(muxNode) != activeMuxNodesAtEdge.end())
          continue;

        switchInfo.dataInfo.iterToInactiveMuxNodes[edgeIterIndex].push_back(
            muxNode);

        // New mux RTL semantics: when the select token is absent, data output
        // is forced to zero rather than leaking port 0.
        std::string selInactiveMuxDataSrcNode = "<inactive_zero>";
        int tmpInactiveMuxOutput = 0;

        IterationValue tmpInactiveMuxValuePair = {tmpInactiveMuxOutput,
                                                  edgeIterIndex};
        if (auto muxDb = getNodeDataBase(muxNode)) {
          muxDb->inactiveDataOut[edgeIterIndex] = tmpInactiveMuxValuePair;
          // Update the originalDataOutVec
          muxDb->originalDataOutVec.push_back(tmpInactiveMuxOutput);
        }
        switchingDebugStream(SwitchingDebugCategory::Mux) << "[MUX_TRACE][BASE]   inactive_mux iter="
                     << edgeIterIndex << " mux=" << muxNode << " src="
                     << (selInactiveMuxDataSrcNode.empty()
                             ? "<empty>"
                             : selInactiveMuxDataSrcNode)
                     << " value=" << tmpInactiveMuxOutput << "\n";
      }
    }
  }
}

void buildSegmentSuccNodesList(SwitchingInfo &switchInfo,
                               SCFProfilingResult &profileResults) {
  //! Testing
  for (const auto &[label, bblist] : switchInfo.staticInfo.segToBBs) {
    switchingDebugStream(SwitchingDebugCategory::Data) << "[DEBUG] \t\tSeg Label: " << label << "\n";
    switchingDebugStream(SwitchingDebugCategory::Data) << "[DEBUG] \t\t\tBB List: ";
    for (const auto &selBB : bblist) {
      switchingDebugStream(SwitchingDebugCategory::Data) << selBB << ", ";
    }
    switchingDebugStream(SwitchingDebugCategory::Data) << "\n";
  }

  for (const auto &selBaseNode :
       switchInfo.staticInfo.dataflowGraph->allDataBaseNode) {
    //! Testing
    // switchingDebugStream(SwitchingDebugCategory::Data) << "[DEBUG] Node Name: " << selBaseNode << "\n";

    // Check the existence of the selected node
    if (!contains(switchInfo.staticInfo.dataflowGraph->nodes, selBaseNode))
      continue;

    if (contains(selBaseNode, "control_merge")) {
      // If this is a control_merge node
      auto selCMNode = dyn_cast<CMergeNode>(
          switchInfo.staticInfo.dataflowGraph->nodes[selBaseNode].get());

      std::string dataSucNode = selCMNode->dataSucNodeName;
      std::string conSucNode = selCMNode->conSucNodeName;

      //! Testing
      switchingDebugStream(SwitchingDebugCategory::Data) << "[DEBUG] \tControl Merge Node: " << selBaseNode << "\n";
      switchingDebugStream(SwitchingDebugCategory::Data) << "[DEBUG] \t\tData Suc Node: " << dataSucNode << "\n";
      switchingDebugStream(SwitchingDebugCategory::Data) << "[DEBUG] \t\tControl Suc Node: " << conSucNode << "\n";

      for (const auto &[label, bblist] : switchInfo.staticInfo.segToBBs) {
        SegmentSuccessorInfo tmpMGInfo;
        std::vector<std::string> controlExcludVec = {dataSucNode};
        std::vector<std::string> dataExcludVec = {conSucNode};
        tmpMGInfo.control = segCtrlMergeSuccSearch(switchInfo, selBaseNode,
                                                   controlExcludVec, label);
        tmpMGInfo.data = segCtrlMergeSuccSearch(switchInfo, selBaseNode,
                                                dataExcludVec, label);
        tmpMGInfo.glitch = segCtrlMergeGlitchSuccSearch(
            switchInfo, selBaseNode, controlExcludVec, label);

        // Update the storing structure
        switchInfo.dataInfo.nodeToDataState[selBaseNode]
            ->segmentSuccessorInfoMap[label.str()] = tmpMGInfo;
      }
    } else {
      for (const auto &[label, bblist] : switchInfo.staticInfo.segToBBs) {
        // Update the stroing structure
        switchInfo.dataInfo.nodeToDataState[selBaseNode]
            ->segmentSuccessorInfoMap[label.str()] =
            segGeneralSuccSearch(switchInfo, selBaseNode, label);
      }
    }
  }
}

void dataGlitchNodeSearch(SwitchingInfo &switchInfo,
                          SCFProfilingResult &profileResults) {
  for (const auto &[label, bblist] : switchInfo.staticInfo.segToBBs) {
    // Skip all "S", "E", and "T" sections
    if (contains(label, "S") || contains(label, "E") || contains(label, "T")) {
      continue;
    }

    // Get the selected AdjGraph
    auto selAdjGraph = switchInfo.staticInfo.segToGraph[label];
    std::string mgBaseNode = selAdjGraph->baseNode;
    auto mgII = switchInfo.staticInfo.cfdfcIIs[std::stoul(std::string(label))];
    if (mgII == 0) {
      llvm::errs() << "[ERROR] The II for CFDFC " << label
                   << " is 0, which is invalid. Skip the glitch node search for"
                   << " this segment.\n";
      continue;
    }

    auto orderedNodes = selAdjGraph->orderedNodeName;
    auto &orderedALUs = switchInfo.dataInfo.segmentToOrderedAluNodes[label];

    //! Testing
    switchingDebugStream(SwitchingDebugCategory::Data) << "[DEBUG] \tCFDFC Label: " << label << "\n";
    switchingDebugStream(SwitchingDebugCategory::Data) << "[DEBUG] \t\tCFDFC Base Node: " << mgBaseNode << "\n";
    switchingDebugStream(SwitchingDebugCategory::Data) << "[DEBUG] \t\tCFDFC II: " << mgII << "\n";

    //
    std::map<std::string, std::vector<NodeGlitchInfo>> tmpGlitchDict;
    std::vector<std::string> tmpGlitchNodes;

    // Iterate over all mapped nodes in the CFDFC
    for (const auto &selNode : orderedALUs) {
      switchingDebugStream(SwitchingDebugCategory::Data) << "[DEBUG] \t\tALU Node: " << selNode << "\n";

      const std::string nodeType = getNodeType(selNode);
      if (GLITCH_NODE.find(nodeType) == GLITCH_NODE.end())
        continue;

      // Glitch model is defined only for binary integer ALU-like nodes.
      auto preNodeLists = selAdjGraph->nodes[selNode]->pres;
      if (preNodeLists.size() != 2) {
        llvm::errs() << "[ERROR] The ALU node " << selNode
                     << " does not have 2 inputs\n";
        continue;
      }

      // Unpack the pre_node_list
      const std::string &preNode1 = preNodeLists[0];
      const std::string &preNode2 = preNodeLists[1];

      // Get the source of the two inputs
      std::string srcNode1 =
          segNodeDataSrcSearch(switchInfo, preNode1, selAdjGraph.get());
      std::string srcNode2 =
          segNodeDataSrcSearch(switchInfo, preNode2, selAdjGraph.get());

      // Check whether this inode can have glitches
      LongestPathResult result1 =
          selLongestPath(switchInfo, preNode1, std::string(label));
      LongestPathResult result2 =
          selLongestPath(switchInfo, preNode2, std::string(label));

      //! Testing
      // switchingDebugStream(SwitchingDebugCategory::Data) << "[DEBUG] \tNode : " << selNode << "\n";
      // switchingDebugStream(SwitchingDebugCategory::Data) << "[DEBUG] \t\tsrcNode1 : " << srcNode1 << "\n";
      // switchingDebugStream(SwitchingDebugCategory::Data) << "[DEBUG] \t\tsrcNode2 : " << srcNode2 << "\n";

      // TODO: Validate the following rounding process
      int finStartPoint1Steady =
          static_cast<int>(
              std::fmod(static_cast<float_t>(result1.maxLatency), mgII)) +
          selAdjGraph->startBaseNodeShiftMap[result1.selStartNode] +
          selAdjGraph->startBaseNodeShiftMap[mgBaseNode];
      int finStartPoint2Steady =
          static_cast<int>(
              std::fmod(static_cast<float_t>(result2.maxLatency), mgII)) +
          selAdjGraph->startBaseNodeShiftMap[result2.selStartNode] +
          selAdjGraph->startBaseNodeShiftMap[mgBaseNode];

      // TODO: Need to validate the following assumptions
      // If the node is directly connected with the srcs(without any buffers
      // in between), the steady state value can not be used As the latency
      // can not be hidden
      bool preSrc1Buffered = false;
      bool preSrc2Buffered = false;

      auto src1It = switchInfo.dataInfo.nodeToDataState.find(srcNode1);
      if (src1It != switchInfo.dataInfo.nodeToDataState.end()) {
        auto seg1It = src1It->second->segmentSuccessorInfoMap.find(label.str());
        if (seg1It != src1It->second->segmentSuccessorInfoMap.end()) {
          for (const auto &selOriNode : seg1It->second.original) {
            if (preNode1 == selOriNode)
              preSrc1Buffered = true;
          }
        }
      }

      auto src2It = switchInfo.dataInfo.nodeToDataState.find(srcNode2);
      if (src2It != switchInfo.dataInfo.nodeToDataState.end()) {
        auto seg2It = src2It->second->segmentSuccessorInfoMap.find(label.str());
        if (seg2It != src2It->second->segmentSuccessorInfoMap.end()) {
          for (const auto &selOriNode : seg2It->second.original) {
            if (preNode2 == selOriNode)
              preSrc2Buffered = true;
          }
        }
      }

      // Assign the faster one with value 0 and slower one with value 1
      if (!preSrc1Buffered && !preSrc2Buffered) {
        finStartPoint1Steady =
            static_cast<int>(result1.maxLatency) +
            selAdjGraph->startBaseNodeShiftMap[result1.selStartNode] +
            selAdjGraph->startBaseNodeShiftMap[mgBaseNode];
        finStartPoint2Steady =
            static_cast<int>(result2.maxLatency) +
            selAdjGraph->startBaseNodeShiftMap[result2.selStartNode] +
            selAdjGraph->startBaseNodeShiftMap[mgBaseNode];

        if (finStartPoint1Steady != finStartPoint2Steady) {
          if (finStartPoint1Steady > finStartPoint2Steady) {
            finStartPoint1Steady = 1;
            finStartPoint2Steady = 0;
          } else {
            finStartPoint1Steady = 0;
            finStartPoint2Steady = 1;
          }
        }
      }

      //! Testing
      // switchingDebugStream(SwitchingDebugCategory::Data) << "\t\tPre_Node_1: " << preNode1 << ", Src_Node_1: " <<
      // srcNode1
      //             << ", Latency: " << result1.maxLatency << "\n"
      //             << "\t\tPre_Node_2: " << preNode2 << ", Src_Node_2: " <<
      //             srcNode2
      //             << ", Latency: " << result2.maxLatency << "\n"
      //             << "\tFin_start_time_1: " << finStartPoint1Steady << "\n"
      //             << "\tFin_start_time_2: " << finStartPoint2Steady << "\n"
      //             << "\tSrc_1_buffered: " << preSrc1Buffered << "\n"
      //             << "\tSrc_2_buffered: " << preSrc2Buffered << "\n"
      //             << "\tTmp_glitching_list: ";
      // for (const auto &n : tmpGlitchNodes)
      //   switchingDebugStream(SwitchingDebugCategory::Data) << n << " ";
      // switchingDebugStream(SwitchingDebugCategory::Data)  << "\n";

      // Define status storing structure
      NodeGlitchInfo tmpNode1, tmpNode2;
      tmpNode1.srcNode = srcNode1;
      tmpNode1.steadyTime = finStartPoint1Steady;
      tmpNode1.buffered = preSrc1Buffered;

      tmpNode2.srcNode = srcNode2;
      tmpNode2.steadyTime = finStartPoint2Steady;
      tmpNode2.buffered = preSrc2Buffered;

      // TODO: Validate the following criterias
      if (finStartPoint1Steady != finStartPoint2Steady) {
        //  Check condition for Case 1
        // C1: None of the src node is a constant
        if ((!contains(srcNode1, "constant")) &&
            (!contains(srcNode2, "constant"))) {
          tmpGlitchDict[selNode].push_back(tmpNode1);
          tmpGlitchDict[selNode].push_back(tmpNode2);
          tmpGlitchNodes.push_back(selNode);
        }
      } else {
        // Check Case 4
        if (containsValue(tmpGlitchNodes, srcNode1) &&
            (!containsValue(tmpGlitchNodes, srcNode2))) {
          if (!preSrc1Buffered) {
            tmpGlitchDict[selNode].push_back(tmpNode1);
            tmpGlitchDict[selNode].push_back(tmpNode2);
            tmpGlitchNodes.push_back(selNode);
          }
        } else if (containsValue(tmpGlitchNodes, srcNode2) &&
                   (!containsValue(tmpGlitchNodes, srcNode1))) {
          if (!preSrc2Buffered) {
            tmpGlitchDict[selNode].push_back(tmpNode1);
            tmpGlitchDict[selNode].push_back(tmpNode2);
            tmpGlitchNodes.push_back(selNode);
          }
        }
      }
    }

    // Store the glitching info into the main storing structure
    switchInfo.dataInfo.glitches[label] = tmpGlitchDict;
  }
}

int calGlitchValue(int op1, int op2, std::string selNode) {
  // TODO: Add supports for other types of nodes in self.GLITCH_NODE
  if (contains(selNode, "add")) {
    return op1 + op2;
  } else if (contains(selNode, "mul")) {
    return op1 * op2;
  } else {
    llvm::errs()
        << "[ERROR] Unexpected node type during glitching calculation: "
        << selNode << "\n";
    return 0;
  }
}

/// Return the first non-buffer predecessor along the **unique** path
/// between `node` and the Mux control port.
/// If you hit more than one predecessor (fan-in) we bail out –
/// only transparent chains are supported.
static std::string peelBufferChain(const SwitchingInfo &SI, std::string node) {
  // Use find() instead of operator[] to handle const StringMap
  while (contains(node, "buffer")) {
    auto nodeIt = SI.dataInfo.nodeToDataState.find(node);
    if (nodeIt == SI.dataInfo.nodeToDataState.end()) {
      llvm::errs() << "[ERROR] Node " << node
                   << " not found in nodeToDataState\n";
      return node;
    }

    const auto &preds =
        nodeIt->second
            ->segmentSuccessorInfoMap; // Note: using segmentSuccessorInfoMap
                                       // as it exists
    if (preds.size() != 1) {
      llvm::errs() << "[ERROR] Buffer " << node
                   << " has multiple or no predecessors; can't peel.\n";
      return node; // give up – caller will still error-out cleanly
    }

    const auto &firstSeg = preds.begin()->second;
    if (firstSeg.original.size() != 1) {
      llvm::errs() << "[ERROR] Buffer " << node
                   << " has multiple original predecessors; can't peel.\n";
      return node;
    }

    node = firstSeg.original.front(); // hop one step upstream
  }
  return node;
}

static unsigned getChannelBitWidth(mlir::Value value) {
  if (auto chanType = dyn_cast<handshake::ChannelType>(value.getType()))
    return chanType.getDataBitWidth();
  return 0;
}

static int signExtendValue(int value, unsigned srcWidth) {
  if (srcWidth == 0 || srcWidth >= 32)
    return value;

  int truncated = reduceBits(value, srcWidth);
  unsigned signBit = 1U << (srcWidth - 1);
  if (static_cast<unsigned>(truncated) & signBit)
    truncated |= static_cast<int>(~((1U << srcWidth) - 1U));
  return truncated;
}

static std::optional<int>
evaluateNodeOperation(mlir::Operation *op,
                      const std::function<int(unsigned)> &resolveOperandValue,
                      std::optional<int> profiledCmpValue = std::nullopt) {
  if (!op)
    return std::nullopt;

  if (auto shliOp = dyn_cast<handshake::ShLIOp>(op))
    return resolveOperandValue(0) << resolveOperandValue(1);

  if (auto shrsiOp = dyn_cast<handshake::ShRSIOp>(op)) {
    unsigned inputWidth = getChannelBitWidth(shrsiOp.getLhs());
    return signExtendValue(resolveOperandValue(0), inputWidth) >>
           resolveOperandValue(1);
  }

  if (auto shruiOp = dyn_cast<handshake::ShRUIOp>(op)) {
    unsigned inputWidth = getChannelBitWidth(shruiOp.getLhs());
    return static_cast<int>(
        static_cast<unsigned>(reduceBits(resolveOperandValue(0), inputWidth)) >>
        resolveOperandValue(1));
  }

  if (auto extsiOp = dyn_cast<handshake::ExtSIOp>(op)) {
    unsigned inputWidth = getChannelBitWidth(extsiOp.getOperand());
    return signExtendValue(resolveOperandValue(0), inputWidth);
  }

  if (auto extuiOp = dyn_cast<handshake::ExtUIOp>(op)) {
    unsigned inputWidth = getChannelBitWidth(extuiOp.getOperand());
    return reduceBits(resolveOperandValue(0), inputWidth);
  }

  if (auto trunciOp = dyn_cast<handshake::TruncIOp>(op)) {
    unsigned outputWidth = getChannelBitWidth(trunciOp.getResult());
    return reduceBits(resolveOperandValue(0), outputWidth);
  }

  if (isa<handshake::AddIOp>(op))
    return resolveOperandValue(0) + resolveOperandValue(1);

  if (isa<handshake::SubIOp>(op))
    return resolveOperandValue(0) - resolveOperandValue(1);

  if (isa<handshake::MulIOp>(op))
    return resolveOperandValue(0) * resolveOperandValue(1);

  if (isa<handshake::AndIOp>(op))
    return resolveOperandValue(0) & resolveOperandValue(1);

  if (isa<handshake::OrIOp>(op))
    return resolveOperandValue(0) | resolveOperandValue(1);

  if (isa<handshake::XOrIOp>(op))
    return resolveOperandValue(0) ^ resolveOperandValue(1);

  if (auto cmpiOp = dyn_cast<handshake::CmpIOp>(op)) {
    if (profiledCmpValue.has_value()) {
      unsigned outputWidth = getChannelBitWidth(cmpiOp.getResult());
      return reduceBits(*profiledCmpValue, outputWidth);
    }

    int lhs = resolveOperandValue(0);
    int rhs = resolveOperandValue(1);
    switch (cmpiOp.getPredicate()) {
    case handshake::CmpIPredicate::eq:
      return lhs == rhs;
    case handshake::CmpIPredicate::ne:
      return lhs != rhs;
    case handshake::CmpIPredicate::uge:
      return static_cast<unsigned>(lhs) >= static_cast<unsigned>(rhs);
    case handshake::CmpIPredicate::sge:
      return lhs >= rhs;
    case handshake::CmpIPredicate::ugt:
      return static_cast<unsigned>(lhs) > static_cast<unsigned>(rhs);
    case handshake::CmpIPredicate::sgt:
      return lhs > rhs;
    case handshake::CmpIPredicate::ule:
      return static_cast<unsigned>(lhs) <= static_cast<unsigned>(rhs);
    case handshake::CmpIPredicate::sle:
      return lhs <= rhs;
    case handshake::CmpIPredicate::ult:
      return static_cast<unsigned>(lhs) < static_cast<unsigned>(rhs);
    case handshake::CmpIPredicate::slt:
      return lhs < rhs;
    }
  }

  return std::nullopt;
}

static std::optional<int>
evaluateNodeFromCurrentSources(SwitchingInfo &switchInfo,
                               const std::string &executedSeg,
                               const std::string &nodeName, unsigned iter) {
  auto graphIt = switchInfo.staticInfo.segToGraph.find(executedSeg);
  if (graphIt == switchInfo.staticInfo.segToGraph.end() || !graphIt->second)
    return std::nullopt;

  auto nodeIt = graphIt->second->nodes.find(nodeName);
  if (nodeIt == graphIt->second->nodes.end() || !nodeIt->second ||
      !nodeIt->second->op)
    return std::nullopt;

  auto resolveOperandValue = [&](unsigned operandIdx) -> int {
    if (operandIdx >= nodeIt->second->pres.size())
      return 0;

    std::string srcNode =
        segNodeDataSrcSearch(switchInfo, nodeIt->second->pres[operandIdx],
                             graphIt->second.get());
    if (srcNode.empty())
      return 0;

    auto srcIt = switchInfo.dataInfo.nodeToDataState.find(srcNode);
    if (srcIt == switchInfo.dataInfo.nodeToDataState.end() || !srcIt->second)
      return 0;

    return getCurrentDataOutValueAtOrBefore(srcIt->second, iter, 0);
  };

  mlir::Operation *op = nodeIt->second->op;
  std::optional<int> profiledCmpValue;
  auto selfStateIt = switchInfo.dataInfo.nodeToDataState.find(nodeName);
  if (selfStateIt != switchInfo.dataInfo.nodeToDataState.end() &&
      selfStateIt->second) {
    profiledCmpValue = getDataOutValueIfPresent(selfStateIt->second, iter);
  }

  return evaluateNodeOperation(op, resolveOperandValue, profiledCmpValue);
}

static std::vector<int>
getCurrentWaveformForIteration(const std::shared_ptr<DataBase> &nodeData,
                               unsigned iter) {
  if (!nodeData)
    return {0};

  auto glitchIt = nodeData->glitchDataOutByIter.find(iter);
  if (glitchIt != nodeData->glitchDataOutByIter.end() &&
      !glitchIt->second.empty())
    return glitchIt->second;

  if (std::optional<int> stableValue =
          getCurrentDataOutValueAtOrBeforeStrict(nodeData, iter))
    return {*stableValue};

  return {0};
}

static std::optional<std::vector<int>>
evaluateNodeWaveformFromCurrentSources(SwitchingInfo &switchInfo,
                                       const std::string &executedSeg,
                                       const std::string &nodeName,
                                       unsigned iter) {
  auto graphIt = switchInfo.staticInfo.segToGraph.find(executedSeg);
  if (graphIt == switchInfo.staticInfo.segToGraph.end() || !graphIt->second)
    return std::nullopt;

  auto nodeIt = graphIt->second->nodes.find(nodeName);
  if (nodeIt == graphIt->second->nodes.end() || !nodeIt->second ||
      !nodeIt->second->op)
    return std::nullopt;

  std::vector<std::vector<int>> operandWaveforms;
  operandWaveforms.reserve(nodeIt->second->pres.size());

  for (const auto &preNode : nodeIt->second->pres) {
    std::string srcNode =
        segNodeDataSrcSearch(switchInfo, preNode, graphIt->second.get());
    if (srcNode.empty()) {
      operandWaveforms.push_back({0});
      continue;
    }

    auto srcIt = switchInfo.dataInfo.nodeToDataState.find(srcNode);
    if (srcIt == switchInfo.dataInfo.nodeToDataState.end() || !srcIt->second) {
      operandWaveforms.push_back({0});
      continue;
    }

    operandWaveforms.push_back(getCurrentWaveformForIteration(srcIt->second, iter));
  }

  if (operandWaveforms.empty())
    return std::nullopt;

  size_t eventCount = 1;
  for (const auto &waveform : operandWaveforms)
    eventCount = std::max(eventCount, waveform.size());

  std::vector<int> outputWaveform;
  outputWaveform.reserve(eventCount);

  for (size_t eventIdx = 0; eventIdx < eventCount; ++eventIdx) {
    auto resolveOperandValue = [&](unsigned operandIdx) -> int {
      if (operandIdx >= operandWaveforms.size() ||
          operandWaveforms[operandIdx].empty())
        return 0;

      const auto &waveform = operandWaveforms[operandIdx];
      const size_t sampleIdx = std::min(eventIdx, waveform.size() - 1);
      return waveform[sampleIdx];
    };

    std::optional<int> value =
        evaluateNodeOperation(nodeIt->second->op, resolveOperandValue);
    if (!value.has_value())
      return std::nullopt;
    outputWaveform.push_back(*value);
  }

  dedupConsecutiveValues(outputWaveform);
  return outputWaveform;
}

static bool shouldRunGlitchUpdateSegment(const std::string &seg);

static std::string
findPreferredSegmentForNode(const SwitchingInfo &switchInfo,
                            const std::string &nodeName) {
  auto nodeIt = switchInfo.staticInfo.dataflowGraph->nodes.find(nodeName);
  if (nodeIt == switchInfo.staticInfo.dataflowGraph->nodes.end() ||
      !nodeIt->second)
    return "";

  const unsigned bbIndex = nodeIt->second->bbindex;
  std::string bestSeg;
  size_t bestBBCount = std::numeric_limits<size_t>::max();

  for (const auto &[segLabelRef, bbList] : switchInfo.staticInfo.segToBBs) {
    const std::string segLabel = segLabelRef.str();
    if (!shouldRunGlitchUpdateSegment(segLabel) ||
        !containsValue(bbList, bbIndex))
      continue;

    auto segGraphIt = switchInfo.staticInfo.segToGraph.find(segLabel);
    if (segGraphIt == switchInfo.staticInfo.segToGraph.end() ||
        !segGraphIt->second ||
        !contains(segGraphIt->second->nodes, nodeName))
      continue;

    if (bbList.size() < bestBBCount) {
      bestSeg = segLabel;
      bestBBCount = bbList.size();
    }
  }

  return bestSeg;
}

static bool hasCurrentIterationEvent(const std::shared_ptr<DataBase> &nodeData,
                                     unsigned iter) {
  if (!nodeData)
    return false;

  if (contains(nodeData->originalDataOut, iter))
    return true;

  auto glitchIt = nodeData->glitchDataOutByIter.find(iter);
  return glitchIt != nodeData->glitchDataOutByIter.end() &&
         !glitchIt->second.empty();
}

static bool shouldReplayCrossSegmentCombinationalNode(
    const SwitchingInfo &switchInfo, const std::string &ownerSeg,
    const std::string &nodeName) {
  if (ownerSeg.empty() || contains(nodeName, "constant") ||
      contains(nodeName, "source") || contains(nodeName, "load") ||
      contains(nodeName, "buffer") || contains(nodeName, "mux") ||
      contains(nodeName, "control_merge"))
    return false;

  auto segGraphIt = switchInfo.staticInfo.segToGraph.find(ownerSeg);
  if (segGraphIt == switchInfo.staticInfo.segToGraph.end() ||
      !segGraphIt->second)
    return false;

  auto graphNodeIt = segGraphIt->second->nodes.find(nodeName);
  if (graphNodeIt == segGraphIt->second->nodes.end() || !graphNodeIt->second ||
      !graphNodeIt->second->op)
    return false;

  if (isa<handshake::CmpIOp>(graphNodeIt->second->op))
    return false;

  auto glitchSegIt = switchInfo.dataInfo.glitches.find(ownerSeg);
  if (glitchSegIt != switchInfo.dataInfo.glitches.end() &&
      contains(glitchSegIt->second, nodeName))
    return false;

  return true;
}

static bool nodeHasOperandEventAtIter(SwitchingInfo &switchInfo,
                                      const std::string &ownerSeg,
                                      const std::string &nodeName,
                                      unsigned iter) {
  auto segGraphIt = switchInfo.staticInfo.segToGraph.find(ownerSeg);
  if (segGraphIt == switchInfo.staticInfo.segToGraph.end() ||
      !segGraphIt->second)
    return false;

  auto graphNodeIt = segGraphIt->second->nodes.find(nodeName);
  if (graphNodeIt == segGraphIt->second->nodes.end() || !graphNodeIt->second)
    return false;

  bool sawDynamicOperand = false;
  bool sawWaveformOperand = false;

  for (const auto &preNode : graphNodeIt->second->pres) {
    std::string srcNode =
        segNodeDataSrcSearch(switchInfo, preNode, segGraphIt->second.get());
    if (srcNode.empty())
      continue;

    if (contains(srcNode, "constant") || contains(srcNode, "source") ||
        contains(srcNode, "start"))
      continue;

    sawDynamicOperand = true;

    auto srcStateIt = switchInfo.dataInfo.nodeToDataState.find(srcNode);
    if (srcStateIt == switchInfo.dataInfo.nodeToDataState.end() ||
        !srcStateIt->second)
      return false;

    if (!hasCurrentIterationEvent(srcStateIt->second, iter))
      return false;

    auto glitchIt = srcStateIt->second->glitchDataOutByIter.find(iter);
    if (glitchIt != srcStateIt->second->glitchDataOutByIter.end() &&
        glitchIt->second.size() > 1)
      sawWaveformOperand = true;
  }

  return sawDynamicOperand && sawWaveformOperand;
}

static void replayCrossSegmentCombinationalWaveforms(
    SwitchingInfo &switchInfo, unsigned iter, const std::string &executedSeg,
    bool debug) {
  for (const auto &nodeName :
       switchInfo.staticInfo.dataflowGraph->orderedNodeName) {
    const std::string ownerSeg = findPreferredSegmentForNode(switchInfo, nodeName);
    if (ownerSeg.empty() || ownerSeg == executedSeg)
      continue;

    if (!shouldReplayCrossSegmentCombinationalNode(switchInfo, ownerSeg,
                                                   nodeName) ||
        !nodeHasOperandEventAtIter(switchInfo, ownerSeg, nodeName, iter))
      continue;

    std::optional<std::vector<int>> waveform =
        evaluateNodeWaveformFromCurrentSources(switchInfo, ownerSeg, nodeName,
                                               iter);
    if (!waveform.has_value() || waveform->empty())
      continue;

    auto stateIt = switchInfo.dataInfo.nodeToDataState.find(nodeName);
    if (stateIt == switchInfo.dataInfo.nodeToDataState.end() || !stateIt->second)
      continue;

    if (contains(stateIt->second->glitchDataOutByIter, iter) &&
        stateIt->second->glitchDataOutByIter[iter] == *waveform)
      continue;

    stateIt->second->glitchDataOutByIter[iter] = *waveform;
    stateIt->second->lastUpdateIndex = iter;

    if (debug) {
      switchingDebugStream(SwitchingDebugCategory::Data)
          << "[DEBUG] \t\t[CROSS_SEG_COMB] iter=" << iter
          << " owner-seg=" << ownerSeg << " node=" << nodeName
          << " waveform=" << formatIntVector(*waveform) << "\n";
    }
  }
}

static bool shouldRunGlitchUpdateSegment(const std::string &seg) {
  return !(contains(seg, "S") || contains(seg, "E") || contains(seg, "T"));
}

static void
updateALUGlitchValuesForSegment(SwitchingInfo &switchInfo,
                                const std::string &executedSeg, unsigned iter,
                                bool glitchUpdateFlag, bool debug,
                                const std::vector<std::string> &segExecTrace,
                                std::vector<std::string> &indexUpdateList) {
  // ==================================================================
  // Step 1: Calculate glitching for inner MG nodes -- All ALUs, other
  // nodes shall be excluded.
  // ==================================================================
  for (const auto &selNode :
       switchInfo.dataInfo.segmentToOrderedDataSourceNodes[executedSeg]) {
    if (contains(selNode, "constant") || contains(selNode, "source") ||
        contains(selNode, "load"))
      continue;

    indexUpdateList.push_back(selNode);

    // Define vector to store the tmp glitch value
    std::vector<int> tmpValue;

    // Check whether we need the glitch value calculation
    if (glitchUpdateFlag) {
      // If we have glitch info for this node
      if (contains(switchInfo.dataInfo.glitches[executedSeg], selNode)) {
        // Get the source node
        std::string preSrc1 =
            switchInfo.dataInfo.glitches[executedSeg][selNode][0].srcNode;
        std::string preSrc2 =
            switchInfo.dataInfo.glitches[executedSeg][selNode][1].srcNode;
        // Get the starting time
        int preSrc1Start =
            switchInfo.dataInfo.glitches[executedSeg][selNode][0].steadyTime;
        int preSrc2Start =
            switchInfo.dataInfo.glitches[executedSeg][selNode][1].steadyTime;
        std::map<std::string, int> srcStartTimeDict = {{preSrc1, preSrc1Start},
                                                       {preSrc2, preSrc2Start}};

        // Get the buffering information
        bool preSrc1Buffered =
            switchInfo.dataInfo.glitches[executedSeg][selNode][0].buffered;
        bool preSrc2Buffered =
            switchInfo.dataInfo.glitches[executedSeg][selNode][1].buffered;
        std::map<std::string, bool> srcBufferedDict = {
            {preSrc1, preSrc1Buffered}, {preSrc2, preSrc2Buffered}};

        // Defining variables for glitch calculation
        int op1 = 0, op2 = 0;
        unsigned op1PreIndex = 0, op2PreIndex = 0;

        // Status Definition
        std::string fasterNode = "", slowerNode = "";

        //! Testing
        if (debug) {
          switchingDebugStream(SwitchingDebugCategory::Data) << "[DEBUG] \t\tGlitch Node: " << selNode << "\n"
                       << "[DEBUG] \t\t\tPre_src_1: " << preSrc1 << "\n"
                       << "[DEBUG] \t\t\tPre_src_2: " << preSrc2 << "\n";
        }

        // If this is the last iteration, we ignore the glitching value,
        // just copy the original value
        if (iter == segExecTrace.size() - 1) {
          int finalValue =
              evaluateNodeWaveformFromCurrentSources(switchInfo, executedSeg,
                                                     selNode, iter)
                  .value_or(std::vector<int>{})
                  .empty()
                  ? evaluateNodeFromCurrentSources(switchInfo, executedSeg,
                                                   selNode, iter)
                        .value_or(getDataOutValueAtOrBefore(
                            switchInfo.dataInfo.nodeToDataState[selNode], iter,
                            0))
                  : evaluateNodeWaveformFromCurrentSources(
                        switchInfo, executedSeg, selNode, iter)
                        ->back();
          //! Testing
          if (debug) {

            switchingDebugStream(SwitchingDebugCategory::Data)
                << "[DEBUG] \t\t(LAST ITER DURING GLITCH CALCULATION)\n";
          }

          tmpValue.push_back(finalValue);
          switchInfo.dataInfo.nodeToDataState[selNode]
              ->glitchDataOutByIter[iter] = tmpValue;
          switchInfo.dataInfo.nodeToDataState[selNode]->lastUpdateIndex = iter;
          continue;
        }

        // Check the starting time
        if (preSrc1Start != preSrc2Start) {
          if (preSrc1Start > preSrc2Start) {
            fasterNode = preSrc2;
            slowerNode = preSrc1;
          } else {
            fasterNode = preSrc1;
            slowerNode = preSrc2;
          }

          // Value 1: F[x - 1] op S[x - 1] if needed
          if (srcStartTimeDict[fasterNode] > 0) {
            // Initialization
            op1PreIndex = switchInfo.dataInfo.nodeToDataState[fasterNode]
                              ->lastUpdateIndex;
            op2PreIndex = switchInfo.dataInfo.nodeToDataState[slowerNode]
                              ->lastUpdateIndex;

            //! Testing
            if (debug) {

              switchingDebugStream(SwitchingDebugCategory::Data) << "[DEBUG] \t\t\t[Value 1]: \n"
                           << "[DEBUG] \t\t\t\tOp_1_src_node: " << fasterNode
                           << "\n"
                           << "[DEBUG] \t\t\t\tOp_1_pre_index: " << op1PreIndex
                           << "\n"
                           << "[DEBUG] \t\t\t\tOp_2_src_node: " << slowerNode
                           << "\n"
                           << "[DEBUG] \t\t\t\tOp_2_pre_index: " << op2PreIndex
                           << "\n";
            }

            // Check the existence of the selected iter
            if (contains(switchInfo.dataInfo.nodeToDataState[fasterNode]
                             ->originalDataOut,
                         op1PreIndex)) {
              op1 = getCurrentDataOutValueAtOrBefore(
                  switchInfo.dataInfo.nodeToDataState[fasterNode], op1PreIndex,
                  0);
            } else {
              op1 = 0;
            }

            // Get operand 2, if op_2_pre not in original_dataout keys, we
            // use 0 instead
            if (!(contains(switchInfo.dataInfo.nodeToDataState[slowerNode]
                               ->originalDataOut,
                           op2PreIndex))) {
              //! Testing
              if (debug) {
                switchingDebugStream(SwitchingDebugCategory::Data) << "[DEBUG] \t\t\t[Value 1]: \n"
                             << "[DEBUG] \t\t\t\t[Warning] S Last Active Iter "
                                "Not in the corresponding storing structure\n";
              }
              op2 = 0;
            } else {
              op2 = getCurrentDataOutValueAtOrBefore(
                  switchInfo.dataInfo.nodeToDataState[slowerNode], op2PreIndex,
                  0);
            }

            tmpValue.push_back(calGlitchValue(op1, op2, selNode));

            //! Testing
            if (debug) {

              switchingDebugStream(SwitchingDebugCategory::Data) << "[DEBUG] \t\t\t\tFaster Node: " << fasterNode
                           << "\n"
                           << "[DEBUG] \t\t\t\tOp_1: " << op1 << "\n"
                           << "[DEBUG] \t\t\t\tSlower Node: " << slowerNode
                           << "\n"
                           << "[DEBUG] \t\t\t\tOp_2: " << op2 << "\n";
            }
          }

          // Value 2: F[x] op S[x - 1]
          //! Testing
          if (debug) {
            switchingDebugStream(SwitchingDebugCategory::Data) << "[DEBUG] \t\t\t[Value 2]: \n";
          }

          int op1 = 0, op2 = 0;
          // Be careful that one of the operand may be generated by the
          // selNode itself, then we need to use the value in the
          // previous iteration in the calculation
          op1PreIndex =
              switchInfo.dataInfo.nodeToDataState[fasterNode]->lastUpdateIndex;
          op2PreIndex =
              switchInfo.dataInfo.nodeToDataState[slowerNode]->lastUpdateIndex;

          // Get the value of op1
          if (fasterNode == selNode) {
            if (contains(switchInfo.dataInfo.nodeToDataState[fasterNode]
                             ->originalDataOut,
                         op1PreIndex)) {
              op1 = getCurrentDataOutValueAtOrBefore(
                  switchInfo.dataInfo.nodeToDataState[fasterNode], op1PreIndex,
                  0);
            }
          } else if (contains(fasterNode, "mux")) {
            // Check whether the source of mux node is the selected node
            // itself
            std::string tmpMuxDataSrcNode =
                getMuxDataSrc(switchInfo, fasterNode, iter);

            if (tmpMuxDataSrcNode == selNode) {
              // TODO: Validate the following checking mechanism
              if (!contains(switchInfo.dataInfo.nodeToDataState[fasterNode]
                                ->originalDataOut,
                            op1PreIndex)) {
                op1 = 0;
              } else
                op1 = getCurrentDataOutValueAtOrBefore(
                    switchInfo.dataInfo.nodeToDataState[fasterNode],
                    op1PreIndex, 0);
            } else {
              op1 = getCurrentDataOutValueAtOrBefore(
                  switchInfo.dataInfo.nodeToDataState[fasterNode], iter, 0);
            }
          } else if (contains(switchInfo.dataInfo.glitches[executedSeg],
                              fasterNode)) {
            // If the faster node is glitching, we check whether the node
            // is buffered or not
            if (srcBufferedDict[fasterNode]) {
              // Buffered
              op1 = getCurrentDataOutValueAtOrBefore(
                  switchInfo.dataInfo.nodeToDataState[fasterNode], iter, 0);
            } else {
              auto &glitchvec = switchInfo.dataInfo.nodeToDataState[fasterNode]
                                    ->glitchDataOutByIter[iter];
              // ! If the list is smaller than 2, something is wrong
              op1 = 0;
              if (glitchvec.size() >= 2) {
                op1 = glitchvec[glitchvec.size() - 2];
              } else if (!glitchvec.empty()) {
                llvm::errs() << "[ERROR] Glitch Vector for " << selNode
                             << " has a size smaller than 2\n";
                op1 = glitchvec.back();
              } else { // only one glitch value – use it
                llvm::errs() << "[ERROR] Use original dataout\n";
                op1 = getCurrentDataOutValueAtOrBefore(
                    switchInfo.dataInfo.nodeToDataState[fasterNode], iter,
                    0); // no glitches recorded
              }
            }
          } else {
            op1 = getCurrentDataOutValueAtOrBefore(
                switchInfo.dataInfo.nodeToDataState[fasterNode], iter, 0);
          }

          // Get the value of op2
          if (contains(switchInfo.dataInfo.nodeToDataState[slowerNode]
                           ->originalDataOut,
                       op2PreIndex)) {
            op2 = getCurrentDataOutValueAtOrBefore(
                switchInfo.dataInfo.nodeToDataState[slowerNode], op2PreIndex,
                0);
          } else {
            op2 = getCurrentDataOutValueAtOrBefore(
                switchInfo.dataInfo.nodeToDataState[slowerNode], iter, 0);
          }

          tmpValue.push_back(calGlitchValue(op1, op2, selNode));

          //! Testing
          if (debug) {

            switchingDebugStream(SwitchingDebugCategory::Data) << "[DEBUG] \t\t\t\tFaster Node: " << fasterNode
                         << "\n"
                         << "[DEBUG] \t\t\t\tF Last Active Iter: "
                         << op1PreIndex << "\n"
                         << "[DEBUG] \t\t\t\tOp_1: " << op1 << "\n"
                         << "[DEBUG] \t\t\t\tSlower Node: " << slowerNode
                         << "\n"
                         << "[DEBUG] \t\t\t\tS Last Active Iter: "
                         << op2PreIndex << "\n"
                         << "[DEBUG] \t\t\t\tOp_2: " << op2 << "\n";
          }

          // Value 3: F[x] op S[x]
          // Check wheter the faster node is a mux node
          if (contains(fasterNode, "mux")) {
            std::string tmpMuxDataSrcNode =
                getMuxDataSrc(switchInfo, fasterNode, iter);

            if (tmpMuxDataSrcNode == selNode) {
              op1PreIndex = switchInfo.dataInfo.nodeToDataState[fasterNode]
                                ->lastUpdateIndex;

              // TODO: Validate the following rounding process
              if (!contains(switchInfo.dataInfo.nodeToDataState[fasterNode]
                                ->originalDataOut,
                            op1PreIndex)) {
                op1 = 0;
              } else {
                op1 = getCurrentDataOutValueAtOrBefore(
                    switchInfo.dataInfo.nodeToDataState[fasterNode],
                    op1PreIndex, 0);
              }
            } else {
              op1 = getCurrentDataOutValueAtOrBefore(
                  switchInfo.dataInfo.nodeToDataState[fasterNode], iter, 0);
            }
          } else {
            op1 = getCurrentDataOutValueAtOrBefore(
                switchInfo.dataInfo.nodeToDataState[fasterNode], iter, 0);
          }

          // Check whether the slower node is a mux node
          // TODO: Check the condition below
          op2 = getCurrentDataOutValueAtOrBefore(
              switchInfo.dataInfo.nodeToDataState[slowerNode], iter, 0);

          tmpValue.push_back(calGlitchValue(op1, op2, selNode));

          if (debug) {
            switchingDebugStream(SwitchingDebugCategory::Data) << "\t[Value 3]: \n"
                         << "\t\tOp_1: " << op1 << "\n"
                         << "\t\tOp_2: " << op2 << "\n"
                         << "\t\t[FINAL] ";
            for (auto v : tmpValue)
              switchingDebugStream(SwitchingDebugCategory::Data) << v << " ";
            switchingDebugStream(SwitchingDebugCategory::Data) << "\n";
          }
        }
      } else {
        tmpValue.push_back(
            evaluateNodeFromCurrentSources(switchInfo, executedSeg, selNode,
                                           iter)
                .value_or(getDataOutValueAtOrBefore(
                    switchInfo.dataInfo.nodeToDataState[selNode], iter, 0)));
      }
    } else {
      // No neeed for glitch value calculation, we directly copy the ori
      // data
      if (std::optional<std::vector<int>> waveform =
              evaluateNodeWaveformFromCurrentSources(switchInfo, executedSeg,
                                                     selNode, iter)) {
        tmpValue = *waveform;
      } else {
        tmpValue.push_back(
            evaluateNodeFromCurrentSources(switchInfo, executedSeg, selNode,
                                           iter)
                .value_or(getDataOutValueAtOrBefore(
                    switchInfo.dataInfo.nodeToDataState[selNode], iter, 0)));
      }
    }
    dedupConsecutiveValues(tmpValue);

    // Update the storing structure
    switchInfo.dataInfo.nodeToDataState[selNode]->glitchDataOutByIter[iter] =
        tmpValue;
  }
}

static void appendMuxValuesForCond(
    SwitchingInfo &switchInfo, const std::string &executedSeg, unsigned iter,
    const std::string &selMuxNode,
    const std::map<std::string, std::string> &muxSrcMap,
    std::optional<int> muxCondValue, bool glitchUpdateFlag, bool debug,
    std::vector<int> &outValues) {
  const size_t oldOutSize = outValues.size();
  std::string resolveCase = "unresolved";
  std::string selDataSrcNode = "";

  if (!muxCondValue.has_value()) {
    outValues.push_back(0);
    resolveCase = "inactive_no_control_default_0";
  } else {
    const int resolvedCond = *muxCondValue;
    selDataSrcNode =
        getMuxReplayDataSrcNode(switchInfo, selMuxNode, resolvedCond);
    const bool selectedInputHasCurrentEvent =
        muxImmediateInputHasCurrentEvent(switchInfo, selMuxNode, resolvedCond,
                                         iter);
    const bool selectedInputAllowsHeld =
        muxImmediateInputAllowsHeldToken(switchInfo, selMuxNode, resolvedCond);

    // TODO: Validate the following indexing mechanism
    // Check whether we have glitches from the srcs or not
    bool validDataSrcNode =
        !selDataSrcNode.empty() &&
        contains(switchInfo.dataInfo.nodeToDataState, selDataSrcNode) &&
        switchInfo.dataInfo.nodeToDataState[selDataSrcNode];

    if (!validDataSrcNode) {
      llvm::errs() << "[WARNING] Invalid data source \"" << selDataSrcNode
                   << "\" for mux node " << selMuxNode << " at iter " << iter
                   << ", using 0\n";
      outValues.push_back(0);
      resolveCase = "invalid_src_default_0";
    } else if (!muxSelectedInputIsValid(switchInfo, selMuxNode, resolvedCond,
                                        iter)) {
      outValues.push_back(0);
      resolveCase = "selected_input_invalid_default_0";
    } else if (contains(selDataSrcNode, "constant") ||
               contains(selDataSrcNode, "source") ||
               contains(selDataSrcNode, "start")) {
      int constantLikeValue = switchInfo.dataInfo.nodeToDataState[selDataSrcNode]
                                  ->originalDataOut[0]
                                  .value;

      outValues.push_back(constantLikeValue);
      resolveCase = "constant_like";
    } else if (selDataSrcNode == selMuxNode) {
      // The src node is the selected node itself
      unsigned preIndex =
          switchInfo.dataInfo.nodeToDataState[selDataSrcNode]
              ->lastUpdateIndex;

      // Check whether the preIndex exists or not
      if (contains(switchInfo.dataInfo.nodeToDataState[selDataSrcNode]
                       ->originalDataOut,
                   preIndex)) {
        int selfValue =
            switchInfo.dataInfo.nodeToDataState[selDataSrcNode]
                            ->originalDataOut[preIndex]
                            .value;

        outValues.push_back(selfValue);
        resolveCase = "self_feedback_last_update";
      } else {
        outValues.push_back(0);
        resolveCase = "self_feedback_missing_last_update_default_0";
      }
    } else if (glitchUpdateFlag &&
               (contains(switchInfo.dataInfo.glitches[executedSeg],
                         selDataSrcNode))) {
      // Check whether there are buffers in between
      std::string preNode = "";
      auto selMuxNodeStructure = dyn_cast<MuxNode>(
          switchInfo.staticInfo.dataflowGraph->nodes[selMuxNode].get());

      // Get the actual preNode
      // TODO: Need to check the portidx to name mapping
      for (const auto &[nodeName, portIdx] :
           selMuxNodeStructure->preNameToPortIdxMap) {
        //* Here we need to do cond + 1, as the port map of muxnode assigns
        // 0 to its control input.
        const unsigned int cond_add1 =
            static_cast<unsigned>(resolvedCond + 1);
        if (portIdx == (cond_add1))
          preNode = nodeName;
      }

      bool bufferedFlag = false;
      auto segSucIt = switchInfo.dataInfo.nodeToDataState[selDataSrcNode]
                          ->segmentSuccessorInfoMap.find(executedSeg);
      if (segSucIt != switchInfo.dataInfo.nodeToDataState[selDataSrcNode]
                          ->segmentSuccessorInfoMap.end() &&
          containsValue(segSucIt->second.original, preNode))
        bufferedFlag = true;

      //! Testing
      if (debug) {
        switchingDebugStream(SwitchingDebugCategory::Data) << "\t\tCur src node has glitches, buffered: "
                     << bufferedFlag << "\n";
      }

      if (bufferedFlag) {
        // The new mux semantics only force zero when the select token is
        // absent. Once a select token is present, buffered feedback paths keep
        // driving their held payload until they are overwritten.
        if (std::optional<int> bufferedValue = getCurrentDataOutValueIfPresent(
                switchInfo.dataInfo.nodeToDataState[selDataSrcNode],
                iter)) {
          outValues.push_back(*bufferedValue);
          resolveCase = "glitch_src_buffered_current";
        } else if (std::optional<int> heldValue =
                       muxSourceAllowsHeldFallback(
                           switchInfo, selMuxNode, resolvedCond, selDataSrcNode)
                           ? getCurrentDataOutValueAtOrBeforeStrict(
                                 switchInfo.dataInfo
                                     .nodeToDataState[selDataSrcNode],
                                 iter)
                           : std::nullopt) {
          outValues.push_back(*heldValue);
          resolveCase = "glitch_src_buffered_held";
        } else {
          outValues.push_back(0);
          resolveCase = "glitch_src_buffered_no_value_default_0";
        }
      } else {
        auto glitchIt = switchInfo.dataInfo.nodeToDataState[selDataSrcNode]
                            ->glitchDataOutByIter.find(iter);
        if (glitchIt !=
                switchInfo.dataInfo.nodeToDataState[selDataSrcNode]
                             ->glitchDataOutByIter.end() &&
            !glitchIt->second.empty()) {
          outValues.insert(outValues.end(), glitchIt->second.begin(),
                           glitchIt->second.end());
          resolveCase = "glitch_src_event_list";
        } else if (std::optional<int> stableValue =
                       getCurrentDataOutValueIfPresent(
                           switchInfo.dataInfo.nodeToDataState[selDataSrcNode],
                           iter)) {
          outValues.push_back(*stableValue);
          resolveCase = "glitch_src_current";
        } else if (std::optional<int> heldValue =
                       muxSourceAllowsHeldFallback(
                           switchInfo, selMuxNode, resolvedCond, selDataSrcNode)
                           ? getCurrentDataOutValueAtOrBeforeStrict(
                                 switchInfo.dataInfo.nodeToDataState[selDataSrcNode],
                                 iter)
                           : std::nullopt) {
          outValues.push_back(*heldValue);
          resolveCase = "glitch_src_held";
        } else {
          outValues.push_back(0);
          resolveCase = "glitch_src_no_value_default_0";
        }
      }
    } else if (glitchUpdateFlag && contains(selDataSrcNode, "mux")) {
      // Cascaded mux source: reuse upstream mux event output when
      // available.
      std::string preNode = "";
      auto selMuxNodeStructure = dyn_cast<MuxNode>(
          switchInfo.staticInfo.dataflowGraph->nodes[selMuxNode].get());
      for (const auto &[nodeName, portIdx] :
           selMuxNodeStructure->preNameToPortIdxMap) {
        const unsigned int condAdd1 = static_cast<unsigned>(resolvedCond + 1);
        if (portIdx == condAdd1)
          preNode = nodeName;
      }

      bool bufferedFlag = false;
      auto segSucIt = switchInfo.dataInfo.nodeToDataState[selDataSrcNode]
                          ->segmentSuccessorInfoMap.find(executedSeg);
      if (segSucIt != switchInfo.dataInfo.nodeToDataState[selDataSrcNode]
                          ->segmentSuccessorInfoMap.end() &&
          containsValue(segSucIt->second.original, preNode))
        bufferedFlag = true;

      auto glitchIt = switchInfo.dataInfo.nodeToDataState[selDataSrcNode]
                          ->glitchDataOutByIter.find(iter);
      if (std::optional<int> cascadedValue = getCurrentDataOutValueIfPresent(
              switchInfo.dataInfo.nodeToDataState[selDataSrcNode], iter)) {
        outValues.push_back(*cascadedValue);
        resolveCase = bufferedFlag ? "cascaded_mux_buffered_current"
                                   : "cascaded_mux_current";
      } else if (!bufferedFlag &&
                 glitchIt != switchInfo.dataInfo.nodeToDataState[selDataSrcNode]
                                 ->glitchDataOutByIter.end() &&
                 !glitchIt->second.empty()) {
        outValues.insert(outValues.end(), glitchIt->second.begin(),
                         glitchIt->second.end());
        resolveCase = "cascaded_mux_event_list";
      } else if (std::optional<int> heldValue =
                     muxSourceAllowsHeldFallback(switchInfo, selMuxNode,
                                                 resolvedCond, selDataSrcNode)
                         ? getCurrentDataOutValueAtOrBeforeStrict(
                               switchInfo.dataInfo.nodeToDataState[selDataSrcNode],
                               iter)
                         : std::nullopt) {
        outValues.push_back(*heldValue);
        resolveCase = bufferedFlag ? "cascaded_mux_buffered_held"
                                   : "cascaded_mux_held";
      } else {
        outValues.push_back(0);
        resolveCase = bufferedFlag
                          ? "cascaded_mux_buffered_no_value_default_0"
                          : "cascaded_mux_no_value_default_0";
      }
    } else {
      std::optional<int> stableValue =
          selectedInputHasCurrentEvent
              ? getCurrentDataOutValueIfPresent(
                    switchInfo.dataInfo.nodeToDataState[selDataSrcNode], iter)
              : std::nullopt;
      if (stableValue.has_value()) {
        outValues.push_back(*stableValue);
        resolveCase = "current_source";
      } else {
        std::optional<int> heldValue =
            (selectedInputHasCurrentEvent || selectedInputAllowsHeld)
                ? getCurrentDataOutValueAtOrBeforeStrict(
                      switchInfo.dataInfo.nodeToDataState[selDataSrcNode], iter)
                : std::nullopt;
        if (heldValue.has_value()) {
          outValues.push_back(*heldValue);
          resolveCase =
              selectedInputHasCurrentEvent ? "current_event_held_source"
                                           : "held_source";
        } else {
          outValues.push_back(0);
          resolveCase = "source_no_live_token_default_0";
        }
      }
    }
  }

  std::vector<int> appendedValues(outValues.begin() + oldOutSize,
                                  outValues.end());
  switchingDebugStream(SwitchingDebugCategory::Mux) << "[MUX_TRACE][GLITCH] iter=" << iter << " seg=" << executedSeg
               << " mux=" << selMuxNode << " cond=";
  if (muxCondValue.has_value())
    switchingDebugStream(SwitchingDebugCategory::Mux) << *muxCondValue;
  else
    switchingDebugStream(SwitchingDebugCategory::Mux) << "<inactive>";
  switchingDebugStream(SwitchingDebugCategory::Mux) << " src="
               << (selDataSrcNode.empty() ? "<empty>" : selDataSrcNode)
               << " immediate="
               << (muxCondValue.has_value()
                       ? getMuxImmediateInputNode(switchInfo, selMuxNode,
                                                  *muxCondValue)
                       : std::string("<inactive>"))
               << " fresh="
               << (muxCondValue.has_value() &&
                           muxImmediateInputHasCurrentEvent(
                               switchInfo, selMuxNode, *muxCondValue, iter)
                       ? "yes"
                       : "no")
               << " holds="
               << (muxCondValue.has_value() &&
                           muxImmediateInputAllowsHeldToken(
                               switchInfo, selMuxNode, *muxCondValue)
                       ? "yes"
                       : "no")
               << " resolve=" << resolveCase
               << " appended=" << formatIntVector(appendedValues) << "\n";
}

static void updateMuxGlitchValuesForSegment(SwitchingInfo &switchInfo,
                                            const std::string &executedSeg,
                                            unsigned iter,
                                            bool glitchUpdateFlag, bool debug) {
  // ====================================================================
  // Step 2: Update value for all MUXs
  // ====================================================================
  const auto &activeMuxNodes =
      switchInfo.dataInfo.segmentControlNodes[executedSeg].muxNodes;
  const unsigned segII = getSegmentII(switchInfo, executedSeg);
  // For each mux node in the executed segment, we update its value
  for (const auto &selMuxNode : activeMuxNodes) {
    auto muxSrcIt =
        switchInfo.staticInfo.dataflowGraph->muxToSrcNodeMap.find(selMuxNode);
    if (muxSrcIt == switchInfo.staticInfo.dataflowGraph->muxToSrcNodeMap.end())
      continue;

    std::string selCondInputNode = "";
    auto controlIt = muxSrcIt->second.find("control");
    if (controlIt != muxSrcIt->second.end())
      selCondInputNode = controlIt->second;

    std::optional<int> condValue;
    if (!selCondInputNode.empty() &&
        contains(switchInfo.dataInfo.nodeToDataState, selCondInputNode)) {
      if (auto *tmpCMBaseNode = dyn_cast<CMergeData>(
              switchInfo.dataInfo.nodeToDataState[selCondInputNode].get()))
        condValue = tmpCMBaseNode->getControlOutputIfPresent(iter);
    }
    switchingDebugStream(SwitchingDebugCategory::Mux) << "[MUX_TRACE][GLITCH] iter=" << iter
                 << " seg=" << executedSeg << " mux=" << selMuxNode
                 << " cond_node="
                 << (selCondInputNode.empty() ? "<empty>" : selCondInputNode)
                 << " cond=";
    if (condValue.has_value())
      switchingDebugStream(SwitchingDebugCategory::Mux) << *condValue;
    else
      switchingDebugStream(SwitchingDebugCategory::Mux) << "<inactive>";
    switchingDebugStream(SwitchingDebugCategory::Mux)
        << " glitch_mode=" << (glitchUpdateFlag ? "on" : "off") << "\n";

    // Temp Value vector definition
    std::vector<int> preValue, curValue, nexValue;

    //! Testing
    if (debug) {
      std::string selDataSrcNode = "";
      if (condValue.has_value()) {
        auto dataSrcIt = muxSrcIt->second.find(std::to_string(*condValue));
        if (dataSrcIt != muxSrcIt->second.end())
          selDataSrcNode = dataSrcIt->second;
      }

      switchingDebugStream(SwitchingDebugCategory::Data) << "Mux Node: " << selMuxNode << "\n"
                   << "\t[CUR_VALUE]\n"
                   << "\t\tCond Node: " << selCondInputNode << "\n"
                   << "\t\tCond_value: ";
      if (condValue.has_value())
        switchingDebugStream(SwitchingDebugCategory::Data) << *condValue;
      else
        switchingDebugStream(SwitchingDebugCategory::Data) << "<inactive>";
      switchingDebugStream(SwitchingDebugCategory::Data)
          << "\n"
          << "\t\tCur data src: " << selDataSrcNode << "\n";
    }

    std::vector<std::optional<int>> muxCondEvents;
    if (condValue.has_value()) {
      // In pipelined loop segments the select token disappears before the next
      // token arrives. Model that invalid-select gap ahead of the next active
      // event so traces match the RTL's 0-then-new-value ordering.
      if (segII > 1)
        muxCondEvents.push_back(std::nullopt);
      muxCondEvents.push_back(condValue);
    } else {
      muxCondEvents.push_back(std::nullopt);
    }

    for (std::optional<int> eventCond : muxCondEvents) {
      appendMuxValuesForCond(switchInfo, executedSeg, iter, selMuxNode,
                             muxSrcIt->second, eventCond, glitchUpdateFlag,
                             debug, curValue);
    }

    //! Testing
    if (debug) {
      switchingDebugStream(SwitchingDebugCategory::Data) << "\t\tCur_value: ";
      for (auto v : curValue)
        switchingDebugStream(SwitchingDebugCategory::Data) << v << " ";
      switchingDebugStream(SwitchingDebugCategory::Data) << "\n\t[TRANSATION GLITCHES]\n";
    }

    // ==============================================================
    // Calculate control flow glitches
    // ==============================================================
    // if (!switchInfo.dataInfo.nodeToDataState[selMuxNode]->skipControlCal) {
    //   std::string nodePreValidSeg =
    //       switchInfo.dataInfo.nodeToDataState[selMuxNode]->lastValidSeg;
    //   bool doubleTransFlag = false;

    //   //! Testing
    //   if (debug) {
    //     switchingDebugStream(SwitchingDebugCategory::Data) << "\t\tNode Pre MG: " << nodePreValidSeg << "\n"
    //                  << "\t\tNode Cur MG: " << executedSeg << "\n";
    //   }

    //   // Check whether we have transition between different MGs
    //   // We have two types of MG transitions
    //   //    TYPE 1: MG 0 -> MG 1 -> MG 1
    //   //    TYPE 2: MG 0 -> MG 1 -> MG 0
    //   if (nodePreValidSeg != "" && (nodePreValidSeg != executedSeg)) {
    //     muxTraceReason += "+cross_mg_transition";
    //     // Transition detected, check the next iter mg label
    //     if (iter != segExecTrace.size() - 1) {
    //       // Check the existence of the mux node
    //       if (std::find(switchInfo.dataInfo
    //                         .segmentControlNodes[segExecTrace[iter + 1]]
    //                         .muxNodes.begin(),
    //                     switchInfo.dataInfo
    //                         .segmentControlNodes[segExecTrace[iter + 1]]
    //                         .muxNodes.end(),
    //                     selMuxNode) !=
    //           switchInfo.dataInfo.segmentControlNodes[segExecTrace[iter +
    //           1]]
    //               .muxNodes.end()) {
    //         std::string nextExecSegLabel = segExecTrace[iter + 1];

    //         if (nextExecSegLabel != executedSeg) {
    //           // Type 2 detected
    //           doubleTransFlag = true;
    //           muxTraceReason += "+double_transition";
    //         }
    //       }
    //     }

    //     // Calculate preList
    //     if (!contains(executedSeg, "E")) {
    //       // Cond value will be the same for the last segment
    //       int preCondValue = 1 - condValue;
    //       auto muxIt =
    //           switchInfo.staticInfo.dataflowGraph->muxToSrcNodeMap.find(
    //               selMuxNode);
    //       if (muxIt ==
    //           switchInfo.staticInfo.dataflowGraph->muxToSrcNodeMap.end())
    //         continue;
    //       auto preSrcIt = muxIt->second.find(std::to_string(preCondValue));
    //       if (preSrcIt == muxIt->second.end())
    //         continue;
    //       std::string preDataSrc = preSrcIt->second;

    //       auto dbIt = switchInfo.dataInfo.nodeToDataState.find(preDataSrc);
    //       if (dbIt == switchInfo.dataInfo.nodeToDataState.end() ||
    //           !dbIt->second)
    //         continue;

    //       // Check whether the value exist or not
    //       if (contains(preDataSrc, "constant")) {
    //         preValue.push_back(
    //             getDataOutValueAtOrBefore(dbIt->second, /*iter=*/0, 0));
    //       } else {
    //         preValue.push_back(
    //             getDataOutValueAtOrBefore(dbIt->second, iter, 0));
    //       }
    //       muxTraceReason += "+pre_mg_value";
    //     }

    //     // If double transition
    //     if (doubleTransFlag) {
    //       switchInfo.dataInfo.nodeToDataState[selMuxNode]->skipControlCal =
    //           true;
    //       muxTraceReason += "+next_value_from_pre";
    //       nexValue = preValue;
    //     }
    //   }

    //   //! Testing
    //   if (debug) {
    //     switchingDebugStream(SwitchingDebugCategory::Data) << "\t\tDouble transition: " << doubleTransFlag <<
    //     "\n"
    //                  << "\t\tPre_value: ";
    //     for (auto v : preValue)
    //       switchingDebugStream(SwitchingDebugCategory::Data) << v << " ";
    //     switchingDebugStream(SwitchingDebugCategory::Data) << "\n\t\tNext value: ";
    //     for (auto v : nexValue)
    //       switchingDebugStream(SwitchingDebugCategory::Data) << v << " ";
    //     switchingDebugStream(SwitchingDebugCategory::Data) << "\n";
    //   }
    // } else {
    switchInfo.dataInfo.nodeToDataState[selMuxNode]->skipControlCal = false;
    // }

    // Get the final mux output data list
    std::vector<int> finalMuxOutputList;
    std::vector<int> inactiveValueList;

    for (const auto &selValue : preValue)
      finalMuxOutputList.push_back(selValue);
    for (const auto &selValue : curValue)
      finalMuxOutputList.push_back(selValue);
    for (const auto &selValue : nexValue)
      finalMuxOutputList.push_back(selValue);

    // If we have glitches as the mux is inactive for sometime in this
    // iteration
    if (!condValue.has_value() &&
        contains(switchInfo.dataInfo.iterToInactiveMuxNodes, iter) &&
        containsValue(switchInfo.dataInfo.iterToInactiveMuxNodes[iter],
                      selMuxNode)) {
      // finalMuxOutputList.push_back(0);
      auto muxDb = switchInfo.dataInfo.nodeToDataState[selMuxNode];
      finalMuxOutputList.push_back(muxDb->inactiveDataOut[iter].value);
      inactiveValueList.push_back(muxDb->inactiveDataOut[iter].value);
    }

    dedupConsecutiveValues(finalMuxOutputList);
    switchingDebugStream(SwitchingDebugCategory::Mux) << "[MUX_TRACE][GLITCH] iter=" << iter
                 << " seg=" << executedSeg << " mux=" << selMuxNode
                 << " pre=" << formatIntVector(preValue)
                 << " cur=" << formatIntVector(curValue)
                 << " next=" << formatIntVector(nexValue)
                 << " inactive=" << formatIntVector(inactiveValueList)
                 << " final=" << formatIntVector(finalMuxOutputList) << "\n";

    //! Testing
    if (debug) {
      switchingDebugStream(SwitchingDebugCategory::Data) << "\t\t[FINAL MUX OUTPUT] ";
      for (auto v : finalMuxOutputList)
        switchingDebugStream(SwitchingDebugCategory::Data) << v << " ";
      switchingDebugStream(SwitchingDebugCategory::Data) << "\n";
    }

    // Update the storing structure
    switchInfo.dataInfo.nodeToDataState[selMuxNode]->glitchDataOutByIter[iter] =
        finalMuxOutputList;
    switchInfo.dataInfo.nodeToDataState[selMuxNode]->lastValidSeg = executedSeg;
    switchInfo.dataInfo.nodeToDataState[selMuxNode]->lastUpdateIndex = iter;
  }
}

static void updateLoadNodesLastUpdateIndexForSegment(
    SwitchingInfo &switchInfo, const std::string &executedSeg, unsigned iter) {
  // Step 3: Relay memory load node's data
  for (const auto &selNode :
       switchInfo.dataInfo.segmentToOrderedDataSourceNodes[executedSeg]) {
    if (contains(selNode, "load")) {
      switchInfo.dataInfo.nodeToDataState[selNode]->lastUpdateIndex = iter;
    }
  }
}

void dataBaseNodeGlitchUpdate(SwitchingInfo &switchInfo,
                              SCFProfilingResult &profileResults, bool debug) {
  if (debug) {
    switchingDebugStream(SwitchingDebugCategory::Data) << "[DEBUG]\n[DEBUG] \t\t[NODE GLITCHING VALUE CALCULATION]\n";
  }
  //
  auto segExecTrace = profileResults.executedSegmentTrace;

  // Iterate through the execution trace
  for (unsigned i = 0; i < segExecTrace.size(); i++) {
    std::string executedSeg = segExecTrace[i];

    if (debug) {
      switchingDebugStream(SwitchingDebugCategory::Data) << "[DEBUG] \t******** Iter: " << i
                   << ", Seg: " << executedSeg << "\n";
    }

    bool glitchUpdateFlag = shouldRunGlitchUpdateSegment(executedSeg);

    std::vector<std::string> indexUpdateList;
    updateALUGlitchValuesForSegment(switchInfo, executedSeg, i,
                                    glitchUpdateFlag, debug, segExecTrace,
                                    indexUpdateList);

    // Step 1.5: update last update index for all data base nodes
    for (const auto &selNode : indexUpdateList) {
      switchInfo.dataInfo.nodeToDataState[selNode]->lastUpdateIndex = i;
    }

    updateMuxGlitchValuesForSegment(switchInfo, executedSeg, i,
                                    glitchUpdateFlag, debug);
    updateLoadNodesLastUpdateIndexForSegment(switchInfo, executedSeg, i);
    replayCrossSegmentCombinationalWaveforms(switchInfo, i, executedSeg,
                                             debug);
  }
}

// Masks 'value' to 'targetBitWidth' bits.
// E.g. reduceBits(0xABCD, 8) => 0xCD
int reduceBits(int value, unsigned targetBitWidth) {
  // If the target bit width is >= 32, we leave 'value' unchanged
  // because an int on most platforms is 32 bits, so no further masking is
  // needed.
  if (targetBitWidth >= 32) {
    return value;
  }

  // Build the mask. E.g. if targetBitWidth=8 => mask=0xFF
  // 1U << targetBitWidth is well-defined for 0 <= targetBitWidth <= 31
  // (shifting by 32 is undefined in 32-bit, hence the check above).
  unsigned mask = (1U << targetBitWidth) - 1U;

  // Apply the mask; cast to int to keep the function's return type int.
  return static_cast<int>(static_cast<unsigned>(value) & mask);
}

int getCondBrNodeCondValue(SwitchingInfo &switchInfo, unsigned iterIndex,
                           std::string nodeName) {
  // Defensive check for condBrToConSrcMap
  if (!contains(switchInfo.staticInfo.dataflowGraph->condBrToConSrcMap,
                nodeName)) {
    llvm::errs() << "[ERROR] Warning: cond_br node " << nodeName
                 << " not found in condBrToConSrcMap\n";
    return 0; // Default value
  }

  std::string condSrcNode =
      switchInfo.staticInfo.dataflowGraph->condBrToConSrcMap[nodeName];

  // Defensive check for nodeToDataState
  if (!contains(switchInfo.dataInfo.nodeToDataState, condSrcNode)) {
    llvm::errs() << "[ERROR] Warning: condSrcNode " << condSrcNode
                 << " not found in nodeToDataState\n";
    return 0; // Default value
  }

  const int rawCond = getCurrentDataOutValueAtOrBefore(
      switchInfo.dataInfo.nodeToDataState[condSrcNode], iterIndex, 0);
  return rawCond != 0 ? 1 : 0;
}

void dfgDataChannelPropagate(SwitchingInfo &switchInfo,
                             SCFProfilingResult &profileResults, bool debug) {
  //
  auto segExecTrace = profileResults.executedSegmentTrace;

  // Defensive reset in case propagation is rerun in the same pass instance.
  for (auto &[nodeName, nodeDb] : switchInfo.dataInfo.nodeToDataState) {
    (void)nodeName;
    if (!nodeDb)
      continue;
    if (auto *cmNode = dyn_cast<CMergeData>(nodeDb.get()))
      cmNode->controlGlitchValues.clear();
  }

  // Iterate through the execution trace and propagate base nodes values in
  // the selected segment
  for (unsigned i = 0; i < segExecTrace.size(); i++) {
    std::string executedSeg = segExecTrace[i];
    const unsigned segII = getSegmentII(switchInfo, executedSeg);

    // List storing nodes that need special treatment
    std::vector<std::string> pendingMemList;
    auto resolveOutWidth = [&](const std::string &srcNode,
                               const std::string &dstNode,
                               unsigned fallback = 1U) -> unsigned {
      unsigned width = fallback;

      auto baseNodeIt = switchInfo.dataInfo.nodeToDataState.find(srcNode);
      if (baseNodeIt != switchInfo.dataInfo.nodeToDataState.end() &&
          baseNodeIt->second &&
          contains(baseNodeIt->second->segmentSuccessorInfoMap, executedSeg)) {
        auto &dwMap = baseNodeIt->second->segmentSuccessorInfoMap[executedSeg]
                          .dataWidthMap;
        auto it = dwMap.find(dstNode);
        if (it != dwMap.end() && it->second > 0)
          width = std::max(width, it->second);
      }

      auto graphNodeIt =
          switchInfo.staticInfo.dataflowGraph->nodes.find(srcNode);
      if (graphNodeIt != switchInfo.staticInfo.dataflowGraph->nodes.end() &&
          graphNodeIt->second != nullptr) {
        auto edgeIt = graphNodeIt->second->sucsDataWidthMap.find(dstNode);
        if (edgeIt != graphNodeIt->second->sucsDataWidthMap.end() &&
            edgeIt->second > 0)
          width = std::max(width, edgeIt->second);
      }

      return width;
    };

    //! Testing
    if (debug)
      switchingDebugStream(SwitchingDebugCategory::Data) << "[DEBUG] ==================================\n";

    // Propagate data values for all non_memory related nodes
    for (const auto &selNode :
         switchInfo.dataInfo.segmentToDataSourceNodes[executedSeg].all) {
      //! Testing
      if (debug)
        switchingDebugStream(SwitchingDebugCategory::Data) << "[DEBUG] Node: " << selNode << "\n";

      // Case 1: Mapped Nodes --> All ALUs
      if (containsValue(
              switchInfo.dataInfo.segmentToOrderedAluNodes[executedSeg],
              selNode)) {
        //! Testing
        if (debug)
          switchingDebugStream(SwitchingDebugCategory::Data) << "[DEBUG] ALU Node Detected \n";

        // Defensive check for ALU node
        if (!contains(switchInfo.dataInfo.nodeToDataState, selNode)) {
          llvm::errs() << "[ERROR] Warning: ALU node " << selNode
                       << " not found in nodeToDataState at iteration " << i
                       << "\n";
          continue;
        }

        // The value shall be obtained from ori_glitch_dataout
        // Step 1: Update the selected node itself
        for (const auto &selValue :
             switchInfo.dataInfo.nodeToDataState[selNode]
                 ->glitchDataOutByIter[i]) {
          switchInfo.staticInfo.dataflowGraph->nodes[selNode]
              ->updateDataoutChannel(selValue);

          // Step 2: Update the glitching suceeding list
          //! Testing
          if (debug)
            switchingDebugStream(SwitchingDebugCategory::Data) << "[DEBUG] \t[Glitch Output]\n";

          // Defensive check for segmentSuccessorInfoMap access
          if (!contains(switchInfo.dataInfo.nodeToDataState[selNode]
                            ->segmentSuccessorInfoMap,
                        executedSeg)) {
            llvm::errs()
                << "[ERROR] Warning: segment " << executedSeg
                << " not found in segmentSuccessorInfoMap for ALU node "
                << selNode << "\n";
            continue;
          }

          for (const auto &selSucNode :
               switchInfo.dataInfo.nodeToDataState[selNode]
                   ->segmentSuccessorInfoMap[executedSeg]
                   .glitch) {
            //! Testing
            if (debug)
              switchingDebugStream(SwitchingDebugCategory::Data) << "[DEBUG] \t\tSucNode: " << selSucNode << "\n";

            // Get the node bitwidth
            unsigned tmpNodeWidth = resolveOutWidth(selNode, selSucNode);

            //! Testing
            if (debug)
              switchingDebugStream(SwitchingDebugCategory::Data) << "[DEBUG] \t\t\tDataWidth: " << tmpNodeWidth
                           << "\n";

            if (contains(selSucNode, "store")) {
              // Get the actual node storing structure
              auto selStoreNode = dyn_cast<DStoreNode>(
                  switchInfo.staticInfo.dataflowGraph->nodes[selSucNode].get());
              selStoreNode->updateDataout(reduceBits(selValue, tmpNodeWidth),
                                          selNode);
            } else if (contains(selSucNode, "cond_br")) {
              // Get the node
              auto selCondStoreNode = dyn_cast<CBrNode>(
                  switchInfo.staticInfo.dataflowGraph->nodes[selSucNode].get());
              // Get the control value
              // TODO: Validate the assumption that cond_value is always
              // updated before the data_in of the node
              int tmpCondValue =
                  getCondBrNodeCondValue(switchInfo, i, selSucNode);

              selCondStoreNode->updateDataout(
                  reduceBits(selValue, tmpNodeWidth), tmpCondValue);
            } else {
              switchInfo.staticInfo.dataflowGraph->nodes[selSucNode]
                  ->updateDataoutChannel(reduceBits(selValue, tmpNodeWidth));
            }
          }

          //! Testing
          if (debug)
            switchingDebugStream(SwitchingDebugCategory::Data) << "[DEBUG] \t[DONE]\n";
        }

        // Step 3: Update ori succeeding node list
        //! Testing
        if (debug)
          switchingDebugStream(SwitchingDebugCategory::Data) << "[DEBUG] \t[Ori Output]\n";
        for (const auto &selSucNode :
             switchInfo.dataInfo.nodeToDataState[selNode]
                 ->segmentSuccessorInfoMap[executedSeg]
                 .original) {
          // Get Node Bitwidth
          unsigned tmpNodeWidth = resolveOutWidth(selNode, selSucNode);

          //! Testing
          if (debug)
            switchingDebugStream(SwitchingDebugCategory::Data) << "[DEBUG] \t\tSuc Node: " << selSucNode << "\n";

          //
          if (contains(selSucNode, "store")) {
            //
            auto selStoreNode = dyn_cast<DStoreNode>(
                switchInfo.staticInfo.dataflowGraph->nodes[selSucNode].get());
            selStoreNode->updateDataout(
                reduceBits(getCurrentDataOutValueAtOrBefore(
                               switchInfo.dataInfo.nodeToDataState[selNode], i,
                               0),
                           tmpNodeWidth),
                selNode);
          } else if (contains(selSucNode, "cond_br")) {
            // Get the node
            // TODO: Validate the assumption that cond_value is always updated
            // before the data_in of the node
            auto selCondStoreNode = dyn_cast<CBrNode>(
                switchInfo.staticInfo.dataflowGraph->nodes[selSucNode].get());
            int tmpCondValue =
                getCondBrNodeCondValue(switchInfo, i, selSucNode);

            selCondStoreNode->updateDataout(
                reduceBits(getCurrentDataOutValueAtOrBefore(
                               switchInfo.dataInfo.nodeToDataState[selNode], i,
                               0),
                           tmpNodeWidth),
                tmpCondValue);
          } else {
            switchInfo.staticInfo.dataflowGraph->nodes[selSucNode]
                ->updateDataoutChannel(reduceBits(
                    getCurrentDataOutValueAtOrBefore(
                        switchInfo.dataInfo.nodeToDataState[selNode], i, 0),
                    tmpNodeWidth));
          }
        }
        //! Testing
        if (debug)
          switchingDebugStream(SwitchingDebugCategory::Data) << "[DEBUG] \t[DONE]\n";
      } else {
        // Case 2: This is a control merge node
        if (contains(selNode, "control_merge")) {
          //! Testing
          if (debug)
            switchingDebugStream(SwitchingDebugCategory::Data) << "[DEBUG] Control Merge Detected \n";

          // Defensive check for control merge node
          if (!contains(switchInfo.dataInfo.nodeToDataState, selNode)) {
            llvm::errs() << "[ERROR] Warning: Control merge node " << selNode
                         << " not found in nodeToDataState at iteration " << i
                         << "\n";
            continue;
          }

          // Get node
          auto selCMNode = dyn_cast<CMergeData>(
              switchInfo.dataInfo.nodeToDataState[selNode].get());
          auto selCMGraphNode = dyn_cast<CMergeNode>(
              switchInfo.staticInfo.dataflowGraph->nodes[selNode].get());
          if (!selCMGraphNode) {
            llvm::errs() << "[ERROR] Warning: Graph node " << selNode
                         << " is not a CMergeNode at iteration " << i << "\n";
            continue;
          }
          int initValue = 0;

          //! For all transition related nodes, we need to check the iter 0 as
          //! well
          if (i == 1) {
            initValue = selCMNode->getControlOutput(0);
            selCMGraphNode->updateDataout(initValue);
          } else {
            initValue = 0;
          }

          // Step 1: Update the node itself
          int selValue = selCMNode->getControlOutput(i);
          std::vector<int> cmControlOutputEvents = {selValue};
          if (selValue == 1 && segII > 1)
            cmControlOutputEvents.push_back(0);

          // Keep a trace of control-side glitch events for debug/inspection.
          for (int v : cmControlOutputEvents)
            selCMNode->controlGlitchValues.push_back(v);

          for (int v : cmControlOutputEvents) {
            selCMGraphNode->updateDataout(v);
          }

          // Step 2: Update all nodes in the control succeeding node list
          //! Testing
          if (debug)
            switchingDebugStream(SwitchingDebugCategory::Data) << "[DEBUG] \t[Control Output]\n";

          // Defensive check for segmentSuccessorInfoMap access
          if (!contains(selCMNode->segmentSuccessorInfoMap, executedSeg)) {
            llvm::errs() << "[ERROR] Segment " << executedSeg
                         << " not found in segmentSuccessorInfoMap for control "
                            "merge node "
                         << selNode << "\n";
            continue;
          }

          for (const auto &selSucNode :
               selCMNode->segmentSuccessorInfoMap[executedSeg].control) {
            //! Testing
            if (debug)
              switchingDebugStream(SwitchingDebugCategory::Data) << "[DEBUG] \t\tSuc Node: " << selSucNode << "\n";

            if (i == 1) {
              switchInfo.staticInfo.dataflowGraph->nodes[selSucNode]
                  ->updateDataoutChannel(initValue);
            }
            for (int v : cmControlOutputEvents) {
              switchInfo.staticInfo.dataflowGraph->nodes[selSucNode]
                  ->updateDataoutChannel(v);
            }
          }

          //! Testing
          if (debug)
            switchingDebugStream(SwitchingDebugCategory::Data) << "[DEBUG] \t[DONE]\n";

          // Step 3: Update all data channel values
          //! Testing
          if (debug)
            switchingDebugStream(SwitchingDebugCategory::Data) << "[DEBUG] \t[Data Output]\n";
          for (const auto &selSucNode :
               selCMNode->cmergeSegmentSuccessorInfoMap[executedSeg].data) {
            //! Testing
            if (debug)
              switchingDebugStream(SwitchingDebugCategory::Data) << "[DEBUG] \t\tSuc Node: " << selSucNode << "\n";

            if (contains(selSucNode, "cond_br")) {
              // Get the node
              auto selCondStoreNode = dyn_cast<CBrNode>(
                  switchInfo.staticInfo.dataflowGraph->nodes[selSucNode].get());

              int tmpCondValue =
                  getCondBrNodeCondValue(switchInfo, i, selSucNode);
              selCondStoreNode->updateDataout(-1, tmpCondValue);
            } else {
              switchInfo.staticInfo.dataflowGraph->nodes[selSucNode]
                  ->updateDataoutChannel(-1);
            }
          }

          //! Testing
          if (debug)
            switchingDebugStream(SwitchingDebugCategory::Data) << "[DEBUG] \t[DONE]\n";
        } else if (contains(selNode, "mux")) {
          // Case 3: MUX Node
          // In the last iteration or MG transitions, we take the
          // ori_glitch_data out value
          //! For all transition related nodes, we need to check the iter 0 as
          //! well Testing
          if (debug)
            switchingDebugStream(SwitchingDebugCategory::Data) << "[DEBUG] MUX Detected \n";

          // Defensive check for MUX node
          if (!contains(switchInfo.dataInfo.nodeToDataState, selNode)) {
            switchingDebugStream(SwitchingDebugCategory::Data) << "[DEBUG] Warning: MUX node " << selNode
                         << " not found in nodeToDataState at iteration " << i
                         << "\n";
            continue;
          }
          auto muxDataState = switchInfo.dataInfo.nodeToDataState[selNode];
          const auto &muxGlitchOut = muxDataState->glitchDataOutByIter[i];
          switchingDebugStream(SwitchingDebugCategory::Mux) << "[MUX_TRACE][PROP] iter=" << i
                       << " seg=" << executedSeg << " mux=" << selNode
                       << " glitch_events=" << formatIntVector(muxGlitchOut);
          if (contains(muxDataState->originalDataOut, i))
            switchingDebugStream(SwitchingDebugCategory::Data) << " original_iter_value="
                         << muxDataState->originalDataOut[i].value;
          else
            switchingDebugStream(SwitchingDebugCategory::Data) << " original_iter_value=<none>";
          switchingDebugStream(SwitchingDebugCategory::Data) << "\n";

          for (const auto &selValue :
               switchInfo.dataInfo.nodeToDataState[selNode]
                   ->glitchDataOutByIter[i]) {
            switchInfo.staticInfo.dataflowGraph->nodes[selNode]
                ->updateDataoutChannel(selValue);

            // Step 2: Update the glitching succeeding list
            //! Testing
            if (debug)
              switchingDebugStream(SwitchingDebugCategory::Data) << "[DEBUG] \t[Glitch Output]\n";

            // Defensive check for segmentSuccessorInfoMap access
            if (!contains(switchInfo.dataInfo.nodeToDataState[selNode]
                              ->segmentSuccessorInfoMap,
                          executedSeg)) {
              llvm::errs() << "[ERROR] Warning: segment " << executedSeg
                           << " not found in segmentSuccessorInfoMap for node "
                           << selNode << "\n";
              continue;
            }

            for (const auto &selSucNode :
                 switchInfo.dataInfo.nodeToDataState[selNode]
                     ->segmentSuccessorInfoMap[executedSeg]
                     .glitch) {
              //! Testing
              if (debug)
                switchingDebugStream(SwitchingDebugCategory::Data) << "[DEBUG] \t\tSuc Node: " << selSucNode << "\n";

              // Get the node output bitwidth
              unsigned tmpNodeWidth = resolveOutWidth(selNode, selSucNode);
              int reducedGlitchValue = reduceBits(selValue, tmpNodeWidth);

              if (contains(selSucNode, "cond_br")) {
                // Get the node
                auto selCondStoreNode = dyn_cast<CBrNode>(
                    switchInfo.staticInfo.dataflowGraph->nodes[selSucNode]
                        .get());

                int tmpCondValue =
                    getCondBrNodeCondValue(switchInfo, i, selSucNode);
                selCondStoreNode->updateDataout(reducedGlitchValue,
                                                tmpCondValue);
                switchingDebugStream(SwitchingDebugCategory::Mux) << "[MUX_TRACE][PROP]   iter=" << i
                             << " seg=" << executedSeg << " mux=" << selNode
                             << " phase=glitch dst=" << selSucNode
                             << " width=" << tmpNodeWidth
                             << " value=" << reducedGlitchValue
                             << " cond=" << tmpCondValue << "\n";
              } else {
                switchInfo.staticInfo.dataflowGraph->nodes[selSucNode]
                    ->updateDataoutChannel(reducedGlitchValue);
                switchingDebugStream(SwitchingDebugCategory::Mux) << "[MUX_TRACE][PROP]   iter=" << i
                             << " seg=" << executedSeg << " mux=" << selNode
                             << " phase=glitch dst=" << selSucNode
                             << " width=" << tmpNodeWidth
                             << " value=" << reducedGlitchValue << "\n";
              }
            }
            //! Testing
            if (debug)
              switchingDebugStream(SwitchingDebugCategory::Data) << "[DEBUG] \t[DONE]\n";
          }

          // Step 3: Update the ori succeeding node list
          //! Testing
          if (debug)
            switchingDebugStream(SwitchingDebugCategory::Data) << "[DEBUG] \t[Ori Output]\n";

          for (const auto &selSucNode :
               switchInfo.dataInfo.nodeToDataState[selNode]
                   ->segmentSuccessorInfoMap[executedSeg]
                   .original) {
            //! Testing
            if (debug)
              switchingDebugStream(SwitchingDebugCategory::Data) << "[DEBUG] \t\tSuc Node: " << selSucNode << "\n";
            // Get bitwidth
            unsigned tmpNodeWidth = resolveOutWidth(selNode, selSucNode);

            //
            if (std::optional<int> originalValue = getCurrentDataOutValueIfPresent(
                    switchInfo.dataInfo.nodeToDataState[selNode], i)) {
              int reducedOriginalValue =
                  reduceBits(*originalValue, tmpNodeWidth);
              if (contains(selSucNode, "store")) {
                auto selStoreNode = dyn_cast<DStoreNode>(
                    switchInfo.staticInfo.dataflowGraph->nodes[selSucNode]
                        .get());
                selStoreNode->updateDataout(reducedOriginalValue, selNode);
                switchingDebugStream(SwitchingDebugCategory::Mux) << "[MUX_TRACE][PROP]   iter=" << i
                             << " seg=" << executedSeg << " mux=" << selNode
                             << " phase=original dst=" << selSucNode
                             << " width=" << tmpNodeWidth
                             << " value=" << reducedOriginalValue << "\n";
              } else if (contains(selSucNode, "cond_br")) {
                auto selCondBrNode = dyn_cast<CBrNode>(
                    switchInfo.staticInfo.dataflowGraph->nodes[selSucNode]
                        .get());
                int tmpCondValue =
                    getCondBrNodeCondValue(switchInfo, i, selSucNode);

                selCondBrNode->updateDataout(reducedOriginalValue,
                                             tmpCondValue);
                switchingDebugStream(SwitchingDebugCategory::Mux) << "[MUX_TRACE][PROP]   iter=" << i
                             << " seg=" << executedSeg << " mux=" << selNode
                             << " phase=original dst=" << selSucNode
                             << " width=" << tmpNodeWidth
                             << " value=" << reducedOriginalValue
                             << " cond=" << tmpCondValue << "\n";
              } else {
                switchInfo.staticInfo.dataflowGraph->nodes[selSucNode]
                    ->updateDataoutChannel(reducedOriginalValue);
                switchingDebugStream(SwitchingDebugCategory::Mux) << "[MUX_TRACE][PROP]   iter=" << i
                             << " seg=" << executedSeg << " mux=" << selNode
                             << " phase=original dst=" << selSucNode
                             << " width=" << tmpNodeWidth
                             << " value=" << reducedOriginalValue << "\n";
              }
            }
          }
          //! Testing
          if (debug)
            switchingDebugStream(SwitchingDebugCategory::Data) << "[DEBUG] \t[DONE]\n";
        } else if (contains(selNode, "load")) {
          //! Testing
          if (debug)
            switchingDebugStream(SwitchingDebugCategory::Data) << "[DEBUG] LOAD Detected \n";
          // Case 4: load node
          pendingMemList.push_back(selNode);
          continue;
        } else {
          //! Testing
          if (debug)
            switchingDebugStream(SwitchingDebugCategory::Data) << "[DEBUG] Other Nodes Detected\n";
          // All other nodes
          int selValue = 0;
          if (contains(selNode, "source") || contains(selNode, "start")) {
            if (i ==
                switchInfo.dataInfo.segmentToFirstExecutionIter[executedSeg]) {
              selValue = switchInfo.dataInfo.nodeToDataState[selNode]
                             ->originalDataOut[0]
                             .value;
            } else
              continue;
          } else if (contains(selNode, "constant")) {
            selValue = switchInfo.dataInfo.nodeToDataState[selNode]
                           ->originalDataOut[0]
                           .value;
          } else {
            // Defensive check to prevent crash
            if (!contains(switchInfo.dataInfo.nodeToDataState, selNode)) {
              llvm::errs() << "[ERROR] Warning: selNode " << selNode
                           << " not found in nodeToDataState at iteration " << i
                           << "\n";
              selValue = 0; // Default value
            } else {
              auto nodeState = switchInfo.dataInfo.nodeToDataState[selNode];
              if (contains(selNode, "buffer")) {
                // Buffer traces are sparse/event-based. Use the latest known
                // value at-or-before this iteration.
                selValue = getDataOutValueAtOrBefore(nodeState, i, 0);
              } else if (!contains(nodeState->originalDataOut, i)) {
                llvm::errs() << "[ERROR] Warning: No data found for node "
                             << selNode << " at iteration " << i << "\n";
                selValue = 0; // Default value
              } else {
                selValue = nodeState->originalDataOut[i].value;
              }
            }
          }

          // Step 1: update the node itself
          switchInfo.staticInfo.dataflowGraph->nodes[selNode]
              ->updateDataoutChannel(selValue);

          // Step 2: Update all glitching nodes in the succeeding list
          // Defensive check for segmentSuccessorInfoMap access
          if (!contains(switchInfo.dataInfo.nodeToDataState[selNode]
                            ->segmentSuccessorInfoMap,
                        executedSeg)) {
            // switchingDebugStream(SwitchingDebugCategory::Data) << "[DEBUG] Warning: segment " << executedSeg << "
            // not found in segmentSuccessorInfoMap for other node " <<
            // selNode
            // << "\n";
            continue;
          }

          for (const auto &selSucNode :
               switchInfo.dataInfo.nodeToDataState[selNode]
                   ->segmentSuccessorInfoMap[executedSeg]
                   .glitch) {
            if (contains(selSucNode, "cond_br")) {
              auto selCondStoreNode = dyn_cast<CBrNode>(
                  switchInfo.staticInfo.dataflowGraph->nodes[selSucNode].get());
              int tmpCondValue =
                  getCondBrNodeCondValue(switchInfo, i, selSucNode);
              selCondStoreNode->updateDataout(selValue, tmpCondValue);
            } else {
              switchInfo.staticInfo.dataflowGraph->nodes[selSucNode]
                  ->updateDataoutChannel(selValue);
            }
          }

          // Step 3: Update all non glitching nodes in the succeeding list
          for (const auto &selSucNode :
               switchInfo.dataInfo.nodeToDataState[selNode]
                   ->segmentSuccessorInfoMap[executedSeg]
                   .original) {
            switchInfo.staticInfo.dataflowGraph->nodes[selSucNode]
                ->updateDataoutChannel(selValue);
          }

          //! Testing
          if (debug)
            switchingDebugStream(SwitchingDebugCategory::Data) << "[DEBUG] [DONE]\n";
        }
      }
    }

    // Case 6: Nodes related to memory accesses
    for (const auto &selNode : pendingMemList) {
      // Defensive check for memory node
      if (!contains(switchInfo.dataInfo.nodeToDataState, selNode)) {
        llvm::errs() << "[ERROR] Warning: Memory node " << selNode
                     << " not found in nodeToDataState at iteration " << i
                     << "\n";
        continue;
      }

      // We need to update the address out and data out separatly
      // Step 1: Get the address source
      auto selMemNode = dyn_cast<DLoadNode>(
          switchInfo.staticInfo.dataflowGraph->nodes[selNode].get());

      std::string tmpAddrPreNode = selMemNode->addressInNodeName;
      std::string tmpAddrPreSrcNode = "";
      if (executedSeg == "E") {
        tmpAddrPreSrcNode =
            segENodeSrcSearch(switchInfo, tmpAddrPreNode, profileResults);

      } else {
        tmpAddrPreSrcNode = memAddrSrcSearch(switchInfo, profileResults,
                                             tmpAddrPreNode, executedSeg, i);
      }

      // We assume there will not be any glitches for the address value
      int addrValue = 0;
      if (contains(tmpAddrPreSrcNode, "constant")) {
        addrValue = switchInfo.dataInfo.nodeToDataState[tmpAddrPreSrcNode]
                        ->originalDataOut[0]
                        .value;
      } else {
        // TODO: Validate the following assumption
        if (contains(switchInfo.dataInfo.nodeToDataState[tmpAddrPreSrcNode]
                         ->originalDataOut,
                     i)) {
          addrValue = switchInfo.dataInfo.nodeToDataState[tmpAddrPreSrcNode]
                          ->originalDataOut[i]
                          .value;
        } else {
          addrValue = -10;
        }
      }

      int selValue = getDataOutValueAtOrBefore(
          switchInfo.dataInfo.nodeToDataState[selNode], i, 0);
      // Step 1: Update the node itself
      selMemNode->updateDataout(selValue, addrValue);

      // Step 2: Update teh nodes in glitching list
      // Defensive check for segmentSuccessorInfoMap access
      if (!contains(switchInfo.dataInfo.nodeToDataState[selNode]
                        ->segmentSuccessorInfoMap,
                    executedSeg)) {
        llvm::errs() << "[ERROR] Warning: segment " << executedSeg
                     << " not found in segmentSuccessorInfoMap for memory node "
                     << selNode << "\n";
        continue;
      }

      for (const auto &selSucNode :
           switchInfo.dataInfo.nodeToDataState[selNode]
               ->segmentSuccessorInfoMap[executedSeg]
               .glitch) {
        //! Exclude the memory controller
        if ((!contains(selSucNode, "mem_controller")) &&
            ((!contains(selSucNode, "end")))) {
          // Get node bitwidth
          unsigned tmpNodeWidth = resolveOutWidth(selNode, selSucNode);

          if (contains(selSucNode, "cond_br")) {
            // Get the node
            auto selCondStoreNode = dyn_cast<CBrNode>(
                switchInfo.staticInfo.dataflowGraph->nodes[selSucNode].get());

            int tmpCondValue =
                getCondBrNodeCondValue(switchInfo, i, selSucNode);
            selCondStoreNode->updateDataout(reduceBits(selValue, tmpNodeWidth),
                                            tmpCondValue);
          } else {
            switchInfo.staticInfo.dataflowGraph->nodes[selSucNode]
                ->updateDataoutChannel(reduceBits(selValue, tmpNodeWidth));
          }
        }
      }

      // Step 3: Update nodes in non glitching list
      for (const auto &selSucNode :
           switchInfo.dataInfo.nodeToDataState[selNode]
               ->segmentSuccessorInfoMap[executedSeg]
               .original) {
        // Get bitwidth
        unsigned tmpNodeWidth = resolveOutWidth(selNode, selSucNode);

        //
        if (contains(selSucNode, "cond_br")) {
          // Get the node
          auto selCondStoreNode = dyn_cast<CBrNode>(
              switchInfo.staticInfo.dataflowGraph->nodes[selSucNode].get());

          int tmpCondValue = getCondBrNodeCondValue(switchInfo, i, selSucNode);
          selCondStoreNode->updateDataout(reduceBits(selValue, tmpNodeWidth),
                                          tmpCondValue);
        } else {
          switchInfo.staticInfo.dataflowGraph->nodes[selSucNode]
              ->updateDataoutChannel(reduceBits(selValue, tmpNodeWidth));
        }
      }
    }
  }

  // Final step, change the results of buffers directly connected to cond_br
  // node for (auto& [selCondNode, bufferList] :
  // switchInfo.staticInfo.dataflowGraph->condBrToBufferMap) {
  //   // Get the node
  //   auto selCondStoringNode =
  //   dyn_cast<CBrNode>(switchInfo.staticInfo.dataflowGraph->nodes[selCondNode].get());
  //   for (auto& [selBuffer, portIdx]: bufferList) {
  //     // Change the resutls of the buffer node
  //     for (auto& [selIdx, valueVec]:
  //     switchInfo.staticInfo.dataflowGraph->nodes[selBuffer]->dataOut) {
  //       valueVec = selCondStoringNode->per_channel_dataout[portIdx];
  //     }
  //   }
  // }
}

//===----------------------------------------------------------------------===//
//
// Functions for finding the data source node in different segments
//
//===----------------------------------------------------------------------===//
std::vector<std::string>
segCtrlMergeSuccSearch(SwitchingInfo &switchInfo, std::string startNode,
                       std::vector<std::string> &excludingList,
                       StringRef segLabel) {
  std::vector<std::string> succNodeList;
  auto &si = switchInfo;
  auto segBBs = llvm::ArrayRef<unsigned>(si.staticInfo.segToBBs[segLabel]);

  // Lambda function for finding the valid successor nodes
  auto succOf = [&si, &excludingList, &segBBs,
                 &segLabel](const std::string &n) -> std::vector<std::string> {
    std::vector<std::string> out;

    for (const auto &m : si.staticInfo.dataflowGraph->nodes[n]->sucs) {
      if (isExcluded(excludingList, m))
        continue;
      if (isDataBase(si, m))
        continue;
      if (isInvalidBackedge(si, segLabel, n, m))
        continue;
      if (crossesCondPort(si, n, m))
        continue;
      if (bufferOutsideSeg(si, m, segBBs))
        continue;

      out.push_back(m);
    }
    return out;
  };

  // Lambda function for pre-order traversal
  auto pre = [&succNodeList, &startNode](const std::string &n) {
    if (n != startNode) // skip the root itself
      succNodeList.push_back(n);
  };

  // Create roots array
  std::vector<std::string> roots{startNode};

  genericDFS(roots, succOf, pre);

  return succNodeList;
}

SegmentSuccessorInfo segGeneralSuccSearch(SwitchingInfo &switchInfo,
                                          std::string startNode,
                                          StringRef segLabel) {
  // Status variables
  unsigned numBuffers = 0;
  unsigned minDataWidth = 32;
  std::string lastNode = "";

  // Initialize the storing structure
  SegmentSuccessorInfo successorInfo;

  // Track the current path for proper buffer counting and cycle detection
  std::vector<std::string> currentPath;
  auto &si = switchInfo;
  auto segBBs = llvm::ArrayRef<unsigned>(si.staticInfo.segToBBs[segLabel]);

  // Lambda function for finding the valid successor nodes
  auto succOf = [&si, &segBBs, &segLabel, &currentPath,
                 &startNode](const std::string &n) -> std::vector<std::string> {
    std::vector<std::string> out;

    if (n == startNode) {
      for (auto &m : si.staticInfo.dataflowGraph->nodes[n]->sucs) {
        if (isDataBase(si, m))
          continue;
        out.push_back(m);
      }
      return out;
    }

    // For other nodes, we need to do more checks
    for (auto &m : si.staticInfo.dataflowGraph->nodes[n]->sucs) {
      if (isDataBase(si, m))
        continue;
      if (std::find(currentPath.begin(), currentPath.end(), m) !=
          currentPath.end())
        continue;
      if (isInvalidBackedge(si, segLabel, n, m))
        continue;
      if (skipCondBrPort(si, n, m) || skipMuxPort(si, n, m))
        continue;
      if (bufferOutsideSeg(si, m, segBBs))
        continue;
      if (isEndNode(m, segLabel))
        continue;
      if (isMemController(m))
        continue;

      out.push_back(m);
    }
    return out;
  };

  // Lambda function for Pre processing when entering a node
  auto onEnter = [&successorInfo, &numBuffers, &minDataWidth, &lastNode,
                  &currentPath, &si, &startNode](const std::string &n) {
    currentPath.push_back(n);

    if (n == startNode)
      return; // skip the root

    // predecessor in the path
    auto preNode = currentPath[currentPath.size() - 2];

    // 1) Update the running min‐width
    unsigned w =
        si.staticInfo.dataflowGraph->nodes[preNode]->sucsDataWidthMap[n];
    if (w < minDataWidth)
      minDataWidth = w;
    // 2) Buffer count
    if (isOpaqueBuffer(si, n))
      ++numBuffers;
    // 3) Classify
    if (numBuffers > 0)
      successorInfo.original.push_back(n);
    else
      successorInfo.glitch.push_back(n);

    lastNode = n;
    // on “push” (old code’s first write):
    successorInfo.dataWidthMap[preNode] = minDataWidth;
    // at end‐of‐iteration (old code’s second write):
    successorInfo.dataWidthMap[n] = minDataWidth;
  };

  auto onExit = [&](const std::string &n) {
    if (!currentPath.empty() && currentPath.back() == n) {
      if (isOpaqueBuffer(si, n))
        --numBuffers;
      currentPath.pop_back();
    }
  };

  // Run DFS starting from the startNode
  std::vector<std::string> roots = {startNode};
  dfsWithEvents(roots, succOf, onEnter, onExit);

  successorInfo.dataWidthMap[lastNode] = minDataWidth;

  return successorInfo;
}

unsigned getExecutionIter(unsigned bbIndex, unsigned curBB,
                          SCFProfilingResult &profileResults) {
  auto it = profileResults.edgeIndexToIterationMap.find(bbIndex);
  if (it != profileResults.edgeIndexToIterationMap.end())
    return it->second;

  // Deterministic fallback: use the closest previous mapped BB index.
  // This avoids off-by-one oscillation and spurious iteration drift when the
  // trace index does not have an exact map entry.
  auto ub = profileResults.edgeIndexToIterationMap.upper_bound(bbIndex);
  if (ub != profileResults.edgeIndexToIterationMap.begin()) {
    --ub;
    llvm::errs()
        << "[WARNING] Missing edgeIndexToIterationMap entry for bbIndex "
        << bbIndex << ", falling back to nearest mapped index " << ub->first
        << " (iter " << ub->second << ")\n";
    return ub->second;
  }

  llvm::errs() << "[ERROR] Cannot find the execution iteration mapping for "
               << "bbIndex " << bbIndex << "\n";
  return 0;
}

std::string segNodeDataSrcSearch(SwitchingInfo &switchInfo,
                                 std::string startNode, AdjGraph *selGraph) {
  // 0) trivial case
  if (isDataBase(switchInfo, startNode))
    return startNode;

  std::string foundDataSrc;
  std::unordered_set<std::string> pathSet;

  // 1) backwards‐successors in reverse order (to match your old .back())
  auto succOf = [&](const std::string &node) -> std::vector<std::string> {
    if (!foundDataSrc.empty())
      return {};
    std::vector<std::string> preds;
    if (contains(node, "cond_br")) {
      if (auto *c = dyn_cast<CBrNode>(
              switchInfo.staticInfo.dataflowGraph->nodes[node].get())) {
        preds.emplace_back(!c->dataPreNodeName.empty() ? c->dataPreNodeName
                                                       : c->condPreNodeName);
      }
    } else if (selGraph->nodes.count(node)) {
      for (auto it = selGraph->nodes[node]->pres.rbegin();
           it != selGraph->nodes[node]->pres.rend(); ++it)
        preds.push_back(*it);
    }
    // drop any already‐in‐path
    preds.erase(std::remove_if(preds.begin(), preds.end(),
                               [&](auto &m) { return isInPath(pathSet, m); }),
                preds.end());
    return preds;
  };

  // 2) onEnter: mark path & catch first base‐node
  auto onEnter = [&](const std::string &n) {
    pathSet.insert(n);
    if (foundDataSrc.empty() && isDataBase(switchInfo, n))
      foundDataSrc = n;
  };

  // 3) onExit: un‐mark path
  auto onExit = [&](const std::string &n) { pathSet.erase(n); };

  // 4) run your existing DFS kernel
  dfsWithEvents({startNode}, succOf, onEnter, onExit);

  // 5) report failure
  if (foundDataSrc.empty()) {
    llvm::errs() << "[ERROR] Could not find base node for " << startNode
                 << "\n";
    return "";
  }

  return foundDataSrc;
}

LongestPathResult selLongestPath(SwitchingInfo &switchInfo, std::string dstNode,
                                 std::string mgLabel) {
  //
  unsigned maxLatency = 0;
  std::string selStartNode = "";
  std::string lastSecondBuff = "";
  LongestPathResult returnValue;
  auto selGraph = switchInfo.staticInfo.segToGraph[mgLabel];

  // Get the list of buffers
  std::vector<std::string> selBuffList;
  for (const auto &selNode : selGraph->orderedNodeName) {
    if (selNode.find("buffer") != std::string::npos) {
      selBuffList.push_back(selNode);
    }
  }

  // For all start node
  for (const auto &startNode : selGraph->segStartNodes) {
    auto tmpPaths = selGraph->findPaths(startNode, dstNode, false, true);

    std::vector<std::pair<unsigned, std::string>> tmpLastSecondBufferList;

    for (auto &selPath : tmpPaths) {
      unsigned tmpPathLatency = selPath.latency;

      //! Testing
      // selPath.printDetail();

      if (tmpPathLatency > maxLatency) {
        maxLatency = tmpPathLatency;
        selStartNode = startNode;
      }

      // Get the last second buffer, if exist
      if (tmpPathLatency == maxLatency) {
        std::vector<std::string> tmpBufferList;
        for (const auto &tmpSelNode : selPath.nodeList) {
          if (tmpSelNode.find("buffer") != std::string::npos) {
            tmpBufferList.push_back(tmpSelNode);
          }
        }

        if (tmpBufferList.size() > 1) {
          tmpLastSecondBufferList.push_back(
              {tmpPathLatency, tmpBufferList[tmpBufferList.size() - 2]});
        }
      }
    }

    // Select the wanted last second buffer, this will be used in the cal of
    // SET_R for transparent buffers
    for (const auto &selPair : tmpLastSecondBufferList) {
      if (selPair.first == maxLatency) {
        // TODO: Validate the following selection criteria
        auto selBuffer =
            dyn_cast<BufferNode>(selGraph->nodes[selPair.second].get());
        float_t oriOcc = 0;
        if (lastSecondBuff != "") {
          auto oriBuffer =
              dyn_cast<BufferNode>(selGraph->nodes[lastSecondBuff].get());
          oriOcc = oriBuffer->occupancy;
        }
        if (lastSecondBuff == "" || (selBuffer->occupancy < oriOcc)) {
          lastSecondBuff = selPair.second;
        }
      }
    }
  }

  // Check the validity of the start node
  if (selStartNode == "") {
    selStartNode = selGraph->baseNode;
  }

  // Construct the return value
  returnValue.lastSecondBuffer = lastSecondBuff;
  returnValue.maxLatency = maxLatency;
  returnValue.selStartNode = selStartNode;

  return returnValue;
}

std::string getMuxDataSrc(SwitchingInfo &switchInfo, std::string selMuxNode,
                          unsigned selIter) {
  auto muxIt =
      switchInfo.staticInfo.dataflowGraph->muxToSrcNodeMap.find(selMuxNode);
  if (muxIt == switchInfo.staticInfo.dataflowGraph->muxToSrcNodeMap.end()) {
    llvm::errs() << "[ERROR] Mux node " << selMuxNode
                 << " not found in muxToSrcNodeMap\n";
    return ""; // Return empty string on error
  }

  // Check if "control" key exists
  auto controlIt = muxIt->second.find("control");
  if (controlIt == muxIt->second.end()) {
    llvm::errs() << "[ERROR] Control input not found for mux node "
                 << selMuxNode << "\n";
    return ""; // Return empty string on error
  }

  // The cond input src will always be a control merge node
  std::string muxCondInput = controlIt->second;
  muxCondInput = peelBufferChain(switchInfo, muxCondInput);

  // TODO: Prevent the following situation from happening
  if (contains(muxCondInput, "buffer")) {
    llvm::errs() << "[ERROR] A buffer is directly preceding " << selMuxNode
                 << "'s control input port\n";
  }

  // Use find() instead of operator[] for safer access
  auto nodeIt = switchInfo.dataInfo.nodeToDataState.find(muxCondInput);
  if (nodeIt == switchInfo.dataInfo.nodeToDataState.end()) {
    llvm::errs() << "[ERROR] Node " << muxCondInput
                 << " not found in nodeToDataState\n";
    return ""; // Return empty string on error
  }

  // Get the corresponding Control Merge Data base node
  auto *selCMBaseNode = dyn_cast<CMergeData>(nodeIt->second.get());
  if (!selCMBaseNode) {
    llvm::errs() << "[ERROR] No selCMBaseNode, got " << muxCondInput
                 << "'s control input port\n";
    return ""; // Return empty string on error
  }

  std::optional<int> ctrlInValue = selCMBaseNode->getControlOutputIfPresent(selIter);
  if (!ctrlInValue.has_value())
    return "";

  if (!muxSelectedInputIsValid(switchInfo, selMuxNode, *ctrlInValue, selIter))
    return "";

  const std::string selDataSrcNode =
      getMuxReplayDataSrcNode(switchInfo, selMuxNode, *ctrlInValue);
  if (contains(selDataSrcNode, "constant") || contains(selDataSrcNode, "source") ||
      contains(selDataSrcNode, "start"))
    return selDataSrcNode;

  auto srcIt = switchInfo.dataInfo.nodeToDataState.find(selDataSrcNode);
  if (srcIt == switchInfo.dataInfo.nodeToDataState.end() || !srcIt->second)
    return "";

  if (getCurrentDataOutValueIfPresent(srcIt->second, selIter).has_value())
    return selDataSrcNode;

  if (muxSourceAllowsHeldFallback(switchInfo, selMuxNode, *ctrlInValue,
                                  selDataSrcNode) &&
      getCurrentDataOutValueAtOrBeforeStrict(srcIt->second, selIter).has_value())
    return selDataSrcNode;

  return "";
}

// This function finds the src node in a specified segment for a given node
static std::optional<std::string>
findSrcInSegment(const SwitchingInfo &si, const std::string &nodeName,
                 const std::string &segLabel) {
  static constexpr std::array<llvm::StringRef, 4> cats{"control", "data",
                                                       "glitch", "original"};

  // 1) Check whether the segment exists
  if (!contains(si.dataInfo.segmentToDataSourceNodes, segLabel)) {
    return std::nullopt;
  }

  // Iterate all mapped roots
  for (auto const &mapped :
       si.dataInfo.segmentToDataSourceNodes.at(segLabel).all) {
    // 2) Direct name-match
    if (contains(nodeName, mapped)) {
      return mapped;
    }

    // 3) Check if DFG node exists
    if (!contains(si.dataInfo.nodeToDataState, mapped)) {
      llvm::errs() << "[ERROR] Node " << mapped
                   << " not found in nodeToDataState\n";
      continue;
    }
    auto const &nodeVal = si.dataInfo.nodeToDataState.at(mapped);

    // 4) Check if segSuc entry exists
    if (!contains(nodeVal->segmentSuccessorInfoMap, segLabel)) {
      llvm::errs() << "[ERROR] Segment " << segLabel
                   << " not found in segmentSuccessorInfoMap for node "
                   << mapped << "\n";
      continue;
    }
    auto const &mSuc = nodeVal->segmentSuccessorInfoMap.at(segLabel);

    // 5) Search in successor lists
    for (auto cat : cats) {
      auto const &lst = (cat == "control"  ? mSuc.control
                         : cat == "data"   ? mSuc.data
                         : cat == "glitch" ? mSuc.glitch
                                           : mSuc.original);
      if (std::any_of(lst.begin(), lst.end(), [&](auto const &s) {
            return s.find(nodeName) != std::string::npos;
          }))
        return mapped;
    }
  }
  return std::nullopt;
}

std::string segENodeSrcSearch(SwitchingInfo &switchInfo, std::string nodeName,
                              SCFProfilingResult &profileResults) {
  if (auto src = findSrcInSegment(switchInfo, nodeName, "E"))
    return *src;

  if (profileResults.executedSegmentTrace.size() >= 2) {
    auto prev =
        profileResults
            .executedSegmentTrace[profileResults.executedSegmentTrace.size() -
                                  2];
    if (auto src = findSrcInSegment(switchInfo, nodeName, prev))
      return *src;
  }
  return "";
}

std::string memAddrSrcSearch(SwitchingInfo &si, SCFProfilingResult &profile,
                             const std::string &nodeName,
                             const std::string &selSeg, unsigned iterIdx) {
  if (auto src = findSrcInSegment(si, nodeName, selSeg))
    return *src;
  // fallback to the previous segment in the trace
  if (iterIdx == 0 || iterIdx > profile.executedSegmentTrace.size() - 1)
    return "";
  auto prev = profile.executedSegmentTrace[iterIdx - 1];
  if (auto src = findSrcInSegment(si, nodeName, prev))
    return *src;
  return "";
}

//===----------------------------------------------------------------------===//
//
// Class for storing data channel values
//
//===----------------------------------------------------------------------===//
DataBase::DataBase(const std::string &node)
    : nodeName(node), lastUpdateIndex(0), skipControlCal(false) {}

void DataBase::printDetail() {
  switchingDebugStream(SwitchingDebugCategory::Data) << "Node Name: " << nodeName << "\n";

  // Print originalDataOut
  switchingDebugStream(SwitchingDebugCategory::Data) << "\tOriginal Dataout:\n";
  for (const auto &kv : originalDataOut) {
    switchingDebugStream(SwitchingDebugCategory::Data) << "\t\tIter " << kv.first << ": (" << kv.second.value << ", "
                 << kv.second.iterIndex << ")\n";
  }

  // Print mg_suc_node_dict, which is a map<mg_label, MgInfo>
  for (const auto &mgPair : segmentSuccessorInfoMap) {
    const auto &mgLabel = mgPair.first;
    const SegmentSuccessorInfo &info = mgPair.second;

    switchingDebugStream(SwitchingDebugCategory::Data) << "\tSegment Label: " << mgLabel << "\n";
    // Print "original"
    switchingDebugStream(SwitchingDebugCategory::Data) << "\t\toriginal = [";
    for (size_t i = 0; i < info.original.size(); ++i) {
      switchingDebugStream(SwitchingDebugCategory::Data) << info.original[i];
      if (i + 1 < info.original.size())
        switchingDebugStream(SwitchingDebugCategory::Data) << ", ";
    }
    switchingDebugStream(SwitchingDebugCategory::Data) << "]\n";
    // Print "glitch"
    switchingDebugStream(SwitchingDebugCategory::Data) << "\t\tglitch = [";
    for (size_t i = 0; i < info.glitch.size(); ++i) {
      switchingDebugStream(SwitchingDebugCategory::Data) << info.glitch[i];
      if (i + 1 < info.glitch.size())
        switchingDebugStream(SwitchingDebugCategory::Data) << ", ";
    }
    switchingDebugStream(SwitchingDebugCategory::Data) << "]\n";
    // Print data_width
    switchingDebugStream(SwitchingDebugCategory::Data) << "\t\tdata_width:\n";
    for (const auto &dw : info.dataWidthMap) {
      switchingDebugStream(SwitchingDebugCategory::Data) << "\t\t  " << dw.first << " => " << dw.second << "\n";
    }
  }

  // if this is a control merge node
  if (controlDataOut.size()) {
    switchingDebugStream(SwitchingDebugCategory::Data) << "\tControl Dataout:\n";
    for (const auto &kv : controlDataOut) {
      switchingDebugStream(SwitchingDebugCategory::Data) << "\t\tIter " << kv.first << ": (" << kv.second.value
                   << ", " << kv.second.iterIndex << ")\n";
    }
  }
}

void CMergeData::printDetail() {
  switchingDebugStream(SwitchingDebugCategory::Data) << "Node Name: " << nodeName << "\n";

  // Print originalDataOut
  switchingDebugStream(SwitchingDebugCategory::Data) << "\tOriginal Dataout:\n";
  for (const auto &kv : originalDataOut) {
    switchingDebugStream(SwitchingDebugCategory::Data) << "\t\tIter " << kv.first << ": (" << kv.second.value << ", "
                 << kv.second.iterIndex << ")\n";
  }

  switchingDebugStream(SwitchingDebugCategory::Data) << "\tControl Dataout:\n";
  for (const auto &kv : controlDataOut) {
    switchingDebugStream(SwitchingDebugCategory::Data) << "\t\tIter " << kv.first << ": (" << kv.second.value << ", "
                 << kv.second.iterIndex << ")\n";
  }

  // Print mg_suc_node_dict, which is a map<mg_label, MgInfo>
  for (const auto &mgPair : cmergeSegmentSuccessorInfoMap) {
    const std::string &mgLabel = mgPair.first;
    const SegmentSuccessorInfo &info = mgPair.second;

    switchingDebugStream(SwitchingDebugCategory::Data) << "\tCFDFC/Segment Label: " << mgLabel << "\n";
    // Print "original"
    switchingDebugStream(SwitchingDebugCategory::Data) << "\t\tcontrol = [";
    for (size_t i = 0; i < info.original.size(); ++i) {
      switchingDebugStream(SwitchingDebugCategory::Data) << info.original[i];
      if (i + 1 < info.original.size())
        switchingDebugStream(SwitchingDebugCategory::Data) << ", ";
    }
    switchingDebugStream(SwitchingDebugCategory::Data) << "]\n";
    // Print "glitch"
    switchingDebugStream(SwitchingDebugCategory::Data) << "\t\tdata = [";
    for (size_t i = 0; i < info.glitch.size(); ++i) {
      switchingDebugStream(SwitchingDebugCategory::Data) << info.glitch[i];
      if (i + 1 < info.glitch.size())
        switchingDebugStream(SwitchingDebugCategory::Data) << ", ";
    }
    switchingDebugStream(SwitchingDebugCategory::Data) << "]\n";
    // Print data_width
    switchingDebugStream(SwitchingDebugCategory::Data) << "\t\tdata_width:\n";
    for (const auto &dw : info.dataWidthMap) {
      switchingDebugStream(SwitchingDebugCategory::Data) << "\t\t  " << dw.first << " => " << dw.second << "\n";
    }
  }

  // if this is a control merge node
  if (controlGlitchValues.size()) {
    switchingDebugStream(SwitchingDebugCategory::Data) << "\tControl glitch Dataout:\n\t";
    for (const auto &kv : controlGlitchValues) {
      switchingDebugStream(SwitchingDebugCategory::Data) << std::to_string(kv) << " ,";
    }
    switchingDebugStream(SwitchingDebugCategory::Data) << "\n";
  }
}

std::optional<int> CMergeData::getControlOutputIfPresent(
    unsigned selIter) const {
  auto it = controlDataOut.find(selIter);
  if (it == controlDataOut.end())
    return std::nullopt;
  return it->second.value;
}

int CMergeData::getControlOutput(unsigned selIter) {
  if (std::optional<int> value = getControlOutputIfPresent(selIter))
    return *value;
  // ControlMerge control output is event-based in this model. If the selected
  // iteration has no event, default to 0 rather than reusing an older value.
  return 0;
}

std::vector<std::string>
segCtrlMergeGlitchSuccSearch(SwitchingInfo &switchInfo, std::string startNode,
                             std::vector<std::string> &excludingList,
                             StringRef segLabel) {
  std::vector<std::string> glitchSuccNodeList;
  auto segBBs =
      llvm::ArrayRef<unsigned>(switchInfo.staticInfo.segToBBs[segLabel]);

  unsigned numBuffers = 0;
  std::vector<std::string> currentPath;

  // A) succOf: special-case the startNode so that its successors
  //    are filtered *only* by isExcluded + isDataBase (just like your
  //    original “initAdjList”), but for all other nodes apply the
  //    full suite of filters.
  auto succOf = [&](const std::string &n) {
    std::vector<std::string> out;
    if (n == startNode) {
      // only the two “root” filters here:
      for (auto &m : switchInfo.staticInfo.dataflowGraph->nodes[n]->sucs) {
        if (isExcluded(excludingList, m))
          continue;
        if (isDataBase(switchInfo, m))
          continue;
        out.push_back(m);
      }
      return out;
    }
    // …else, do the full filtering…
    for (auto &m : switchInfo.staticInfo.dataflowGraph->nodes[n]->sucs) {
      if (isExcluded(excludingList, m))
        continue;
      if (isDataBase(switchInfo, m))
        continue;
      if (std::find(currentPath.begin(), currentPath.end(), m) !=
          currentPath.end())
        continue;
      if (isInvalidBackedge(switchInfo, segLabel, n, m))
        continue;
      if (crossesCondPort(switchInfo, n, m))
        continue;
      if (bufferOutsideSeg(switchInfo, m, segBBs))
        continue;
      if (isEndNode(m, segLabel))
        continue;
      if (isMemController(m))
        continue;
      out.push_back(m);
    }
    return out;
  };

  // B) pre: push every node into currentPath so cycle-checks work
  //    but *don’t* record the startNode itself in glitchSuccNodeList
  auto pre = [&](const std::string &n) {
    currentPath.push_back(n);
    if (n == startNode)
      return; // <-- skip root itself

    if (isOpaqueBuffer(switchInfo, n))
      numBuffers++;
    if (numBuffers == 0) {
      glitchSuccNodeList.push_back(n);
    }
  };

  // C) post: exactly as before
  auto post = [&](const std::string &n) {
    if (!currentPath.empty() && currentPath.back() == n) {
      if (isOpaqueBuffer(switchInfo, n))
        numBuffers--;
      currentPath.pop_back();
    }
  };

  // now kick off DFS *from* the startNode itself
  std::vector<std::string> roots = {startNode};
  dfsWithEvents(roots, succOf, pre, post);

  return glitchSuccNodeList;
}

//===----------------------------------------------------------------------===//
//
// Helper function for debuging
//
//===----------------------------------------------------------------------===//
void printSegmentDataSourceNodes(const SegmentDataSourceNodes &segmentNodes) {
  switchingDebugStream(SwitchingDebugCategory::Data) << "[DEBUG] \t\tSegmentDataSourceNodes:\n";

  switchingDebugStream(SwitchingDebugCategory::Data) << "[DEBUG] \t\t  All: [";
  for (size_t i = 0; i < segmentNodes.all.size(); ++i) {
    switchingDebugStream(SwitchingDebugCategory::Data) << segmentNodes.all[i];
    if (i + 1 < segmentNodes.all.size()) {
      switchingDebugStream(SwitchingDebugCategory::Data) << ", ";
    }
  }
  switchingDebugStream(SwitchingDebugCategory::Data) << "]\n";

  switchingDebugStream(SwitchingDebugCategory::Data) << "[DEBUG] \t\t  Control: [";
  for (size_t i = 0; i < segmentNodes.control.size(); ++i) {
    switchingDebugStream(SwitchingDebugCategory::Data) << segmentNodes.control[i];
    if (i + 1 < segmentNodes.control.size()) {
      switchingDebugStream(SwitchingDebugCategory::Data) << ", ";
    }
  }
  switchingDebugStream(SwitchingDebugCategory::Data) << "]\n";

  switchingDebugStream(SwitchingDebugCategory::Data) << "[DEBUG] \t\t  Data: [";
  for (size_t i = 0; i < segmentNodes.data.size(); ++i) {
    switchingDebugStream(SwitchingDebugCategory::Data) << segmentNodes.data[i];
    if (i + 1 < segmentNodes.data.size()) {
      switchingDebugStream(SwitchingDebugCategory::Data) << ", ";
    }
  }
  switchingDebugStream(SwitchingDebugCategory::Data) << "]\n";

  switchingDebugStream(SwitchingDebugCategory::Data) << "[DEBUG] \t\t  Opaque Buffers: [";
  for (size_t i = 0; i < segmentNodes.opaque_buffers.size(); ++i) {
    switchingDebugStream(SwitchingDebugCategory::Data) << segmentNodes.opaque_buffers[i];
    if (i + 1 < segmentNodes.opaque_buffers.size()) {
      switchingDebugStream(SwitchingDebugCategory::Data) << ", ";
    }
  }
  switchingDebugStream(SwitchingDebugCategory::Data) << "]\n";
}

// 1) Print the muxToSrcNodeMap
// Format: {"mux_node_name" : {"control" : ctrlSrcName, "0" : srcName0, "1" :
// srcName1}}
void printMuxToSrcNodeMap(
    const llvm::StringMap<std::map<std::string, std::string>>
        &muxToSrcNodeMap) {
  if (muxToSrcNodeMap.empty()) {
    llvm::errs() << "[ERROR] \tmuxToSrcNodeMap is empty\n";

    return;
  }
  switchingDebugStream(SwitchingDebugCategory::Data) << "[DEBUG] \tmuxToSrcNodeMap:\n";
  for (const auto &muxEntry : muxToSrcNodeMap) {
    // muxEntry.first => mux node name
    // muxEntry.second => map from { "control", "0", "1" } to source node name
    const auto &muxNodeName = muxEntry.first();
    const auto &innerMap = muxEntry.second;

    switchingDebugStream(SwitchingDebugCategory::Data) << "[DEBUG] \t\tMux Node: " << muxNodeName << " => {\n";
    for (const auto &kv : innerMap) {
      switchingDebugStream(SwitchingDebugCategory::Data) << "[DEBUG] \t\t\t\"" << kv.first << "\" : \"" << kv.second
                   << "\",\n";
    }
    switchingDebugStream(SwitchingDebugCategory::Data) << "[DEBUG] \t\t}\n";
  }
}

// 2) Print the srcNodeToMuxMap
// Format: {"src_node_name" : [ (mux_node_name, portId), ... ]}
void printSrcNodeToMuxMap(
    const llvm::StringMap<std::vector<std::pair<std::string, unsigned>>>
        &srcNodeToMuxMap) {
  switchingDebugStream(SwitchingDebugCategory::Data) << "[DEBUG] \tsrcNodeToMuxMap:\n";
  for (const auto &srcEntry : srcNodeToMuxMap) {
    // srcEntry.first => source node name
    // srcEntry.second => vector of pairs
    const auto &srcNodeName = srcEntry.first();
    const auto &muxList = srcEntry.second;

    switchingDebugStream(SwitchingDebugCategory::Data) << "[DEBUG] \t\tSource Node: " << srcNodeName << " => [";
    for (size_t i = 0; i < muxList.size(); ++i) {
      const auto &pairVal = muxList[i];
      switchingDebugStream(SwitchingDebugCategory::Data) << "(" << pairVal.first << ", " << pairVal.second << ")";
      if (i + 1 < muxList.size()) {
        switchingDebugStream(SwitchingDebugCategory::Data) << ", ";
      }
    }
    switchingDebugStream(SwitchingDebugCategory::Data) << "]\n";
  }
}
