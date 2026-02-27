//===- DataChannelCal.cpp - Estimate Swithicng Activities ------*- C++ -*-===//
//
// This file declares all functions used for data channel switching calculation
//
//===----------------------------------------------------------------------===//

#include "experimental/Analysis/SwitchingEstimation/DataChannelCal.h"
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

namespace {
bool getLatestKnownValue(const std::map<unsigned, IterationValue> &omap,
                         unsigned iterIndex, int &valueOut) {
  if (omap.empty())
    return false;
  auto it = omap.upper_bound(iterIndex);
  if (it == omap.begin())
    return false;
  --it;
  valueOut = it->second.value;
  return true;
}

// Keep meaningful toggles while removing adjacent duplicates
// (e.g. [0,0,1,1,0] -> [0,1,0]).
void dedupConsecutiveValues(std::vector<int> &values) {
  values.erase(std::unique(values.begin(), values.end()), values.end());
}

// Safely read a node's profiled output value at `iter`, falling back to the
// latest known value not newer than `iter` (or `fallback` if unavailable).
int getDataOutValueAtOrBefore(const std::shared_ptr<DataBase> &nodeData,
                              unsigned iter, int fallback = 0) {
  if (!nodeData) {
    llvm::dbgs() << "[WARNING] Node data is null, return fallback " << fallback
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
  auto updateBufferFromSource = [&](const std::string &srcNode, int value,
                                    unsigned iterIndex) {
    auto writeSingleBuffer = [&](const std::string &srcNodeName,
                                 const std::string &bufferNode,
                                 unsigned targetIter, int newValue) {
      auto bufferDbIt = switchInfo.dataInfo.nodeToDataState.find(bufferNode);
      if (bufferDbIt == switchInfo.dataInfo.nodeToDataState.end() ||
          !bufferDbIt->second) {
        return;
      }

      bool hasExisting =
          contains(bufferDbIt->second->originalDataOut, targetIter);

      if (!hasExisting) {
        bufferDbIt->second->originalDataOut[targetIter] = {newValue,
                                                           targetIter};
      }
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

      writeSingleBuffer(currentSrcNode, bufferNode, targetIter, value);

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
        // Create the value struct
        IterationValue tmpValuePair = {std::abs(value), iterIdx};
        switchInfo.dataInfo.nodeToDataState[selNode]->originalDataOut[iterIdx] =
            tmpValuePair;
        updateBufferFromSource(selNode, tmpValuePair.value, iterIdx);
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

      // Update the value in the vec list
      IterationValue tmpValuePair = {constantValue, 0};
      switchInfo.dataInfo.nodeToDataState[selNode]->originalDataOut[0] =
          tmpValuePair;
      updateBufferFromSource(selNode, tmpValuePair.value, 0);

      //! Testing
      // switchInfo.dataInfo.nodeToDataState[selNode]->printDetail();
    } else if (contains(selNode, "source")) {
      IterationValue tmpValuePair = {0, 0};
      switchInfo.dataInfo.nodeToDataState[selNode]->originalDataOut[0] =
          tmpValuePair;
      updateBufferFromSource(selNode, tmpValuePair.value, 0);
    } else {
      for (const auto &[value, iterIdx] :
           profileResults.nodeToValueTrace[selNode]) {
        IterationValue tmpValuePair = {value, iterIdx};
        switchInfo.dataInfo.nodeToDataState[selNode]->originalDataOut[iterIdx] =
            tmpValuePair;
        updateBufferFromSource(selNode, tmpValuePair.value, iterIdx);
      }

      //! Testing
      // switchInfo.dataInfo.nodeToDataState[selNode]->printDetail();
    }
  }

  // Define temporary storing structure
  std::map<std::string, int> tmpMuxOutputMap;
  std::vector<std::string> allMuxNodes;
  allMuxNodes.reserve(
      switchInfo.staticInfo.dataflowGraph->muxToSrcNodeMap.size());
  for (const auto &muxEntry :
       switchInfo.staticInfo.dataflowGraph->muxToSrcNodeMap) {
    allMuxNodes.push_back(muxEntry.first().str());
  }

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

      // Iterate over all (CMNode, output value) pairs
      for (auto selValuePair : itCM->second) {
        auto nodeName = selValuePair.first;
        int outValue = selValuePair.second;
        // `executedBasicBlockTrace[i]` corresponds to edge-index `(i - 1)` in
        // the profiler log. Align the query index with
        // `edgeIndexToIterationMap` keys.
        unsigned iterIndex = edgeIterIndex;

        //! Testing
        // llvm::dbgs() << "[DEBUG] \t\tCtrlMerge Node: " << nodeName << "\n";
        // llvm::dbgs() << "[DEBUG] \t\t\tCon output Value: " << outValue <<
        // "\n"; llvm::dbgs() << "[DEBUG] \t\t\tIter Index: " << iterIndex <<
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
        if (!contains(selCtrlMergeNode->originalDataOut, iterIndex)) {
          IterationValue tmpDataValuePair = {-1, iterIndex};
          selCtrlMergeNode->originalDataOut[iterIndex] = tmpDataValuePair;
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

        // Build a dependency graph for the multiplexers
        // We may encounter different situation when updating the value for
        // MUX nodes
        for (const auto &selMuxNode : selMuxVec) {
          // Only update muxes this control-merge actually influences
          if (!containsValue(selMuxVec, selMuxNode))
            continue;
          std::string selMuxDataSrcNode;
          if (auto muxIt =
                  switchInfo.staticInfo.dataflowGraph->muxToSrcNodeMap.find(
                      selMuxNode);
              muxIt !=
              switchInfo.staticInfo.dataflowGraph->muxToSrcNodeMap.end()) {
            auto srcIt = muxIt->second.find(std::to_string(outValue));
            if (srcIt != muxIt->second.end())
              selMuxDataSrcNode = srcIt->second;
          }
          int tmpMuxOutput = 0;

          //! Testing
          // llvm::dbgs() << "[DEBUG] \t\t\t\tMux Node: " << selMuxNode <<
          // "\n"; llvm::dbgs() << "[DEBUG] \t\t\t\tSel Data Src Node: " <<
          // selMuxDataSrcNode << "\n";

          if (contains(selMuxDataSrcNode, "constant") ||
              contains(selMuxDataSrcNode, "source")) {
            auto srcDb = getNodeDataBase(selMuxDataSrcNode);
            if (srcDb && !srcDb->originalDataOut.empty())
              tmpMuxOutput = getDataOutValueAtOrBefore(srcDb, 0, 0);
            else
              tmpMuxOutput = 0;
          } else {
            auto srcDb = getNodeDataBase(selMuxDataSrcNode);
            if (!srcDb) {
              llvm::errs() << "[ERROR] Data source node " << selMuxDataSrcNode
                           << " not found for mux node " << selMuxNode << "\n";
              tmpMuxOutput = 0;
              IterationValue tmpMuxValuePair = {tmpMuxOutput, iterIndex};
              if (auto muxDb = getNodeDataBase(selMuxNode))
                muxDb->originalDataOut[iterIndex] = tmpMuxValuePair;
              tmpMuxOutputMap[selMuxNode] = tmpMuxOutput;
              continue;
            }
            // If the src node is a mux node
            if (contains(selMuxDataSrcNode, "mux")) {
              // Case 1: the source node is the mux node itself

              if (selMuxDataSrcNode == selMuxNode) {
                //! Testing
                // llvm::dbgs()
                //     << "[MUX_DEBUG][BASE] resolve_case=SELF_FEEDBACK\n";
                // Use the previous value
                if (contains(tmpMuxOutputMap, selMuxDataSrcNode)) {
                  tmpMuxOutput = tmpMuxOutputMap[selMuxDataSrcNode];
                } else {
                  llvm::errs()
                      << "[ERROR] Data src(" << selMuxDataSrcNode
                      << ") of Mux Node: " << selMuxNode << ", not found.\n";
                }
              } else if (contains(srcDb->originalDataOut, iterIndex)) {
                //! Testing
                // llvm::dbgs() << "[MUX_DEBUG][BASE]     "
                //                 "resolve_case=SRC_HAS_ITER_VALUE\n";
                tmpMuxOutput = srcDb->originalDataOut[iterIndex].value;
              } else if (contains(tmpMuxOutputMap, selMuxDataSrcNode)) {
                //! Testing
                // llvm::dbgs()
                //     << "[MUX_DEBUG][BASE] resolve_case=SRC_FROM_TMP_MAP\n";
                tmpMuxOutput = tmpMuxOutputMap[selMuxDataSrcNode];
              } else {
                //! Testing
                // llvm::dbgs() << "[MUX_DEBUG][BASE]     "
                //                 "resolve_case=DEFERRED_FROM_PREV_ITER\n";
                // TODO: Check the following update logic
                unsigned fallbackIter = (iterIndex == 0) ? 0 : iterIndex - 1;
                IterationValue tmpInnerMuxValuePair = {tmpMuxOutput,
                                                       fallbackIter};
                srcDb->originalDataOut[fallbackIter] = tmpInnerMuxValuePair;
                tmpUnUpdatedMux.push_back(selMuxNode);
                continue;
              }
            } else {
              if (selMuxDataSrcNode.empty()) {
                llvm::errs()
                    << "[ERROR] The data src node for Mux Node: " << selMuxNode
                    << " is empty\n";
                // Unconnected , use default vlue
                tmpMuxOutput = 0;
              } else {
                auto &omap = srcDb->originalDataOut;
                if (omap.find(iterIndex) != omap.end()) {
                  tmpMuxOutput = omap[iterIndex].value;
                } else if (contains(tmpMuxOutputMap, selMuxDataSrcNode)) {
                  tmpMuxOutput = tmpMuxOutputMap[selMuxDataSrcNode];
                } else if (getLatestKnownValue(omap, iterIndex, tmpMuxOutput)) {
                  //! Testing
                  // llvm::dbgs() << "[DEBUG] The value for src node " <<
                  // selMuxDataSrcNode << " not found at iteration " <<
                  // iterIndex
                  //              << ", fallback to latest known value\n";
                } else if (contains(selMuxDataSrcNode, "buffer")) {
                  auto srcBufDb = getNodeDataBase(selMuxDataSrcNode);
                  if (srcBufDb &&
                      getLatestKnownValue(srcBufDb->originalDataOut, iterIndex,
                                          tmpMuxOutput)) {
                    // llvm::dbgs()
                    //     << "[DEBUG] The value for buffer source node "
                    //        "not found at iteration "
                    //     << iterIndex << ", fallback to latest known value\n";
                  } else {
                    // If no value is found, we use 0 for the buffer source.
                    tmpMuxOutput = 0;
                  }
                } else {
                  llvm::errs()
                      << "[ERROR] Data src(" << selMuxDataSrcNode
                      << ") of Mux Node: " << selMuxNode << ", not found.\n";
                  tmpMuxOutput = 0;
                }
              }
            }
          }

          // Update the storing structure and the tmp dict
          //! Testing
          // llvm::dbgs() << "[DEBUG] \t\t\t\t[OutValue] " << tmpMuxOutput <<
          // "\n";

          IterationValue tmpMuxValuePair = {tmpMuxOutput, iterIndex};
          if (auto muxDb = getNodeDataBase(selMuxNode))
            if (!contains(muxDb->originalDataOut, iterIndex)) {
              muxDb->originalDataOut[iterIndex] = tmpMuxValuePair;
            }
          tmpMuxOutputMap[selMuxNode] = tmpMuxOutput;
          updateBufferFromSource(selMuxNode, tmpMuxValuePair.value, iterIndex);
        }

        // Update all remaining mux nodes
        // TODO: This maybe useless, check it out
        for (const auto &leftMuxNode : tmpUnUpdatedMux) {
          std::string selMuxDataSrcNode;
          if (auto muxIt =
                  switchInfo.staticInfo.dataflowGraph->muxToSrcNodeMap.find(
                      leftMuxNode);
              muxIt !=
              switchInfo.staticInfo.dataflowGraph->muxToSrcNodeMap.end()) {
            auto srcIt = muxIt->second.find(std::to_string(outValue));
            if (srcIt != muxIt->second.end())
              selMuxDataSrcNode = srcIt->second;
          } else {
            llvm::errs() << "[ERROR] Mux node " << leftMuxNode
                         << " not found in muxToSrcNodeMap\n";
          }
          int tmpMuxOutput = 0;

          if (contains(tmpMuxOutputMap, selMuxDataSrcNode)) {
            tmpMuxOutput = tmpMuxOutputMap[selMuxDataSrcNode];
          } else {
            llvm::errs() << "[ERROR] Data src(" << selMuxDataSrcNode
                         << ") of Mux Node: " << leftMuxNode
                         << ", not found in tmpMuxOutputMap.\n";
          }

          IterationValue tmpMuxValuePair = {tmpMuxOutput, iterIndex};
          if (auto leftDb = getNodeDataBase(leftMuxNode))
            if (!contains(leftDb->originalDataOut, iterIndex))
              leftDb->originalDataOut[iterIndex] = tmpMuxValuePair;
          updateBufferFromSource(leftMuxNode, tmpMuxValuePair.value, iterIndex);
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
      }
    }
  }
}

void buildSegmentSuccNodesList(SwitchingInfo &switchInfo,
                               SCFProfilingResult &profileResults) {
  //! Testing
  for (const auto &[label, bblist] : switchInfo.staticInfo.segToBBs) {
    llvm::dbgs() << "[DEBUG] \t\tSeg Label: " << label << "\n";
    llvm::dbgs() << "[DEBUG] \t\t\tBB List: ";
    for (const auto &selBB : bblist) {
      llvm::dbgs() << selBB << ", ";
    }
    llvm::dbgs() << "\n";
  }

  for (const auto &selBaseNode :
       switchInfo.staticInfo.dataflowGraph->allDataBaseNode) {
    //! Testing
    // llvm::dbgs() << "[DEBUG] Node Name: " << selBaseNode << "\n";

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
      llvm::dbgs() << "[DEBUG] \tControl Merge Node: " << selBaseNode << "\n";
      llvm::dbgs() << "[DEBUG] \t\tData Suc Node: " << dataSucNode << "\n";
      llvm::dbgs() << "[DEBUG] \t\tControl Suc Node: " << conSucNode << "\n";

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
    llvm::dbgs() << "[DEBUG] \tCFDFC Label: " << label << "\n";
    llvm::dbgs() << "[DEBUG] \t\tCFDFC Base Node: " << mgBaseNode << "\n";
    llvm::dbgs() << "[DEBUG] \t\tCFDFC II: " << mgII << "\n";

    //
    std::map<std::string, std::vector<NodeGlitchInfo>> tmpGlitchDict;
    std::vector<std::string> tmpGlitchNodes;

    // Iterate over all mapped nodes in the CFDFC
    for (const auto &selNode : orderedALUs) {
      llvm::dbgs() << "[DEBUG] \t\tALU Node: " << selNode << "\n";

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
      // llvm::dbgs() << "[DEBUG] \tNode : " << selNode << "\n";
      // llvm::dbgs() << "[DEBUG] \t\tsrcNode1 : " << srcNode1 << "\n";
      // llvm::dbgs() << "[DEBUG] \t\tsrcNode2 : " << srcNode2 << "\n";

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
      // llvm::dbgs() << "\t\tPre_Node_1: " << preNode1 << ", Src_Node_1: " <<
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
      //   llvm::dbgs() << n << " ";
      // llvm::dbgs()  << "\n";

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

static bool shouldRunGlitchUpdateSegment(const std::string &seg) {
  return !(contains(seg, "S") || contains(seg, "E") || contains(seg, "T"));
}

static void updateALUGlitchValuesForSegment(
    SwitchingInfo &switchInfo, const std::string &executedSeg, unsigned iter,
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
          llvm::dbgs() << "[DEBUG] \t\tGlitch Node: " << selNode << "\n"
                       << "[DEBUG] \t\t\tPre_src_1: " << preSrc1 << "\n"
                       << "[DEBUG] \t\t\tPre_src_2: " << preSrc2 << "\n";
        }

        // If this is the last iteration, we ignore the glitching value,
        // just copy the original value
        if (iter == segExecTrace.size() - 1) {
          int finalValue = getDataOutValueAtOrBefore(
              switchInfo.dataInfo.nodeToDataState[selNode], iter, 0);
          //! Testing
          if (debug) {

            llvm::dbgs()
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

              llvm::dbgs() << "[DEBUG] \t\t\t[Value 1]: \n"
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
              op1 = switchInfo.dataInfo.nodeToDataState[fasterNode]
                        ->originalDataOut[op1PreIndex]
                        .value;
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
                llvm::dbgs() << "[DEBUG] \t\t\t[Value 1]: \n"
                             << "[DEBUG] \t\t\t\t[Warning] S Last Active Iter "
                                "Not in the corresponding storing structure\n";
              }
              op2 = 0;
            } else {
              op2 = switchInfo.dataInfo.nodeToDataState[slowerNode]
                        ->originalDataOut[op2PreIndex]
                        .value;
            }

            tmpValue.push_back(calGlitchValue(op1, op2, selNode));

            //! Testing
            if (debug) {

              llvm::dbgs() << "[DEBUG] \t\t\t\tFaster Node: " << fasterNode
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
            llvm::dbgs() << "[DEBUG] \t\t\t[Value 2]: \n";
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
              op1 = switchInfo.dataInfo.nodeToDataState[fasterNode]
                        ->originalDataOut[op1PreIndex]
                        .value;
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
                op1 = switchInfo.dataInfo.nodeToDataState[fasterNode]
                          ->originalDataOut[op1PreIndex]
                          .value;
            } else {
              op1 = getDataOutValueAtOrBefore(
                  switchInfo.dataInfo.nodeToDataState[fasterNode], iter, 0);
            }
          } else if (contains(switchInfo.dataInfo.glitches[executedSeg],
                              fasterNode)) {
            // If the faster node is glitching, we check whether the node
            // is buffered or not
            if (srcBufferedDict[fasterNode]) {
              // Buffered
              op1 = getDataOutValueAtOrBefore(
                  switchInfo.dataInfo.nodeToDataState[fasterNode], iter, 0);
            } else {
              auto &glitchvec = switchInfo.dataInfo.nodeToDataState[fasterNode]
                                    ->glitchDataOutByIter[iter];
              // ! If the list is smaller than 2, something is wrong
              int op1 = 0;
              if (glitchvec.size() >= 2) {
                op1 = glitchvec[glitchvec.size() - 2];
              } else if (!glitchvec.empty()) {
                llvm::errs() << "[ERROR] Glitch Vector for " << selNode
                             << " has a size smaller than 2\n";
                op1 = glitchvec.back();
              } else { // only one glitch value – use it
                llvm::errs() << "[ERROR] Use original dataout\n";
                op1 = getDataOutValueAtOrBefore(
                    switchInfo.dataInfo.nodeToDataState[fasterNode], iter,
                    0); // no glitches recorded
              }
            }
          } else {
            op1 = getDataOutValueAtOrBefore(
                switchInfo.dataInfo.nodeToDataState[fasterNode], iter, 0);
          }

          // Get the value of op2
          if (contains(switchInfo.dataInfo.nodeToDataState[slowerNode]
                           ->originalDataOut,
                       op2PreIndex)) {
            op2 = switchInfo.dataInfo.nodeToDataState[slowerNode]
                      ->originalDataOut[op2PreIndex]
                      .value;
          } else {
            op2 = getDataOutValueAtOrBefore(
                switchInfo.dataInfo.nodeToDataState[slowerNode], iter, 0);
          }

          tmpValue.push_back(calGlitchValue(op1, op2, selNode));

          //! Testing
          if (debug) {

            llvm::dbgs() << "[DEBUG] \t\t\t\tFaster Node: " << fasterNode
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
                op1 = switchInfo.dataInfo.nodeToDataState[fasterNode]
                          ->originalDataOut[op1PreIndex]
                          .value;
              }
            } else {
              getDataOutValueAtOrBefore(
                  switchInfo.dataInfo.nodeToDataState[fasterNode], iter, 0);
            }
          } else {
            op1 = getDataOutValueAtOrBefore(
                switchInfo.dataInfo.nodeToDataState[fasterNode], iter, 0);
          }

          // Check whether the slower node is a mux node
          // TODO: Check the condition below
          op2 = getDataOutValueAtOrBefore(
              switchInfo.dataInfo.nodeToDataState[slowerNode], iter, 0);

          tmpValue.push_back(calGlitchValue(op1, op2, selNode));

          if (debug) {
            llvm::dbgs() << "\t[Value 3]: \n"
                         << "\t\tOp_1: " << op1 << "\n"
                         << "\t\tOp_2: " << op2 << "\n"
                         << "\t\t[FINAL] ";
            for (auto v : tmpValue)
              llvm::dbgs() << v << " ";
            llvm::dbgs() << "\n";
          }
        }
      } else {
        tmpValue.push_back(getDataOutValueAtOrBefore(
            switchInfo.dataInfo.nodeToDataState[selNode], iter, 0));
      }
    } else {
      // No neeed for glitch value calculation, we directly copy the ori
      // data
      tmpValue.push_back(getDataOutValueAtOrBefore(
          switchInfo.dataInfo.nodeToDataState[selNode], iter, 0));
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
    const std::map<std::string, std::string> &muxSrcMap, int muxCondValue,
    bool glitchUpdateFlag, bool debug, std::vector<int> &outValues) {
  std::string selDataSrcNode = "";
  auto dataSrcIt = muxSrcMap.find(std::to_string(muxCondValue));
  if (dataSrcIt != muxSrcMap.end())
    selDataSrcNode = dataSrcIt->second;

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
  } else if (contains(selDataSrcNode, "constant") ||
             contains(selDataSrcNode, "source") ||
             contains(selDataSrcNode, "start")) {
    int constantLikeValue = switchInfo.dataInfo.nodeToDataState[selDataSrcNode]
                                ->originalDataOut[0]
                                .value;

    outValues.push_back(constantLikeValue);
  } else if (selDataSrcNode == selMuxNode) {
    // The src node is the selected node itself
    unsigned preIndex =
        switchInfo.dataInfo.nodeToDataState[selDataSrcNode]->lastUpdateIndex;

    // Check whether the preIndex exists or not
    if (contains(switchInfo.dataInfo.nodeToDataState[selDataSrcNode]
                     ->originalDataOut,
                 preIndex)) {
      int selfValue = switchInfo.dataInfo.nodeToDataState[selDataSrcNode]
                          ->originalDataOut[preIndex]
                          .value;

      outValues.push_back(selfValue);
    } else {
      outValues.push_back(0);
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
      const unsigned int cond_add1 = static_cast<unsigned>(muxCondValue + 1);
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
      llvm::dbgs() << "\t\tCur src node has glitches, buffered: "
                   << bufferedFlag << "\n";
    }

    if (bufferedFlag) {
      // Src glitching but buffered
      int bufferedValue = getDataOutValueAtOrBefore(
          switchInfo.dataInfo.nodeToDataState[selDataSrcNode], iter, 0);

      outValues.push_back(bufferedValue);
    } else {
      auto &glitchValues = switchInfo.dataInfo.nodeToDataState[selDataSrcNode]
                               ->glitchDataOutByIter[iter];

      outValues.insert(outValues.end(), glitchValues.begin(),
                       glitchValues.end());
    }
  } else if (glitchUpdateFlag && contains(selDataSrcNode, "mux")) {
    // Cascaded mux source: reuse upstream mux event output when
    // available.
    std::string preNode = "";
    auto selMuxNodeStructure = dyn_cast<MuxNode>(
        switchInfo.staticInfo.dataflowGraph->nodes[selMuxNode].get());
    for (const auto &[nodeName, portIdx] :
         selMuxNodeStructure->preNameToPortIdxMap) {
      const unsigned int condAdd1 = static_cast<unsigned>(muxCondValue + 1);
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
    if (bufferedFlag ||
        glitchIt == switchInfo.dataInfo.nodeToDataState[selDataSrcNode]
                        ->glitchDataOutByIter.end() ||
        glitchIt->second.empty()) {
      int cascadedStableValue = getDataOutValueAtOrBefore(
          switchInfo.dataInfo.nodeToDataState[selDataSrcNode], iter, 0);

      outValues.push_back(cascadedStableValue);
    } else {
      outValues.insert(outValues.end(), glitchIt->second.begin(),
                       glitchIt->second.end());
    }
  } else {
    // Handling special case for cascaded Muxes
    int stableValue = getDataOutValueAtOrBefore(
        switchInfo.dataInfo.nodeToDataState[selDataSrcNode], iter, 0);

    outValues.push_back(stableValue);
  }
}

static void updateMuxGlitchValuesForSegment(
    SwitchingInfo &switchInfo, const std::string &executedSeg, unsigned iter,
    bool glitchUpdateFlag, bool debug) {
  // ====================================================================
  // Step 2: Update value for all MUXs
  // ====================================================================
  const auto &activeMuxNodes =
      switchInfo.dataInfo.segmentControlNodes[executedSeg].muxNodes;
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

    int condValue = 0;
    if (!selCondInputNode.empty() &&
        contains(switchInfo.dataInfo.nodeToDataState, selCondInputNode)) {
      if (auto *tmpCMBaseNode = dyn_cast<CMergeData>(
              switchInfo.dataInfo.nodeToDataState[selCondInputNode].get()))
        condValue = tmpCMBaseNode->getControlOutput(iter);
    }

    // Temp Value vector definition
    std::vector<int> preValue, curValue, nexValue;

    //! Testing
    if (debug) {
      std::string selDataSrcNode = "";
      auto dataSrcIt = muxSrcIt->second.find(std::to_string(condValue));
      if (dataSrcIt != muxSrcIt->second.end())
        selDataSrcNode = dataSrcIt->second;

      llvm::dbgs() << "Mux Node: " << selMuxNode << "\n"
                   << "\t[CUR_VALUE]\n"
                   << "\t\tCond Node: " << selCondInputNode << "\n"
                   << "\t\tCond_value: " << condValue << "\n"
                   << "\t\tCur data src: " << selDataSrcNode << "\n";
    }

    appendMuxValuesForCond(switchInfo, executedSeg, iter, selMuxNode,
                           muxSrcIt->second, condValue, glitchUpdateFlag, debug,
                           curValue);

    //! Testing
    if (debug) {
      llvm::dbgs() << "\t\tCur_value: ";
      for (auto v : curValue)
        llvm::dbgs() << v << " ";
      llvm::dbgs() << "\n\t[TRANSATION GLITCHES]\n";
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
    //     llvm::dbgs() << "\t\tNode Pre MG: " << nodePreValidSeg << "\n"
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
    //           switchInfo.dataInfo.segmentControlNodes[segExecTrace[iter + 1]]
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
    //     llvm::dbgs() << "\t\tDouble transition: " << doubleTransFlag << "\n"
    //                  << "\t\tPre_value: ";
    //     for (auto v : preValue)
    //       llvm::dbgs() << v << " ";
    //     llvm::dbgs() << "\n\t\tNext value: ";
    //     for (auto v : nexValue)
    //       llvm::dbgs() << v << " ";
    //     llvm::dbgs() << "\n";
    //   }
    // } else {
    switchInfo.dataInfo.nodeToDataState[selMuxNode]->skipControlCal = false;
    // }

    // Get the final mux output data list
    std::vector<int> finalMuxOutputList;

    // If we have glitches as the mux is inactive for sometime in this iteration
    if (contains(switchInfo.dataInfo.iterToInactiveMuxNodes, iter) &&
        containsValue(switchInfo.dataInfo.iterToInactiveMuxNodes[iter],
                      selMuxNode)) {
      finalMuxOutputList.push_back(0);
    }

    for (const auto &selValue : preValue)
      finalMuxOutputList.push_back(selValue);
    for (const auto &selValue : curValue)
      finalMuxOutputList.push_back(selValue);
    for (const auto &selValue : nexValue)
      finalMuxOutputList.push_back(selValue);

    dedupConsecutiveValues(finalMuxOutputList);

    //! Testing
    if (debug) {
      llvm::dbgs() << "\t\t[FINAL MUX OUTPUT] ";
      for (auto v : finalMuxOutputList)
        llvm::dbgs() << v << " ";
      llvm::dbgs() << "\n";
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
    llvm::dbgs() << "[DEBUG]\n[DEBUG] \t\t[NODE GLITCHING VALUE CALCULATION]\n";
  }
  //
  auto segExecTrace = profileResults.executedSegmentTrace;

  // Iterate through the execution trace
  for (unsigned i = 0; i < segExecTrace.size(); i++) {
    std::string executedSeg = segExecTrace[i];

    if (debug) {
      llvm::dbgs() << "[DEBUG] \t******** Iter: " << i
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
  int condValue = 1;

  // Defensive check for nodeToDataState
  if (!contains(switchInfo.dataInfo.nodeToDataState, condSrcNode)) {
    llvm::errs() << "[ERROR] Warning: condSrcNode " << condSrcNode
                 << " not found in nodeToDataState\n";
    return 0; // Default value
  }

  const int rawCond = getDataOutValueAtOrBefore(
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
      llvm::dbgs() << "[DEBUG] ==================================\n";

    // Propagate data values for all non_memory related nodes
    for (const auto &selNode :
         switchInfo.dataInfo.segmentToDataSourceNodes[executedSeg].all) {
      //! Testing
      if (debug)
        llvm::dbgs() << "[DEBUG] Node: " << selNode << "\n";

      // Case 1: Mapped Nodes --> All ALUs
      if (containsValue(
              switchInfo.dataInfo.segmentToOrderedAluNodes[executedSeg],
              selNode)) {
        //! Testing
        if (debug)
          llvm::dbgs() << "[DEBUG] ALU Node Detected \n";

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
            llvm::dbgs() << "[DEBUG] \t[Glitch Output]\n";

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
              llvm::dbgs() << "[DEBUG] \t\tSucNode: " << selSucNode << "\n";

            // Get the node bitwidth
            unsigned tmpNodeWidth = resolveOutWidth(selNode, selSucNode);

            //! Testing
            if (debug)
              llvm::dbgs() << "[DEBUG] \t\t\tDataWidth: " << tmpNodeWidth
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
            llvm::dbgs() << "[DEBUG] \t[DONE]\n";
        }

        // Step 3: Update ori succeeding node list
        //! Testing
        if (debug)
          llvm::dbgs() << "[DEBUG] \t[Ori Output]\n";
        for (const auto &selSucNode :
             switchInfo.dataInfo.nodeToDataState[selNode]
                 ->segmentSuccessorInfoMap[executedSeg]
                 .original) {
          // Get Node Bitwidth
          unsigned tmpNodeWidth = resolveOutWidth(selNode, selSucNode);

          //! Testing
          if (debug)
            llvm::dbgs() << "[DEBUG] \t\tSuc Node: " << selSucNode << "\n";

          //
          if (contains(selSucNode, "store")) {
            //
            auto selStoreNode = dyn_cast<DStoreNode>(
                switchInfo.staticInfo.dataflowGraph->nodes[selSucNode].get());
            selStoreNode->updateDataout(
                reduceBits(
                    getDataOutValueAtOrBefore(
                        switchInfo.dataInfo.nodeToDataState[selNode], i, 0),
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
                reduceBits(
                    getDataOutValueAtOrBefore(
                        switchInfo.dataInfo.nodeToDataState[selNode], i, 0),
                    tmpNodeWidth),
                tmpCondValue);
          } else {
            switchInfo.staticInfo.dataflowGraph->nodes[selSucNode]
                ->updateDataoutChannel(reduceBits(
                    getDataOutValueAtOrBefore(
                        switchInfo.dataInfo.nodeToDataState[selNode], i, 0),
                    tmpNodeWidth));
          }
        }
        //! Testing
        if (debug)
          llvm::dbgs() << "[DEBUG] \t[DONE]\n";
      } else {
        // Case 2: This is a control merge node
        if (contains(selNode, "control_merge")) {
          //! Testing
          if (debug)
            llvm::dbgs() << "[DEBUG] Control Merge Detected \n";

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
            llvm::dbgs() << "[DEBUG] \t[Control Output]\n";

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
              llvm::dbgs() << "[DEBUG] \t\tSuc Node: " << selSucNode << "\n";

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
            llvm::dbgs() << "[DEBUG] \t[DONE]\n";

          // Step 3: Update all data channel values
          //! Testing
          if (debug)
            llvm::dbgs() << "[DEBUG] \t[Data Output]\n";
          for (const auto &selSucNode :
               selCMNode->cmergeSegmentSuccessorInfoMap[executedSeg].data) {
            //! Testing
            if (debug)
              llvm::dbgs() << "[DEBUG] \t\tSuc Node: " << selSucNode << "\n";

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
            llvm::dbgs() << "[DEBUG] \t[DONE]\n";
        } else if (contains(selNode, "mux")) {
          // Case 3: MUX Node
          // In the last iteration or MG transitions, we take the
          // ori_glitch_data out value
          //! For all transition related nodes, we need to check the iter 0 as
          //! well Testing
          if (debug)
            llvm::dbgs() << "[DEBUG] MUX Detected \n";

          // Defensive check for MUX node
          if (!contains(switchInfo.dataInfo.nodeToDataState, selNode)) {
            llvm::dbgs() << "[DEBUG] Warning: MUX node " << selNode
                         << " not found in nodeToDataState at iteration " << i
                         << "\n";
            continue;
          }

          for (const auto &selValue :
               switchInfo.dataInfo.nodeToDataState[selNode]
                   ->glitchDataOutByIter[i]) {
            switchInfo.staticInfo.dataflowGraph->nodes[selNode]
                ->updateDataoutChannel(selValue);

            // Step 2: Update the glitching succeeding list
            //! Testing
            if (debug)
              llvm::dbgs() << "[DEBUG] \t[Glitch Output]\n";

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
                llvm::dbgs() << "[DEBUG] \t\tSuc Node: " << selSucNode << "\n";

              // Get the node output bitwidth
              unsigned tmpNodeWidth = resolveOutWidth(selNode, selSucNode);

              if (contains(selSucNode, "cond_br")) {
                // Get the node
                auto selCondStoreNode = dyn_cast<CBrNode>(
                    switchInfo.staticInfo.dataflowGraph->nodes[selSucNode]
                        .get());

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
              llvm::dbgs() << "[DEBUG] \t[DONE]\n";
          }

          // Step 3: Update the ori succeeding node list
          //! Testing
          if (debug)
            llvm::dbgs() << "[DEBUG] \t[Ori Output]\n";

          for (const auto &selSucNode :
               switchInfo.dataInfo.nodeToDataState[selNode]
                   ->segmentSuccessorInfoMap[executedSeg]
                   .original) {
            //! Testing
            if (debug)
              llvm::dbgs() << "[DEBUG] \t\tSuc Node: " << selSucNode << "\n";
            // Get bitwidth
            unsigned tmpNodeWidth = resolveOutWidth(selNode, selSucNode);

            //
            if (contains(switchInfo.dataInfo.nodeToDataState[selNode]
                             ->originalDataOut,
                         i)) {
              if (contains(selSucNode, "store")) {
                auto selStoreNode = dyn_cast<DStoreNode>(
                    switchInfo.staticInfo.dataflowGraph->nodes[selSucNode]
                        .get());
                selStoreNode->updateDataout(
                    reduceBits(switchInfo.dataInfo.nodeToDataState[selNode]
                                   ->originalDataOut[i]
                                   .value,
                               tmpNodeWidth),
                    selNode);
              } else if (contains(selSucNode, "cond_br")) {
                auto selCondBrNode = dyn_cast<CBrNode>(
                    switchInfo.staticInfo.dataflowGraph->nodes[selSucNode]
                        .get());
                int tmpCondValue =
                    getCondBrNodeCondValue(switchInfo, i, selSucNode);

                selCondBrNode->updateDataout(
                    reduceBits(switchInfo.dataInfo.nodeToDataState[selNode]
                                   ->originalDataOut[i]
                                   .value,
                               tmpNodeWidth),
                    tmpCondValue);
              } else {
                switchInfo.staticInfo.dataflowGraph->nodes[selSucNode]
                    ->updateDataoutChannel(
                        reduceBits(switchInfo.dataInfo.nodeToDataState[selNode]
                                       ->originalDataOut[i]
                                       .value,
                                   tmpNodeWidth));
              }
            }
          }
          //! Testing
          if (debug)
            llvm::dbgs() << "[DEBUG] \t[DONE]\n";
        } else if (contains(selNode, "load")) {
          //! Testing
          if (debug)
            llvm::dbgs() << "[DEBUG] LOAD Detected \n";
          // Case 4: load node
          pendingMemList.push_back(selNode);
          continue;
        } else {
          //! Testing
          if (debug)
            llvm::dbgs() << "[DEBUG] Other Nodes Detected\n";
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
            } else if (!contains(switchInfo.dataInfo.nodeToDataState[selNode]
                                     ->originalDataOut,
                                 i)) {
              llvm::errs() << "[ERROR] Warning: No data found for node "
                           << selNode << " at iteration " << i << "\n";
              selValue = 0; // Default value
            } else {
              selValue = switchInfo.dataInfo.nodeToDataState[selNode]
                             ->originalDataOut[i]
                             .value;
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
            // llvm::dbgs() << "[DEBUG] Warning: segment " << executedSeg << "
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
            llvm::dbgs() << "[DEBUG] [DONE]\n";
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
    llvm::dbgs() << "[ERROR] Could not find base node for " << startNode
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
  auto selMGII = switchInfo.staticInfo.cfdfcIIs[std::stoul(mgLabel)];

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
  const auto &selMuxSrcMap =
      switchInfo.staticInfo.dataflowGraph->muxToSrcNodeMap;

  // Check if the mux node exists in the map
  auto muxIt = selMuxSrcMap.find(selMuxNode);
  if (muxIt == selMuxSrcMap.end()) {
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

  int ctrlInValue = selCMBaseNode->getControlOutput(selIter);

  // Check if the control value key exists
  auto dataIt = muxIt->second.find(std::to_string(ctrlInValue));
  if (dataIt == muxIt->second.end()) {
    llvm::errs() << "[ERROR] Data input for control value " << ctrlInValue
                 << " not found for mux node " << selMuxNode << "\n";
    return ""; // Return empty string on error
  }

  return dataIt->second;
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
  llvm::dbgs() << "Node Name: " << nodeName << "\n";

  // Print originalDataOut
  llvm::dbgs() << "\tOriginal Dataout:\n";
  for (const auto &kv : originalDataOut) {
    llvm::dbgs() << "\t\tIter " << kv.first << ": (" << kv.second.value << ", "
                 << kv.second.iterIndex << ")\n";
  }

  // Print mg_suc_node_dict, which is a map<mg_label, MgInfo>
  for (const auto &mgPair : segmentSuccessorInfoMap) {
    const auto &mgLabel = mgPair.first;
    const SegmentSuccessorInfo &info = mgPair.second;

    llvm::dbgs() << "\tSegment Label: " << mgLabel << "\n";
    // Print "original"
    llvm::dbgs() << "\t\toriginal = [";
    for (size_t i = 0; i < info.original.size(); ++i) {
      llvm::dbgs() << info.original[i];
      if (i + 1 < info.original.size())
        llvm::dbgs() << ", ";
    }
    llvm::dbgs() << "]\n";
    // Print "glitch"
    llvm::dbgs() << "\t\tglitch = [";
    for (size_t i = 0; i < info.glitch.size(); ++i) {
      llvm::dbgs() << info.glitch[i];
      if (i + 1 < info.glitch.size())
        llvm::dbgs() << ", ";
    }
    llvm::dbgs() << "]\n";
    // Print data_width
    llvm::dbgs() << "\t\tdata_width:\n";
    for (const auto &dw : info.dataWidthMap) {
      llvm::dbgs() << "\t\t  " << dw.first << " => " << dw.second << "\n";
    }
  }

  // if this is a control merge node
  if (controlDataOut.size()) {
    llvm::dbgs() << "\tControl Dataout:\n";
    for (const auto &kv : controlDataOut) {
      llvm::dbgs() << "\t\tIter " << kv.first << ": (" << kv.second.value
                   << ", " << kv.second.iterIndex << ")\n";
    }
  }
}

void CMergeData::printDetail() {
  llvm::dbgs() << "Node Name: " << nodeName << "\n";

  // Print originalDataOut
  llvm::dbgs() << "\tOriginal Dataout:\n";
  for (const auto &kv : originalDataOut) {
    llvm::dbgs() << "\t\tIter " << kv.first << ": (" << kv.second.value << ", "
                 << kv.second.iterIndex << ")\n";
  }

  llvm::dbgs() << "\tControl Dataout:\n";
  for (const auto &kv : controlDataOut) {
    llvm::dbgs() << "\t\tIter " << kv.first << ": (" << kv.second.value << ", "
                 << kv.second.iterIndex << ")\n";
  }

  // Print mg_suc_node_dict, which is a map<mg_label, MgInfo>
  for (const auto &mgPair : cmergeSegmentSuccessorInfoMap) {
    const std::string &mgLabel = mgPair.first;
    const SegmentSuccessorInfo &info = mgPair.second;

    llvm::dbgs() << "\tCFDFC/Segment Label: " << mgLabel << "\n";
    // Print "original"
    llvm::dbgs() << "\t\tcontrol = [";
    for (size_t i = 0; i < info.original.size(); ++i) {
      llvm::dbgs() << info.original[i];
      if (i + 1 < info.original.size())
        llvm::dbgs() << ", ";
    }
    llvm::dbgs() << "]\n";
    // Print "glitch"
    llvm::dbgs() << "\t\tdata = [";
    for (size_t i = 0; i < info.glitch.size(); ++i) {
      llvm::dbgs() << info.glitch[i];
      if (i + 1 < info.glitch.size())
        llvm::dbgs() << ", ";
    }
    llvm::dbgs() << "]\n";
    // Print data_width
    llvm::dbgs() << "\t\tdata_width:\n";
    for (const auto &dw : info.dataWidthMap) {
      llvm::dbgs() << "\t\t  " << dw.first << " => " << dw.second << "\n";
    }
  }

  // if this is a control merge node
  if (controlGlitchValues.size()) {
    llvm::dbgs() << "\tControl glitch Dataout:\n\t";
    for (const auto &kv : controlGlitchValues) {
      llvm::dbgs() << std::to_string(kv) << " ,";
    }
    llvm::dbgs() << "\n";
  }
}

int CMergeData::getControlOutput(unsigned selIter) {
  if (contains(controlDataOut, selIter))
    return controlDataOut[selIter].value;

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
  llvm::dbgs() << "[DEBUG] \t\tSegmentDataSourceNodes:\n";

  llvm::dbgs() << "[DEBUG] \t\t  All: [";
  for (size_t i = 0; i < segmentNodes.all.size(); ++i) {
    llvm::dbgs() << segmentNodes.all[i];
    if (i + 1 < segmentNodes.all.size()) {
      llvm::dbgs() << ", ";
    }
  }
  llvm::dbgs() << "]\n";

  llvm::dbgs() << "[DEBUG] \t\t  Control: [";
  for (size_t i = 0; i < segmentNodes.control.size(); ++i) {
    llvm::dbgs() << segmentNodes.control[i];
    if (i + 1 < segmentNodes.control.size()) {
      llvm::dbgs() << ", ";
    }
  }
  llvm::dbgs() << "]\n";

  llvm::dbgs() << "[DEBUG] \t\t  Data: [";
  for (size_t i = 0; i < segmentNodes.data.size(); ++i) {
    llvm::dbgs() << segmentNodes.data[i];
    if (i + 1 < segmentNodes.data.size()) {
      llvm::dbgs() << ", ";
    }
  }
  llvm::dbgs() << "]\n";

  llvm::dbgs() << "[DEBUG] \t\t  Opaque Buffers: [";
  for (size_t i = 0; i < segmentNodes.opaque_buffers.size(); ++i) {
    llvm::dbgs() << segmentNodes.opaque_buffers[i];
    if (i + 1 < segmentNodes.opaque_buffers.size()) {
      llvm::dbgs() << ", ";
    }
  }
  llvm::dbgs() << "]\n";
}

// 1) Print the muxToSrcNodeMap
// Format: {"mux_node_name" : {"control" : ctrlSrcName, "0" : srcName0, "1" :
// srcName1}}
void printMuxToSrcNodeMap(
    const llvm::StringMap<std::map<std::string, std::string>>
        &muxToSrcNodeMap) {
  if (muxToSrcNodeMap.empty()) {
    llvm::dbgs() << "[ERROR] \tmuxToSrcNodeMap is empty\n";

    return;
  }
  llvm::dbgs() << "[DEBUG] \tmuxToSrcNodeMap:\n";
  for (const auto &muxEntry : muxToSrcNodeMap) {
    // muxEntry.first => mux node name
    // muxEntry.second => map from { "control", "0", "1" } to source node name
    const auto &muxNodeName = muxEntry.first();
    const auto &innerMap = muxEntry.second;

    llvm::dbgs() << "[DEBUG] \t\tMux Node: " << muxNodeName << " => {\n";
    for (const auto &kv : innerMap) {
      llvm::dbgs() << "[DEBUG] \t\t\t\"" << kv.first << "\" : \"" << kv.second
                   << "\",\n";
    }
    llvm::dbgs() << "[DEBUG] \t\t}\n";
  }
}

// 2) Print the srcNodeToMuxMap
// Format: {"src_node_name" : [ (mux_node_name, portId), ... ]}
void printSrcNodeToMuxMap(
    const llvm::StringMap<std::vector<std::pair<std::string, unsigned>>>
        &srcNodeToMuxMap) {
  llvm::dbgs() << "[DEBUG] \tsrcNodeToMuxMap:\n";
  for (const auto &srcEntry : srcNodeToMuxMap) {
    // srcEntry.first => source node name
    // srcEntry.second => vector of pairs
    const auto &srcNodeName = srcEntry.first();
    const auto &muxList = srcEntry.second;

    llvm::dbgs() << "[DEBUG] \t\tSource Node: " << srcNodeName << " => [";
    for (size_t i = 0; i < muxList.size(); ++i) {
      const auto &pairVal = muxList[i];
      llvm::dbgs() << "(" << pairVal.first << ", " << pairVal.second << ")";
      if (i + 1 < muxList.size()) {
        llvm::dbgs() << ", ";
      }
    }
    llvm::dbgs() << "]\n";
  }
}
