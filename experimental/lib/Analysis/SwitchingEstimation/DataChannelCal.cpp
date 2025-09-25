//===- DataChannelCal.cpp - Estimate Swithicng Activities ------*- C++ -*-===//
//
// This file declares all functions used for data channel switching calculation
//
//===----------------------------------------------------------------------===//

#include "experimental/Analysis/SwitchingEstimation/DataChannelCal.h"
#include "experimental/Analysis/SwitchingEstimation/DFSKernel.h"
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

#include <cassert>

void mapBBPairToControlMerge(SwitchingInfo &switchInfo) {
  // *Some control merge nodes with only 1 input is also included in the map
  for (const auto &nodeName : switchInfo.staticinfo.dataflowGraph->orderedNodeName) {
    auto *op = switchInfo.staticinfo.dataflowGraph->nodes[nodeName]->op;
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
        if (contains(switchInfo.data.bbPairToCtrlMerge, std::make_pair(preBB, CMBB))) {
          switchInfo.data.bbPairToCtrlMerge[std::make_pair(preBB, CMBB)].push_back(
              std::make_pair(nodeName, i));
        } else {
          switchInfo.data.bbPairToCtrlMerge[std::make_pair(preBB, CMBB)] = {std::make_pair(nodeName, i)};
        }
      }
    }
  }
}

void getDataBaseNodes(SwitchingInfo &switchInfo,
                      SCFProfilingResult &profileResults) {
  // Instantiate all DataBaseNodesTriple for all segments
  for (const auto &[segLabel, BBVec] : switchInfo.staticinfo.segToBBs) {
    DataBaseNodesTriple tmpDataBaseNodes;
    switchInfo.data.segToDataBaseVec[segLabel] = tmpDataBaseNodes;

    // Update the ordered mapped list
    switchInfo.data.segToOrderedDataBaseNodes[segLabel] = {};
    switchInfo.data.segToOrderedALUNodes[segLabel] = {};
    muxCMNodesList tmpList;
    switchInfo.data.controlNodes[segLabel] = tmpList;
  }

  // Traverse all the nodes in the dataflow graph
  for (const auto &nodeName : switchInfo.staticinfo.dataflowGraph->orderedNodeName) {
    unsigned nodeBB;
    auto op = switchInfo.staticinfo.dataflowGraph->nodes[nodeName]->op;
    if (!op) {
      llvm::errs() << "[WARNING] The op pointer for node " << nodeName
                   << " is null, skip it.\n";
      continue;
    }
    if (std::optional<unsigned> optBB = getLogicBB(op); !optBB.has_value())
      continue;
    else
      nodeBB = *optBB;

    auto dataflowGraph = switchInfo.staticinfo.dataflowGraph;
    // Check the types of the nodes
    // [TYPE 1]: DATA nodes from scf profiling
    if ((contains(profileResults.opNameToValueListMap, nodeName))||
        (contains(nodeName, "constant")) || (contains(nodeName, "source"))) {
      // Check each segment
      for (const auto &[segLabel, BBVec] : switchInfo.staticinfo.segToBBs) {
        if (std::find(BBVec.begin(), BBVec.end(), nodeBB) != BBVec.end()) {
          // If this is a CFDFC
          if (contains(switchInfo.staticinfo.segToGraph, segLabel)) {
            switchInfo.staticinfo.segToGraph[segLabel]->profileBaseNodes.insert(nodeName);
            switchInfo.staticinfo.segToGraph[segLabel]->allDataBaseNode.insert(nodeName);
          }

          switchInfo.data.segToDataBaseVec[segLabel].data.push_back(nodeName);
          switchInfo.data.segToDataBaseVec[segLabel].all.push_back(nodeName);
          switchInfo.data.segToOrderedDataBaseNodes[segLabel].push_back(nodeName);

          if ((!contains(nodeName, "constant")) &&(!contains(nodeName, "load")) && (!contains(nodeName, "source")))
            switchInfo.data.segToOrderedALUNodes[segLabel].push_back(nodeName);
        }
      }
      switchInfo.staticinfo.dataflowGraph->profileBaseNodes.insert(nodeName);
      switchInfo.staticinfo.dataflowGraph->allDataBaseNode.insert(nodeName);
      // [TYPE 2]: Control Nodes
    } else if (contains(nodeName, "control_merge") ||
               contains(nodeName, "mux")) {
      // Check each segment
      for (const auto &[segLabel, BBVec] : switchInfo.staticinfo.segToBBs) { 
        if (containsValue(BBVec, nodeBB)) {
          // If this is a CFDFC
          if (contains(switchInfo.staticinfo.segToGraph, segLabel)) {
            switchInfo.staticinfo.segToGraph[segLabel]->allDataBaseNode.insert(nodeName);
          }

          switchInfo.data.segToDataBaseVec[segLabel].control.push_back(nodeName);
          switchInfo.data.segToDataBaseVec[segLabel].all.push_back(nodeName);

          if (contains(nodeName, "control_merge")) {
            switchInfo.data.controlNodes[segLabel].cmNodeList.push_back(nodeName);
          } else {
            switchInfo.data.controlNodes[segLabel].muxNodeList.push_back(nodeName);
          }
        }
      }

      switchInfo.staticinfo.dataflowGraph->allDataBaseNode.insert(nodeName);
    }
  }

  // Add all arguments into data base nodes
  for (const auto &selArg : profileResults.argNamesVec) {
    switchInfo.staticinfo.dataflowGraph->profileBaseNodes.insert(selArg);
    switchInfo.staticinfo.dataflowGraph->allDataBaseNode.insert(selArg);
  }
}

void dataChannelBaseNodesValueUpdate(SwitchingInfo &switchInfo,
                                     SCFProfilingResult &profileResults) {
  // [STEP 1] Iterate all profile base node in the dataflow graph
  for (const auto& selNode : switchInfo.staticinfo.dataflowGraph->profileBaseNodes) {
    // Define tmp storing structure
    switchInfo.data.dfgBaseNodeValue[selNode] = std::make_shared<DataBase>(selNode);

    // Check the type of the node
    if (contains(selNode, "cmp")) {
      for (const auto &[value, iterIdx] : profileResults.opNameToValueListMap[selNode]) {
        // Create the value struct
        ValueIter tmpValuePair = {std::abs(value), iterIdx};
        switchInfo.data.dfgBaseNodeValue[selNode]->originalDataOut[iterIdx] = tmpValuePair;
      }

      //! Testing
      // switchInfo.data.dfgBaseNodeValue[selNode]->printDetail();
    } else if (contains(selNode, "constant")) {
      // We need to get the constant value from the attribute
      // Get the mlir op
      auto constantOp = dyn_cast<handshake::ConstantOp>(
          switchInfo.staticinfo.dataflowGraph->nodes[selNode]->op);
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
      ValueIter tmpValuePair = {constantValue, 0};
      switchInfo.data.dfgBaseNodeValue[selNode]->originalDataOut[0] = tmpValuePair;

      //! Testing
      // switchInfo.data.dfgBaseNodeValue[selNode]->printDetail();
    } else if (contains(selNode, "source")) {
      ValueIter tmpValuePair = {0, 0};
      switchInfo.data.dfgBaseNodeValue[selNode]->originalDataOut[0] = tmpValuePair;
    } else {
      for (const auto& [value, iterIdx] : profileResults.opNameToValueListMap[selNode]) {
        ValueIter tmpValuePair = {value, iterIdx};
        switchInfo.data.dfgBaseNodeValue[selNode]->originalDataOut[iterIdx] = tmpValuePair;
      }

      //! Testing
      // switchInfo.data.dfgBaseNodeValue[selNode]->printDetail();
    }
  }

  // [STEP 2] Iterate over all CMerge and MUX node
  for (const auto &selNode : switchInfo.staticinfo.dataflowGraph->allDataBaseNode) {
    // Check the type of the nodes
    if (contains(selNode, "control_merge")) {
      switchInfo.data.dfgBaseNodeValue[selNode] =std::make_shared<CMergeData>(selNode);
    } else if (contains(selNode, "mux")) {
      switchInfo.data.dfgBaseNodeValue[selNode] =std::make_shared<DataBase>(selNode);
    }
  }

  // Define temporary storing structure
  std::map<std::string, int> tmpMuxOutputMap;

  // [STEP 3] Traverse the executed BB trace
  // Skip the first BB, as it will always be BB 0
  for (unsigned i = 1; i < profileResults.executedBBTrace.size(); i++) {
    unsigned preBB = profileResults.executedBBTrace[i - 1];
    unsigned curBB = profileResults.executedBBTrace[i];

    std::pair<unsigned, unsigned> key_pair(preBB, curBB);

    // Get the corresponding control merge value
    // [Step 3.1] We first update the value of all influenced control_merge node in the circuit
    auto itCM = switchInfo.data.bbPairToCtrlMerge.find(key_pair);
    if (itCM != switchInfo.data.bbPairToCtrlMerge.end()) {
      // Iterate over all (CMNode, output value) pairs
      // TODO: [Urgent] Check whether the value stored match the simulation or not, 22/09/2025
      for (auto selValuePair : itCM->second) {
        auto nodeName = selValuePair.first;
        int outValue = selValuePair.second;
        unsigned iterIndex = getExecutionIter(i, curBB, profileResults);

        //! Testing
        // llvm::dbgs() << "[DEBUG] \t\tCtrlMerge Node: " << nodeName << "\n";
        // llvm::dbgs() << "[DEBUG] \t\t\tCon output Value: " << outValue <<
        // "\n"; llvm::dbgs() << "[DEBUG] \t\t\tIter Index: " << iterIndex <<
        // "\n";

        // Update the value
        auto selCtrlMergeNode = dyn_cast<CMergeData>(switchInfo.data.dfgBaseNodeValue[nodeName].get());

        // Data output for the control merge node, we assign -1 to it
        ValueIter tmpDataValuePair = {-1, iterIndex};
        selCtrlMergeNode->originalDataOut[iterIndex] = tmpDataValuePair;

        // Control output
        ValueIter tmpConValuePair = {outValue, iterIndex};
        selCtrlMergeNode->controlDataOut[iterIndex] = tmpConValuePair;

        // Define the Vector for unupdated mux
        std::vector<std::string> tmpUnUpdatedMux;
        // [Step 3.2] Update all related MUX node
        // Get the list of influenced mux node
        std::vector<std::string> selMuxVec = switchInfo.staticinfo.dataflowGraph->cmToMuxMap[nodeName];

        // Build a dependency graph for the multiplexers
        // We may encounter different situation when updating the value for MUX
        // nodes
        for (const auto &selMuxNode : selMuxVec) {
          // Only update muxes this control-merge actually influences
          if (!containsValue(selMuxVec, selMuxNode)) continue;
          auto selMuxSrcMap =
              switchInfo.staticinfo.dataflowGraph->muxToSrcNodeMap[selMuxNode];
          std::string selMuxDataSrcNode = selMuxSrcMap[std::to_string(outValue)];
          int tmpMuxOutput = 0;

          //! Testing
          // llvm::dbgs() << "[DEBUG] \t\t\t\tMux Node: " << selMuxNode << ";
          // Sel Data Src Node: " << selMuxDataSrcNode << "\n";

          if (contains(selMuxDataSrcNode, "constant") || contains(selMuxDataSrcNode, "source")) {
            tmpMuxOutput = switchInfo.data.dfgBaseNodeValue[selMuxDataSrcNode]->originalDataOut[0].value;
          } else {
            // If the src node is a mux node
            // TODO: Check the following update logic
            if (contains(selMuxDataSrcNode, "mux")) {
              // Case 1: the source node is the mux node itself
              //! Testing
              // llvm::dbgs() << "[DEBUG] \t\t\t(CASE1)\n";
              if (selMuxDataSrcNode == selMuxNode) {
                // Use the previous value
                if (contains(tmpMuxOutputMap, selMuxDataSrcNode)) {
                  tmpMuxOutput = tmpMuxOutputMap[selMuxDataSrcNode];
                } else {
                  llvm::errs()
                      << "[ERROR] Data src(" << selMuxDataSrcNode
                      << ") of Mux Node: " << selMuxDataSrcNode
                      << ", not found.\n";
                }
              } else if (contains(switchInfo.data.dfgBaseNodeValue[selMuxDataSrcNode]->originalDataOut, iterIndex) ) {
                //! Testing
                // llvm::dbgs() << "[DEBUG] \t\t\t(CASE2)\n";
                tmpMuxOutput =
                    switchInfo.data.dfgBaseNodeValue[selMuxDataSrcNode]->originalDataOut[iterIndex].value;
              } else if (contains(tmpMuxOutputMap, selMuxDataSrcNode)) {
                //! Testing
                // llvm::dbgs() << "[DEBUG] \t\t\t(CASE3)\n";
                tmpMuxOutput = tmpMuxOutputMap[selMuxDataSrcNode];
              } else {
                //! Testing
                // llvm::dbgs() << "[DEBUG] \t\t\t(CASE4)\n";
                // TODO: Check the following update logic
                ValueIter tmpInnerMuxValuePair = {tmpMuxOutput, iterIndex - 1};
                switchInfo.data.dfgBaseNodeValue[selMuxDataSrcNode]->originalDataOut[iterIndex - 1] = tmpInnerMuxValuePair;
                tmpUnUpdatedMux.push_back(selMuxNode);
                continue;
              }
            } else {
              if (selMuxDataSrcNode.empty()) {
                llvm::errs() << "[ERROR] The data src node for Mux Node: " << selMuxNode << " is empty\n";
                // Unconnected , use default vlue
                tmpMuxOutput = 0;
              } else {
                auto &omap = switchInfo.data.dfgBaseNodeValue[selMuxDataSrcNode]->originalDataOut;
                if (omap.find(iterIndex) != omap.end()) {
                  tmpMuxOutput = omap[iterIndex].value;
                } else if (iterIndex > 0 &&
                           omap.find(iterIndex - 1) != omap.end()) {
                  // Fallback to previous
                  llvm::dbgs() << "[WARNING] The value for node " << selMuxDataSrcNode
                               << " at iteration " << iterIndex
                               << " not found, fallback to previous\n";
                  tmpMuxOutput =
                      omap[iterIndex - 1].value; // fallback to previous
                } else {
                  llvm::errs() << "[ERROR] Data src(" << selMuxDataSrcNode
                               << ") of Mux Node: " << selMuxNode
                               << ", not found.\n";
                  tmpMuxOutput = 0; 
                }
              }
            }
          }

          // Update the storing structure and the tmp dict
          //! Testing
          // llvm::dbgs() << "[DEBUG] \t\t\t\t[OutValue] " << tmpMuxOutput <<
          // "\n";

          ValueIter tmpMuxValuePair = {tmpMuxOutput, iterIndex};
          switchInfo.data.dfgBaseNodeValue[selMuxNode]->originalDataOut[iterIndex] = tmpMuxValuePair;
          tmpMuxOutputMap[selMuxNode] = tmpMuxOutput;
        }

        //! Testing
        // llvm::dbgs() << "[DEBUG] \t\t\tRemaining Mux Nodes:\n";
        // llvm::dbgs() << "[DEBUG] \t\t\t\tTmp Mux Output Map\n";
        // for (const auto& [key, value] : tmpMuxOutputMap) {
        //   llvm::dbgs() << "[DEBUG] \t\t\t\t\tNode: " << key << ", Value: " <<
        //   value << "\n";
        // }

        // Update all remaining mux nodes
        // TODO: This maybe useless, check it out
        for (const auto &leftMuxNode : tmpUnUpdatedMux) {
          auto selMuxSrcMap = switchInfo.staticinfo.dataflowGraph->muxToSrcNodeMap[leftMuxNode];
          std::string selMuxDataSrcNode = selMuxSrcMap[std::to_string(outValue)];
          int tmpMuxOutput = 0;

          if (contains(tmpMuxOutputMap, selMuxDataSrcNode)) {
            tmpMuxOutput = tmpMuxOutputMap[selMuxDataSrcNode];
          } else {
            llvm::dbgs() << "[ERROR] Data src(" << selMuxDataSrcNode
                         << ") of Mux Node: " << leftMuxNode
                         << ", not found in tmpMuxOutputMap.\n";
          }

          ValueIter tmpMuxValuePair = {tmpMuxOutput, iterIndex};
          switchInfo.data.dfgBaseNodeValue[leftMuxNode]
              ->originalDataOut[iterIndex] = tmpMuxValuePair;
        }
      }
    }
  }
}

void buildSegmentSuccNodesList(SwitchingInfo &switchInfo,
                         SCFProfilingResult &profileResults) {
  //! Testing
  for (const auto &[label, bblist] : switchInfo.staticinfo.segToBBs) {
    LLVM_DEBUG(llvm::dbgs() << "[DEBUG] \t\tSeg Label: " << label << "\n");
    LLVM_DEBUG(llvm::dbgs() << "[DEBUG] \t\t\tBB List: ");
    for (const auto &selBB : bblist) {
      LLVM_DEBUG(llvm::dbgs() << selBB << ", ");
    }
    LLVM_DEBUG(llvm::dbgs() << "\n");
  }

  for (const auto &selBaseNode : switchInfo.staticinfo.dataflowGraph->allDataBaseNode) {
    //! Testing
    // llvm::dbgs() << "[DEBUG] Node Name: " << selBaseNode << "\n";

    // Check the existence of the selected node
    if (!contains(switchInfo.staticinfo.dataflowGraph->nodes, selBaseNode))
      continue;

    if (contains(selBaseNode, "control_merge")) {
      // If this is a control_merge node
      auto selCMNode = dyn_cast<CMergeNode>(
          switchInfo.staticinfo.dataflowGraph->nodes[selBaseNode].get());

      std::string dataSucNode = selCMNode->dataSucNodeName;
      std::string conSucNode = selCMNode->conSucNodeName;

      for (const auto &[label, bblist] : switchInfo.staticinfo.segToBBs) {
        MgNodeInfo tmpMGInfo;
        std::vector<std::string> controlExcludVec = {dataSucNode};
        std::vector<std::string> dataExcludVec = {conSucNode};
        tmpMGInfo.control = segCtrlMergeSuccSearch(switchInfo, selBaseNode, controlExcludVec, label);
        tmpMGInfo.data = segCtrlMergeSuccSearch(switchInfo, selBaseNode, dataExcludVec, label);
        tmpMGInfo.glitch = segCtrlMergeGlitchSuccSearch(switchInfo, selBaseNode, controlExcludVec, label);

        // Update the stroing structure
        switchInfo.data.dfgBaseNodeValue[selBaseNode]->segSucNodeMap[label.str()] = tmpMGInfo;
      }
    } else {
      for (const auto &[label, bblist] : switchInfo.staticinfo.segToBBs) {
        // Update the stroing structure
        switchInfo.data.dfgBaseNodeValue[selBaseNode]->segSucNodeMap[label.str()] = segGeneralSuccSearch(switchInfo, selBaseNode, label);
      }
    }
  }
}


void dataGlitchNodeSearch(SwitchingInfo &switchInfo, SCFProfilingResult &profileResults) {
  for (const auto & [label, bblist] : switchInfo.staticinfo.segToBBs) {
    // Skip all "S", "E", and "T" sections
    if (contains(label,"S") || contains(label,"E")||contains(label,"T")) {
      continue;
    }

    // Get the selected AdjGraph
    auto selAdjGraph = switchInfo.staticinfo.segToGraph[label];
    std::string mgBaseNode = selAdjGraph->baseNode;
    auto mgII = switchInfo.staticinfo.cfdfcIIs[std::stoul(std::string(label))];

    auto orderedNodes =selAdjGraph->orderedNodeName;
    auto &orderedALUs = switchInfo.data.segToOrderedALUNodes[label];


    //! Testing
    LLVM_DEBUG(llvm::dbgs() << "[DEBUG] \tCFDFC Label: " << label << "\n");
    LLVM_DEBUG(llvm::dbgs() << "[DEBUG] \t\tCFDFC Base Node: " << mgBaseNode << "\n");
    LLVM_DEBUG(llvm::dbgs() << "[DEBUG] \t\tCFDFC II: " << mgII << "\n");
    
    //
    std::map<std::string, std::vector<NodeGlitchInfo>> tmpGlitchDict;
    std::vector<std::string> tmpGlitchNodes;

    // Iterate over all mapped nodes in the CFDFC
    for (const auto & selNode : orderedALUs) {
      LLVM_DEBUG(llvm::dbgs() << "[DEBUG] \t\tALU Node: " << selNode << "\n");

      // ALU will only have two inputs
      auto preNodeLists = selAdjGraph->nodes[selNode]->pres;
      if (preNodeLists.size() != 2) {
        llvm::errs() << "[ERROR] The ALU node " << selNode << " does not have 2 inputs\n";
        continue;
      }

      // Unpack the pre_node_list
      const std::string& preNode1 = preNodeLists[0];
      const std::string& preNode2 = preNodeLists[1];

      // Get the source of the two inputs
      std::string srcNode1 = segNodeDataSrcSearch(switchInfo, preNode1, selAdjGraph.get());
      std::string srcNode2 = segNodeDataSrcSearch(switchInfo, preNode2, selAdjGraph.get());

      // Check whether this inode can have glitches
      LongestPathResult result1 = selLongestPath(switchInfo, preNode1, std::string(label));
      LongestPathResult result2 = selLongestPath(switchInfo, preNode2, std::string(label));

      //! Testing
      // llvm::dbgs() << "[DEBUG] \tNode : " << selNode << "\n";
      // llvm::dbgs() << "[DEBUG] \t\tsrcNode1 : " << srcNode1 << "\n";
      // llvm::dbgs() << "[DEBUG] \t\tsrcNode2 : " << srcNode2 << "\n";

      // TODO: Validate the following rounding process
      int finStartPoint1Steady = static_cast<int>(std::fmod(static_cast<float_t>(result1.maxLatency), mgII)) + 
                                      selAdjGraph->startBaseNodeShiftMap[result1.selStartNode] + selAdjGraph->startBaseNodeShiftMap[mgBaseNode];
      int finStartPoint2Steady = static_cast<int>(std::fmod(static_cast<float_t>(result2.maxLatency), mgII)) + 
                                  selAdjGraph->startBaseNodeShiftMap[result2.selStartNode] + selAdjGraph->startBaseNodeShiftMap[mgBaseNode];

      // TODO: Need to validate the following assumptions
      // If the node is directly connected with the srcs(without any buffers in between), the steady state value can not be used
      // As the latency can not be hidden
      bool preSrc1Buffered = false;
      bool preSrc2Buffered = false;

      for (const auto &selOriNode: switchInfo.data.dfgBaseNodeValue[srcNode1]->segSucNodeMap[label.str()].original) {
        if (preNode1 == selOriNode) preSrc1Buffered = true;
      }

      for (const auto &selOriNode: switchInfo.data.dfgBaseNodeValue[srcNode2]->segSucNodeMap[label.str()].original) {
        if (preNode2 == selOriNode) preSrc2Buffered = true;
      }

      // Assign the faster one with value 0 and slower one with value 1
      if (!preSrc1Buffered && !preSrc2Buffered) {
        finStartPoint1Steady = static_cast<int>(result1.maxLatency) + 
                                  selAdjGraph->startBaseNodeShiftMap[result1.selStartNode] + selAdjGraph->startBaseNodeShiftMap[mgBaseNode];
        finStartPoint2Steady = static_cast<int>(result2.maxLatency) + 
                                  selAdjGraph->startBaseNodeShiftMap[result2.selStartNode] + selAdjGraph->startBaseNodeShiftMap[mgBaseNode];

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
      // llvm::dbgs() << "\t\tPre_Node_1: " << preNode1 << ", Src_Node_1: " << srcNode1
      //             << ", Latency: " << result1.maxLatency << "\n"
      //             << "\t\tPre_Node_2: " << preNode2 << ", Src_Node_2: " << srcNode2
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
        if ((!contains(srcNode1,"constant"))  && (!contains(srcNode2,"constant"))) {
          tmpGlitchDict[selNode].push_back(tmpNode1);
          tmpGlitchDict[selNode].push_back(tmpNode2);
          tmpGlitchNodes.push_back(selNode);
        }
      } else {
        // Check Case 4
        if (containsValue(tmpGlitchNodes, srcNode1) &&(!containsValue(tmpGlitchNodes ,srcNode2) )){
          if (!preSrc1Buffered) {
            tmpGlitchDict[selNode].push_back(tmpNode1);
            tmpGlitchDict[selNode].push_back(tmpNode2);
            tmpGlitchNodes.push_back(selNode);
          }
        } else if (containsValue(tmpGlitchNodes, srcNode2) &&(!containsValue(tmpGlitchNodes ,srcNode1) )) {
          if (!preSrc2Buffered) {
            tmpGlitchDict[selNode].push_back(tmpNode1);
            tmpGlitchDict[selNode].push_back(tmpNode2);
            tmpGlitchNodes.push_back(selNode);
          }
        }
      }
    }

    // Store the glitching info into the main storing structure
    switchInfo.data.glitches[label] = tmpGlitchDict;
  }
}

int calGlitchValue(int op1, int op2, std::string selNode) {
  // TODO: Add supports for other types of nodes in self.GLITCH_NODE
  if ( contains(selNode,"add")) {return op1 + op2;} 
  else if ( contains(selNode,"mul")) {return op1 * op2; } 
  else {
    llvm::errs() << "[ERROR] Unexpected node type during glitching calculation: " << selNode << "\n";
    return 0;
  }
}

/// Return the first non-buffer predecessor along the **unique** path
/// between `node` and the Mux control port.
/// If you hit more than one predecessor (fan-in) we bail out –
/// only transparent chains are supported.
static std::string peelBufferChain(const SwitchingInfo &SI,
                                   std::string node) {
  // Use find() instead of operator[] to handle const StringMap
  while (contains(node, "buffer")) {
    auto nodeIt = SI.data.dfgBaseNodeValue.find(node);
    if (nodeIt == SI.data.dfgBaseNodeValue.end()) {
      llvm::errs() << "[ERROR] Node " << node << " not found in dfgBaseNodeValue\n";
      return node;
    }
    
    const auto &preds = nodeIt->second->segSucNodeMap; // Note: using segSucNodeMap as it exists
    if (preds.size() != 1) {
      llvm::errs() << "[ERROR] Buffer " << node
                   << " has multiple or no predecessors; can't peel.\n";
      return node;        // give up – caller will still error-out cleanly
    }
    
    const auto& firstSeg = preds.begin()->second;
    if (firstSeg.original.size() != 1) {
      llvm::errs() << "[ERROR] Buffer " << node
                   << " has multiple original predecessors; can't peel.\n";
      return node;
    }
    
    node = firstSeg.original.front(); // hop one step upstream
  }
  return node;
}

void dataBaseNodeGlitchUpdate(SwitchingInfo &switchInfo, SCFProfilingResult &profileResults, bool debug) {
  if (debug) {
    LLVM_DEBUG(llvm::dbgs() << "[DEBUG]\n[DEBUG] \t\t[NODE GLITCHING VALUE CALCULATION]\n");
  }

  //
  auto segExecTrace = profileResults.executedSegTrace;
  std::map<std::string, unsigned> lastSegIterMap;

  // Iterate through the execution trace
  for (unsigned i = 0; i < segExecTrace.size(); i++) {
    std::string executedSeg = segExecTrace[i];
    lastSegIterMap[executedSeg] = i;
    bool glitchUpdateFlag = false;

    if (debug) {
      LLVM_DEBUG(llvm::dbgs() << "[DEBUG] \t******** Iter: " << i << ", Seg: " << executedSeg << "\n");
    }

    // Check whether we need to update the glitch value for this seg
    if  (contains(executedSeg, "S") ||contains(executedSeg, "E") ||contains(executedSeg, "T")) {
      glitchUpdateFlag = false;
    } else {
      glitchUpdateFlag = true;
    }

    // Step 1: Calculate glitching for inner MG nodes -- All ALUs, other nodes shall be excluded
    //! For this type of glitching nodes (only ALUs), we only deal with the following glitching cases:
    //!     - CASE 1: No glitching for the two input operands
    //!     - CASE 2: One of the operand is glitching, but the original arriving time of the two inputs are the same.
    //! We ignore all other glitching cases and directly use the value from software profiling(original value)
    // TODO: Need to optimize the following process for glitching update
    // Iterate through all data base nodes
    std::vector<std::string> indexUpdateList;
    for (const auto& selNode: switchInfo.data.segToOrderedDataBaseNodes[executedSeg]) {
      if (contains(selNode,"constant") || contains(selNode,"source")  ||contains(selNode,"load")  ) continue;
          
      indexUpdateList.push_back(selNode);

      // Define vector to store the tmp glitch value
      std::vector<int> tmpValue;

      // Check whether we need the glitch value calculation
      if (glitchUpdateFlag) {
        // If we have glitch info for this node
        if (contains(switchInfo.data.glitches[executedSeg], selNode)) {
          // Get the source node
          std::string preSrc1 = switchInfo.data.glitches[executedSeg][selNode][0].srcNode;
          std::string preSrc2 = switchInfo.data.glitches[executedSeg][selNode][1].srcNode;
          // Get the starting time
          int preSrc1Start = switchInfo.data.glitches[executedSeg][selNode][0].steadyTime;
          int preSrc2Start = switchInfo.data.glitches[executedSeg][selNode][1].steadyTime;
          std::map<std::string, int> srcStartTimeDict = {{preSrc1, preSrc1Start}, {preSrc2, preSrc2Start}};

          // Get the buffering information
          bool preSrc1Buffered = switchInfo.data.glitches[executedSeg][selNode][0].buffered;
          bool preSrc2Buffered = switchInfo.data.glitches[executedSeg][selNode][1].buffered;
          std::map<std::string, bool> srcBufferedDict = {{preSrc1, preSrc1Buffered}, {preSrc2, preSrc2Buffered}};

          // Defining variables for glitch calculation
          int op1 = 0, op2 = 0;
          unsigned op1PreIndex = 0, op2PreIndex = 0;

          // Status Definition
          std::string fasterNode = "", slowerNode = "";

          //! Testing
          if (debug) {
            LLVM_DEBUG(llvm::dbgs() << "[DEBUG] \t\tGlitch Node: " << selNode << "\n"
                      << "[DEBUG] \t\t\tPre_src_1: " << preSrc1 << "\n"
                      << "[DEBUG] \t\t\tPre_src_2: " << preSrc2 << "\n");
          }

          // If this is the last iteration, we ignore the glitching value, just copy the original value
          if (i == segExecTrace.size() - 1) {
            //! Testing
            if (debug) {
              LLVM_DEBUG(llvm::dbgs() << "[DEBUG] \t\t(LAST ITER DURING GLITCH CALCULATION)\n");
            }

            switchInfo.data.dfgBaseNodeValue[selNode]->oriGlitchDataOut[i] = 
                {switchInfo.data.dfgBaseNodeValue[selNode]->originalDataOut[i].value};
            switchInfo.data.dfgBaseNodeValue[selNode]->lastUpdateIndex = i;
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
              op1PreIndex = switchInfo.data.dfgBaseNodeValue[fasterNode]->lastUpdateIndex;
              op2PreIndex = switchInfo.data.dfgBaseNodeValue[slowerNode]->lastUpdateIndex;

              //! Testing
              if (debug) {
                LLVM_DEBUG(llvm::dbgs() << "[DEBUG] \t\t\t[Value 1]: \n"
                          << "[DEBUG] \t\t\t\tOp_1_src_node: " << fasterNode << "\n"
                          << "[DEBUG] \t\t\t\tOp_1_pre_index: " << op1PreIndex << "\n"
                          << "[DEBUG] \t\t\t\tOp_2_src_node: " << slowerNode << "\n"
                          << "[DEBUG] \t\t\t\tOp_2_pre_index: " << op2PreIndex << "\n");
              }

              // Check the existence of the selected iter
              if (contains(switchInfo.data.dfgBaseNodeValue[fasterNode]->originalDataOut, op1PreIndex)) {
                op1 = switchInfo.data.dfgBaseNodeValue[fasterNode]->originalDataOut[op1PreIndex].value;
              } else {
                op1 = 0;
              }

              // Get operand 2, if op_2_pre not in original_dataout keys, we use 0 instead
              if (!(contains(switchInfo.data.dfgBaseNodeValue[slowerNode]->originalDataOut,op2PreIndex))){
                //! Testing
                if (debug) {
                  LLVM_DEBUG(llvm::dbgs() << "[DEBUG] \t\t\t[Value 1]: \n"
                              << "[DEBUG] \t\t\t\t[Warning] S Last Active Iter Not in the corresponding storing structure\n");
                }
                op2 = 0;
              } else {
                op2 = switchInfo.data.dfgBaseNodeValue[slowerNode]->originalDataOut[op2PreIndex].value;
              }

              tmpValue.push_back(calGlitchValue(op1, op2, selNode));

              //! Testing
              if (debug) {
                LLVM_DEBUG(llvm::dbgs() << "[DEBUG] \t\t\t\tFaster Node: " << fasterNode << "\n"
                          << "[DEBUG] \t\t\t\tOp_1: " << op1 << "\n"
                          << "[DEBUG] \t\t\t\tSlower Node: " << slowerNode << "\n"
                          << "[DEBUG] \t\t\t\tOp_2: " << op2 << "\n");
              }
            }

            // Value 2: F[x] op S[x - 1]
            //! Testing
            if (debug) {
              llvm::dbgs() << "[DEBUG] \t\t\t[Value 2]: \n";
            }

            int op1 = 0, op2 = 0;
            // Be careful that one of the operand may be generated by the selNode itself, then we need to use
            // the value in the previous iteration in the calculation
            op1PreIndex = switchInfo.data.dfgBaseNodeValue[fasterNode]->lastUpdateIndex;
            op2PreIndex = switchInfo.data.dfgBaseNodeValue[slowerNode]->lastUpdateIndex;

            // Get the value of op1
            if (fasterNode == selNode) {
              if (contains(switchInfo.data.dfgBaseNodeValue[fasterNode]->originalDataOut,op1PreIndex)){
                op1 = switchInfo.data.dfgBaseNodeValue[fasterNode]->originalDataOut[op1PreIndex].value;
              }
            } else if (contains(fasterNode,"mux")) {
              // Check whether the source of mux node is the selected node itself
              std::string tmpMuxDataSrcNode = getMuxDataSrc(switchInfo, fasterNode, i);

              if (tmpMuxDataSrcNode == selNode) {
                // TODO: Validate the following checking mechanism
                if (!contains(switchInfo.data.dfgBaseNodeValue[fasterNode]->originalDataOut, op1PreIndex)) {
                  op1 = 0;
                } else op1 = switchInfo.data.dfgBaseNodeValue[fasterNode]->originalDataOut[op1PreIndex].value;
              } else {
                op1 = switchInfo.data.dfgBaseNodeValue[fasterNode]->originalDataOut[i].value;
              }
            } else if (contains(switchInfo.data.glitches[executedSeg], fasterNode) ) {
              // If the faster node is glitching, we check whether the node is buffered or not
              if (srcBufferedDict[fasterNode]) {
                // Buffered
                op1 = switchInfo.data.dfgBaseNodeValue[fasterNode]->originalDataOut[i].value;
              } else {
                // unsigned tmpSize = switchInfo.data.dfgBaseNodeValue[fasterNode]->oriGlitchDataOut[i].size();
                auto  &glitchvec =switchInfo.data.dfgBaseNodeValue[fasterNode]->oriGlitchDataOut[i];
                // ! If the list is smaller than 2, something is wrong
                int op1 = 0;
                if (glitchvec.size() >= 2) {
                  op1 = glitchvec[glitchvec.size() - 2];
                } else if (!glitchvec.empty()) {
                  llvm::errs() << "[ERROR] Glitch Vector for " << selNode << " has a size smaller than 2\n";
                  op1 = glitchvec.back();
                } else { // only one glitch value – use it
                  llvm::errs() << "[ERROR] Use original dataout\n";
                  op1 = switchInfo.data.dfgBaseNodeValue[fasterNode]->originalDataOut[i].value; // no glitches recorded
                }
              }
            } else {
              op1 = switchInfo.data.dfgBaseNodeValue[fasterNode]->originalDataOut[i].value;
            }

            // Get the value of op2
            if (!contains(switchInfo.data.dfgBaseNodeValue[slowerNode]->originalDataOut, op2PreIndex)) {
              op2 = switchInfo.data.dfgBaseNodeValue[slowerNode]->originalDataOut[op2PreIndex].value;
            }

            tmpValue.push_back(calGlitchValue(op1, op2, selNode));

            //! Testing
            if (debug) {
              LLVM_DEBUG(llvm::dbgs() << "[DEBUG] \t\t\t\tFaster Node: " << fasterNode << "\n"
                        << "[DEBUG] \t\t\t\tF Last Active Iter: " << op1PreIndex << "\n"
                        << "[DEBUG] \t\t\t\tOp_1: " << op1 << "\n"
                        << "[DEBUG] \t\t\t\tSlower Node: " << slowerNode << "\n"
                        << "[DEBUG] \t\t\t\tS Last Active Iter: " << op2PreIndex << "\n"
                        << "[DEBUG] \t\t\t\tOp_2: " << op2 << "\n");
            }

            // Value 3: F[x] op S[x]
            // Check wheter the faster node is a mux node
            if (contains(fasterNode, "mux")) {
              std::string tmpMuxDataSrcNode = getMuxDataSrc(switchInfo, fasterNode, i);
            
              if (tmpMuxDataSrcNode == selNode) {
                op1PreIndex = switchInfo.data.dfgBaseNodeValue[fasterNode]->lastUpdateIndex;

                // TODO: Validate the following rounding process
                if (!contains(switchInfo.data.dfgBaseNodeValue[fasterNode]->originalDataOut, op1PreIndex)) {
                  op1 = 0;
                } else {
                  op1 = switchInfo.data.dfgBaseNodeValue[fasterNode]->originalDataOut[op1PreIndex].value;
                }
              } else {
                op1 = switchInfo.data.dfgBaseNodeValue[fasterNode]->originalDataOut[i].value;
              }
            } else {
              op1 = switchInfo.data.dfgBaseNodeValue[fasterNode]->originalDataOut[i].value;
            }

            // Check whether the slower node is a mux node
            // TODO: Check the condition below
            op2 = switchInfo.data.dfgBaseNodeValue[slowerNode]->originalDataOut[i].value;

            tmpValue.push_back(calGlitchValue(op1, op2, selNode));

            if (debug) {
              LLVM_DEBUG(llvm::dbgs() << "\t[Value 3]: \n"
                        << "\t\tOp_1: " << op1 << "\n"
                        << "\t\tOp_2: " << op2 << "\n"
                        << "\t\t[FINAL] ");
              for (auto v : tmpValue) llvm::dbgs() << v << " ";
              LLVM_DEBUG(llvm::dbgs() << "\n");
            }
          }
        } else {
          tmpValue.push_back(switchInfo.data.dfgBaseNodeValue[selNode]->originalDataOut[i].value);
        }
      } else {
        // No neeed for glitch value calculation, we directly copy the ori data
        tmpValue.push_back(switchInfo.data.dfgBaseNodeValue[selNode]->originalDataOut[i].value);
      }

      // Update the storing structure
      switchInfo.data.dfgBaseNodeValue[selNode]->oriGlitchDataOut[i] = tmpValue;
    }

    // Step 1.5: Update the list of last update index for all data base nodes
    for (const auto& selNode: indexUpdateList) {
      switchInfo.data.dfgBaseNodeValue[selNode]->lastUpdateIndex = i;
    }

    // Step 2: Update value for all MUXs
    // TODO: Pre store the information like mux etc.
    for (const auto& selMuxNode: switchInfo.data.controlNodes[executedSeg].muxNodeList) {
      std::string selCondInputNode = switchInfo.staticinfo.dataflowGraph->muxToSrcNodeMap[selMuxNode]["control"];

      // We need to check the existence of the control value output
      auto selCMBaseNode = dyn_cast<CMergeData>(switchInfo.data.dfgBaseNodeValue[selCondInputNode].get());
      int condValue = selCMBaseNode->getControlOutput(i);
      std::string selDataSrcNode = switchInfo.staticinfo.dataflowGraph->muxToSrcNodeMap[selMuxNode][std::to_string(condValue)];

      // Temp Value vector definition
      std::vector<int> preValue, curValue, nexValue;

      //! Testing
      if (debug) {
        LLVM_DEBUG(llvm::dbgs() << "Mux Node: " << selMuxNode << "\n"
                    << "\t[CUR_VALUE]\n"
                    << "\t\tCond Node: " << selCondInputNode << "\n"
                    << "\t\tCond_value: " << condValue << "\n"
                    << "\t\tCur data src: " << selDataSrcNode << "\n");
      }

      // TODO: Validate the following indexing mechanism   
      // Check whether we have glitches from the srcs or not
      if (contains(selDataSrcNode, "constant") || contains(selDataSrcNode, "source") || contains(selDataSrcNode, "start")) {
        curValue.push_back(switchInfo.data.dfgBaseNodeValue[selDataSrcNode]->originalDataOut[0].value);
      } else if (selDataSrcNode == selMuxNode) {
        // The src node is the selected node itself
        unsigned preIndex = switchInfo.data.dfgBaseNodeValue[selDataSrcNode]->lastUpdateIndex;
        // Check whether the preIndex exists or not
        if (contains(switchInfo.data.dfgBaseNodeValue[selDataSrcNode]->originalDataOut, preIndex)) {
          curValue.push_back(switchInfo.data.dfgBaseNodeValue[selDataSrcNode]->originalDataOut[preIndex].value);
        } else curValue.push_back(0);
      } else if (glitchUpdateFlag &&
                  (contains(switchInfo.data.glitches[executedSeg], selDataSrcNode))) {
        // Check whether there are buffers in between
        std::string preNode = "";
        auto selMuxNodeStructure = dyn_cast<MuxNode>(switchInfo.staticinfo.dataflowGraph->nodes[selMuxNode].get());

        // Get the actual preNode
        // TODO: Need to check the portidx to name mapping
        for (const auto& [nodeName, portIdx] : selMuxNodeStructure->preNameToPortIdxMap) {
          //* Here we need to do cond + 1, as the port map of muxnode assigns 0 to its control input.
          const unsigned int cond_add1= condValue + 1;
          if (portIdx == (cond_add1)) preNode = nodeName;
        }

        bool bufferedFlag = false;
        if (containsValue(switchInfo.data.dfgBaseNodeValue[selDataSrcNode]->segSucNodeMap[executedSeg].original, preNode)) bufferedFlag = true;
        
        //! Testing
        if (debug) {
          LLVM_DEBUG(llvm::dbgs() << "\t\tCur src node has glitches, buffered: " << bufferedFlag << "\n");
        }

        if (bufferedFlag) {
          // Src glitching but buffered
          curValue.push_back(switchInfo.data.dfgBaseNodeValue[selDataSrcNode]->originalDataOut[i].value);
        } else {
          curValue = switchInfo.data.dfgBaseNodeValue[selDataSrcNode]->oriGlitchDataOut[i];
        }
      } else {
        // Handling special case for cascaded Muxes
        if (contains(selDataSrcNode,"mux") && 
            (switchInfo.data.dfgBaseNodeValue[selDataSrcNode]->originalDataOut.find(i) == switchInfo.data.dfgBaseNodeValue[selDataSrcNode]->originalDataOut.end())) {
          curValue.push_back(switchInfo.data.dfgBaseNodeValue[selDataSrcNode]->originalDataOut[i - 1].value);
        } else {
          curValue.push_back(switchInfo.data.dfgBaseNodeValue[selDataSrcNode]->originalDataOut[i].value);
        }
      }
      
      //! Testing
      if (debug) {
        LLVM_DEBUG(llvm::dbgs() << "\t\tCur_value: ");
        for (auto v : curValue) LLVM_DEBUG(llvm::dbgs() << v << " ");
        LLVM_DEBUG(llvm::dbgs() << "\n\t[TRANSATION GLITCHES]\n");
      }

      // Calculate control flow glitches
      if (!switchInfo.data.dfgBaseNodeValue[selMuxNode]->skipControlCal) {
        std::string nodePreValidSeg = switchInfo.data.dfgBaseNodeValue[selMuxNode]->lastValidSeg;
        bool doubleTransFlag = false;

        //! Testing
        if (debug) {
          llvm::dbgs() << "\t\tNode Pre MG: " << nodePreValidSeg << "\n"
                    << "\t\tNode Cur MG: " << executedSeg << "\n";
        }

        // Check whether we have transition between different MGs
        // We have two types of MG transitions
        //    TYPE 1: MG 0 -> MG 1 -> MG 1
        //    TYPE 2: MG 0 -> MG 1 -> MG 0
        if (nodePreValidSeg != "" && (nodePreValidSeg != executedSeg)) {
          // Transition detected, check the next iter mg label
          if (i != segExecTrace.size() - 1) {
            // Check the existence of the mux node
            if (std::find(switchInfo.data.controlNodes[segExecTrace[i + 1]].muxNodeList.begin(), 
                        switchInfo.data.controlNodes[segExecTrace[i + 1]].muxNodeList.end(), selMuxNode) != switchInfo.data.controlNodes[segExecTrace[i + 1]].muxNodeList.end()) {
              std::string nextExecSegLabel = segExecTrace[i + 1];

              if (nextExecSegLabel != executedSeg) {
                // Type 2 detected
                doubleTransFlag = true;
              }
            }
          }

          // Calculate preList
          if (!contains(executedSeg,"E")) {
            // Cond value will be the same for the last segment
            int preCondValue = 1 - condValue;
            std::string preDataSrc = switchInfo.staticinfo.dataflowGraph->muxToSrcNodeMap[selMuxNode][std::to_string(preCondValue)];

            // Check whether the value exist or not
            if (contains(preDataSrc,"constant")) {
              preValue.push_back(switchInfo.data.dfgBaseNodeValue[preDataSrc]->originalDataOut[0].value);
            } else if (switchInfo.data.dfgBaseNodeValue[preDataSrc]->originalDataOut.find(i) == switchInfo.data.dfgBaseNodeValue[preDataSrc]->originalDataOut.end()) {
              preValue.push_back(switchInfo.data.dfgBaseNodeValue[preDataSrc]->originalDataOut[i].value);
            }
          } 

          // If double transition
          if (doubleTransFlag) {
            switchInfo.data.dfgBaseNodeValue[selMuxNode]->skipControlCal = true;
            nexValue = preValue;
          }
          // TODO: Check the following CondValue condition, it maybe 0 now
        } else if (nodePreValidSeg != "" && condValue == 1) {
          //! Con_Merge's control output automatically go back to 0 need to check why!!!!!!!
          if (!contains(executedSeg, "E")) {
            // Cond value will be the same for the last segment
            int preCondValue = 1 - condValue;
            std::string preDataSrc = switchInfo.staticinfo.dataflowGraph->muxToSrcNodeMap[selMuxNode][std::to_string(preCondValue)];
            [[maybe_unused]] unsigned newIterIdx = i;

            // If we have the same sources for both cond_value
            if (preDataSrc == selDataSrcNode) {
              // Change the iterindex
              if (contains(switchInfo.segToBackedgePairMap,executedSeg)) {
                auto mgIndexList = switchInfo.staticinfo.backEdgeToCFDFC[switchInfo.segToBackedgePairMap[executedSeg]];
                if (mgIndexList.size() > 1) {
                  // Get the first one
                  for (const auto & selMG: mgIndexList) {
                    if (std::to_string(selMG) != executedSeg) {
                      newIterIdx = lastSegIterMap[std::to_string(selMG)];
                      break;
                    }
                  }
                }
              } else {
                newIterIdx = i;
              }
            } else {
              newIterIdx = i;
            }

            // Check whether the value exist or not
            if (contains(preDataSrc,"constant")){
              preValue.push_back(switchInfo.data.dfgBaseNodeValue[preDataSrc]->originalDataOut[0].value);
            } else if (switchInfo.data.dfgBaseNodeValue[preDataSrc]->originalDataOut.find(i) != switchInfo.data.dfgBaseNodeValue[preDataSrc]->originalDataOut.end()) {
              preValue.push_back(switchInfo.data.dfgBaseNodeValue[preDataSrc]->originalDataOut[i].value);
            }
          }
        }

        //! Testing
        if (debug) {
          LLVM_DEBUG(llvm::dbgs() << "\t\tDouble transition: " << doubleTransFlag << "\n"
                        << "\t\tPre_value: ");
              for (auto v : preValue) LLVM_DEBUG(llvm::dbgs() << v << " ");
              LLVM_DEBUG(llvm::dbgs() << "\n\t\tNext value: ");
              for (auto v : nexValue) LLVM_DEBUG(llvm::dbgs() << v << " ");
              LLVM_DEBUG(llvm::dbgs() << "\n");
        }
      } else {
        switchInfo.data.dfgBaseNodeValue[selMuxNode]->skipControlCal = false;
      }

      // Get the final mux output data list
      std::vector<int> finalMuxOutputList;

      for (const auto& selValue: preValue) finalMuxOutputList.push_back(selValue);
      for (const auto& selValue: curValue) finalMuxOutputList.push_back(selValue);
      for (const auto& selValue: nexValue) finalMuxOutputList.push_back(selValue);
    }

    // Step 3: Relay memory load node's data
    for (const auto& selNode: switchInfo.data.segToOrderedDataBaseNodes[executedSeg]) {
      if (contains(selNode, "load")) {
        switchInfo.data.dfgBaseNodeValue[selNode]->lastUpdateIndex = i;
      }
    }
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

int getCondBrNodeCondValue(SwitchingInfo &switchInfo, unsigned iterIndex, std::string nodeName) {
  // Defensive check for condBrToConSrcMap
  if (!contains(switchInfo.staticinfo.dataflowGraph->condBrToConSrcMap, nodeName)) {
    llvm::errs() << "[ERROR] Warning: cond_br node " << nodeName << " not found in condBrToConSrcMap\n";
    return 1; // Default value
  }

  std::string condSrcNode = switchInfo.staticinfo.dataflowGraph->condBrToConSrcMap[nodeName];
  int condValue = 1;

  // Defensive check for dfgBaseNodeValue
  if (!contains(switchInfo.data.dfgBaseNodeValue, condSrcNode)) {
    llvm::errs() << "[ERROR] Warning: condSrcNode " << condSrcNode << " not found in dfgBaseNodeValue\n";
    return 1; // Default value
  }

  if (contains(switchInfo.data.dfgBaseNodeValue[condSrcNode]->originalDataOut, iterIndex)) {
    condValue = switchInfo.data.dfgBaseNodeValue[condSrcNode]->originalDataOut[iterIndex].value;
  } else {
    condValue = 1; 
  }

  return condValue;
}

void dfgDataChannelPropagate(SwitchingInfo &switchInfo,
                             SCFProfilingResult &profileResults, bool debug) {
  //
  auto segExecTrace = profileResults.executedSegTrace;

  // Iterate through the execution trace and propagate base nodes values in the
  // selected segment
  for (unsigned i = 0; i < segExecTrace.size(); i++) {
    std::string executedSeg = segExecTrace[i];

    // List storing nodes that need special treatment
    std::vector<std::string> pendingMemList;

    //! Testing
    if (debug)
      LLVM_DEBUG(llvm::dbgs() << "[DEBUG] ==================================\n");

    // Propagate data values for all non_memory related nodes
    for (const auto &selNode :
         switchInfo.data.segToDataBaseVec[executedSeg].all) {
      //! Testing
      if (debug)
        LLVM_DEBUG(llvm::dbgs() << "[DEBUG] Node: " << selNode << "\n");

      // Case 1: Mapped Nodes --> All ALUs 
      if (containsValue(switchInfo.data.segToOrderedALUNodes[executedSeg], selNode)) {
        //! Testing
        if (debug)
          LLVM_DEBUG(llvm::dbgs() << "[DEBUG] ALU Node Detected \n");
        
        // Defensive check for ALU node
        if ( !contains(switchInfo.data.dfgBaseNodeValue, selNode) ){
          llvm::errs() << "[ERROR] Warning: ALU node " << selNode << " not found in dfgBaseNodeValue at iteration " << i << "\n";
          continue;
        }
        
        // The value shall be obtained from ori_glitch_dataout
        // Step 1: Update the selected node itself
        for (const auto &selValue :
             switchInfo.data.dfgBaseNodeValue[selNode]->oriGlitchDataOut[i]) {
          switchInfo.staticinfo.dataflowGraph->nodes[selNode]->updateDataoutChannel(selValue);

          // Step 2: Update the glitching suceeding list
          //! Testing
          if (debug)
            LLVM_DEBUG(llvm::dbgs() << "[DEBUG] \t[Glitch Output]\n");
          
          // Defensive check for segSucNodeMap access
          if (!contains(switchInfo.data.dfgBaseNodeValue[selNode]->segSucNodeMap, executedSeg)) {
            llvm::errs() << "[ERROR] Warning: segment " << executedSeg << " not found in segSucNodeMap for ALU node " << selNode << "\n";
            continue;
          }
          
          for (const auto &selSucNode : switchInfo.data.dfgBaseNodeValue[selNode]->segSucNodeMap[executedSeg].glitch) {
            //! Testing
            if (debug)
              llvm::dbgs() << "[DEBUG] \t\tSucNode: " << selSucNode << "\n";

            // Get the node bitwidth
            unsigned tmpNodeWidth = switchInfo.data.dfgBaseNodeValue[selNode]->segSucNodeMap[executedSeg].dataWidthMap[selSucNode];

            //! Testing
            if (debug)
              llvm::dbgs() << "[DEBUG] \t\t\tDataWidth: " << tmpNodeWidth
                           << "\n";

            if (contains(selSucNode, "store")) {
              // Get the actual node storing structure
              auto selStoreNode = dyn_cast<DStoreNode>(
                  switchInfo.staticinfo.dataflowGraph->nodes[selSucNode].get());
              selStoreNode->updateDataout(reduceBits(selValue, tmpNodeWidth),
                                          selNode);
            } else if (contains(selSucNode, "cond_br")) {
              // Get the node
              auto selCondStoreNode = dyn_cast<CBrNode>(switchInfo.staticinfo.dataflowGraph->nodes[selSucNode].get());
              // Get the control value
              // TODO: Validate the assumption that cond_value is always updated
              // before the data_in of the node
              int tmpCondValue = getCondBrNodeCondValue(switchInfo, i, selSucNode);

              selCondStoreNode->updateDataout(reduceBits(selValue, tmpNodeWidth), tmpCondValue);
            } else {
              switchInfo.staticinfo.dataflowGraph->nodes[selSucNode]->updateDataoutChannel(reduceBits(selValue, tmpNodeWidth));
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
        for (const auto &selSucNode : switchInfo.data.dfgBaseNodeValue[selNode]->segSucNodeMap[executedSeg].original) {
          // Get Node Bitwidth
          unsigned tmpNodeWidth = switchInfo.data.dfgBaseNodeValue[selNode]->segSucNodeMap[executedSeg].dataWidthMap[selSucNode];

          //! Testing
          if (debug)
            llvm::dbgs() << "[DEBUG] \t\tSuc Node: " << selSucNode << "\n";

          //
          if (contains(selSucNode, "store")) {
            //
            auto selStoreNode = dyn_cast<DStoreNode>(
                switchInfo.staticinfo.dataflowGraph->nodes[selSucNode].get());
            selStoreNode->updateDataout(
                reduceBits(switchInfo.data.dfgBaseNodeValue[selNode]->originalDataOut[i].value, tmpNodeWidth),
                selNode);
          } else if (contains(selSucNode, "cond_br")) {
            // Get the node
            // TODO: Validate the assumption that cond_value is always updated
            // before the data_in of the node
            auto selCondStoreNode = dyn_cast<CBrNode>(
                switchInfo.staticinfo.dataflowGraph->nodes[selSucNode].get());
            int tmpCondValue =
                getCondBrNodeCondValue(switchInfo, i, selSucNode);

            selCondStoreNode->updateDataout(
                reduceBits(switchInfo.data.dfgBaseNodeValue[selNode]
                               ->originalDataOut[i].value, tmpNodeWidth), tmpCondValue);
          } else {
            switchInfo.staticinfo.dataflowGraph->nodes[selSucNode]->updateDataoutChannel(
                reduceBits(switchInfo.data.dfgBaseNodeValue[selNode]
                               ->originalDataOut[i].value, tmpNodeWidth));
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
          if (!contains(switchInfo.data.dfgBaseNodeValue, selNode)) {
            llvm::errs() << "[ERROR] Warning: Control merge node " << selNode << " not found in dfgBaseNodeValue at iteration " << i << "\n";
            continue;
          }

          // Get node
          auto selCMNode = dyn_cast<CMergeData>(
              switchInfo.data.dfgBaseNodeValue[selNode].get());
          int initValue = 0;

          //! For all transition related nodes, we need to check the iter 0 as
          //! well
          if (i == 1) {
            initValue = selCMNode->getControlOutput(0);
            switchInfo.staticinfo.dataflowGraph->nodes[selNode]->updateDataoutChannel(
                initValue);
          } else {
            initValue = 0;
          }

          // Step 1: Update the node itself
          int selValue = selCMNode->getControlOutput(i);
          switchInfo.staticinfo.dataflowGraph->nodes[selNode]->updateDataoutChannel(
              selValue);

          if (selValue == 1) {
            switchInfo.staticinfo.dataflowGraph->nodes[selNode]->updateDataoutChannel(0);
          }

          // Step 2: Update all nodes in the control succeeding node list
          //! Testing
          if (debug)
            llvm::dbgs() << "[DEBUG] \t[Control Output]\n";
          
          // Defensive check for mgSucNodeDict access
          if (!contains(selCMNode->mgSucNodeDict, executedSeg)) {
            llvm::errs() << "[ERROR] Segment " << executedSeg << " not found in mgSucNodeDict for control merge node " << selNode << "\n";
            continue;
          }
          
          for (const auto &selSucNode :
               selCMNode->mgSucNodeDict[executedSeg].control) {
            //! Testing
            if (debug)
              llvm::dbgs() << "[DEBUG] \t\tSuc Node: " << selSucNode << "\n";

            if (i == 1) {
              switchInfo.staticinfo.dataflowGraph->nodes[selSucNode]->updateDataoutChannel(
                  initValue);
            }
            switchInfo.staticinfo.dataflowGraph->nodes[selSucNode]->updateDataoutChannel(
                selValue);

            // Step 2.5: Update all glitch node, if cm_value = 1, then change it
            // back to 0, unless II == 1
            if (selValue == 1) {
              if (std::find(
                      selCMNode->mgSucNodeDict[executedSeg].glitch.begin(),
                      selCMNode->mgSucNodeDict[executedSeg].glitch.end(),
                      selSucNode) !=
                  selCMNode->mgSucNodeDict[executedSeg].glitch.end()) {
                switchInfo.staticinfo.dataflowGraph->nodes[selSucNode]
                    ->updateDataoutChannel(0);
              }
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
               selCMNode->mgSucNodeDict[executedSeg].data) {
            //! Testing
            if (debug)
              llvm::dbgs() << "[DEBUG] \t\tSuc Node: " << selSucNode << "\n";

            if (contains(selSucNode, "cond_br")) {
              // Get the node
              auto selCondStoreNode = dyn_cast<CBrNode>(
                  switchInfo.staticinfo.dataflowGraph->nodes[selSucNode].get());

              int tmpCondValue =
                  getCondBrNodeCondValue(switchInfo, i, selSucNode);
              selCondStoreNode->updateDataout(-1, tmpCondValue);
            } else {
              switchInfo.staticinfo.dataflowGraph->nodes[selSucNode]->updateDataoutChannel(
                  -1);
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
          if (!contains(switchInfo.data.dfgBaseNodeValue, selNode)) {
            llvm::dbgs() << "[DEBUG] Warning: MUX node " << selNode << " not found in dfgBaseNodeValue at iteration " << i << "\n";
            continue;
          }
          
          for (const auto &selValue :
               switchInfo.data.dfgBaseNodeValue[selNode]->oriGlitchDataOut[i]) {
            switchInfo.staticinfo.dataflowGraph->nodes[selNode]->updateDataoutChannel(
                selValue);

            // Step 2: Update the glitching succeeding list
            //! Testing
            if (debug)
              llvm::dbgs() << "[DEBUG] \t[Glitch Output]\n";
            
            // Defensive check for segSucNodeMap access
            if (!contains(switchInfo.data.dfgBaseNodeValue[selNode]->segSucNodeMap, executedSeg)) {
              llvm::errs() << "[ERROR] Warning: segment " << executedSeg << " not found in segSucNodeMap for node " << selNode << "\n";
              continue;
            }
            
            for (const auto &selSucNode :
                 switchInfo.data.dfgBaseNodeValue[selNode]
                     ->segSucNodeMap[executedSeg]
                     .glitch) {
              //! Testing
              if (debug)
                llvm::dbgs() << "[DEBUG] \t\tSuc Node: " << selSucNode << "\n";

              // Get the node output bitwidth
              unsigned tmpNodeWidth = switchInfo.data.dfgBaseNodeValue[selNode]->segSucNodeMap[executedSeg].dataWidthMap[selSucNode];

              if (contains(selSucNode, "cond_br")) {
                // Get the node
                auto selCondStoreNode = dyn_cast<CBrNode>(switchInfo.staticinfo.dataflowGraph->nodes[selSucNode].get());

                int tmpCondValue =
                    getCondBrNodeCondValue(switchInfo, i, selSucNode);
                selCondStoreNode->updateDataout(
                    reduceBits(selValue, tmpNodeWidth), tmpCondValue);
              } else {
                switchInfo.staticinfo.dataflowGraph->nodes[selSucNode]
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

          for (const auto &selSucNode : switchInfo.data.dfgBaseNodeValue[selNode]
                                            ->segSucNodeMap[executedSeg]
                                            .original) {
            //! Testing
            if (debug)
              llvm::dbgs() << "[DEBUG] \t\tSuc Node: " << selSucNode << "\n";
            // Get bitwidth
            unsigned tmpNodeWidth = switchInfo.data.dfgBaseNodeValue[selNode]
                                        ->segSucNodeMap[executedSeg]
                                        .dataWidthMap[selSucNode];

            //
            if (contains(switchInfo.data.dfgBaseNodeValue[selNode]->originalDataOut, i)) {
              if (contains(selSucNode, "store")) {
                auto selStoreNode = dyn_cast<DStoreNode>(switchInfo.staticinfo.dataflowGraph->nodes[selSucNode].get());
                selStoreNode->updateDataout(reduceBits(switchInfo.data.dfgBaseNodeValue[selNode] ->originalDataOut[i] .value,
                               tmpNodeWidth),
                    selNode);
              } else if (contains(selSucNode, "cond_br")) {
                auto selCondBrNode = dyn_cast<CBrNode>(
                    switchInfo.staticinfo.dataflowGraph->nodes[selSucNode].get());
                int tmpCondValue =
                    getCondBrNodeCondValue(switchInfo, i, selSucNode);

                selCondBrNode->updateDataout(
                    reduceBits(switchInfo.data.dfgBaseNodeValue[selNode]
                                   ->originalDataOut[i]
                                   .value,
                               tmpNodeWidth),
                    tmpCondValue);
              } else {
                switchInfo.staticinfo.dataflowGraph->nodes[selSucNode]
                    ->updateDataoutChannel(
                        reduceBits(switchInfo.data.dfgBaseNodeValue[selNode]
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
            if (i == switchInfo.data.firstExecutedIter[executedSeg]) {
              selValue = switchInfo.data.dfgBaseNodeValue[selNode]
                             ->originalDataOut[0]
                             .value;
            } else
              continue;
          } else if (contains(selNode, "constant")) {
            selValue = switchInfo.data.dfgBaseNodeValue[selNode]
                           ->originalDataOut[0]
                           .value;
          } else {
            // Defensive check to prevent crash
            if (!contains(switchInfo.data.dfgBaseNodeValue, selNode)) {
              llvm::errs() << "[ERROR] Warning: selNode " << selNode << " not found in dfgBaseNodeValue at iteration " << i << "\n";
              selValue = 0; // Default value
            } else if (!contains(switchInfo.data.dfgBaseNodeValue[selNode]->originalDataOut, i)) {
              llvm::errs() << "[ERROR] Warning: No data found for node " << selNode << " at iteration " << i << "\n";
              selValue = 0; // Default value
            } else {
              selValue = switchInfo.data.dfgBaseNodeValue[selNode]->originalDataOut[i].value;
            }
          }

          // Step 1: update the node itself
          switchInfo.staticinfo.dataflowGraph->nodes[selNode]->updateDataoutChannel(selValue);

          // Step 2: Update all glitching nodes in the succeeding list
          // Defensive check for segSucNodeMap access
          if (!contains(switchInfo.data.dfgBaseNodeValue[selNode]->segSucNodeMap, executedSeg)) {
            // llvm::dbgs() << "[DEBUG] Warning: segment " << executedSeg << " not found in segSucNodeMap for other node " << selNode << "\n";
            continue;
          }
          
          for (const auto &selSucNode : switchInfo.data.dfgBaseNodeValue[selNode]
                                            ->segSucNodeMap[executedSeg]
                                            .glitch) {
            if (contains(selSucNode, "cond_br")) {
              auto selCondStoreNode = dyn_cast<CBrNode>(
                  switchInfo.staticinfo.dataflowGraph->nodes[selSucNode].get());
              int tmpCondValue =
                  getCondBrNodeCondValue(switchInfo, i, selSucNode);
              selCondStoreNode->updateDataout(selValue, tmpCondValue);
            } else {
              switchInfo.staticinfo.dataflowGraph->nodes[selSucNode]->updateDataoutChannel(
                  selValue);
            }
          }

          // Step 3: Update all non glitching nodes in the succeeding list
          for (const auto &selSucNode : switchInfo.data.dfgBaseNodeValue[selNode]
                                            ->segSucNodeMap[executedSeg]
                                            .original) {
            switchInfo.staticinfo.dataflowGraph->nodes[selSucNode]->updateDataoutChannel(
                selValue);
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
      if (!contains(switchInfo.data.dfgBaseNodeValue, selNode)) {
        llvm::errs() << "[ERROR] Warning: Memory node " << selNode << " not found in dfgBaseNodeValue at iteration " << i << "\n";
        continue;
      }
      
      // We need to update the address out and data out separatly
      // Step 1: Get the address source
      auto selMemNode =
          dyn_cast<DLoadNode>(switchInfo.staticinfo.dataflowGraph->nodes[selNode].get());

      std::string tmpAddrPreNode = selMemNode->addressInNodeName;
      std::string tmpAddrPreSrcNode = "";
      if (executedSeg == "E") {
        tmpAddrPreSrcNode =segENodeSrcSearch(switchInfo, tmpAddrPreNode, profileResults);


      } else {
        tmpAddrPreSrcNode = memAddrSrcSearch(switchInfo, profileResults,tmpAddrPreNode, executedSeg, i);

      }

      // We assume there will not be any glitches for the address value
      int addrValue = 0;
      if (contains(tmpAddrPreSrcNode,"constant") ) {
        addrValue = switchInfo.data.dfgBaseNodeValue[tmpAddrPreSrcNode]
                        ->originalDataOut[0]
                        .value;
      } else {
        // TODO: Validate the following assumption
        if (contains(switchInfo.data.dfgBaseNodeValue[tmpAddrPreSrcNode]->originalDataOut,i)) {
          addrValue = switchInfo.data.dfgBaseNodeValue[tmpAddrPreSrcNode]->originalDataOut[i].value;
        } else {
          addrValue = -10;
        }
      }

      int selValue =
          switchInfo.data.dfgBaseNodeValue[selNode]->originalDataOut[i].value;

      // Step 1: Update the node itself
      selMemNode->updateDataout(selValue, addrValue);

      // Step 2: Update teh nodes in glitching list
      // Defensive check for segSucNodeMap access
      if (!contains(switchInfo.data.dfgBaseNodeValue[selNode]->segSucNodeMap, executedSeg)) {
        llvm::errs() << "[ERROR] Warning: segment " << executedSeg << " not found in segSucNodeMap for memory node " << selNode << "\n";
        continue;
      }
      
      for (const auto &selSucNode : switchInfo.data.dfgBaseNodeValue[selNode]->segSucNodeMap[executedSeg].glitch) {
        //! Exclude the memory controller
        if (  (!contains(selSucNode,"mem_controller"))   && (  (!contains(selSucNode,"end"))   ))  {
          // Get node bitwidth
          unsigned tmpNodeWidth = switchInfo.data.dfgBaseNodeValue[selNode]->segSucNodeMap[executedSeg].dataWidthMap[selSucNode];

          if (contains(selSucNode, "cond_br")) {
            // Get the node
            auto selCondStoreNode = dyn_cast<CBrNode>(switchInfo.staticinfo.dataflowGraph->nodes[selSucNode].get());

            int tmpCondValue =getCondBrNodeCondValue(switchInfo, i, selSucNode);
            selCondStoreNode->updateDataout(reduceBits(selValue, tmpNodeWidth),
                                            tmpCondValue);
          } else {
            switchInfo.staticinfo.dataflowGraph->nodes[selSucNode]->updateDataoutChannel(
                reduceBits(selValue, tmpNodeWidth));
          }
        }
      }

      // Step 3: Update nodes in non glitching list
      for (const auto &selSucNode : switchInfo.data.dfgBaseNodeValue[selNode]->segSucNodeMap[executedSeg].original) {
        // Get bitwidth
        unsigned tmpNodeWidth = switchInfo.data.dfgBaseNodeValue[selNode]->segSucNodeMap[executedSeg].dataWidthMap[selSucNode];

        //
        if (contains(selSucNode, "cond_br")) {
          // Get the node
          auto selCondStoreNode = dyn_cast<CBrNode>(
              switchInfo.staticinfo.dataflowGraph->nodes[selSucNode].get());

          int tmpCondValue = getCondBrNodeCondValue(switchInfo, i, selSucNode);
          selCondStoreNode->updateDataout(reduceBits(selValue, tmpNodeWidth),
                                          tmpCondValue);
        } else {
          switchInfo.staticinfo.dataflowGraph->nodes[selSucNode]->updateDataoutChannel(
              reduceBits(selValue, tmpNodeWidth));
        }
      }
    }
  }

  // Final step, change the results of buffers directly connected to cond_br
  // node for (auto& [selCondNode, bufferList] :
  // switchInfo.staticinfo.dataflowGraph->condBrToBufferMap) {
  //   // Get the node
  //   auto selCondStoringNode =
  //   dyn_cast<CBrNode>(switchInfo.staticinfo.dataflowGraph->nodes[selCondNode].get());
  //   for (auto& [selBuffer, portIdx]: bufferList) {
  //     // Change the resutls of the buffer node
  //     for (auto& [selIdx, valueVec]:
  //     switchInfo.staticinfo.dataflowGraph->nodes[selBuffer]->dataOut) {
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
std::vector<std::string> segCtrlMergeSuccSearch(SwitchingInfo &switchInfo, std::string startNode,
                      std::vector<std::string> &excludingList,
                      StringRef segLabel) {
  std::vector<std::string> succNodeList;
  auto &si = switchInfo;
  auto segBBs = llvm::ArrayRef<unsigned>(si.staticinfo.segToBBs[segLabel]);
  
  // Lambda function for finding the valid successor nodes
  auto succOf = [&si, &excludingList, &segBBs, &segLabel](const std::string &n) -> std::vector<std::string> {
    std::vector<std::string> out;

    for (const auto &m : si.staticinfo.dataflowGraph->nodes[n]->sucs)  {
      if (isExcluded(excludingList, m)) continue;
      if (isDataBase(si, m)) continue;
      if (isInvalidBackedge(si, segLabel, n, m)) continue;
      if (crossesCondPort(si, n, m)) continue;
      if (bufferOutsideSeg(si, m, segBBs))  continue;
      
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

MgNodeInfo segGeneralSuccSearch(SwitchingInfo &switchInfo,
                                  std::string startNode, StringRef segLabel) {
  // Status variables
  unsigned numBuffers = 0;
  unsigned minDataWidth = 32;
  std::string lastNode = "";

  // Initialize the storing structure
  MgNodeInfo tmpMgNodeInfo;
  
  // Track the current path for proper buffer counting and cycle detection
  std::vector<std::string> currentPath;
  auto &si = switchInfo;
  auto segBBs = llvm::ArrayRef<unsigned>(si.staticinfo.segToBBs[segLabel]);

  // Lambda function for finding the valid successor nodes
  auto succOf = [&si, &segBBs, &segLabel, &currentPath, &startNode]
               (const std::string &n) -> std::vector<std::string> {
    std::vector<std::string> out;

    if (n == startNode) {
      for (auto &m : si.staticinfo.dataflowGraph->nodes[n]->sucs) {
        if (isDataBase(si, m)) continue;
        out.push_back(m);
      }
      return out;
    }

    // For other nodes, we need to do more checks
    for (auto &m : si.staticinfo.dataflowGraph->nodes[n]->sucs) {
      if (isDataBase(si, m)) continue;
      if (std::find(currentPath.begin(), currentPath.end(), m)
            != currentPath.end()) continue;
      if (isInvalidBackedge(si, segLabel, n, m)) continue;
      if (skipCondBrPort(si, n, m) || skipMuxPort(si, n, m)) continue;
      if (bufferOutsideSeg(si, m, segBBs)) continue;
      if (isEndNode(m, segLabel)) continue;
      if (isMemController(m)) continue;

      out.push_back(m);
    }
    return out;
  };

  // Lambda function for Pre processing when entering a node
  auto onEnter = [&tmpMgNodeInfo,
                &numBuffers,
                &minDataWidth,
                &lastNode,
                &currentPath,
                &si,
                &startNode]
               (const std::string &n) {
    currentPath.push_back(n);

    if (n == startNode)
      return;              // skip the root

    // predecessor in the path
    auto preNode = currentPath[currentPath.size() - 2];

    // 1) Update the running min‐width
    unsigned w = si.staticinfo.dataflowGraph->nodes[preNode]->sucsDataWidthMap[n];
    if (w < minDataWidth) minDataWidth = w;
    // 2) Buffer count
    if (contains(n, "buffer"))
      ++numBuffers;
    // 3) Classify
    if (numBuffers > 0)
      tmpMgNodeInfo.original.push_back(n);
    else
      tmpMgNodeInfo.glitch.push_back(n);

    lastNode = n;
    // on “push” (old code’s first write):
    tmpMgNodeInfo.dataWidthMap[preNode] = minDataWidth;
    // at end‐of‐iteration (old code’s second write):
    tmpMgNodeInfo.dataWidthMap[n]       = minDataWidth;
  };

  auto onExit = [&](const std::string &n) {
    if (!currentPath.empty() && currentPath.back() == n) {
      if (contains(n, "buffer"))
        --numBuffers;
      currentPath.pop_back();
    }
  };

  // Run DFS starting from the startNode
  std::vector<std::string> roots = {startNode};
  dfsWithEvents(roots, succOf, onEnter, onExit);

  tmpMgNodeInfo.dataWidthMap[lastNode] = minDataWidth;

  return tmpMgNodeInfo;
}

unsigned getExecutionIter(unsigned bbIndex, unsigned curBB, SCFProfilingResult &profileResults) {
  if (!contains(profileResults.bbToIterMap, bbIndex)) {
    llvm::errs() << "[ERROR] Cannot find the execution iteration for BB " << curBB
                 << " at index " << bbIndex << "\n";
    return 0;
  }

  // Get the execution iteration
  unsigned iterIndex = profileResults.bbToIterMap[bbIndex];

  if (iterIndex == 0) {
    llvm::errs() << "[ERROR] The execution iteration for BB " << curBB
                 << " at index " << bbIndex << " is 0\n";
    return 0;
  }

  if (iterIndex - 1 >= profileResults.executedBBTrace.size()) {
    llvm::dbgs() << "[ERROR] iterIndex-1 (" << (iterIndex-1) 
                 << ") out of bounds for executedBBTrace (size: " 
                 << profileResults.executedBBTrace.size() << ")\n";
    return iterIndex; // Return iterIndex instead of underflowing
  }

  if (curBB == profileResults.executedBBTrace[iterIndex - 1] ||   (bbIndex == 1)) {
    return iterIndex - 1;
  }
  return iterIndex;
}

std::string segNodeDataSrcSearch(SwitchingInfo &switchInfo,
                                 std::string startNode,
                                 AdjGraph *selGraph) {
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
      if (auto *c = dyn_cast<CBrNode>(switchInfo.staticinfo.dataflowGraph->nodes[node].get())) {
        preds.emplace_back(
          !c->dataPreNodeName.empty() ? c->dataPreNodeName
                                      : c->condPreNodeName
        );
      }
    } else if (selGraph->nodes.count(node)) {
      for (auto it = selGraph->nodes[node]->pres.rbegin();
               it != selGraph->nodes[node]->pres.rend(); ++it)
        preds.push_back(*it);
    }
    // drop any already‐in‐path
    preds.erase(std::remove_if(preds.begin(), preds.end(),
                               [&](auto &m){ return isInPath(pathSet, m); }),
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
  auto onExit = [&](const std::string &n) {
    pathSet.erase(n);
  };

  // 4) run your existing DFS kernel
  dfsWithEvents({startNode}, succOf, onEnter, onExit);

  // 5) report failure
  if (foundDataSrc.empty()) {
    llvm::dbgs() << "[ERROR] Could not find base node for " << startNode << "\n";
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
  auto selGraph = switchInfo.staticinfo.segToGraph[mgLabel];
  auto selMGII = switchInfo.staticinfo.cfdfcIIs[std::stoul(mgLabel)];

  // Get the list of buffers
  std::vector<std::string> selBuffList;
  for (const auto& selNode: selGraph->orderedNodeName) {
    if (selNode.find("buffer") != std::string::npos) {
      selBuffList.push_back(selNode);
    }
  }

  // For all start node
  for (const auto &startNode : selGraph->segStartNodes) {
    auto tmpPaths = selGraph->findPaths(startNode, dstNode, false, true);

    std::vector<std::pair<unsigned, std::string>> tmpLastSecondBufferList;

    for (auto& selPath: tmpPaths) {
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
        for (const auto& tmpSelNode: selPath.nodeList) {
          if (tmpSelNode.find("buffer") != std::string::npos) {
            tmpBufferList.push_back(tmpSelNode);
          }
        }

        if (tmpBufferList.size() > 1) {
          tmpLastSecondBufferList.push_back({tmpPathLatency, tmpBufferList[tmpBufferList.size() - 2]});
        }
      }
    }

    // Select the wanted last second buffer, this will be used in the cal of SET_R for transparent buffers
    for (const auto &selPair: tmpLastSecondBufferList) {
      if (selPair.first == maxLatency) {
        // TODO: Validate the following selection criteria
        auto selBuffer = dyn_cast<BufferNode>(selGraph->nodes[selPair.second].get());
        float_t oriOcc = 0;
        if (lastSecondBuff != "") {
          auto oriBuffer = dyn_cast<BufferNode>(selGraph->nodes[lastSecondBuff].get());
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
  const auto& selMuxSrcMap = switchInfo.staticinfo.dataflowGraph->muxToSrcNodeMap;
  
  // Check if the mux node exists in the map
  auto muxIt = selMuxSrcMap.find(selMuxNode);
  if (muxIt == selMuxSrcMap.end()) {
    llvm::errs() << "[ERROR] Mux node " << selMuxNode << " not found in muxToSrcNodeMap\n";
    return ""; // Return empty string on error
  }
  
  // Check if "control" key exists
  auto controlIt = muxIt->second.find("control");
  if (controlIt == muxIt->second.end()) {
    llvm::errs() << "[ERROR] Control input not found for mux node " << selMuxNode << "\n";
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
  auto nodeIt = switchInfo.data.dfgBaseNodeValue.find(muxCondInput);
  if (nodeIt == switchInfo.data.dfgBaseNodeValue.end()) {
    llvm::errs() << "[ERROR] Node " << muxCondInput << " not found in dfgBaseNodeValue\n";
    return ""; // Return empty string on error
  }
  
  // Get the corresponding Control Merge Data base node
  auto* selCMBaseNode = dyn_cast<CMergeData>(nodeIt->second.get());
  if (!selCMBaseNode) {
    llvm::errs() << "[ERROR] No selCMBaseNode, got " << muxCondInput << "'s control input port\n";
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
static std::optional<std::string> findSrcInSegment(const SwitchingInfo &si,
                                                    const std::string &nodeName,
                                                    const std::string &segLabel) {
  static constexpr std::array<llvm::StringRef, 4> cats {
    "control", "data", "glitch", "original"
  };

  // 1) Check whether the segment exists
  if (!contains(si.data.segToDataBaseVec, segLabel)) {
    return std::nullopt;
  }

  // Iterate all mapped roots
  for (auto const &mapped : si.data.segToDataBaseVec.at(segLabel).all) {
    // 2) Direct name-match
    if (contains(nodeName, mapped)) {
      return mapped;
    }

    // 3) Check if DFG node exists
    if (!contains(si.data.dfgBaseNodeValue, mapped)) {
      llvm::errs() << "[ERROR] Node " << mapped << " not found in dfgBaseNodeValue\n";
      continue;
    }
    auto const &nodeVal = si.data.dfgBaseNodeValue.at(mapped);

    // 4) Check if segSuc entry exists
    if (!contains(nodeVal->segSucNodeMap, segLabel)) {
      llvm::errs() << "[ERROR] Segment " << segLabel << " not found in segSucNodeMap for node " << mapped << "\n";
      continue;
    }
    auto const &mSuc = nodeVal->segSucNodeMap.at(segLabel);

    // 5) Search in successor lists
    for (auto cat : cats) {
      auto const &lst = (cat == "control" ? mSuc.control
                        : cat == "data" ? mSuc.data
                        : cat == "glitch" ? mSuc.glitch
                        : mSuc.original);
      if (std::any_of(lst.begin(), lst.end(),
                      [&](auto const &s){ return s.find(nodeName) != std::string::npos;})) return mapped;
    }
  }
  return std::nullopt;
} 

std::string segENodeSrcSearch(SwitchingInfo &switchInfo, std::string nodeName, SCFProfilingResult &profileResults) {
  if (auto src = findSrcInSegment(switchInfo, nodeName, "E")) return *src;

  if (profileResults.executedSegTrace.size() >= 2) {
    auto prev = profileResults.executedSegTrace[profileResults.executedSegTrace.size() - 2];
    if (auto src = findSrcInSegment(switchInfo, nodeName, prev)) return *src;
  }
  return "";
}

std::string memAddrSrcSearch(SwitchingInfo &si,
                              SCFProfilingResult &profile,
                              const std::string&  nodeName,
                              const std::string&  selSeg,
                              unsigned iterIdx) {
  if (auto src = findSrcInSegment(si, nodeName, selSeg))
    return *src;
  // fallback to the previous segment in the trace
  if (iterIdx == 0 || iterIdx > profile.executedSegTrace.size()-1)
    return "";
  auto prev = profile.executedSegTrace[iterIdx-1];
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
  LLVM_DEBUG(llvm::dbgs() << "Node Name: " << nodeName << "\n");

  // Print originalDataOut
  LLVM_DEBUG(llvm::dbgs() << "\tOriginal Dataout:\n");
  for (const auto &kv : originalDataOut) {
    LLVM_DEBUG(llvm::dbgs() << "\t\tIter " << kv.first << ": (" << kv.second.value << ", "
                 << kv.second.iterIndex << ")\n");
  }

  // Print mg_suc_node_dict, which is a map<mg_label, MgInfo>
  for (const auto &mgPair : segSucNodeMap) {
    const auto &mgLabel = mgPair.first;
    const MgNodeInfo &info = mgPair.second;

    LLVM_DEBUG(llvm::dbgs() << "\tSegment Label: " << mgLabel << "\n");
    // Print "original"
    LLVM_DEBUG(llvm::dbgs() << "\t\toriginal = [");
    for (size_t i = 0; i < info.original.size(); ++i) {
      LLVM_DEBUG(llvm::dbgs() << info.original[i]);
      if (i + 1 < info.original.size())
        LLVM_DEBUG(llvm::dbgs() << ", ");
    }
    LLVM_DEBUG(llvm::dbgs() << "]\n");
    // Print "glitch"
    LLVM_DEBUG(llvm::dbgs() << "\t\tglitch = [");
    for (size_t i = 0; i < info.glitch.size(); ++i) {
      LLVM_DEBUG(llvm::dbgs() << info.glitch[i]);
      if (i + 1 < info.glitch.size())
        LLVM_DEBUG(llvm::dbgs() << ", ");
    }
    LLVM_DEBUG(llvm::dbgs() << "]\n");
    // Print data_width
    LLVM_DEBUG(llvm::dbgs() << "\t\tdata_width:\n");
    for (const auto &dw : info.dataWidthMap) {
      LLVM_DEBUG(llvm::dbgs() << "\t\t  " << dw.first << " => " << dw.second << "\n");
    }
  }

  // if this is a control merge node
  if (controlDataOut.size()) {
    LLVM_DEBUG(llvm::dbgs() << "\tControl Dataout:\n");
    for (const auto &kv : controlDataOut) {
      LLVM_DEBUG(llvm::dbgs() << "\t\tIter " << kv.first << ": (" << kv.second.value
                   << ", " << kv.second.iterIndex << ")\n");
    }
  }
}

void CMergeData::printDetail() {
  LLVM_DEBUG(llvm::dbgs() << "Node Name: " << nodeName << "\n");

  // Print originalDataOut
  LLVM_DEBUG(llvm::dbgs() << "\tOriginal Dataout:\n");
  for (const auto &kv : originalDataOut) {
    LLVM_DEBUG(llvm::dbgs() << "\t\tIter " << kv.first << ": (" << kv.second.value << ", "
                 << kv.second.iterIndex << ")\n");
  }

  LLVM_DEBUG(llvm::dbgs() << "\tControl Dataout:\n");
  for (const auto &kv : controlDataOut) {
    LLVM_DEBUG(llvm::dbgs() << "\t\tIter " << kv.first << ": (" << kv.second.value << ", "
                 << kv.second.iterIndex << ")\n");
  }

  // Print mg_suc_node_dict, which is a map<mg_label, MgInfo>
  for (const auto &mgPair : mgSucNodeDict) {
    const std::string &mgLabel = mgPair.first;
    const MgNodeInfo &info = mgPair.second;

    LLVM_DEBUG(llvm::dbgs() << "\tCFDFC/Segment Label: " << mgLabel << "\n");
    // Print "original"
    LLVM_DEBUG(llvm::dbgs() << "\t\tcontrol = [");
    for (size_t i = 0; i < info.original.size(); ++i) {
      LLVM_DEBUG(llvm::dbgs() << info.original[i]);
      if (i + 1 < info.original.size())
        LLVM_DEBUG(llvm::dbgs() << ", ");
    }
    LLVM_DEBUG(llvm::dbgs() << "]\n");
    // Print "glitch"
    LLVM_DEBUG(llvm::dbgs() << "\t\tdata = [");
    for (size_t i = 0; i < info.glitch.size(); ++i) {
      LLVM_DEBUG(llvm::dbgs() << info.glitch[i]);
      if (i + 1 < info.glitch.size())
        LLVM_DEBUG(llvm::dbgs() << ", ");
    }
    LLVM_DEBUG(llvm::dbgs() << "]\n");
    // Print data_width
    LLVM_DEBUG(llvm::dbgs() << "\t\tdata_width:\n");
    for (const auto &dw : info.dataWidthMap) {
      LLVM_DEBUG(llvm::dbgs() << "\t\t  " << dw.first << " => " << dw.second << "\n");
    }
  }

  // if this is a control merge node
  if (controlGlitchVec.size()) {
    LLVM_DEBUG(llvm::dbgs() << "\tControl glitch Dataout:\n\t");
    for (const auto &kv : controlGlitchVec) {
      LLVM_DEBUG(llvm::dbgs() << std::to_string(kv) << " ,");
    }
    LLVM_DEBUG(llvm::dbgs() << "\n");
  }
}

int CMergeData::getControlOutput(unsigned selIter) {
  if (contains(controlDataOut,selIter) ) {
    return controlDataOut[selIter].value;
  } else {
    return 0;
  }
}

std::vector<std::string> segCtrlMergeGlitchSuccSearch(SwitchingInfo &switchInfo, std::string startNode,
                            std::vector<std::string> &excludingList,
                            StringRef segLabel) {
  std::vector<std::string> glitchSuccNodeList;
  auto segBBs = llvm::ArrayRef<unsigned>(switchInfo.staticinfo.segToBBs[segLabel]);

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
      for (auto &m : switchInfo.staticinfo.dataflowGraph->nodes[n]->sucs) {
        if (isExcluded(excludingList, m))    continue;
        if (isDataBase(switchInfo, m))                continue;
        out.push_back(m);
      }
      return out;
    }
    // …else, do the full filtering…
    for (auto &m : switchInfo.staticinfo.dataflowGraph->nodes[n]->sucs) {
      if (isExcluded(excludingList, m))                 continue;
      if (isDataBase(switchInfo, m))                            continue;
      if (std::find(currentPath.begin(), currentPath.end(), m) != currentPath.end()) continue;
      if (isInvalidBackedge(switchInfo, segLabel, n, m))        continue;
      if (crossesCondPort(switchInfo, n, m))                    continue;
      if (bufferOutsideSeg(switchInfo, m, segBBs))              continue;
      if (isEndNode(m, segLabel))                       continue;
      if (isMemController(m))                           continue;
      out.push_back(m);
    }
    return out;
  };

  // B) pre: push every node into currentPath so cycle-checks work
  //    but *don’t* record the startNode itself in glitchSuccNodeList
  auto pre = [&](const std::string &n) {
    currentPath.push_back(n);
    if (n == startNode) return;          // <-- skip root itself

    if (contains(n, "buffer")) numBuffers++;
    if (numBuffers == 0) {
      glitchSuccNodeList.push_back(n);
    }
  };

  // C) post: exactly as before
  auto post = [&](const std::string &n) {
    if (!currentPath.empty() && currentPath.back() == n) {
      if (contains(n, "buffer")) numBuffers--;
      currentPath.pop_back();
    }
  };

  // now kick off DFS *from* the startNode itself
  std::vector<std::string> roots = { startNode };
  dfsWithEvents(roots, succOf, pre, post);

  return glitchSuccNodeList;
}

//===----------------------------------------------------------------------===//
//
// Helper function for debuging
//
//===----------------------------------------------------------------------===//
void printDataBaseNodesTriple(DataBaseNodesTriple dbnt) {
  LLVM_DEBUG(llvm::dbgs() << "[DEBUG] \t\tDataBaseNodesTriple:\n");

  LLVM_DEBUG(llvm::dbgs() << "[DEBUG] \t\t  All: [");
  for (size_t i = 0; i < dbnt.all.size(); ++i) {
    LLVM_DEBUG(llvm::dbgs() << dbnt.all[i]);
    if (i + 1 < dbnt.all.size()) {
      LLVM_DEBUG(llvm::dbgs() << ", ");
    }
  }
  LLVM_DEBUG(llvm::dbgs() << "]\n");

  LLVM_DEBUG(llvm::dbgs() << "[DEBUG] \t\t  Control: [");
  for (size_t i = 0; i < dbnt.control.size(); ++i) {
    LLVM_DEBUG(llvm::dbgs() << dbnt.control[i]);
    if (i + 1 < dbnt.control.size()) {
      LLVM_DEBUG(llvm::dbgs() << ", ");
    }
  }
  LLVM_DEBUG(llvm::dbgs() << "]\n");

  LLVM_DEBUG(llvm::dbgs() << "[DEBUG] \t\t  Data: [");
  for (size_t i = 0; i < dbnt.data.size(); ++i) {
    LLVM_DEBUG(llvm::dbgs() << dbnt.data[i]);
    if (i + 1 < dbnt.data.size()) {
      LLVM_DEBUG(llvm::dbgs() << ", ");
    }
  }
  LLVM_DEBUG(llvm::dbgs() << "]\n");
}

// 1) Print the muxToSrcNodeMap
// Format: {"mux_node_name" : {"control" : ctrlSrcName, "0" : srcName0, "1" :
// srcName1}}
void printMuxToSrcNodeMap(
    const llvm::StringMap<  std::map<std::string, std::string>>
        &muxToSrcNodeMap) {
  if(muxToSrcNodeMap.empty()){
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
    const llvm::StringMap<  std::vector<std::pair<std::string, unsigned>>>
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


