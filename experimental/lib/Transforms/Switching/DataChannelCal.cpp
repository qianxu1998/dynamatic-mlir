//===- DataChannelCal.cpp - Estimate Swithicng Activities ------*- C++ -*-===//
//
// This file declares all functions used for data channel switching calculation
//
//===----------------------------------------------------------------------===//

#include "experimental/Transforms/Switching/DataChannelCal.h"
#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Dialect/Handshake/HandshakeAttributes.h"
#include "dynamatic/Support/DynamaticPass.h"
#include "dynamatic/Support/LLVM.h"
#include "dynamatic/Support/CFG.h"
#include "dynamatic/Support/Logging.h"
#include "dynamatic/Support/Backedge.h"
#include "dynamatic/Support/TimingModels.h"
#include "dynamatic/Support/Attribute.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Debug.h"

void constructBBPairToCMResMap(SwitchingInfo &switchInfo) {
  for (const auto& nodeName: switchInfo.dataflowGraph->orderedNodeName) {
   if (auto CMOp = dyn_cast<handshake::ControlMergeOp>(switchInfo.dataflowGraph->nodes[nodeName]->op)) {
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
    for (int i = 0; i < opOperands.size(); i++) {
      auto inputSrcOp = opOperands[i].getDefiningOp();
      unsigned preBB;
      if (std::optional<unsigned> optBB = getLogicBB(inputSrcOp); !optBB.has_value())
        continue;
      else
        preBB = *optBB;
      
      // Update the storing dict
      if (switchInfo.bbPairToCMResultMap.find(std::make_pair(preBB, CMBB)) != switchInfo.bbPairToCMResultMap.end()) {
        switchInfo.bbPairToCMResultMap[std::make_pair(preBB, CMBB)].push_back(std::make_pair(nodeName, i));
      } else {
        switchInfo.bbPairToCMResultMap[std::make_pair(preBB, CMBB)] = { std::make_pair(nodeName, i) };
      }
    }
   } 
  }
}

void getDataBaseNodes(SwitchingInfo &switchInfo, SCFProfilingResult &profileResults) {
  // Instantiate all DataBaseNodesTriple for all segments
  for (const auto& [segLabel, BBVec]: switchInfo.segToBBListMap) {
    DataBaseNodesTriple tmpDataBaseNodes;
    switchInfo.segToDataBaseVecMap[segLabel] = tmpDataBaseNodes;

    // Update the ordered mapped list
    switchInfo.segToOrderedDataBaseNodes[segLabel] = {};
    muxCMNodesList tmpList;
    switchInfo.segToControlNodeList[segLabel] = tmpList;
  }

  // Traverse all the nodes in the dataflow graph
  for (const auto& nodeName: switchInfo.dataflowGraph->orderedNodeName) {
    unsigned nodeBB;
    if (std::optional<unsigned> optBB = getLogicBB(switchInfo.dataflowGraph->nodes[nodeName]->op); !optBB.has_value())
      continue;
    else
      nodeBB = *optBB;

    auto dataflowGraph = switchInfo.dataflowGraph;
    // Check the types of the nodes
    // TYPE 1: DATA nodes from scf profiling
    if (profileResults.opNameToValueListMap.find(nodeName) != profileResults.opNameToValueListMap.end() ||
          (nodeName.find("constant") != std::string::npos) ||
          (nodeName.find("source") != std::string::npos)) {
      // Check each segment
      for (const auto& [segLabel, BBVec]: switchInfo.segToBBListMap) {
        if (std::find(BBVec.begin(), BBVec.end(), nodeBB) != BBVec.end()) {
          // If this is a MG
          if (switchInfo.segToAdjGraphMap.find(segLabel) != switchInfo.segToAdjGraphMap.end()) {
            switchInfo.segToAdjGraphMap[segLabel]->profileBaseNodes.insert(nodeName);
            switchInfo.segToAdjGraphMap[segLabel]->allDataBaseNode.insert(nodeName);
          }

          switchInfo.segToDataBaseVecMap[segLabel].data.push_back(nodeName);
          switchInfo.segToDataBaseVecMap[segLabel].all.push_back(nodeName);
          switchInfo.segToOrderedDataBaseNodes[segLabel].push_back(nodeName);
        }
      }
      switchInfo.dataflowGraph->profileBaseNodes.insert(nodeName);
      switchInfo.dataflowGraph->allDataBaseNode.insert(nodeName);
    // TYPE 2: Control Nodes
    } else if ((nodeName.find("control_merge") != std::string::npos) ||
                (nodeName.find("mux") != std::string::npos)) {
      // Check each segment
      for (const auto& [segLabel, BBVec]: switchInfo.segToBBListMap) {
        if (std::find(BBVec.begin(), BBVec.end(), nodeBB) != BBVec.end()) {
          // If this is a MG
          if (switchInfo.segToAdjGraphMap.find(segLabel) != switchInfo.segToAdjGraphMap.end()) {
            switchInfo.segToAdjGraphMap[segLabel]->allDataBaseNode.insert(nodeName);
          }

          switchInfo.segToDataBaseVecMap[segLabel].control.push_back(nodeName);
          switchInfo.segToDataBaseVecMap[segLabel].all.push_back(nodeName);

          if (nodeName.find("control_merge") != std::string::npos) {
            switchInfo.segToControlNodeList[segLabel].cmNodeList.push_back(nodeName);
          } else {
            switchInfo.segToControlNodeList[segLabel].muxNodeList.push_back(nodeName);
          }
        }
      }

      switchInfo.dataflowGraph->allDataBaseNode.insert(nodeName);
    }
  }

  // Add all arguments into data base nodes
  for (const auto& selArg: profileResults.argNamesVec) {

    switchInfo.dataflowGraph->profileBaseNodes.insert(selArg);
    switchInfo.dataflowGraph->allDataBaseNode.insert(selArg);
  }
}

//===----------------------------------------------------------------------===//
//
// Class for storing data channel values
//
//===----------------------------------------------------------------------===//
DataBase::DataBase(const std::string &node): nodeName(node), lastUpdateIndex(0), skipControlCal(false) {}

void DataBase::printDetail() {
  llvm::dbgs() << "Node Name: " << nodeName << "\n";

  // Print originalDataOut
  llvm::dbgs() << "\tOriginal Dataout:\n";
  for (const auto &kv : originalDataOut) {
    llvm::dbgs() << "\t\tIter " << kv.first << ": ("
              << kv.second.value << ", " << kv.second.iterIndex << ")\n";
  }

  // Print mg_suc_node_dict, which is a map<mg_label, MgInfo>
  for (const auto &mgPair : segSucNodeMap) {
    const std::string &mgLabel = mgPair.first;
    const MgNodeInfo &info = mgPair.second;

    llvm::dbgs() << "\tCFDFC/Segment Label: " << mgLabel << "\n";
    // Print "original"
    llvm::dbgs() << "\t\toriginal = [";
    for (size_t i = 0; i < info.original.size(); ++i) {
      llvm::dbgs() << info.original[i];
      if (i + 1 < info.original.size()) llvm::dbgs() << ", ";
    }
    llvm::dbgs() << "]\n";
    // Print "glitch"
    llvm::dbgs() << "\t\tglitch = [";
    for (size_t i = 0; i < info.glitch.size(); ++i) {
      llvm::dbgs() << info.glitch[i];
      if (i + 1 < info.glitch.size()) llvm::dbgs() << ", ";
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
      llvm::dbgs() << "\t\tIter " << kv.first << ": ("
                << kv.second.value << ", " << kv.second.iterIndex << ")\n";
    }
  }
}

void CMergeData::printDetail() {
  llvm::dbgs() << "Node Name: " << nodeName << "\n";

  // Print originalDataOut
  llvm::dbgs() << "\tOriginal Dataout:\n";
  for (const auto &kv : originalDataOut) {
    llvm::dbgs() << "\t\tIter " << kv.first << ": ("
              << kv.second.value << ", " << kv.second.iterIndex << ")\n";
  }

  llvm::dbgs() << "\tControl Dataout:\n";
  for (const auto &kv : controlDataOut) {
    llvm::dbgs() << "\t\tIter " << kv.first << ": ("
              << kv.second.value << ", " << kv.second.iterIndex << ")\n";
  }

  // Print mg_suc_node_dict, which is a map<mg_label, MgInfo>
  for (const auto &mgPair : mgSucNodeDict) {
    const std::string &mgLabel = mgPair.first;
    const MgNodeInfo &info = mgPair.second;

    llvm::dbgs() << "\tCFDFC/Segment Label: " << mgLabel << "\n";
    // Print "original"
    llvm::dbgs() << "\t\tcontrol = [";
    for (size_t i = 0; i < info.original.size(); ++i) {
      llvm::dbgs() << info.original[i];
      if (i + 1 < info.original.size()) llvm::dbgs() << ", ";
    }
    llvm::dbgs() << "]\n";
    // Print "glitch"
    llvm::dbgs() << "\t\tdata = [";
    for (size_t i = 0; i < info.glitch.size(); ++i) {
      llvm::dbgs() << info.glitch[i];
      if (i + 1 < info.glitch.size()) llvm::dbgs() << ", ";
    }
    llvm::dbgs() << "]\n";
    // Print data_width
    llvm::dbgs() << "\t\tdata_width:\n";
    for (const auto &dw : info.dataWidthMap) {
      llvm::dbgs() << "\t\t  " << dw.first << " => " << dw.second << "\n";
    }
  }

  // if this is a control merge node
  if (controlGlitchVec.size()) {
    llvm::dbgs() << "\tControl glitch Dataout:\n\t";
    for (const auto &kv : controlGlitchVec) {
      llvm::dbgs() << std::to_string(kv) << " ,";
    }
    llvm::dbgs() << "\n";
  }
}

int CMergeData::getControlOutput(unsigned selIter) {
  if (controlDataOut.find(selIter) != controlDataOut.end()) {
    return controlDataOut[selIter].value;
  } else {
    return 0;
  }
}

unsigned getExecutionIter(unsigned bbIndex, unsigned curBB, SCFProfilingResult &profileResults) {
  unsigned iterIndex = profileResults.bbToIterMap[bbIndex];

  if (curBB == profileResults.executedBBTrace[iterIndex - 1] || (bbIndex == 1)) {
    return iterIndex - 1;
  }
  return iterIndex;
}

void dataChannelBaseNodesValueUpdate(SwitchingInfo &switchInfo, SCFProfilingResult &profileResults) {
  // [Step 1]
  // Iterate all profile base node in the dataflow graph
  for (const auto& selNode: switchInfo.dataflowGraph->profileBaseNodes) {
    // Define tmp storing structure
    switchInfo.dfgBaseNodeValueMap[selNode] = std::make_shared<DataBase>(selNode);

    // Check the type of node
    if (selNode.find("cmp") != std::string::npos) {
      for (const auto& [value, iterIdx] : profileResults.opNameToValueListMap[selNode]) {
        // Create the value struct
        ValueIter tmpValuePair = {std::abs(value), iterIdx};
        switchInfo.dfgBaseNodeValueMap[selNode]->originalDataOut[iterIdx] = tmpValuePair;
      }

      //! Testing
      // switchInfo.dfgBaseNodeValueMap[selNode]->printDetail();
    } else if (selNode.find("constant") != std::string::npos) {
      // We need to get the constant value from the attribute
      // Get the mlir op
      auto nodeOp = dyn_cast<handshake::ConstantOp>(switchInfo.dataflowGraph->nodes[selNode]->op);
      auto valueAttr = nodeOp->getAttrOfType<IntegerAttr>("value");
      if (!valueAttr) {
        llvm::errs() << "[ERROR] Can't get the value for the constant op\n";
      }
      int constantValue = valueAttr.getInt();
      
      // Update the value in the vec list
      ValueIter tmpValuePair = {constantValue, 0};
      switchInfo.dfgBaseNodeValueMap[selNode]->originalDataOut[0] = tmpValuePair;

      //! Testing
      // switchInfo.dfgBaseNodeValueMap[selNode]->printDetail();
    } else if (selNode.find("source") != std::string::npos) {
      ValueIter tmpValuePair = {0, 0};
      switchInfo.dfgBaseNodeValueMap[selNode]->originalDataOut[0] = tmpValuePair;
    } else {
      for (const auto& [value, iterIdx] : profileResults.opNameToValueListMap[selNode]) {
        ValueIter tmpValuePair = {value, iterIdx};
        switchInfo.dfgBaseNodeValueMap[selNode]->originalDataOut[iterIdx] = tmpValuePair;
      }

      //! Testing
      // switchInfo.dfgBaseNodeValueMap[selNode]->printDetail();
    }
  }

  // [Step 2]
  // Iterate over all CMerge and MUX node
  for (const auto& selNode: switchInfo.dataflowGraph->allDataBaseNode) {
    // Check the type of the nodes
    if (selNode.find("control_merge") != std::string::npos) {
      switchInfo.dfgBaseNodeValueMap[selNode] = std::make_shared<CMergeData>(selNode);
    } else if (selNode.find("mux") != std::string::npos) {
      switchInfo.dfgBaseNodeValueMap[selNode] = std::make_shared<DataBase>(selNode);
    }
  }

  // Define temporary storing structure
  std::map<std::string, int> tmpMuxOutputMap;

  // [Step 3]
  // Define 
  // Traverse the executed BB trace
  // Skip the first BB, as it will always be BB 0
  for (unsigned i = 1; i < profileResults.executedBBTrace.size(); i++) {
    unsigned preBB = profileResults.executedBBTrace[i - 1];
    unsigned curBB = profileResults.executedBBTrace[i];

    std::pair<unsigned,unsigned> key_pair(preBB, curBB);

    // Get the corresponding control merge value
    // [Step 3.1] We first update the value of all influenced control_merge node in the circuit
    auto itCM = switchInfo.bbPairToCMResultMap.find(key_pair);
    if (itCM != switchInfo.bbPairToCMResultMap.end()) {
      // Iterate over all (CMNode, output value) pairs
      // TODO: Check whether the value stored match the simulation or not
      for (auto selValuePair : itCM->second) {
        auto nodeName = selValuePair.first;
        int  outValue = selValuePair.second;
        unsigned iterIndex = getExecutionIter(i, curBB, profileResults);
        
        //! Testing
        // llvm::dbgs() << "[DEBUG] \t\tConMerge Node: " << nodeName << "\n";
        // llvm::dbgs() << "[DEBUG] \t\t\tCon output Value: " << outValue << "\n";
        // llvm::dbgs() << "[DEBUG] \t\t\tIter Index: " << iterIndex << "\n";
        // llvm::dbgs() << "[DEBUG] \t\t\tIter Index: " << iterIndex << "\n";

        // Update the value
        auto selConMergeNode = dyn_cast<CMergeData>(switchInfo.dfgBaseNodeValueMap[nodeName].get());

        // Data output for the control merge node, we assign -1 to it 
        ValueIter tmpDataValuePair = {-1, iterIndex};
        selConMergeNode->originalDataOut[iterIndex] = tmpDataValuePair;

        // Control output
        ValueIter tmpConValuePair = {outValue, iterIndex};
        selConMergeNode->controlDataOut[iterIndex] = tmpConValuePair;

        // Define the Vector for unupdated mux
        std::vector<std::string> tmpUnUpdatedMux;
        // [Step 3.2] Update all related MUX node
        // Get the list of influenced mux node
        std::vector<std::string> selMuxVec = switchInfo.dataflowGraph->cmToMuxMap[nodeName];
        // We may encounter different situation when updating the value for MUX nodes
        for (const auto& selMuxNode: selMuxVec) {
          auto selMuxSrcMap = switchInfo.dataflowGraph->muxToSrcNodeMap[selMuxNode];
          std::string selMuxDataSrcNode = selMuxSrcMap[std::to_string(outValue)];

          int tmpMuxOutput = 0;

          //! Testing
          // llvm::dbgs() << "[DEBUG] \t\t\t\tMux Node: " << selMuxNode << "; Sel Data Src Node: " << selMuxDataSrcNode << "\n";

          if (selMuxDataSrcNode.find("constant") != std::string::npos || (selMuxDataSrcNode.find("source") != std::string::npos)) {
            tmpMuxOutput = switchInfo.dfgBaseNodeValueMap[selMuxDataSrcNode]->originalDataOut[0].value;
          } else {
            // If the src node is a mux node
            // TODO: Check the following update logic
            if (selMuxDataSrcNode.find("mux") != std::string::npos) {
              // Case 1: the source node is the mux node itself
              if (selMuxDataSrcNode == selMuxNode){
                // Use the previous value
                if (tmpMuxOutputMap.find(selMuxDataSrcNode) != tmpMuxOutputMap.end()) {
                  tmpMuxOutput = tmpMuxOutputMap[selMuxDataSrcNode];
                } else {
                  llvm::dbgs() << "[ERROR-line367] Data src(" << selMuxDataSrcNode << ") of Mux Node: " << selMuxDataSrcNode << ", not found in tmpMuxOutputMap.\n";
                }
              } else if (switchInfo.dfgBaseNodeValueMap[selMuxDataSrcNode]->originalDataOut.find(iterIndex) != switchInfo.dfgBaseNodeValueMap[selMuxDataSrcNode]->originalDataOut.end()) {
                tmpMuxOutput = switchInfo.dfgBaseNodeValueMap[selMuxDataSrcNode]->originalDataOut[iterIndex].value;
              } else if (tmpMuxOutputMap.find(selMuxDataSrcNode) != tmpMuxOutputMap.end()) {
                tmpMuxOutput = tmpMuxOutputMap[selMuxDataSrcNode];
              } else {
                // TODO: Check the following update logic
                ValueIter tmpInnerMuxValuePair = {tmpMuxOutput, iterIndex - 1};
                switchInfo.dfgBaseNodeValueMap[selMuxDataSrcNode]->originalDataOut[iterIndex - 1] = tmpInnerMuxValuePair;
                tmpUnUpdatedMux.push_back(selMuxNode);

                continue;
              }
            } else {
              tmpMuxOutput = switchInfo.dfgBaseNodeValueMap[selMuxDataSrcNode]->originalDataOut[iterIndex].value;
            }
          }

          // Update the storing structure and the tmp dict
          //! Testing
          // llvm::dbgs() << "[DEBUG] \t\t\t\t[OutValue] " << tmpMuxOutput << "\n";

          ValueIter tmpMuxValuePair = {tmpMuxOutput, iterIndex};
          switchInfo.dfgBaseNodeValueMap[selMuxNode]->originalDataOut[iterIndex] = tmpMuxValuePair;
          tmpMuxOutputMap[selMuxNode] = tmpMuxOutput;
        }

        //! Testing
        // llvm::dbgs() << "[DEBUG] \t\t\tRemaining Mux Nodes:\n";
        // llvm::dbgs() << "[DEBUG] \t\t\t\tTmp Mux Output Map\n";
        // for (const auto& [key, value] : tmpMuxOutputMap) {
        //   llvm::dbgs() << "[DEBUG] \t\t\t\t\tNode: " << key << ", Value: " << value << "\n";
        // }

        // Update all remaining mux nodes
        // TODO: This maybe useless, please check
        for (const auto& leftMuxNode: tmpUnUpdatedMux) {
          auto selMuxSrcMap = switchInfo.dataflowGraph->muxToSrcNodeMap[leftMuxNode];
          std::string selMuxDataSrcNode = selMuxSrcMap[std::to_string(outValue)];
          int tmpMuxOutput = 0;

          if (tmpMuxOutputMap.find(selMuxDataSrcNode) != tmpMuxOutputMap.end()) {
            tmpMuxOutput = tmpMuxOutputMap[selMuxDataSrcNode];
          } else {
            llvm::dbgs() << "[ERROR-line407] Data src(" << selMuxDataSrcNode << ") of Mux Node: " << leftMuxNode << ", not found in tmpMuxOutputMap.\n";
          }
          
          ValueIter tmpMuxValuePair = {tmpMuxOutput, iterIndex};
          switchInfo.dfgBaseNodeValueMap[leftMuxNode]->originalDataOut[iterIndex] = tmpMuxValuePair;
        }
      }
    }
  }

}

void conSegSuccNodesList(SwitchingInfo &switchInfo, SCFProfilingResult &profileResults) {
  //! Testing
  for (const auto& [label, bblist]: switchInfo.segToBBListMap) {
    llvm::dbgs() << "[DEBUG] \t\tSeg Label: " << label << "\n";
    llvm::dbgs() << "[DEBUG] \t\t\tBB List: ";
    for (const auto& selBB : bblist) {
      llvm::dbgs() << selBB << ", ";
    }
    llvm::dbgs() << "\n";
  }

  for (const auto& selBaseNode: switchInfo.dataflowGraph->allDataBaseNode) {
    //! Testing
    // llvm::dbgs() << "[DEBUG] Node Name: " << selBaseNode << "\n";

    // Check the existence of the selected node
    if (switchInfo.dataflowGraph->nodes.find(selBaseNode) == switchInfo.dataflowGraph->nodes.end()) continue;

    if (selBaseNode.find("control_merge") != std::string::npos) {
      // If this is a control_merge node
      auto selCMNode = dyn_cast<CMergeNode>(switchInfo.dataflowGraph->nodes[selBaseNode].get());

      std::string dataSucNode = selCMNode->dataSucNodeName;
      std::string conSucNode = selCMNode->conSucNodeName;

      for (const auto& [label, bblist]: switchInfo.segToBBListMap) {
        MgNodeInfo tmpMGInfo;
        std::vector<std::string> controlExcludVec = {dataSucNode};
        std::vector<std::string> dataExcludVec = {conSucNode};
        tmpMGInfo.control = segConMergeSuccSearch(switchInfo, selBaseNode, controlExcludVec, label);
        tmpMGInfo.data = segConMergeSuccSearch(switchInfo, selBaseNode, dataExcludVec, label);
        tmpMGInfo.glitch = segConMergeGlitchSuccSearch(switchInfo, selBaseNode, controlExcludVec, label);

        // Update the stroing structure
        switchInfo.dfgBaseNodeValueMap[selBaseNode]->segSucNodeMap[label] = tmpMGInfo;
      }
    } else {
      for (const auto& [label, bblist]: switchInfo.segToBBListMap) {
        // Update the stroing structure
        switchInfo.dfgBaseNodeValueMap[selBaseNode]->segSucNodeMap[label] = segGeneralSuccSearch(switchInfo, selBaseNode, label);
      }
    }
  }
}

void dataGlitchNodeSearch(SwitchingInfo &switchInfo, SCFProfilingResult &profileResults) {
  for (const auto & [label, bblist] : switchInfo.segToBBListMap) {
    // Skip all "S", "E", and "T" sections
    if (label.find("S") != std::string::npos || label.find("E") != std::string::npos || label.find("T") != std::string::npos) {
      continue;
    }

    // Get the selected AdjGraph
    auto selAdjGraph = switchInfo.segToAdjGraphMap[label];
    std::string mgBaseNode = selAdjGraph->baseNode;
    auto mgII = switchInfo.cfdfcIIs[std::stoul(label)];

    //! Testing
    llvm::dbgs() << "[DEBUG] \t\tMG II: " << mgII << "\n";

    std::map<std::string, std::vector<NodeGlitchInfo>> tmpGlitchDict;
    std::vector<std::string> tmpGlitchNodes;

    // Iterate over all mapped nodes (orderedMappedList).
    for (const auto& selNode: switchInfo.segToAdjGraphMap[label]->orderedNodeName) {

      auto selDataBaseNodes = switchInfo.segToDataBaseVecMap[label].data;
      if (std::find(selDataBaseNodes.begin(), selDataBaseNodes.end(), selNode) != selDataBaseNodes.end()) {
        // Get the type of the node
        auto nodeType = getNodeType(selNode);

        if (GLITCH_NODE.find(nodeType) != GLITCH_NODE.end()) {
          // ALU will only have two inputs
          auto preNodeLists = selAdjGraph->nodes[selNode]->pres;

          // Unpack the pre_node_list
          std::string preNode1 = preNodeLists[0];
          std::string preNode2 = preNodeLists[1];

          // Get the source of the two inputs
          std::string srcNode1 = segNodeDataSrcSearch(switchInfo, preNode1, selAdjGraph.get());
          std::string srcNode2 = segNodeDataSrcSearch(switchInfo, preNode2, selAdjGraph.get());

          // Check whether this inode can have glitches
          LongestPathResult result1 = selLongestPath(switchInfo, preNode1, label);
          LongestPathResult result2 = selLongestPath(switchInfo, preNode2, label);

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

          for (const auto &selOriNode: switchInfo.dfgBaseNodeValueMap[srcNode1]->segSucNodeMap[label].original) {
            if (preNode1 == selOriNode) preSrc1Buffered = true;
          }

          for (const auto &selOriNode: switchInfo.dfgBaseNodeValueMap[srcNode2]->segSucNodeMap[label].original) {
            if (preNode2 == selOriNode) preSrc2Buffered = true;
          }

          // We assign the faster one with value 0 and slower one with value 1
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
            if (srcNode1.find("constant") == std::string::npos && srcNode2.find("constant") == std::string::npos) {
              tmpGlitchDict[selNode].push_back(tmpNode1);
              tmpGlitchDict[selNode].push_back(tmpNode2);
              tmpGlitchNodes.push_back(selNode);
            }
          } else {
            // Check Case 4
            if (std::find(tmpGlitchNodes.begin(), tmpGlitchNodes.end(), srcNode1) != tmpGlitchNodes.end() &&
                  (std::find(tmpGlitchNodes.begin(), tmpGlitchNodes.end(), srcNode2) == tmpGlitchNodes.end())){
              if (!preSrc1Buffered) {
                tmpGlitchDict[selNode].push_back(tmpNode1);
                tmpGlitchDict[selNode].push_back(tmpNode2);
                tmpGlitchNodes.push_back(selNode);
              }
            } else if (std::find(tmpGlitchNodes.begin(), tmpGlitchNodes.end(), srcNode2) != tmpGlitchNodes.end() &&
                  (std::find(tmpGlitchNodes.begin(), tmpGlitchNodes.end(), srcNode1) == tmpGlitchNodes.end())) {
              if (!preSrc2Buffered) {
                tmpGlitchDict[selNode].push_back(tmpNode1);
                tmpGlitchDict[selNode].push_back(tmpNode2);
                tmpGlitchNodes.push_back(selNode);
              }
            }
          }
        } 
      }
    }
    
    // Store the glitching info
    switchInfo.mgGlitchNodeDict[label] = tmpGlitchDict;
  }
}

int calGlitchValue(int op1, int op2, std::string selNode) {
  // TODO: Add supports for other types of nodes in self.GLITCH_NODE
  if (selNode.find("add") != std::string::npos) {
    return op1 + op2;
  } else if (selNode.find("mul") != std::string::npos) {
    return op1 * op2;
  } else {
    llvm::errs() << "[ERROR] Unexpected node type during glitching calculation: " << selNode << "\n";
    return 0;
  }
}

std::string getMuxDataSrc(SwitchingInfo &switchInfo, std::string selMuxNode, unsigned selIter) {
  auto selMuxSrcMap = switchInfo.dataflowGraph->muxToSrcNodeMap;
  // The cond input src will always be a control merge node
  std::string muxCondInput = selMuxSrcMap[selMuxNode]["control"];

  // TODO: Prevent the following situation from happening
  if (muxCondInput.find("buffer")) {
    llvm::errs() << "[ERROR] A buffer is directly preceding " << selMuxNode << "'s control input port\n";
  }
  // Get the corresponding Control Merge Data base node
  auto selCMBaseNode = dyn_cast<CMergeData>(switchInfo.dfgBaseNodeValueMap[muxCondInput].get());
  int controlInputValue = selCMBaseNode->getControlOutput(selIter);

  return selMuxSrcMap[selMuxNode][std::to_string(controlInputValue)];
}

void dataBaseNodeGlitchUpdate(SwitchingInfo &switchInfo, SCFProfilingResult &profileResults, bool debug) {
  if (debug) {
    llvm::dbgs() << "[DEBUG]\n[DEBUG] \t\t[NODE GLITCHING VALUE CALCULATION]\n";
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
      llvm::dbgs() << "[DEBUG] \t******** Iter: " << i << ", Seg: " << executedSeg << "\n";
    }

    // Check wehther we need to update the glitch value for this seg
    if (executedSeg.find("S") != std::string::npos || 
          executedSeg.find("E") != std::string::npos ||
          executedSeg.find("T") != std::string::npos) {
      glitchUpdateFlag = false;
    } else {
      glitchUpdateFlag = true;
    }

    // Step 1: Calculate glitching for inner MG nodes -- All ALUs, other nodes shall be excluded
    //! For this type of glitching nodes (only ALUs), we only deal with the following glitching cases:
    //!     - CASE 1: No glitching for the two input operands
    //!     - CASE 2: One of the operand is glitching, but the original arriving time of the two inputs are the same.
    //! We ignore all other glitching cases and directly use the value from software profiling(original value)
    // Iterate through all data base nodes
    std::vector<std::string> indexUpdateList;
    for (const auto& selNode: switchInfo.segToOrderedDataBaseNodes[executedSeg]) {
      if (selNode.find("constant") != std::string::npos ||
          selNode.find("source") != std::string::npos ||
          selNode.find("load") != std::string::npos) continue;

      indexUpdateList.push_back(selNode);

      // Define vector to store the tmp glitch value
      std::vector<int> tmpValue;

      // Check whether we need the glitch value calculation
      if (glitchUpdateFlag) {
        // If we have glitch info for this node
        if (switchInfo.mgGlitchNodeDict[executedSeg].find(selNode) != switchInfo.mgGlitchNodeDict[executedSeg].end()) {
          // Get the source node
          std::string preSrc1 = switchInfo.mgGlitchNodeDict[executedSeg][selNode][0].srcNode;
          std::string preSrc2 = switchInfo.mgGlitchNodeDict[executedSeg][selNode][1].srcNode;
          // Get the starting time
          int preSrc1Start = switchInfo.mgGlitchNodeDict[executedSeg][selNode][0].steadyTime;
          int preSrc2Start = switchInfo.mgGlitchNodeDict[executedSeg][selNode][1].steadyTime;
          std::map<std::string, int> srcStartTimeDict = {{preSrc1, preSrc1Start}, {preSrc2, preSrc2Start}};

          // Get the buffering information
          bool preSrc1Buffered = switchInfo.mgGlitchNodeDict[executedSeg][selNode][0].buffered;
          bool preSrc2Buffered = switchInfo.mgGlitchNodeDict[executedSeg][selNode][1].buffered;
          std::map<std::string, bool> srcBufferedDict = {{preSrc1, preSrc1Buffered}, {preSrc2, preSrc2Buffered}};

          // Defining variables for glitch calculation
          int op1 = 0, op2 = 0;
          unsigned op1PreIndex = 0, op2PreIndex = 0;

          // Status Definition
          std::string fasterNode = "";
          std::string slowerNode = "";


          //! Testing
          if (debug) {
            llvm::dbgs() << "[DEBUG] \t\tGlitch Node: " << selNode << "\n"
                      << "[DEBUG] \t\t\tPre_src_1: " << preSrc1 << "\n"
                      << "[DEBUG] \t\t\tPre_src_2: " << preSrc2 << "\n";
          }

          // If this is the last iteration, we ignore the glitching value, just copy the original value
          if (i == segExecTrace.size() - 1) {
            //! Testing
            if (debug) {
              llvm::dbgs() << "[DEBUG] \t\t(LAST ITER DURING GLITCH CALCULATION)\n";
            }

            switchInfo.dfgBaseNodeValueMap[selNode]->oriGlitchDataOut[i] = 
                {switchInfo.dfgBaseNodeValueMap[selNode]->originalDataOut[i].value};
            switchInfo.dfgBaseNodeValueMap[selNode]->lastUpdateIndex = i;
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
              op1PreIndex = switchInfo.dfgBaseNodeValueMap[fasterNode]->lastUpdateIndex;
              op2PreIndex = switchInfo.dfgBaseNodeValueMap[slowerNode]->lastUpdateIndex;

              //! Testing
              if (debug) {
                llvm::dbgs() << "[DEBUG] \t\t\t[Value 1]: \n"
                          << "[DEBUG] \t\t\t\tOp_1_src_node: " << fasterNode << "\n"
                          << "[DEBUG] \t\t\t\tOp_1_pre_index: " << op1PreIndex << "\n"
                          << "[DEBUG] \t\t\t\tOp_2_src_node: " << slowerNode << "\n"
                          << "[DEBUG] \t\t\t\tOp_2_pre_index: " << op2PreIndex << "\n";
              }

              // Check the existence of the selected iter
              if (switchInfo.dfgBaseNodeValueMap[fasterNode]->originalDataOut.find(op1PreIndex) != switchInfo.dfgBaseNodeValueMap[fasterNode]->originalDataOut.end()) {
                op1 = switchInfo.dfgBaseNodeValueMap[fasterNode]->originalDataOut[op1PreIndex].value;
              } else {
                op1 = 0;
              }

              // Get operand 2, if op_2_pre not in original_dataout keys, we use 0 instead
              if (switchInfo.dfgBaseNodeValueMap[slowerNode]->originalDataOut.find(op2PreIndex) == switchInfo.dfgBaseNodeValueMap[slowerNode]->originalDataOut.end()) {
                //! Testing
                if (debug) {
                  llvm::dbgs() << "[DEBUG] \t\t\t[Value 1]: \n"
                              << "[DEBUG] \t\t\t\t[Warning] S Last Active Iter Not in the corresponding storing structure\n";
                }
                op2 = 0;
              } else {
                op2 = switchInfo.dfgBaseNodeValueMap[slowerNode]->originalDataOut[op2PreIndex].value;
              }

              tmpValue.push_back(calGlitchValue(op1, op2, selNode));

              //! Testing
              if (debug) {
                llvm::dbgs() << "[DEBUG] \t\t\t\tFaster Node: " << fasterNode << "\n"
                          << "[DEBUG] \t\t\t\tOp_1: " << op1 << "\n"
                          << "[DEBUG] \t\t\t\tSlower Node: " << slowerNode << "\n"
                          << "[DEBUG] \t\t\t\tOp_2: " << op2 << "\n";
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
            op1PreIndex = switchInfo.dfgBaseNodeValueMap[fasterNode]->lastUpdateIndex;
            op2PreIndex = switchInfo.dfgBaseNodeValueMap[slowerNode]->lastUpdateIndex;

            // Get the value of op1
            if (fasterNode == selNode) {
              if (switchInfo.dfgBaseNodeValueMap[fasterNode]->originalDataOut.find(op1PreIndex) != switchInfo.dfgBaseNodeValueMap[fasterNode]->originalDataOut.end()) {
                op1 = switchInfo.dfgBaseNodeValueMap[fasterNode]->originalDataOut[op1PreIndex].value;
              }
            } else if (fasterNode.find("mux")) {
              // Check whether the source of mux node is the selected node itself
              std::string tmpMuxDataSrcNode = getMuxDataSrc(switchInfo, fasterNode, i);

              if (tmpMuxDataSrcNode == selNode) {
                // TODO: Validate the following checking mechanism
                if (switchInfo.dfgBaseNodeValueMap[fasterNode]->originalDataOut.find(op1PreIndex) == switchInfo.dfgBaseNodeValueMap[fasterNode]->originalDataOut.end()) {
                  op1 = 0;
                } else op1 = switchInfo.dfgBaseNodeValueMap[fasterNode]->originalDataOut[op1PreIndex].value;
              } else {
                op1 = switchInfo.dfgBaseNodeValueMap[fasterNode]->originalDataOut[i].value;
              }
            } else if (switchInfo.mgGlitchNodeDict[executedSeg].find(fasterNode) != switchInfo.mgGlitchNodeDict[executedSeg].end()) {
              // If the faster node is glitching, we check whether the node is buffered or not
              if (srcBufferedDict[fasterNode]) {
                // Buffered
                op1 = switchInfo.dfgBaseNodeValueMap[fasterNode]->originalDataOut[i].value;
              } else {
                unsigned tmpSize = switchInfo.dfgBaseNodeValueMap[fasterNode]->oriGlitchDataOut[i].size();

                // ! If the list is smaller than 2, something is wrong
                if (tmpSize < 2) {
                  llvm::errs() << "[ERROR] Glitch Vector for " << selNode << " has a size smaller than 2\n";
                }
                op1 = switchInfo.dfgBaseNodeValueMap[fasterNode]->oriGlitchDataOut[i][tmpSize - 2];
              }
            } else {
              op1 = switchInfo.dfgBaseNodeValueMap[fasterNode]->originalDataOut[i].value;
            }

            // Get the value of op2
            if (switchInfo.dfgBaseNodeValueMap[slowerNode]->originalDataOut.find(op2PreIndex) == switchInfo.dfgBaseNodeValueMap[slowerNode]->originalDataOut.end()) {
              op2 = switchInfo.dfgBaseNodeValueMap[slowerNode]->originalDataOut[op2PreIndex].value;
            }

            tmpValue.push_back(calGlitchValue(op1, op2, selNode));

            //! Testing
            if (debug) {
              llvm::dbgs() << "[DEBUG] \t\t\t\tFaster Node: " << fasterNode << "\n"
                        << "[DEBUG] \t\t\t\tF Last Active Iter: " << op1PreIndex << "\n"
                        << "[DEBUG] \t\t\t\tOp_1: " << op1 << "\n"
                        << "[DEBUG] \t\t\t\tSlower Node: " << slowerNode << "\n"
                        << "[DEBUG] \t\t\t\tS Last Active Iter: " << op2PreIndex << "\n"
                        << "[DEBUG] \t\t\t\tOp_2: " << op2 << "\n";
            }

            // Value 3: F[x] op S[x]
            // Check wheter the faster node is a mux node
            if (fasterNode.find("mux") != std::string::npos) {
              std::string tmpMuxDataSrcNode = getMuxDataSrc(switchInfo, fasterNode, i);

              if (tmpMuxDataSrcNode == selNode) {
                op1PreIndex = switchInfo.dfgBaseNodeValueMap[fasterNode]->lastUpdateIndex;

                // TODO: Validate the following rounding process
                if (switchInfo.dfgBaseNodeValueMap[fasterNode]->originalDataOut.find(op1PreIndex) == switchInfo.dfgBaseNodeValueMap[fasterNode]->originalDataOut.end()) {
                  op1 = 0;
                } else {
                  op1 = switchInfo.dfgBaseNodeValueMap[fasterNode]->originalDataOut[op1PreIndex].value;
                }
              } else {
                op1 = switchInfo.dfgBaseNodeValueMap[fasterNode]->originalDataOut[i].value;
              }
            } else {
              op1 = switchInfo.dfgBaseNodeValueMap[fasterNode]->originalDataOut[i].value;
            }

            // Check whether the slower node is a mux node
            // TODO: Check the condition below
            op2 = switchInfo.dfgBaseNodeValueMap[slowerNode]->originalDataOut[i].value;

            tmpValue.push_back(calGlitchValue(op1, op2, selNode));

            if (debug) {
              llvm::dbgs() << "\t[Value 3]: \n"
                        << "\t\tOp_1: " << op1 << "\n"
                        << "\t\tOp_2: " << op2 << "\n"
                        << "\t\t[FINAL] ";
              for (auto v : tmpValue) llvm::dbgs() << v << " ";
              llvm::dbgs() << "\n";
            }
          }
        } else {
          tmpValue.push_back(switchInfo.dfgBaseNodeValueMap[selNode]->originalDataOut[i].value);
        }
      } else {
        // No neeed for glitch value calculation, we directly copy the ori data
        tmpValue.push_back(switchInfo.dfgBaseNodeValueMap[selNode]->originalDataOut[i].value);
      }

      // Update the storing structure
      switchInfo.dfgBaseNodeValueMap[selNode]->oriGlitchDataOut[i] = tmpValue;
    }

    // Step 1.5: Update the list of last update index for all data base nodes
    for (const auto& selNode: indexUpdateList) {
      switchInfo.dfgBaseNodeValueMap[selNode]->lastUpdateIndex = i;
    }

    // Step 2: Update value for all MUXs
    // TODO: Pre store the information like mux etc.
    for (const auto& selMuxNode: switchInfo.segToControlNodeList[executedSeg].muxNodeList) {
      std::string selCondInputNode = switchInfo.dataflowGraph->muxToSrcNodeMap[selMuxNode]["control"];

      // We need to check the existence of the control value output
      auto selCMBaseNode = dyn_cast<CMergeData>(switchInfo.dfgBaseNodeValueMap[selCondInputNode].get());
      int condValue = selCMBaseNode->getControlOutput(i);
      std::string selDataSrcNode = switchInfo.dataflowGraph->muxToSrcNodeMap[selMuxNode][std::to_string(condValue)];

      // Temp Value vector definition
      std::vector<int> preValue, curValue, nexValue;

      //! Testing
      if (debug) {
        llvm::dbgs() << "Mux Node: " << selMuxNode << "\n"
                    << "\t[CUR_VALUE]\n"
                    << "\t\tCond Node: " << selCondInputNode << "\n"
                    << "\t\tCond_value: " << condValue << "\n"
                    << "\t\tCur data src: " << selDataSrcNode << "\n";
      }

      // TODO: Validate the following indexing mechanism
      // Check whether we have glitches from the srcs or not
      if (selDataSrcNode.find("constant") != std::string::npos ||
          selDataSrcNode.find("source") != std::string::npos ||
          selDataSrcNode.find("start") != std::string::npos) {
        curValue.push_back(switchInfo.dfgBaseNodeValueMap[selDataSrcNode]->originalDataOut[0].value);
      } else if (selDataSrcNode == selMuxNode) {
        // The src node is the selected node itself
        unsigned preIndex = switchInfo.dfgBaseNodeValueMap[selDataSrcNode]->lastUpdateIndex;
        // Check whether the preIndex exists or not
        if (switchInfo.dfgBaseNodeValueMap[selDataSrcNode]->originalDataOut.find(preIndex) != switchInfo.dfgBaseNodeValueMap[selDataSrcNode]->originalDataOut.end()) {
          curValue.push_back(switchInfo.dfgBaseNodeValueMap[selDataSrcNode]->originalDataOut[preIndex].value);
        } else curValue.push_back(0);
      } else if (glitchUpdateFlag &&
                  (switchInfo.mgGlitchNodeDict[executedSeg].find(selDataSrcNode) != switchInfo.mgGlitchNodeDict[executedSeg].end())) {
        // Check whether there are buffers in between
        std::string preNode = "";
        auto selMuxNodeStructure = dyn_cast<MuxNode>(switchInfo.dataflowGraph->nodes[selMuxNode].get());

        // Get the actual preNode
        // TODO: Need to check the portidx to name mapping
        for (const auto& [nodeName, portIdx] : selMuxNodeStructure->preNameToPortIdxMap) {
          if (portIdx == condValue) preNode = nodeName;
        }

        bool bufferedFlag = false;
        if (std::find(switchInfo.dfgBaseNodeValueMap[selDataSrcNode]->segSucNodeMap[executedSeg].original.begin(),
              switchInfo.dfgBaseNodeValueMap[selDataSrcNode]->segSucNodeMap[executedSeg].original.end(), preNode) != 
              switchInfo.dfgBaseNodeValueMap[selDataSrcNode]->segSucNodeMap[executedSeg].original.end()) bufferedFlag = true;
        
        //! Testing
        if (debug) {
          llvm::dbgs() << "\t\tCur src node has glitches, buffered: " << bufferedFlag << "\n";
        }

        if (bufferedFlag) {
          // Src glitching but buffered
          curValue.push_back(switchInfo.dfgBaseNodeValueMap[selDataSrcNode]->originalDataOut[i].value);
        } else {
          curValue = switchInfo.dfgBaseNodeValueMap[selDataSrcNode]->oriGlitchDataOut[i];
        }
      } else {
        // Handling special case for cascaded Muxes
        if (selDataSrcNode.find("mux") != std::string::npos && 
            (switchInfo.dfgBaseNodeValueMap[selDataSrcNode]->originalDataOut.find(i) == switchInfo.dfgBaseNodeValueMap[selDataSrcNode]->originalDataOut.end())) {
          curValue.push_back(switchInfo.dfgBaseNodeValueMap[selDataSrcNode]->originalDataOut[i - 1].value);
        } else {
          curValue.push_back(switchInfo.dfgBaseNodeValueMap[selDataSrcNode]->originalDataOut[i].value);
        }
      }

      //! Testing
      if (debug) {
        llvm::dbgs() << "\t\tCur_value: ";
        for (auto v : curValue) llvm::dbgs() << v << " ";
        llvm::dbgs() << "\n\t[TRANSATION GLITCHES]\n";
      }

      // Calculate control flow glitches
      if (!switchInfo.dfgBaseNodeValueMap[selMuxNode]->skipControlCal) {
        std::string nodePreValidSeg = switchInfo.dfgBaseNodeValueMap[selMuxNode]->lastValidSeg;
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
            if (std::find(switchInfo.segToControlNodeList[segExecTrace[i + 1]].muxNodeList.begin(), 
                        switchInfo.segToControlNodeList[segExecTrace[i + 1]].muxNodeList.end(), selMuxNode) != switchInfo.segToControlNodeList[segExecTrace[i + 1]].muxNodeList.end()) {
              std::string nextExecSegLabel = segExecTrace[i + 1];

              if (nextExecSegLabel != executedSeg) {
                // Type 2 detected
                doubleTransFlag = true;
              }
            }
          }

          // Calculate preList
          if (executedSeg.find("E") == std::string::npos) {
            // Cond value will be the same for the last segment
            int preCondValue = 1 - condValue;
            std::string preDataSrc = switchInfo.dataflowGraph->muxToSrcNodeMap[selMuxNode][std::to_string(preCondValue)];

            // Check whether the value exist or not
            if (preDataSrc.find("constant") != std::string::npos) {
              preValue.push_back(switchInfo.dfgBaseNodeValueMap[preDataSrc]->originalDataOut[0].value);
            } else if (switchInfo.dfgBaseNodeValueMap[preDataSrc]->originalDataOut.find(i) == switchInfo.dfgBaseNodeValueMap[preDataSrc]->originalDataOut.end()) {
              preValue.push_back(switchInfo.dfgBaseNodeValueMap[preDataSrc]->originalDataOut[i].value);
            }
          } 

          // If double transition
          if (doubleTransFlag) {
            switchInfo.dfgBaseNodeValueMap[selMuxNode]->skipControlCal = true;
            nexValue = preValue;
          }
          // TODO: Check the following CondValue condition, it maybe 0 now
        } else if (nodePreValidSeg != "" && condValue == 1) {
          //! Con_Merge's control output automatically go back to 0 need to check why!!!!!!!
          if (executedSeg.find("E") == std::string::npos) {
            // Cond value will be the same for the last segment
            int preCondValue = 1 - condValue;
            std::string preDataSrc = switchInfo.dataflowGraph->muxToSrcNodeMap[selMuxNode][std::to_string(preCondValue)];
            unsigned newIterIdx = i;

            // If we have the same sources for both cond_value
            if (preDataSrc == selDataSrcNode) {
              // Change the iterindex
              if (switchInfo.segToBackedgePairMap.find(executedSeg) != switchInfo.segToBackedgePairMap.end()) {
                auto mgIndexList = switchInfo.backEdgeToCFDFCMap[switchInfo.segToBackedgePairMap[executedSeg]];
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
            if (preDataSrc.find("constant")){
              preValue.push_back(switchInfo.dfgBaseNodeValueMap[preDataSrc]->originalDataOut[0].value);
            } else if (switchInfo.dfgBaseNodeValueMap[preDataSrc]->originalDataOut.find(i) != switchInfo.dfgBaseNodeValueMap[preDataSrc]->originalDataOut.end()) {
              preValue.push_back(switchInfo.dfgBaseNodeValueMap[preDataSrc]->originalDataOut[i].value);
            }
          }
        }

        //! Testing
        if (debug) {
          llvm::dbgs() << "\t\tDouble transition: " << doubleTransFlag << "\n"
                        << "\t\tPre_value: ";
              for (auto v : preValue) llvm::dbgs() << v << " ";
              llvm::dbgs() << "\n\t\tNext value: ";
              for (auto v : nexValue) llvm::dbgs() << v << " ";
              llvm::dbgs() << "\n";
        }
      } else {
        switchInfo.dfgBaseNodeValueMap[selMuxNode]->skipControlCal = false;
      }

      // Get the final mux output data list
      std::vector<int> finalMuxOutputList;

      for (const auto& selValue: preValue) finalMuxOutputList.push_back(selValue);
      for (const auto& selValue: curValue) finalMuxOutputList.push_back(selValue);
      for (const auto& selValue: nexValue) finalMuxOutputList.push_back(selValue);
    }

    // Step 3: Relay memory load node's data
    for (const auto& selNode: switchInfo.segToOrderedDataBaseNodes[executedSeg]) {
      if (selNode.find("load") != std::string::npos) {
        switchInfo.dfgBaseNodeValueMap[selNode]->lastUpdateIndex = i;
      }
    }
  }
}

void dfgDataChannelPropagate(SwitchingInfo &switchInfo, SCFProfilingResult &profileResults) {
  
}

//===----------------------------------------------------------------------===//
//
// Functions for finding the data source node in different segments
//
//===----------------------------------------------------------------------===//
// TODO: Merge all following functions into a single function

std::vector<std::string> segConMergeSuccSearch(SwitchingInfo &switchInfo, std::string startNode, 
                                                std::vector<std::string> &excludingList, std::string segLabel) {
  std::vector<std::string> succNodeList;

  // DFS STACK: None recrusive approach
  std::vector<std::string> mainStack;
  std::vector<std::vector<std::string>> adjStack;
  std::vector<unsigned> segBBList = switchInfo.segToBBListMap[segLabel];

  // Initialization
  mainStack.push_back(startNode);

  // Construct AdjList
  std::vector<std::string> initAdjList;
  for (const auto& selNode: switchInfo.dataflowGraph->nodes[startNode]->sucs) {
    if (std::find(excludingList.begin(), excludingList.end(), selNode) == excludingList.end() &&
        (switchInfo.dataflowGraph->allDataBaseNode.find(selNode) == switchInfo.dataflowGraph->allDataBaseNode.end())) initAdjList.push_back(selNode);
  }
  adjStack.push_back(initAdjList);

  // We conduct dfs starting from the specified start node in the dfg
  // The search will stop when encountered following situations:
  //  - Encountered a node in the data base node list
  //  - Entering a cond_br / mux node through the conditional channel
  //  - No node left
  // We also check all buffer nodes, whether it's inside a specific cfdfc or not
  while (!mainStack.empty()) {
    std::vector<std::string> curAdjList = adjStack.back();
    adjStack.pop_back();

    if (!curAdjList.empty()) {
      std::string curNode = curAdjList.back();
      curAdjList.pop_back();

      // Update the adjStack
      adjStack.push_back(curAdjList);

      // Flag for conditional port
      bool nonCondFlag = true;
      bool computeFlag = false;

      // Update compute flag
      if (switchInfo.dataflowGraph->allDataBaseNode.find(curNode) == switchInfo.dataflowGraph->allDataBaseNode.end()) {
        // This is not a invalid backedge
        auto selInvalidBEList = switchInfo.segInvalidBackedgesMap[segLabel];
        std::pair<std::string, std::string> selPair = std::make_pair(mainStack.back(), curNode);
        if (std::find(selInvalidBEList.begin(), selInvalidBEList.end(), selPair) == selInvalidBEList.end()) computeFlag = true;
      }

      // Check whether we can update mainStack or not
      if (computeFlag) {
        std::string preNode = mainStack.back();

        if (curNode.find("cond_br") != std::string::npos) {
          // Check whether this is the conditional channel
          auto selNode = dyn_cast<CBrNode>(switchInfo.dataflowGraph->nodes[curNode].get());
          if (preNode == selNode->condPreNodeName) {
            // TODO: Check this updating condition
            if (selNode->dataPreNodeName != "") {
              nonCondFlag = false;
            }
          }
        } else if (curNode.find("mux") != std::string::npos) {
          // Check whether this is the conditional channel
          auto selNode = dyn_cast<MuxNode>(switchInfo.dataflowGraph->nodes[curNode].get());
          if (preNode == selNode->conPreNodeName) nonCondFlag = false;
        } else if (curNode.find("buffer") != std::string::npos) {
          unsigned curNodeBB = switchInfo.dataflowGraph->nodes[curNode]->bbindex;

          if (std::find(segBBList.begin(), segBBList.end(), curNodeBB) == segBBList.end()) nonCondFlag = false;
        }

        // If this is the data channel
        if (nonCondFlag) {
          mainStack.push_back(curNode);
          succNodeList.push_back(curNode);

          // Insert new adjStack
          std::vector<std::string> tmpAdjList;
          auto newAdjList = switchInfo.dataflowGraph->nodes[curNode]->sucs;

          for (const auto& selNextNode: newAdjList) {
            bool inStack = (std::find(mainStack.begin(), mainStack.end(), selNextNode) != mainStack.end());
            bool inBaseNodeList = (switchInfo.dataflowGraph->allDataBaseNode.find(selNextNode) != switchInfo.dataflowGraph->allDataBaseNode.end());
            bool isExcluded = (std::find(excludingList.begin(), excludingList.end(), selNextNode) != excludingList.end());
            if (!inStack && !inBaseNodeList && !isExcluded) tmpAdjList.push_back(selNextNode);
          }

          adjStack.push_back(tmpAdjList);
        }
      }
    } else {
      mainStack.pop_back();
    }
  }

  return succNodeList;
}

MgNodeInfo segGeneralSuccSearch(SwitchingInfo &switchInfo, std::string startNode, std::string segLabel) {
  // Status variale definition
  unsigned numBuffers = 0;
  unsigned minDataWidth = 32;
  std::string lastNode = "";

  // Initialize the stroing structure
  MgNodeInfo tmpMgNodeInfo;

  // DFS STACK: None recrusive approach
  std::vector<std::string> mainStack;
  std::vector<std::vector<std::string>> adjStack;
  std::vector<unsigned> segBBList = switchInfo.segToBBListMap[segLabel];

  // Initialization
  mainStack.push_back(startNode);

  // Construct AdjList
  std::vector<std::string> initAdjList;
  for (const auto& selNode: switchInfo.dataflowGraph->nodes[startNode]->sucs) {
    if (switchInfo.dataflowGraph->allDataBaseNode.find(selNode) == switchInfo.dataflowGraph->allDataBaseNode.end()) initAdjList.push_back(selNode);
  }
  adjStack.push_back(initAdjList);

  // We conduct dfs starting from the specified start node in the dfg
  // The search will stop when encountered following situations:
  //  - Encountered a node in the data base node list
  //  - Entering a cond_br / mux node through the conditional channel
  //  - No node left
  // We also check all buffer nodes, whether it's inside a specific cfdfc or not
  while (!mainStack.empty()) {
    std::vector<std::string> curAdjList = adjStack.back();
    adjStack.pop_back();

    if (!curAdjList.empty()) {
      std::string curNode = curAdjList.back();
      curAdjList.pop_back();

      // Update the adjStack
      adjStack.push_back(curAdjList);

      // Flag for conditional port
      bool nonCondFlag = true;
      bool computeFlag = false;

      // Update compute flag
      if (switchInfo.dataflowGraph->allDataBaseNode.find(curNode) == switchInfo.dataflowGraph->allDataBaseNode.end()) {
        // This is not a invalid backedge
        auto selInvalidBEList = switchInfo.segInvalidBackedgesMap[segLabel];
        std::pair<std::string, std::string> selPair = std::make_pair(mainStack.back(), curNode);
        if (std::find(selInvalidBEList.begin(), selInvalidBEList.end(), selPair) == selInvalidBEList.end()) computeFlag = true;
      }

      // Check whether we can update mainStack or not
      if (computeFlag) {
        std::string preNode = mainStack.back();

        if (curNode.find("cond_br") != std::string::npos) {
          auto selNode = dyn_cast<CBrNode>(switchInfo.dataflowGraph->nodes[curNode].get());
          if (preNode == selNode->condPreNodeName) {
            // TODO: Check this updating condition
            if (selNode->dataPreNodeName != "") {
              nonCondFlag = false;
            }
          }
        } else if (curNode.find("mux") != std::string::npos) {
          // Check whether this is the conditional channel
          auto selNode = dyn_cast<MuxNode>(switchInfo.dataflowGraph->nodes[curNode].get());
          if (preNode == selNode->conPreNodeName) nonCondFlag = false;
        } else if (curNode.find("buffer") != std::string::npos) {
          // TODO: Maybe we need to check not only the name also the type, only opaque buffer is blocking the transmission of the data
          unsigned curNodeBB = switchInfo.dataflowGraph->nodes[curNode]->bbindex;
          if (std::find(segBBList.begin(), segBBList.end(), curNodeBB) == segBBList.end()) nonCondFlag = false;
        } else if (curNode.find("end") != std::string::npos && (segLabel != "E")) {
          // TODO: Check the following update logic
          nonCondFlag = false;
        } else if (curNode.find("mem_controller") != std::string::npos) {
          // TODO: Check the following update logic
          nonCondFlag = false;
        }
        
        // If this is a data channel
        if (nonCondFlag) {
          mainStack.push_back(curNode);

          // Update the buffer information
          if (curNode.find("buffer") != std::string::npos) numBuffers += 1;

          // Update the bitwidth information
          unsigned curNodeBitWidth = switchInfo.dataflowGraph->nodes[preNode]->sucsDataWidthMap[curNode];
          if (curNodeBitWidth < minDataWidth) minDataWidth = curNodeBitWidth;

          if (numBuffers > 0) {
            tmpMgNodeInfo.original.push_back(curNode);
            lastNode = curNode;

            // Update the bitwidth infomration
            tmpMgNodeInfo.dataWidthMap[preNode] = minDataWidth;
          } else {
            tmpMgNodeInfo.glitch.push_back(curNode);
            lastNode = curNode;

            // Update the bitwidth infomration
            tmpMgNodeInfo.dataWidthMap[preNode] = minDataWidth;
          }

          // Insert new AdjList
          std::vector<std::string> tmpAdjList;
          auto newAdjList = switchInfo.dataflowGraph->nodes[curNode]->sucs;

          for (const auto& selNextNode: newAdjList) {
            bool inStack = (std::find(mainStack.begin(), mainStack.end(), selNextNode) != mainStack.end());
            bool inBaseNodeList = (switchInfo.dataflowGraph->allDataBaseNode.find(selNextNode) != switchInfo.dataflowGraph->allDataBaseNode.end());
            if (!inStack && !inBaseNodeList) tmpAdjList.push_back(selNextNode);
          }

          adjStack.push_back(tmpAdjList);
        }
      }
    } else {
      std::string lastMainStackNode = mainStack.back();
      if (lastMainStackNode.find("buffer") != std::string::npos) numBuffers -= 1;

      mainStack.pop_back();
    }

    // Add the last unit to the data_width list
    tmpMgNodeInfo.dataWidthMap[lastNode] = minDataWidth;
  }

  // Return the storing structure
  return tmpMgNodeInfo;
}

std::vector<std::string> segConMergeGlitchSuccSearch(SwitchingInfo &switchInfo, std::string startNode, 
                                                      std::vector<std::string> &excludingList, std::string segLabel) {
  std::vector<std::string> glitchSuccNodeList;

  //
  unsigned numBuffers = 0;
  // DFS STACK: None recrusive approach
  std::vector<std::string> mainStack;
  std::vector<std::vector<std::string>> adjStack;
  std::vector<unsigned> segBBList = switchInfo.segToBBListMap[segLabel];

  // Initialization
  mainStack.push_back(startNode);

  // Construct AdjList
  std::vector<std::string> initAdjList;
  for (const auto& selNode: switchInfo.dataflowGraph->nodes[startNode]->sucs) {
    if (std::find(excludingList.begin(), excludingList.end(), selNode) == excludingList.end() &&
        (switchInfo.dataflowGraph->allDataBaseNode.find(selNode) == switchInfo.dataflowGraph->allDataBaseNode.end())) initAdjList.push_back(selNode);
  }
  adjStack.push_back(initAdjList);

  // We conduct dfs starting from the specified start node in the dfg
  // The search will stop when encountered following situations:
  //  - Encountered a node in the data base node list
  //  - Entering a cond_br / mux node through the conditional channel
  //  - No node left
  // We also check all buffer nodes, whether it's inside a specific cfdfc or not
  while (!mainStack.empty()) {
    std::vector<std::string> curAdjList = adjStack.back();
    adjStack.pop_back();

    if (!curAdjList.empty()) {
      std::string curNode = curAdjList.back();
      curAdjList.pop_back();

      // Update the adjStack
      adjStack.push_back(curAdjList);

      // Flag for conditional port
      bool nonCondFlag = true;
      bool computeFlag = false;

      // Update compute flag
      if (switchInfo.dataflowGraph->allDataBaseNode.find(curNode) == switchInfo.dataflowGraph->allDataBaseNode.end()) {
        // This is not a invalid backedge
        auto selInvalidBEList = switchInfo.segInvalidBackedgesMap[segLabel];
        std::pair<std::string, std::string> selPair = std::make_pair(mainStack.back(), curNode);
        if (std::find(selInvalidBEList.begin(), selInvalidBEList.end(), selPair) == selInvalidBEList.end()) computeFlag = true;
      }

      // Check whether we can update mainStack or not
      if (computeFlag) {
        std::string preNode = mainStack.back();

        if (curNode.find("cond_br") != std::string::npos) {
          // Check whether this is the conditional channel
          auto selNode = dyn_cast<CBrNode>(switchInfo.dataflowGraph->nodes[curNode].get());
          if (preNode == selNode->condPreNodeName) {
            // TODO: Check this updating condition
            if (selNode->dataPreNodeName != "") {
              nonCondFlag = false;
            }
          }
        } else if (curNode.find("mux") != std::string::npos) {
          // Check whether this is the conditional channel
          auto selNode = dyn_cast<MuxNode>(switchInfo.dataflowGraph->nodes[curNode].get());
          if (preNode == selNode->conPreNodeName) nonCondFlag = false;
        } else if (curNode.find("buffer") != std::string::npos) {
          unsigned curNodeBB = switchInfo.dataflowGraph->nodes[curNode]->bbindex;

          if (std::find(segBBList.begin(), segBBList.end(), curNodeBB) == segBBList.end()) nonCondFlag = false;
        } else if (curNode.find("end") != std::string::npos && (segLabel != "E")) {
          // TODO: Check the following update logic
          nonCondFlag = false;
        } else if (curNode.find("mem_controller") != std::string::npos) {
          // TODO: Check the following update logic
          nonCondFlag = false;
        }

        //! Testing
        // llvm::dbgs() << "==================\n";
        // printMainStack(mainStack);
        // printAdjStack(adjStack);
        // llvm::dbgs() << "nonCondFlag: " << nonCondFlag << "\n";

        // If this is a data channel
        if (nonCondFlag) {
          mainStack.push_back(curNode);

          // Update the buffer information
          if (curNode.find("buffer") != std::string::npos) numBuffers += 1;

          if (numBuffers == 0) glitchSuccNodeList.push_back(curNode);

          // Insert new AdjList
          std::vector<std::string> tmpAdjList;
          auto newAdjList = switchInfo.dataflowGraph->nodes[curNode]->sucs;

          for (const auto& selNextNode: newAdjList) {
            bool inStack = (std::find(mainStack.begin(), mainStack.end(), selNextNode) != mainStack.end());
            bool inBaseNodeList = (switchInfo.dataflowGraph->allDataBaseNode.find(selNextNode) != switchInfo.dataflowGraph->allDataBaseNode.end());
            bool isExcluded = (std::find(excludingList.begin(), excludingList.end(), selNextNode) != excludingList.end());
            if (!inStack && !inBaseNodeList && !isExcluded) tmpAdjList.push_back(selNextNode);
          }

          adjStack.push_back(tmpAdjList);
        }
      }
    } else {
      std::string lastMainStackNode = mainStack.back();
      if (lastMainStackNode.find("buffer") != std::string::npos) numBuffers -= 1;

      mainStack.pop_back();
    }
  }

  return glitchSuccNodeList;
}

std::string segNodeDataSrcSearch(SwitchingInfo &switchInfo, std::string startNode, AdjGraph *selGraph) {
  // Check the start node
  if (switchInfo.dataflowGraph->allDataBaseNode.find(startNode) != switchInfo.dataflowGraph->allDataBaseNode.end()) return startNode;

  // Define storing structure
  std::vector<std::string> mainStack;
  std::vector<std::vector<std::string>> adjStack;

  // Initialization
  mainStack.push_back(startNode);

  if (startNode.find("cond_br") != std::string::npos) {
    // This is a cond_br node
    if (auto *cbrNode = dyn_cast<CBrNode>(switchInfo.dataflowGraph->nodes[startNode].get())) {
      std::string tmpDataPreNode = cbrNode->dataPreNodeName;
      std::string tmpCondPreNode = cbrNode->condPreNodeName;
      if (tmpDataPreNode != "") {
        adjStack.push_back({ tmpDataPreNode });
      } else {
        adjStack.push_back({ tmpCondPreNode });
      }
    } 
  } else {
    adjStack.push_back(selGraph->nodes[startNode]->pres);
  }

  while (!mainStack.empty()) {
    std::vector<std::string> curAdjList = adjStack.back();
    adjStack.pop_back();

    if (!curAdjList.empty()) {
      std::string curNode = curAdjList.back();
      curAdjList.pop_back();
      adjStack.push_back(curAdjList);

      //! Testing
      // llvm::dbgs() << "[DEBUG] ============================\n";
      // printMainStack(mainStack);
      // printAdjStack(adjStack);

      //
      if (switchInfo.dataflowGraph->allDataBaseNode.find(curNode) != switchInfo.dataflowGraph->allDataBaseNode.end()) {
        return curNode;
      } else {
        mainStack.push_back(curNode);

        if (curNode.find("cond_br") != std::string::npos) {
          // This is a cond_br node
          if (auto *cbrNode = dyn_cast<CBrNode>(switchInfo.dataflowGraph->nodes[curNode].get())) {
            std::string tmpDataPreNode = cbrNode->dataPreNodeName;
            std::string tmpCondPreNode = cbrNode->condPreNodeName;

            if (tmpDataPreNode != "") {
              if (std::find(mainStack.begin(), mainStack.end(), tmpDataPreNode) == mainStack.end()) {
                adjStack.push_back({ tmpDataPreNode });
              }
            } else {
              if (std::find(mainStack.begin(), mainStack.end(), tmpCondPreNode) == mainStack.end()) {
                adjStack.push_back({ tmpCondPreNode });
              }
            }
          }
        } else {
          std::vector<std::string> tmpAdjList;
          std::vector<std::string> newAdjList = selGraph->nodes[curNode]->pres;

          for (const auto&n : newAdjList) {
            bool inStack = (std::find(mainStack.begin(), mainStack.end(), n) != mainStack.end());
            if (!inStack) tmpAdjList.push_back(n);
          }

          adjStack.push_back(tmpAdjList);
        }

      }
    } else {
      mainStack.pop_back();
    }
  }

  llvm::dbgs() << "[ERROR] Could not find base node for " << startNode << "\n";
  return "";
}

LongestPathResult selLongestPath(SwitchingInfo &switchInfo, std::string dstNode, std::string mgLabel) {
  //
  unsigned maxLatency = 0;
  std::string selStartNode = "";
  std::string lastSecondBuff = "";
  LongestPathResult returnValue;
  auto selGraph = switchInfo.segToAdjGraphMap[mgLabel];
  auto selMGII = switchInfo.cfdfcIIs[std::stoul(mgLabel)];

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

//===----------------------------------------------------------------------===//
//
// Helper function for debuging
//
//===----------------------------------------------------------------------===//
// Function to print the contents of a DataBaseNodesTriple
void printDataBaseNodesTriple(DataBaseNodesTriple dbnt) {
  llvm::dbgs() << "[DEBUG] \t\tDataBaseNodesTriple:\n";

  llvm::dbgs() << "[DEBUG] \t\t  All: [";
  for (size_t i = 0; i < dbnt.all.size(); ++i) {
    llvm::dbgs() << dbnt.all[i];
    if (i + 1 < dbnt.all.size()) {
      llvm::dbgs() << ", ";
    }
  }
  llvm::dbgs() << "]\n";

  llvm::dbgs() << "[DEBUG] \t\t  Control: [";
  for (size_t i = 0; i < dbnt.control.size(); ++i) {
    llvm::dbgs() << dbnt.control[i];
    if (i + 1 < dbnt.control.size()) {
      llvm::dbgs() << ", ";
    }
  }
  llvm::dbgs() << "]\n";

  llvm::dbgs() << "[DEBUG] \t\t  Data: [";
  for (size_t i = 0; i < dbnt.data.size(); ++i) {
    llvm::dbgs() << dbnt.data[i];
    if (i + 1 < dbnt.data.size()) {
      llvm::dbgs() << ", ";
    }
  }
  llvm::dbgs() << "]\n";
}

// 1) Print the muxToSrcNodeMap
// Format: {"mux_node_name" : {"control" : ctrlSrcName, "0" : srcName0, "1" : srcName1}}
void printMuxToSrcNodeMap(const std::map<std::string, std::map<std::string, std::string>> &muxToSrcNodeMap) {
  llvm::dbgs() << "[DEBUG] \tmuxToSrcNodeMap:\n";
  for (const auto &muxEntry : muxToSrcNodeMap) {
    // muxEntry.first => mux node name
    // muxEntry.second => map from { "control", "0", "1" } to source node name
    const std::string &muxNodeName = muxEntry.first;
    const auto &innerMap = muxEntry.second;

    llvm::dbgs() << "[DEBUG] \t\tMux Node: " << muxNodeName << " => {\n";
    for (const auto &kv : innerMap) {
      llvm::dbgs() << "[DEBUG] \t\t\t\"" << kv.first << "\" : \"" << kv.second << "\",\n";
    }
    llvm::dbgs() << "[DEBUG] \t\t}\n";
  }
}

// 2) Print the srcNodeToMuxMap
// Format: {"src_node_name" : [ (mux_node_name, portId), ... ]}
void printSrcNodeToMuxMap(const std::map<std::string, std::vector<std::pair<std::string, unsigned>>> &srcNodeToMuxMap) {
  llvm::dbgs() << "[DEBUG] \tsrcNodeToMuxMap:\n";
  for (const auto &srcEntry : srcNodeToMuxMap) {
    // srcEntry.first => source node name
    // srcEntry.second => vector of pairs
    const std::string &srcNodeName = srcEntry.first;
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
