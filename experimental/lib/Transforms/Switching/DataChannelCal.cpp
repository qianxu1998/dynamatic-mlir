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
  // for (const auto& [label, bblist]: switchInfo.segToBBListMap) {
  //   llvm::dbgs() << "[DEBUG] \t\tSeg Label: " << label << "\n";
  //   llvm::dbgs() << "[DEBUG] \t\t\tBB List: ";
  //   for (const auto& selBB : bblist) {
  //     llvm::dbgs() << selBB << ", ";
  //   }
  //   llvm::dbgs() << "\n";
  // }

  for (const auto& selBaseNode: switchInfo.dataflowGraph->allDataBaseNode) {
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
