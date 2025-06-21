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
   auto *op = switchInfo.dataflowGraph->nodes[nodeName]->op;
   if(!op){// first check if the pointer is null pointer or not (some op's irrelevat )
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
    switchInfo.segToOrderedALUNodes[segLabel] = {};
    muxCMNodesList tmpList;
    switchInfo.segToControlNodeList[segLabel] = tmpList;
  }

  // Traverse all the nodes in the dataflow graph
  for (const auto& nodeName: switchInfo.dataflowGraph->orderedNodeName) {
    unsigned nodeBB;
    auto op = switchInfo.dataflowGraph->nodes[nodeName]->op;
    if(!op ){continue;}
    if (std::optional<unsigned> optBB = getLogicBB(op); !optBB.has_value())
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

          if ((nodeName.find("constant") == std::string::npos) && (nodeName.find("load") == std::string::npos) &&
          (nodeName.find("source") == std::string::npos)) switchInfo.segToOrderedALUNodes[segLabel].push_back(nodeName);
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

// Masks 'value' to 'targetBitWidth' bits.
// E.g. reduceBits(0xABCD, 8) => 0xCD
int reduceBits(int value, unsigned targetBitWidth) {
  // If the target bit width is >= 32, we leave 'value' unchanged
  // because an int on most platforms is 32 bits, so no further masking is needed.
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
  std::string condSrcNode = switchInfo.dataflowGraph->condBrToConSrcMap[nodeName];
  int condValue = 1;

  if (switchInfo.dfgBaseNodeValueMap[condSrcNode]->originalDataOut.find(iterIndex) != switchInfo.dfgBaseNodeValueMap[condSrcNode]->originalDataOut.end()) {
    condValue = switchInfo.dfgBaseNodeValueMap[condSrcNode]->originalDataOut[iterIndex].value;
  } else  condSrcNode = 1;

  return condValue;
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

        // build a dependency graph for the multiplexers
        // We may encounter different situation when updating the value for MUX nodes
        for (const auto& selMuxNode: selMuxVec) {
          // Only update muxes this control-merge actually influences
          if (std::find(selMuxVec.begin(), selMuxVec.end(), selMuxNode) == selMuxVec.end())
            continue;
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
              //! Testing
              // llvm::dbgs() << "[DEBUG] \t\t\t(CASE1)\n";
              if (selMuxDataSrcNode == selMuxNode){
                // Use the previous value
                if (tmpMuxOutputMap.find(selMuxDataSrcNode) != tmpMuxOutputMap.end()) {
                  tmpMuxOutput = tmpMuxOutputMap[selMuxDataSrcNode];
                } else {
                  llvm::dbgs() << "[ERROR-line367] Data src(" << selMuxDataSrcNode << ") of Mux Node: " << selMuxDataSrcNode << ", not found in tmpMuxOutputMap.\n";
                }
              } else if (switchInfo.dfgBaseNodeValueMap[selMuxDataSrcNode]->originalDataOut.find(iterIndex) != switchInfo.dfgBaseNodeValueMap[selMuxDataSrcNode]->originalDataOut.end()) {
                //! Testing
                // llvm::dbgs() << "[DEBUG] \t\t\t(CASE2)\n";
                tmpMuxOutput = switchInfo.dfgBaseNodeValueMap[selMuxDataSrcNode]->originalDataOut[iterIndex].value;
              } else if (tmpMuxOutputMap.find(selMuxDataSrcNode) != tmpMuxOutputMap.end()) {
                //! Testing
                // llvm::dbgs() << "[DEBUG] \t\t\t(CASE3)\n";
                tmpMuxOutput = tmpMuxOutputMap[selMuxDataSrcNode];
              } else {
                //! Testing
                // llvm::dbgs() << "[DEBUG] \t\t\t(CASE4)\n";
                // TODO: Check the following update logic
                ValueIter tmpInnerMuxValuePair = {tmpMuxOutput, iterIndex - 1};
                switchInfo.dfgBaseNodeValueMap[selMuxDataSrcNode]->originalDataOut[iterIndex - 1] = tmpInnerMuxValuePair;
                tmpUnUpdatedMux.push_back(selMuxNode);

                continue;
              }
            } else {
              if (selMuxDataSrcNode.empty()) {
                // Unconnected , use default vlue
                tmpMuxOutput = 0;}
              else{  
              // tmpMuxOutput = switchInfo.dfgBaseNodeValueMap[selMuxDataSrcNode]->originalDataOut[iterIndex].value;
              auto &omap = switchInfo.dfgBaseNodeValueMap[selMuxDataSrcNode]->originalDataOut;
              if (omap.find(iterIndex) != omap.end()){
                  tmpMuxOutput = omap[iterIndex].value;
                }
                else if (iterIndex > 0 && omap.find(iterIndex-1) != omap.end())
                tmpMuxOutput = omap[iterIndex-1].value; // fallback to previous
              else
              tmpMuxOutput = 0; // or  default
            }
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


void dfgDataChannelPropagate(SwitchingInfo &switchInfo, SCFProfilingResult &profileResults, bool debug) {
  //
  auto segExecTrace = profileResults.executedSegTrace;

  // Iterate through the execution trace and propagate base nodes values in the selected segment
  for (unsigned i = 0; i < segExecTrace.size(); i++) {
    std::string executedSeg = segExecTrace[i];

    // List storing nodes that need special treatment
    std::vector<std::string> pendingMemList;

    //! Testing
    if (debug)
      llvm::dbgs() << "[DEBUG] ==================================\n";

    // Propagate data values for all non_memory related nodes
    for (const auto& selNode: switchInfo.segToDataBaseVecMap[executedSeg].all) {
      //! Testing
      if (debug)
        llvm::dbgs() << "[DEBUG] Node: " << selNode << "\n";

      // Case 1: Mapped Nodes --> All ALUs
      if (std::find(switchInfo.segToOrderedALUNodes[executedSeg].begin(), switchInfo.segToOrderedALUNodes[executedSeg].end(), selNode) != switchInfo.segToOrderedALUNodes[executedSeg].end()) {
        //! Testing
        if (debug)
          llvm::dbgs() << "[DEBUG] ALU Node Detected \n";
        // The value shall be obtained from ori_glitch_dataout
        // Step 1: Update the selected node itself
        for (const auto& selValue: switchInfo.dfgBaseNodeValueMap[selNode]->oriGlitchDataOut[i]) {
          switchInfo.dataflowGraph->nodes[selNode]->updateDataoutChannel(selValue);

          // Step 2: Update the glitching suceeding list
          //! Testing
          if (debug)
            llvm::dbgs() << "[DEBUG] \t[Glitch Output]\n";
          for (const auto& selSucNode: switchInfo.dfgBaseNodeValueMap[selNode]->segSucNodeMap[executedSeg].glitch) {
            //! Testing
            if (debug)
              llvm::dbgs() << "[DEBUG] \t\tSucNode: " << selSucNode << "\n";

            // Get the node bitwidth
            unsigned tmpNodeWidth = switchInfo.dfgBaseNodeValueMap[selNode]->segSucNodeMap[executedSeg].dataWidthMap[selSucNode];
            
            //! Testing
            if (debug)
              llvm::dbgs() << "[DEBUG] \t\t\tDataWidth: " << tmpNodeWidth << "\n";

            if (selSucNode.find("store") != std::string::npos) {
              // Get the actual node storing structure
              auto selStoreNode = dyn_cast<DStoreNode>(switchInfo.dataflowGraph->nodes[selSucNode].get());
              selStoreNode->updateDataout(reduceBits(selValue, tmpNodeWidth), selNode);
            } else if (selSucNode.find("cond_br") != std::string::npos) {
              // Get the node
              auto selCondStoreNode = dyn_cast<CBrNode>(switchInfo.dataflowGraph->nodes[selSucNode].get());
              // Get the control value
              // TODO: Validate the assumption that cond_value is always updated before the data_in of the node
              int tmpCondValue = getCondBrNodeCondValue(switchInfo, i, selSucNode);

              selCondStoreNode->updateDataout(reduceBits(selValue, tmpNodeWidth), tmpCondValue);
            } else {
              switchInfo.dataflowGraph->nodes[selSucNode]->updateDataoutChannel(reduceBits(selValue, tmpNodeWidth));
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
        for (const auto& selSucNode: switchInfo.dfgBaseNodeValueMap[selNode]->segSucNodeMap[executedSeg].original) {
          // Get Node Bitwidth
          unsigned tmpNodeWidth = switchInfo.dfgBaseNodeValueMap[selNode]->segSucNodeMap[executedSeg].dataWidthMap[selSucNode];
          
          //! Testing
          if (debug)
            llvm::dbgs() << "[DEBUG] \t\tSuc Node: " << selSucNode << "\n";
          
          //
          if (selSucNode.find("store") != std::string::npos) {
            // 
            auto selStoreNode = dyn_cast<DStoreNode>(switchInfo.dataflowGraph->nodes[selSucNode].get());
            selStoreNode->updateDataout(reduceBits(switchInfo.dfgBaseNodeValueMap[selNode]->originalDataOut[i].value, tmpNodeWidth), selNode);
          } else if (selSucNode.find("cond_br") != std::string::npos) {
            // Get the node
            // TODO: Validate the assumption that cond_value is always updated before the data_in of the node
            auto selCondStoreNode = dyn_cast<CBrNode>(switchInfo.dataflowGraph->nodes[selSucNode].get());
            int tmpCondValue = getCondBrNodeCondValue(switchInfo, i, selSucNode);

            selCondStoreNode->updateDataout(reduceBits(switchInfo.dfgBaseNodeValueMap[selNode]->originalDataOut[i].value , tmpNodeWidth), tmpCondValue);
          } else {
            switchInfo.dataflowGraph->nodes[selSucNode]->updateDataoutChannel(reduceBits(switchInfo.dfgBaseNodeValueMap[selNode]->originalDataOut[i].value , tmpNodeWidth));
          }
        }
        //! Testing
        if (debug)
          llvm::dbgs() << "[DEBUG] \t[DONE]\n";
      } else {
        // Case 2: This is a control merge node
        if (selNode.find("control_merge") != std::string::npos) {
          //! Testing
          if (debug)
            llvm::dbgs() << "[DEBUG] Control Merge Detected \n";

          // Get node
          auto selCMNode = dyn_cast<CMergeData>(switchInfo.dfgBaseNodeValueMap[selNode].get());
          int initValue = 0;

          //! For all transition related nodes, we need to check the iter 0 as well
          if (i == 1) {
            initValue = selCMNode->getControlOutput(0);
            switchInfo.dataflowGraph->nodes[selNode]->updateDataoutChannel(initValue);
          } else {
            initValue = 0;
          }

          // Step 1: Update the node itself
          int selValue = selCMNode->getControlOutput(i);
          switchInfo.dataflowGraph->nodes[selNode]->updateDataoutChannel(selValue);

          if (selValue == 1){
            switchInfo.dataflowGraph->nodes[selNode]->updateDataoutChannel(0);
          }

          // Step 2: Update all nodes in the control succeeding node list
          //! Testing
          if (debug)
            llvm::dbgs() << "[DEBUG] \t[Control Output]\n";
          for (const auto& selSucNode: selCMNode->mgSucNodeDict[executedSeg].control) {
            //! Testing
            if (debug)
              llvm::dbgs() << "[DEBUG] \t\tSuc Node: " << selSucNode << "\n";

            if (i == 1) {
              switchInfo.dataflowGraph->nodes[selSucNode]->updateDataoutChannel(initValue);
            }
            switchInfo.dataflowGraph->nodes[selSucNode]->updateDataoutChannel(selValue);

            // Step 2.5: Update all glitch node, if cm_value = 1, then change it back to 0, unless II == 1
            if (selValue == 1) {
              if (std::find(selCMNode->mgSucNodeDict[executedSeg].glitch.begin(), selCMNode->mgSucNodeDict[executedSeg].glitch.end(), selSucNode) != selCMNode->mgSucNodeDict[executedSeg].glitch.end()) {
                switchInfo.dataflowGraph->nodes[selSucNode]->updateDataoutChannel(0);
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
          for (const auto& selSucNode: selCMNode->mgSucNodeDict[executedSeg].data) {
            //! Testing
            if (debug)
              llvm::dbgs() << "[DEBUG] \t\tSuc Node: " << selSucNode << "\n";

            if (selSucNode.find("cond_br") != std::string::npos) {
              // Get the node
              auto selCondStoreNode = dyn_cast<CBrNode>(switchInfo.dataflowGraph->nodes[selSucNode].get());

              int tmpCondValue = getCondBrNodeCondValue(switchInfo, i, selSucNode);
              selCondStoreNode->updateDataout(-1, tmpCondValue);
            } else {
              switchInfo.dataflowGraph->nodes[selSucNode]->updateDataoutChannel(-1);
            }
          }

          //! Testing
          if (debug)
            llvm::dbgs() << "[DEBUG] \t[DONE]\n";
        } else if (selNode.find("mux") != std::string::npos) {
          // Case 3: MUX Node
          // In the last iteration or MG transitions, we take the ori_glitch_data out value
          //! For all transition related nodes, we need to check the iter 0 as well
          //! Testing
          if (debug)
            llvm::dbgs() << "[DEBUG] MUX Detected \n";
          for (const auto& selValue: switchInfo.dfgBaseNodeValueMap[selNode]->oriGlitchDataOut[i]) {
            switchInfo.dataflowGraph->nodes[selNode]->updateDataoutChannel(selValue);

            // Step 2: Update the glitching succeeding list
            //! Testing
            if (debug)
              llvm::dbgs() << "[DEBUG] \t[Glitch Output]\n";
            for (const auto& selSucNode: switchInfo.dfgBaseNodeValueMap[selNode]->segSucNodeMap[executedSeg].glitch) {
              //! Testing
              if (debug)
                llvm::dbgs() << "[DEBUG] \t\tSuc Node: " << selSucNode << "\n";
              
              // Get the node output bitwidth
              unsigned tmpNodeWidth = switchInfo.dfgBaseNodeValueMap[selNode]->segSucNodeMap[executedSeg].dataWidthMap[selSucNode];

              if (selSucNode.find("cond_br") != std::string::npos) {
                // Get the node
                auto selCondStoreNode = dyn_cast<CBrNode>(switchInfo.dataflowGraph->nodes[selSucNode].get());

                int tmpCondValue = getCondBrNodeCondValue(switchInfo, i, selSucNode);
                selCondStoreNode->updateDataout(reduceBits(selValue, tmpNodeWidth), tmpCondValue);
              } else {
                switchInfo.dataflowGraph->nodes[selSucNode]->updateDataoutChannel(reduceBits(selValue, tmpNodeWidth));
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
          for (const auto& selSucNode: switchInfo.dfgBaseNodeValueMap[selNode]->segSucNodeMap[executedSeg].original) {
            //! Testing
            if (debug)
              llvm::dbgs() << "[DEBUG] \t\tSuc Node: " << selSucNode << "\n";
            // Get bitwidth
            unsigned tmpNodeWidth = switchInfo.dfgBaseNodeValueMap[selNode]->segSucNodeMap[executedSeg].dataWidthMap[selSucNode];

            //
            if (switchInfo.dfgBaseNodeValueMap[selNode]->originalDataOut.find(i) != switchInfo.dfgBaseNodeValueMap[selNode]->originalDataOut.end()) {
              if (selSucNode.find("store") != std::string::npos) {
                auto selStoreNode = dyn_cast<DStoreNode>(switchInfo.dataflowGraph->nodes[selSucNode].get());
                selStoreNode->updateDataout(reduceBits(switchInfo.dfgBaseNodeValueMap[selNode]->originalDataOut[i].value, tmpNodeWidth), selNode);
              } else if (selSucNode.find("cond_br") != std::string::npos) {
                auto selCondBrNode = dyn_cast<CBrNode>(switchInfo.dataflowGraph->nodes[selSucNode].get());
                int tmpCondValue = getCondBrNodeCondValue(switchInfo, i, selSucNode);

                selCondBrNode->updateDataout(reduceBits(switchInfo.dfgBaseNodeValueMap[selNode]->originalDataOut[i].value, tmpNodeWidth), tmpCondValue);
              } else {
                switchInfo.dataflowGraph->nodes[selSucNode]->updateDataoutChannel(reduceBits(switchInfo.dfgBaseNodeValueMap[selNode]->originalDataOut[i].value , tmpNodeWidth));
              }
            }
          }
          //! Testing
          if (debug)
            llvm::dbgs() << "[DEBUG] \t[DONE]\n";
        } else if (selNode.find("load") != std::string::npos) {
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
          if (selNode.find("source") != std::string::npos || (selNode.find("start") != std::string::npos)) {
            if (i == switchInfo.segToExecutedIter[executedSeg]) {
              selValue = switchInfo.dfgBaseNodeValueMap[selNode]->originalDataOut[0].value;
            } else continue;
          } else if (selNode.find("constant") != std::string::npos) {
            selValue = switchInfo.dfgBaseNodeValueMap[selNode]->originalDataOut[0].value;
          } else {
            selValue = switchInfo.dfgBaseNodeValueMap[selNode]->originalDataOut[i].value;
          }

          // Step 1: update the node itself
          switchInfo.dataflowGraph->nodes[selNode]->updateDataoutChannel(selValue);

          // Step 2: Update all glitching nodes in the succeeding list
          for(const auto& selSucNode: switchInfo.dfgBaseNodeValueMap[selNode]->segSucNodeMap[executedSeg].glitch) {
            if (selSucNode.find("cond_br") != std::string::npos) {
              auto selCondStoreNode = dyn_cast<CBrNode>(switchInfo.dataflowGraph->nodes[selSucNode].get());
              int tmpCondValue = getCondBrNodeCondValue(switchInfo, i, selSucNode);
              selCondStoreNode->updateDataout(selValue, tmpCondValue);
            } else {
              switchInfo.dataflowGraph->nodes[selSucNode]->updateDataoutChannel(selValue);
            }
          }

          // Step 3: Update all non glitching nodes in the succeeding list
          for(const auto& selSucNode: switchInfo.dfgBaseNodeValueMap[selNode]->segSucNodeMap[executedSeg].original) {
            switchInfo.dataflowGraph->nodes[selSucNode]->updateDataoutChannel(selValue);
          }

          //! Testing
          if (debug)
            llvm::dbgs() << "[DEBUG] [DONE]\n";
        }
      }
    }

    // Case 6: Nodes related to memory accesses
    for (const auto& selNode: pendingMemList) {
      // We need to update the address out and data out separatly
      // Step 1: Get the address source      
      auto selMemNode = dyn_cast<DLoadNode>(switchInfo.dataflowGraph->nodes[selNode].get());

      std::string tmpAddrPreNode = selMemNode->addressInNodeName;
      std::string tmpAddrPreSrcNode = "";
      if (executedSeg == "E") {
        tmpAddrPreSrcNode = segENodeSrcSearch(switchInfo, tmpAddrPreNode, profileResults);
      } else {
        tmpAddrPreSrcNode = memAddrSrcSearch(switchInfo, profileResults, tmpAddrPreNode, executedSeg, i);
      }

      // We assume there will not be any glitches for the address value
      int addrValue = 0;
      if (tmpAddrPreSrcNode.find("constant") != std::string::npos) {
        addrValue = switchInfo.dfgBaseNodeValueMap[tmpAddrPreSrcNode]->originalDataOut[0].value;
      } else {
        // TODO: Validate the following assumption
        if (switchInfo.dfgBaseNodeValueMap[tmpAddrPreSrcNode]->originalDataOut.find(i) != switchInfo.dfgBaseNodeValueMap[tmpAddrPreSrcNode]->originalDataOut.end()) {
          addrValue = switchInfo.dfgBaseNodeValueMap[tmpAddrPreSrcNode]->originalDataOut[i].value;
        } else {
          addrValue = -10;
        }
      }

      int selValue = switchInfo.dfgBaseNodeValueMap[selNode]->originalDataOut[i].value;

      // Step 1: Update the node itself
      selMemNode->updateDataout(selValue, addrValue);

      // Step 2: Update teh nodes in glitching list
      for (const auto& selSucNode: switchInfo.dfgBaseNodeValueMap[selNode]->segSucNodeMap[executedSeg].glitch) {
        //! Exclude the memory controller
        if (selSucNode.find("mem_controller") == std::string::npos && (selSucNode.find("end") == std::string::npos)) {
          // Get node bitwidth
          unsigned tmpNodeWidth = switchInfo.dfgBaseNodeValueMap[selNode]->segSucNodeMap[executedSeg].dataWidthMap[selSucNode];

          if (selSucNode.find("cond_br") != std::string::npos) {
            // Get the node
            auto selCondStoreNode = dyn_cast<CBrNode>(switchInfo.dataflowGraph->nodes[selSucNode].get());

            int tmpCondValue = getCondBrNodeCondValue(switchInfo, i, selSucNode);
            selCondStoreNode->updateDataout(reduceBits(selValue, tmpNodeWidth), tmpCondValue);
          } else {
            switchInfo.dataflowGraph->nodes[selSucNode]->updateDataoutChannel(reduceBits(selValue, tmpNodeWidth));
          }
        }
      }

      // Step 3: Update nodes in non glitching list
      for (const auto& selSucNode: switchInfo.dfgBaseNodeValueMap[selNode]->segSucNodeMap[executedSeg].original) {
        // Get bitwidth
        unsigned tmpNodeWidth = switchInfo.dfgBaseNodeValueMap[selNode]->segSucNodeMap[executedSeg].dataWidthMap[selSucNode];

        //
        if (selSucNode.find("cond_br") != std::string::npos) {
          // Get the node
          auto selCondStoreNode = dyn_cast<CBrNode>(switchInfo.dataflowGraph->nodes[selSucNode].get());

          int tmpCondValue = getCondBrNodeCondValue(switchInfo, i, selSucNode);
          selCondStoreNode->updateDataout(reduceBits(selValue, tmpNodeWidth), tmpCondValue);
        } else {
          switchInfo.dataflowGraph->nodes[selSucNode]->updateDataoutChannel(reduceBits(selValue, tmpNodeWidth));
        }
      }
    }
  }

  // Final step, change the results of buffers directly connected to cond_br node
  // for (auto& [selCondNode, bufferList] : switchInfo.dataflowGraph->condBrToBufferMap) {
  //   // Get the node
  //   auto selCondStoringNode = dyn_cast<CBrNode>(switchInfo.dataflowGraph->nodes[selCondNode].get());
  //   for (auto& [selBuffer, portIdx]: bufferList) {
  //     // Change the resutls of the buffer node
  //     for (auto& [selIdx, valueVec]: switchInfo.dataflowGraph->nodes[selBuffer]->dataOut) {
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

std::string segENodeSrcSearch(SwitchingInfo &switchInfo, std::string nodeName, SCFProfilingResult &profileResults) {
  std::string dataSrcNode = "";

  for (const auto& selMappedNode: switchInfo.segToDataBaseVecMap["E"].all) {
    // The node itself is mapped
    if (nodeName.find(selMappedNode) != std::string::npos) {
      dataSrcNode = selMappedNode;
      break;
    }

    for (const auto& selSucNode: switchInfo.dfgBaseNodeValueMap[selMappedNode]->segSucNodeMap["E"].control) {
      if (selSucNode == nodeName) dataSrcNode = selMappedNode;
    }

    for (const auto& selSucNode: switchInfo.dfgBaseNodeValueMap[selMappedNode]->segSucNodeMap["E"].data) {
      if (selSucNode == nodeName) dataSrcNode = selMappedNode;
    }

    for (const auto& selSucNode: switchInfo.dfgBaseNodeValueMap[selMappedNode]->segSucNodeMap["E"].glitch) {
      if (selSucNode == nodeName) dataSrcNode = selMappedNode;
    }

    for (const auto& selSucNode: switchInfo.dfgBaseNodeValueMap[selMappedNode]->segSucNodeMap["E"].original) {
      if (selSucNode == nodeName) dataSrcNode = selMappedNode;
    }
  }

  if (dataSrcNode == "") {
    // No src node found in previous search process
    std::string selSeg = profileResults.executedSegTrace[profileResults.executedSegTrace.size() - 2];

    for (const auto& selMappedNode: switchInfo.segToDataBaseVecMap[selSeg].all) {
      if (nodeName.find(selMappedNode) != std::string::npos) {
        dataSrcNode = selMappedNode;
        break;
      }

      for (const auto& selSucNode: switchInfo.dfgBaseNodeValueMap[selMappedNode]->segSucNodeMap[selSeg].control) {
        if (selSucNode.find(nodeName) != std::string::npos) {
          dataSrcNode = selMappedNode;
          break;
        }
      }
  
      for (const auto& selSucNode: switchInfo.dfgBaseNodeValueMap[selMappedNode]->segSucNodeMap[selSeg].data) {
        if (selSucNode.find(nodeName) != std::string::npos) {
          dataSrcNode = selMappedNode;
          break;
        }
      }
  
      for (const auto& selSucNode: switchInfo.dfgBaseNodeValueMap[selMappedNode]->segSucNodeMap[selSeg].glitch) {
        if (selSucNode.find(nodeName) != std::string::npos) {
          dataSrcNode = selMappedNode;
          break;
        }
      }
  
      for (const auto& selSucNode: switchInfo.dfgBaseNodeValueMap[selMappedNode]->segSucNodeMap[selSeg].original) {
        if (selSucNode.find(nodeName) != std::string::npos) {
          dataSrcNode = selMappedNode;
          break;
        }
      }
    }
  }

  return dataSrcNode;
}

std::string memAddrSrcSearch(SwitchingInfo &switchInfo, SCFProfilingResult &profileResults, std::string nodeName, std::string selSeg, unsigned iterIdx) {
  std::string dataSrcNode = "";

  for (const auto& selMappedNode: switchInfo.segToDataBaseVecMap[selSeg].all) {
    // The node itself is mapped
    if (nodeName.find(selMappedNode) != std::string::npos) {
      dataSrcNode = selMappedNode;
      break;
    }

    for (const auto& selSucNode: switchInfo.dfgBaseNodeValueMap[selMappedNode]->segSucNodeMap[selSeg].control) {
      if (selSucNode.find(nodeName) != std::string::npos) dataSrcNode = selMappedNode;
    }

    for (const auto& selSucNode: switchInfo.dfgBaseNodeValueMap[selMappedNode]->segSucNodeMap[selSeg].data) {
      if (selSucNode.find(nodeName) != std::string::npos) dataSrcNode = selMappedNode;
    }

    for (const auto& selSucNode: switchInfo.dfgBaseNodeValueMap[selMappedNode]->segSucNodeMap[selSeg].glitch) {
      if (selSucNode.find(nodeName) != std::string::npos) dataSrcNode = selMappedNode;
    }

    for (const auto& selSucNode: switchInfo.dfgBaseNodeValueMap[selMappedNode]->segSucNodeMap[selSeg].original) {
      if (selSucNode.find(nodeName) != std::string::npos) dataSrcNode = selMappedNode;
    }
  }

  if (dataSrcNode == "") {
    // No src node found in previous search process
    std::string selSegInner = profileResults.executedSegTrace[iterIdx - 1];

    for (const auto& selMappedNode: switchInfo.segToDataBaseVecMap[selSegInner].all) {
      if (nodeName.find(selMappedNode) != std::string::npos) {
        dataSrcNode = selMappedNode;
        break;
      }

      for (const auto& selSucNode: switchInfo.dfgBaseNodeValueMap[selMappedNode]->segSucNodeMap[selSegInner].control) {
        if (selSucNode.find(nodeName) != std::string::npos) {
          dataSrcNode = selMappedNode;
          break;
        }
      }
  
      for (const auto& selSucNode: switchInfo.dfgBaseNodeValueMap[selMappedNode]->segSucNodeMap[selSegInner].data) {
        if (selSucNode.find(nodeName) != std::string::npos) {
          dataSrcNode = selMappedNode;
          break;
        }
      }
  
      for (const auto& selSucNode: switchInfo.dfgBaseNodeValueMap[selMappedNode]->segSucNodeMap[selSegInner].glitch) {
        if (selSucNode.find(nodeName) != std::string::npos) {
          dataSrcNode = selMappedNode;
          break;
        }
      }
  
      for (const auto& selSucNode: switchInfo.dfgBaseNodeValueMap[selMappedNode]->segSucNodeMap[selSegInner].original) {
        if (selSucNode.find(nodeName) != std::string::npos) {
          dataSrcNode = selMappedNode;
          break;
        }
      }
    }
  }

  return dataSrcNode;
}

LongestPathResult selLongestPath(SwitchingInfo &switchInfo, std::string dstNode, std::string mgLabel) {
  //
  unsigned maxLatency = 0;
  std::string selStartNode = "";
  std::string lastSecondBuff = "";
  LongestPathResult returnValue;
  auto selGraph = switchInfo.segToAdjGraphMap[mgLabel];
  [[maybe_unused]] auto selMGII = switchInfo.cfdfcIIs[std::stoul(mgLabel)];

  // Get the list of buffers
  std::vector<std::string> selBuffList;
  for (const auto& selNode: selGraph->orderedNodeName) {
    if (selNode.find("buffer") != std::string::npos) {
      selBuffList.push_back(selNode);
    }
  }

  // For all start node
  for (const auto &startNode : selGraph->segStartNodes) {
    auto [latency, bestsrcnode] = selGraph->getMaxLatency(startNode, dstNode, false, true);
    // llvm::dbgs() << "step5 longest path data channel "<< startNode<<" "<< dstNode <<"\n";

    std::vector<std::pair<unsigned, std::string>> tmpLastSecondBufferList;
    if (latency > maxLatency) {
      maxLatency = latency;
      selStartNode = bestsrcnode;
    }
  }


    // 2. walk the paths just for the last second buffer
    std::vector<std::pair<unsigned, std::string>> tmpLastSecondBufferList;
    for (const auto &startNode : selGraph->segStartNodes) {
        // Find all paths from startNode to dstNode
        auto paths = selGraph->findPaths(startNode, dstNode, false, true);
        for (const auto &selPath : paths) {
            if (selPath.latency == maxLatency) {
                // Find the list of buffer nodes along the path
                std::vector<std::string> tmpBufferList;
                for (const auto &tmpSelNode : selPath.nodeList) {
                    if (tmpSelNode.find("buffer") != std::string::npos) {
                        tmpBufferList.push_back(tmpSelNode);
                    }
                }
                // if there are more than 2 buffers
                if (tmpBufferList.size() > 1) {
                    tmpLastSecondBufferList.push_back({maxLatency, tmpBufferList[tmpBufferList.size() - 2]});
                }
            }
        }
    }


    // Select the wanted last second buffer for set_r calc
    for (const auto &selPair : tmpLastSecondBufferList) {
      if (selPair.first == maxLatency) {
          // pick the buffer with the lowest occupancy??
          auto selBuffer = dyn_cast<BufferNode>(selGraph->nodes[selPair.second].get());
          float_t oriOcc = 0;
          if (!lastSecondBuff.empty()) {
              auto oriBuffer = dyn_cast<BufferNode>(selGraph->nodes[lastSecondBuff].get());
              oriOcc = oriBuffer->occupancy;
          }
          if (lastSecondBuff.empty() || (selBuffer->occupancy < oriOcc)) {
              lastSecondBuff = selPair.second;
          }
      }
  }

  // Check the validity of the start node
    if (selStartNode.empty()) {
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
