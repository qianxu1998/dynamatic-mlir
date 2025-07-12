//===- DataChannelCal.cpp - Estimate Swithicng Activities ------*- C++ -*-===//
//
// This file declares all functions used for data channel switching calculation
//
//===----------------------------------------------------------------------===//

#include "experimental/Transforms/Switching/DataChannelCal.h"
#include "experimental/Transforms/Switching/DFSKernel.h"
#include "experimental/Transforms/Switching/utils.h"
#include <cassert>
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

void mapBBPairToControlMerge(SwitchingInfo &switchInfo) {
  for (const auto &nodeName : switchInfo.staticinfo.dataflowGraph->orderedNodeName) {
    auto *op = switchInfo.staticinfo.dataflowGraph->nodes[nodeName]->op;
    if (!op) { // first check if the pointer is null pointer or not (some op's
               // irrelevat )
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
        if (contains(  switchInfo.data.bbPairToCtrlMerge, std::make_pair(preBB, CMBB)  )  ) {
          switchInfo.data.bbPairToCtrlMerge[std::make_pair(preBB, CMBB)].push_back(
              std::make_pair(nodeName, i));
        } else {
          switchInfo.data.bbPairToCtrlMerge[std::make_pair(preBB, CMBB)] = {std::make_pair(nodeName, i)};
        }
      }
    }
  }
}

void getDataBaseNodes(SwitchingInfo &switchInfo,SCFProfilingResult &profileResults) {
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
      continue;
    }
    if (std::optional<unsigned> optBB = getLogicBB(op); !optBB.has_value())
      continue;
    else
      nodeBB = *optBB;

    auto dataflowGraph = switchInfo.staticinfo.dataflowGraph;
    // Check the types of the nodes
    // TYPE 1: DATA nodes from scf profiling
    if ((contains(profileResults.opNameToValueListMap,nodeName))||
        (contains(nodeName, "constant")) || (contains(nodeName, "source"))) {
      // Check each segment
      for (const auto &[segLabel, BBVec] : switchInfo.staticinfo.segToBBs) {
        if (std::find(BBVec.begin(), BBVec.end(), nodeBB) != BBVec.end()) {
          // If this is a MG
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
      // TYPE 2: Control Nodes
    } else if (contains(nodeName, "control_merge") ||
               contains(nodeName, "mux")) {
      // Check each segment
      for (const auto &[segLabel, BBVec] : switchInfo.staticinfo.segToBBs) { 
        if (containsValue(BBVec, nodeBB)) {
          // If this is a MG
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
  for (const auto &mgPair : segSucNodeMap) {
    const auto &mgLabel = mgPair.first();
    const MgNodeInfo &info = mgPair.second;

    llvm::dbgs() << "\tCFDFC/Segment Label: " << mgLabel << "\n";
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
  for (const auto &mgPair : mgSucNodeDict) {
    const std::string &mgLabel = mgPair.first;
    const MgNodeInfo &info = mgPair.second;

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
  if (controlGlitchVec.size()) {
    llvm::dbgs() << "\tControl glitch Dataout:\n\t";
    for (const auto &kv : controlGlitchVec) {
      llvm::dbgs() << std::to_string(kv) << " ,";
    }
    llvm::dbgs() << "\n";
  }
}

int CMergeData::getControlOutput(unsigned selIter) {
  if (contains(controlDataOut,selIter) ) {
    return controlDataOut[selIter].value;
  } else {
    return 0;
  }
}

unsigned getExecutionIter(unsigned bbIndex, unsigned curBB,
                          SCFProfilingResult &profileResults) {
                            ///bounds check for matvec error
  if (!contains(profileResults.bbToIterMap, bbIndex)) {
    llvm::dbgs() << "[ERROR] bbIndex " << bbIndex << " not found in bbToIterMap\n";
    return 0; 
  }
  unsigned iterIndex = profileResults.bbToIterMap[bbIndex];
 if(iterIndex==0){
  llvm::dbgs() << "[ERROR] iterIndex " << iterIndex << " is 0 ,can't get previous value\n";
  return 0;
 }
// llvm::dbgs()<<" iterindex " <<iterIndex -1 <<" \n";
  // Add bounds checking for executedBBTrace access
  if (iterIndex - 1 >= profileResults.executedBBTrace.size()) {
    llvm::dbgs() << "[ERROR] iterIndex-1 (" << (iterIndex-1) 
                 << ") out of bounds for executedBBTrace (size: " 
                 << profileResults.executedBBTrace.size() << ")\n";
    return iterIndex; // Return iterIndex instead of underflowing
  }

  if (curBB == profileResults.executedBBTrace[iterIndex - 1] ||   (bbIndex == 1)  ) {
    return iterIndex - 1;
  }
  return iterIndex;
}

void dataChannelBaseNodesValueUpdate(SwitchingInfo &switchInfo,
                                     SCFProfilingResult &profileResults) {
  // [Step 1]
  // Iterate all profile base node in the dataflow graph
  for (const auto &selNode : switchInfo.staticinfo.dataflowGraph->profileBaseNodes) {
    // Define tmp storing structure
    switchInfo.data.dfgBaseNodeValue[selNode] = std::make_shared<DataBase>(selNode);

    // Check the type of node
    if (contains(selNode, "cmp")) {
      for (const auto &[value, iterIdx] : profileResults.opNameToValueListMap[selNode]) {
        // Create the value struct
        ValueIter tmpValuePair = {std::abs(value), iterIdx};
        switchInfo.data.dfgBaseNodeValue[selNode]->originalDataOut[iterIdx] =
            tmpValuePair;
      }

      //! Testing
      // switchInfo.data.dfgBaseNodeValue[selNode]->printDetail();
    } else if (contains(selNode, "constant")) {
      // We need to get the constant value from the attribute 
      // Get the mlir op
      auto constantOp = dyn_cast<handshake::ConstantOp>(
          switchInfo.staticinfo.dataflowGraph->nodes[selNode]->op);
      if(!constantOp){
        llvm::errs() << "[ERROR] Node isn't a constant op\n";
        continue;
      }
      auto valueAttr = constantOp->getAttrOfType<IntegerAttr>("value");
      if(!valueAttr){
        llvm::errs() << "[ERROR] Constant node doesn't have a value\n";
        continue;
      }
      auto valueAttr2 = constantOp.getValue();

      llvm::dbgs() << "[assert]  comparing valueattr "<<valueAttr  <<" "<< valueAttr2 << " \n";

      // if (!valueAttr) {
      //   llvm::errs() << "[ERROR] Can't get the value for the constant op\n";
      // }
      // int constantValue = valueAttr.getInt();
      // int constantValue {0};
        // 3) Dispatch on the actual attribute kind
  int constantValue = 0;
  if (auto intAttr = valueAttr.dyn_cast<IntegerAttr>()) {
    constantValue = intAttr.getValue().getSExtValue();
  }  else {
    llvm::errs() << "[ERROR] Unexpected attr type on “"<< selNode << "”: "
                 << valueAttr.getType() << "\n";
    continue;
  }








      // Update the value in the vec list
      ValueIter tmpValuePair = {constantValue, 0};
      switchInfo.data.dfgBaseNodeValue[selNode]->originalDataOut[0] =tmpValuePair;

      //! Testing
      // switchInfo.data.dfgBaseNodeValue[selNode]->printDetail();
    } else if (contains(selNode, "source")) {
      ValueIter tmpValuePair = {0, 0};
      switchInfo.data.dfgBaseNodeValue[selNode]->originalDataOut[0] =
          tmpValuePair;
    } else {
      for (const auto &[value, iterIdx] :
           profileResults.opNameToValueListMap[selNode]) {
        ValueIter tmpValuePair = {value, iterIdx};
        switchInfo.data.dfgBaseNodeValue[selNode]->originalDataOut[iterIdx] =
            tmpValuePair;
      }

      //! Testing
      // switchInfo.data.dfgBaseNodeValue[selNode]->printDetail();
    }
  }

  // [Step 2]
  // Iterate over all CMerge and MUX node
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

  // [Step 3]
  // Define
  // Traverse the executed BB trace
  // Skip the first BB, as it will always be BB 0
  for (unsigned i = 1; i < profileResults.executedBBTrace.size(); i++) {
    unsigned preBB = profileResults.executedBBTrace[i - 1];
    unsigned curBB = profileResults.executedBBTrace[i];

    std::pair<unsigned, unsigned> key_pair(preBB, curBB);

    // Get the corresponding control merge value
    // [Step 3.1] We first update the value of all influenced control_merge node
    // in the circuit
    auto itCM = switchInfo.data.bbPairToCtrlMerge.find(key_pair);
    if (itCM != switchInfo.data.bbPairToCtrlMerge.end()) {
      // Iterate over all (CMNode, output value) pairs
      // TODO: Check whether the value stored match the simulation or not
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
        std::vector<std::string> selMuxVec =switchInfo.staticinfo.dataflowGraph->cmToMuxMap[nodeName];

        // build a dependency graph for the multiplexers
        // We may encounter different situation when updating the value for MUX
        // nodes
        for (const auto &selMuxNode : selMuxVec) {
          // Only update muxes this control-merge actually influences
          if (!containsValue(selMuxVec ,selMuxNode))
            continue;
          auto selMuxSrcMap =
              switchInfo.staticinfo.dataflowGraph->muxToSrcNodeMap[selMuxNode];
          std::string selMuxDataSrcNode =selMuxSrcMap[std::to_string(outValue)];
          int tmpMuxOutput { 0} ;

          //! Testing
          // llvm::dbgs() << "[DEBUG] \t\t\t\tMux Node: " << selMuxNode << ";
          // Sel Data Src Node: " << selMuxDataSrcNode << "\n";

          if (contains(selMuxDataSrcNode, "constant") ||contains(selMuxDataSrcNode, "source")) {
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
                if (contains(tmpMuxOutputMap,selMuxDataSrcNode)) {
                  tmpMuxOutput = tmpMuxOutputMap[selMuxDataSrcNode];
                } else {
                  llvm::dbgs()
                      << "[ERROR] Data src(" << selMuxDataSrcNode
                      << ") of Mux Node: " << selMuxDataSrcNode
                      << ", not found in tmpMuxOutputMap.\n";
                }
              } else if (contains(switchInfo.data.dfgBaseNodeValue[selMuxDataSrcNode]->originalDataOut,iterIndex) ) {
                //! Testing
                // llvm::dbgs() << "[DEBUG] \t\t\t(CASE2)\n";
                tmpMuxOutput =
                    switchInfo.data.dfgBaseNodeValue[selMuxDataSrcNode]->originalDataOut[iterIndex].value;
              } else if (contains(tmpMuxOutputMap,selMuxDataSrcNode)) {
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
                // Unconnected , use default vlue
                tmpMuxOutput = 0;
              } else {
                // tmpMuxOutput =
                // switchInfo.data.dfgBaseNodeValue[selMuxDataSrcNode]->originalDataOut[iterIndex].value;
                auto &omap = switchInfo.data.dfgBaseNodeValue[selMuxDataSrcNode]->originalDataOut;
                if (omap.find(iterIndex) != omap.end()) {
                  tmpMuxOutput = omap[iterIndex].value;
                } else if (iterIndex > 0 &&
                           omap.find(iterIndex - 1) != omap.end())
                  tmpMuxOutput =
                      omap[iterIndex - 1].value; // fallback to previous
                else
                  tmpMuxOutput = 0; // or  default
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
        // TODO: This maybe useless, please check
        for (const auto &leftMuxNode : tmpUnUpdatedMux) {
          auto selMuxSrcMap =switchInfo.staticinfo.dataflowGraph->muxToSrcNodeMap[leftMuxNode];
          std::string selMuxDataSrcNode =selMuxSrcMap[std::to_string(outValue)];
          int tmpMuxOutput = 0;

          if (contains(tmpMuxOutputMap,selMuxDataSrcNode)) {
            tmpMuxOutput = tmpMuxOutputMap[selMuxDataSrcNode];
          } else {
            llvm::dbgs() << "[ERROR-line407] Data src(" << selMuxDataSrcNode
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
    llvm::dbgs() << "[DEBUG] \t\tSeg Label: " << label << "\n";
    llvm::dbgs() << "[DEBUG] \t\t\tBB List: ";
    for (const auto &selBB : bblist) {
      llvm::dbgs() << selBB << ", ";
    }
    llvm::dbgs() << "\n";
  }

  for (const auto &selBaseNode : switchInfo.staticinfo.dataflowGraph->allDataBaseNode) {
    //! Testing
    // llvm::dbgs() << "[DEBUG] Node Name: " << selBaseNode << "\n";

    // Check the existence of the selected node
    if (!contains(switchInfo.staticinfo.dataflowGraph->nodes,selBaseNode))
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
        tmpMGInfo.control = segCtrlMergeSuccSearch(switchInfo, selBaseNode,
                                                  controlExcludVec, label);


        // for (int i=0; i<controlori.size();++i){

        //   llvm::dbgs()<<" segCtrlMergeSuccSearc original " << controlori[i] << " \n"  ;

        // }
        // for (int i=0;  i<tmpMGInfo.control.size() ;++i){
        //   llvm::dbgs()<<" segCtrlMergeSuccSearch new" <<tmpMGInfo.control[i] << " \n"  ;
        // }
        
        tmpMGInfo.data = segCtrlMergeSuccSearch(switchInfo, selBaseNode,
          dataExcludVec, label);
          // auto dataori = segCtrlMergeSuccSearch(switchInfo, selBaseNode,
          //   dataExcludVec, label);
          // llvm::dbgs()<<" segCtrlMergeSuccSearch new size " <<  tmpMGInfo.data.size() << " \n"  ;
          // llvm::dbgs()<<" segCtrlMergeSuccSearch original size " << dataori.size() << " \n"  ;
          
          // llvm::dbgs()<<" segCtrlMergeSuccSearch assert " << (tmpMGInfo.data == dataori) << " \n"  ;
        tmpMGInfo.glitch = segCtrlMergeGlitchSuccSearch(switchInfo, selBaseNode,
                                                       controlExcludVec, label);
          auto glitchori = segCtrlMergeGlitchSuccSearch(switchInfo, selBaseNode,
            controlExcludVec, label);

            // llvm::dbgs()<<" segCtrlMergeGlitchSuccSearch assert" <<(tmpMGInfo.glitch ==glitchori) << " \n"  ;
        // Update the stroing structure
        switchInfo.data.dfgBaseNodeValue[selBaseNode]->segSucNodeMap[label] =
            tmpMGInfo;
      }
    } else {
      for (const auto &[label, bblist] : switchInfo.staticinfo.segToBBs) {
        // Update the stroing structure
        switchInfo.data.dfgBaseNodeValue[selBaseNode]->segSucNodeMap[label] =
        segGeneralSuccSearch(switchInfo, selBaseNode, label);

      }
    }
  }
}



//===----------------------------------------------------------------------===//
//
// Functions for finding the data source node in different segments
//
//===----------------------------------------------------------------------===//







std::vector<std::string>
segCtrlMergeSuccSearch(SwitchingInfo &switchInfo, std::string startNode,
                      std::vector<std::string> &excludingList,
                      llvm::StringRef segLabel) {
  std::vector<std::string> succNodeList;
  auto &si = switchInfo;
  auto segBBs = llvm::ArrayRef<unsigned>(si.staticinfo.segToBBs[segLabel]);

  // Successor getter using helpers
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

  // Pre-visit callback to collect nodes
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
  std::string startNode,llvm::StringRef segLabel) {
  // Status variable definition
  unsigned numBuffers = 0;
  unsigned minDataWidth = 32;
  std::string lastNode = "";

  // Initialize the storing structure
  MgNodeInfo tmpMgNodeInfo;

  // Track the current path for proper buffer counting and cycle detection
  std::vector<std::string> currentPath;
  auto &si = switchInfo;
  auto segBBs = llvm::ArrayRef<unsigned>(si.staticinfo.segToBBs[segLabel]);

  // Successor getter with proper filtering
  auto succOf = [&si, &segBBs, &segLabel, &currentPath, &startNode]
               (const std::string &n) -> std::vector<std::string> {
  std::vector<std::string> out;

  if (n == startNode) {
    // *only* strip out the “base” nodes here—everything else is deferred
    for (auto &m : si.staticinfo.dataflowGraph->nodes[n]->sucs) {
      if (isDataBase(si, m)) continue;
      out.push_back(m);
    }
    return out;
  }

  // …all your existing full‐filtering code for non‐root nodes…
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
  // Pre-visit callback (entering a node)
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



std::vector<std::string>
segCtrlMergeGlitchSuccSearch(SwitchingInfo &si,
                            std::string startNode,
                            std::vector<std::string> &excludingList,
                            llvm::StringRef segLabel) {
  std::vector<std::string> glitchSuccNodeList;
  auto segBBs = llvm::ArrayRef<unsigned>(si.staticinfo.segToBBs[segLabel]);

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
      for (auto &m : si.staticinfo.dataflowGraph->nodes[n]->sucs) {
        if (isExcluded(excludingList, m))    continue;
        if (isDataBase(si, m))                continue;
        out.push_back(m);
      }
      return out;
    }
    // …else, do the full filtering…
    for (auto &m : si.staticinfo.dataflowGraph->nodes[n]->sucs) {
      if (isExcluded(excludingList, m))                 continue;
      if (isDataBase(si, m))                            continue;
      if (std::find(currentPath.begin(), currentPath.end(), m) != currentPath.end()) continue;
      if (isInvalidBackedge(si, segLabel, n, m))        continue;
      if (crossesCondPort(si, n, m))                    continue;
      if (bufferOutsideSeg(si, m, segBBs))              continue;
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
