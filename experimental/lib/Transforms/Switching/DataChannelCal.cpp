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
    for (unsigned i = 0; i < opOperands.size(); i++) {
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

void dataChannelBaseNodesValueUpdate(SwitchingInfo &switchInfo, SCFProfilingResult &profileResults) {
  // Iterate all profile base node in the dataflow graph
  for (const auto& selNode: switchInfo.dataflowGraph->profileBaseNodes) {
    
  }
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
