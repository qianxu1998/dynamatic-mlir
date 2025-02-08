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
  for (const auto& [nodeName, node]: switchInfo.dataflowGraph->nodes) {
   if (auto CMOp = dyn_cast<handshake::ControlMergeOp>(node->op)) {
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

// void getDataBaseNodes(SwitchingInfo &switchInfo, SCFProfilingResult &profileResults) {
//   // Instantiate all DataBaseNodesTriple for all segments
//   for (const auto& [segLabel, BBVec]: switchInfo.segToBBListMap) {
//     DataBaseNodesTriple tmpDataBaseNodes;
//     switchInfo.segToDataBaseVecMap[segLabel] = tmpDataBaseNodes;
//   }

//   // Traverse all the nodes in the dataflow graph
//   for (const auto& [nodeName, node]: switchInfo.dataflowGraph->nodes) {
//     unsigned nodeBB;
//     if (std::optional<unsigned> optBB = getLogicBB(node->op); !optBB.has_value())
//       continue;
//     else
//       nodeBB = *optBB;

//     auto dataflowGraph = switchInfo.dataflowGraph;
//     // Check the types of the nodes
//     // TYPE 1: DATA nodes from scf profiling
//     if (profileResults.opNameToValueListMap.find(nodeName) != profileResults.opNameToValueListMap.end() ||
//           (nodeName.find("constant") != std::string::npos) ||
//           (nodeName.find("source") != std::string::npos)) {
//       // Check each segment
//       for (const auto& [segLabel, BBVec]: switchInfo.segToBBListMap) {
//         if (std::find(BBVec.begin(), BBVec.end(), nodeBB) != BBVec.end()) {
//           // If this is a MG
//           if (switchInfo.segToAdjGraphMap.find(segLabel) != switchInfo.segToAdjGraphMap.end()) {
//             switchInfo.segToAdjGraphMap[segLabel]->profileBaseNodes.insert(nodeName);
//             switchInfo.segToAdjGraphMap[segLabel]->allDataBaseNode.insert(nodeName);
//           }

//           switchInfo.segToDataBaseVecMap[segLabel].data.push_back(nodeName);
//           switchInfo.segToDataBaseVecMap[segLabel].all.push_back(nodeName);
//         }
//       }
//     // TYPE 2: Control Nodes
//     } else if ((nodeName.find("control_merge") != std::string::npos) ||
//                 (nodeName.find("mux") != std::string::npos)) {
//       // Check each segment
//       for (const auto& [segLabel, BBVec]: switchInfo.segToBBListMap) {
//         if (std::find(BBVec.begin(), BBVec.end(), nodeBB) != BBVec.end()) {
//           // If this is a MG
//           if (switchInfo.segToAdjGraphMap.find(segLabel) != switchInfo.segToAdjGraphMap.end()) {
//             switchInfo.segToAdjGraphMap[segLabel]->allDataBaseNode.insert(nodeName);
//           }

//           switchInfo.segToDataBaseVecMap[segLabel].control.push_back(nodeName);
//           switchInfo.segToDataBaseVecMap[segLabel].all.push_back(nodeName);
//         }
//       }
//     }
//   }
// }






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
