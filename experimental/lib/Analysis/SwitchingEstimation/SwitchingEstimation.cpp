//===- SwitchingEstimation.cpp - Switching estimation ------------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// This file implements the --switching-estimation pass.
//===----------------------------------------------------------------------===//

#include "experimental/Analysis/SwitchingEstimation/SwitchingEstimation.h"
#include "experimental/Analysis/SwitchingEstimation/SwitchingSupport.h"
#include "experimental/Analysis/SwitchingEstimation/utils.h"
#include "experimental/Analysis/SwitchingEstimation/ProfilingAnalyzer.h"
#include "experimental/Analysis/SwitchingEstimation/DataChannelCal.h"

#include "dynamatic/Analysis/CFDFCAnalysis.h"
#include "dynamatic/Analysis/NameAnalysis.h"
#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Support/CFG.h"
#include "dynamatic/Support/LLVM.h"
#include "dynamatic/Support/TimingModels.h"
#include "dynamatic/Transforms/BufferPlacement/BufferingSupport.h"
#include "dynamatic/Transforms/BufferPlacement/CFDFC.h"
#include "dynamatic/Transforms/HandshakeMaterialize.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "switching-estimation"

using namespace llvm;
using namespace mlir;
using namespace dynamatic;
using namespace dynamatic::experimental;

namespace dynamatic {
namespace experimental {
#define GEN_PASS_DEF_SWITCHINGESTIMATION
#include "experimental/Analysis/Passes.h.inc"
}; // namespace experimental
}; // namespace dynamatic

struct SwitchingEstimationPass
    : public dynamatic::experimental::impl::SwitchingEstimationBase<
          SwitchingEstimationPass> {

  using SwitchingEstimationBase::SwitchingEstimationBase;
  // TODO: 1. Add one more data channel update method. 2. Update Buffer handshake switching model based on the buffer type
  void runOnOperation() override;

  //
  // Define global storing structure for switching information
  //
  SwitchingInfo switchingInfo;
  
  // 
  //  Information Extraction Related Functions
  //
  // This function extract all CFDFC related information from the CFDFC analysis
  LogicalResult parseCFDFCInfo(handshake::FuncOp& funcOp, CFDFCAnalysis &cfdfcAnalysis);

  // This function extracts and store the names of the ALU nodes in order
  void extractALUNodesInOrder(handshake::FuncOp& funcOp);

  // Dump the switching information
  void dumpSwitchingResults(const SwitchingInfo& switchInfo, StringRef dumpPath);
  
  // 
  //  DataChannel Switching Calculation
  //
  // This function calculates the data channel switching number based on the profiling results
  void calDataChannelSwitching(mlir::ModuleOp& topModule, SCFProfilingResult &profileResults);

};

void SwitchingEstimationPass::runOnOperation() {
  LLVM_DEBUG(llvm::dbgs() << "[DEBUG] Running switching estimation pass\n");
  // Read Component Timing Models
  TimingDatabase timingDB;
  if (failed(TimingDatabase::readFromJSON(timingModels, timingDB)))
    return signalPassFailure();

  // Step 0: Get the CFDFC info
  LLVM_DEBUG(llvm::dbgs() << "[DEBUG] [Step 0] Getting CFDFC info\n");
  auto topModule = dyn_cast<ModuleOp>(getOperation());
  if (failed(verifyIRMaterialized(topModule))) {
    topModule->emitError() << "The module is not fully materialized!";
    return signalPassFailure();
  }

  auto performanceAnalysis = getCachedAnalysis<CFDFCAnalysis>();
  if (!performanceAnalysis.has_value()) {
    topModule->emitError("CFDFCAnalysis not available; "
        "run handshake-place-buffers (fpga20/fpl22/costaware/mapbuf) before switching-estimation.");
    LLVM_DEBUG(llvm::dbgs() << "[DEBUG] [Step 0] Failed to get CFDFC analysis\n");
    return signalPassFailure();
  }

  // [STEP 0] Extract CFDFC info
  auto analysis = performanceAnalysis.value();
  for (handshake::FuncOp funcOp: topModule.getOps<handshake::FuncOp>()) {
    LLVM_DEBUG(llvm::dbgs() << "[DEBUG] [Step 0] Processing function: " << funcOp.getName() << "\n");

    if (failed(parseCFDFCInfo(funcOp, analysis))) {
      topModule->emitError() << "Failed to parse CFDFC info for switching estimation";
      return signalPassFailure();
    }
  }

  // [STEP 1] Parse the SCF level profiling results
  LLVM_DEBUG(llvm::dbgs() << "[DEBUG] [Step 1] Parsing SCF profiling results\n");
  LLVM_DEBUG(llvm::dbgs() << "[DEBUG] Data trace file path: " << dataTrace << "\n");
  SCFProfilingResult profilingResults(dataTrace, switchingInfo);

  // [STEP 2] Build Adjacency graph for each CFDFC
  std::vector<std::pair<std::string, std::string>> allBackedges;
  LLVM_DEBUG(llvm::dbgs() << "[DEBUG] [Step 2] Building adjacency graph for each CFDFC\n");

  for (auto [mgIndex, selCFDFC] : switchingInfo.staticinfo.cfdfcInfoMap) {
    LLVM_DEBUG(llvm::dbgs() << "[DEBUG] [Step 2] \tProcessing CFDFC index: " << mgIndex << "\n");
    auto adj = std::make_shared<AdjGraph>(selCFDFC, timingDB,
                                          switchingInfo.staticinfo.cfdfcIIs[mgIndex],
                                          mgIndex, targetPeriod);
    switchingInfo.staticinfo.segToGraph.insert_or_assign(std::to_string(mgIndex), adj);

    // Update the global backedge list
    for (const auto& selPair : adj->backedges) {
      allBackedges.push_back(selPair);
    }
  }

  // [STEP 3] Build the graph for the entire dataflow graph (Final storage structure)
  LLVM_DEBUG(llvm::dbgs() << "[DEBUG] [Step 3] Building the graph for the entire dataflow graph\n");
  // [STEP 3.1] First store the information of the backedges in the circuit
  for (const auto& [selSegLabel, segBBList] : switchingInfo.staticinfo.segToBBs) {
    if (contains(selSegLabel,"S")) {
      switchingInfo.segInvalidBackedgesMap[selSegLabel] = allBackedges;
    } else if (contains(selSegLabel,"E")) {
      switchingInfo.segInvalidBackedgesMap[selSegLabel] = allBackedges;
    } else if (contains(selSegLabel,"T")) {
      auto sucMg = switchingInfo.staticinfo.transToSucMGMap[selSegLabel];
      std::vector<std::pair<std::string, std::string>> tmpInvalidBackedges; 
      for (const auto& [selCFDFCIndex, selGraph]: switchingInfo.staticinfo.segToGraph) {
        if (selCFDFCIndex != sucMg) {
          for (const auto& selPair: selGraph->backedges) {
            tmpInvalidBackedges.push_back(selPair);
          }
        }
      }
    }
  }

  //! Testing
  LLVM_DEBUG(llvm::dbgs() << "[DEBUG] \tInvalue Backedge list map:\n");
  for (const auto& [segLabel, edgeList] : switchingInfo.segInvalidBackedgesMap) {
    LLVM_DEBUG(llvm::dbgs() << "[DEBUG] \t\tsegLabel: " << segLabel << " : ");
    for (const auto& selPair: edgeList) {
      LLVM_DEBUG(llvm::dbgs() << "(" << selPair.first << ", " << selPair.second << "), ");
    }
    LLVM_DEBUG(llvm::dbgs() << "\n");
  }

  // [STEP 3.2] Create the storing structure for the entire dataflow graph
  for (handshake::FuncOp funcOp : topModule.getOps<handshake::FuncOp>()) {
    // Contruct the Adj graph for the entire dataflow circuit
    switchingInfo.staticinfo.dataflowGraph = std::make_shared<AdjGraph>(timingDB, switchingInfo.staticinfo.cfdfcIIs[0], funcOp, allBackedges, targetPeriod);
  }

  // [STEP 4] Determining the Global start time and shifting for each CFDFC
  LLVM_DEBUG(llvm::dbgs() << "[DEBUG] [Step 4] Determining the global start time and shifting for each CFDFC\n");
  for (auto& [mgIndex, selCFDFC] : switchingInfo.staticinfo.segToGraph) {
    // [STEP 4.1] Determine the latest start time for each cfdfc
    selCFDFC->obtainNodeGlobalOrder();

    // [STEP 4.2] Determine the shifting for each start node in the cfdfc
    selCFDFC->computeStartNodeShifts();
  }

  // [STEP 5] Calculate Data channel switching
  LLVM_DEBUG(llvm::dbgs() << "[DEBUG] [Step 5] Calculating data channel switching\n");
  calDataChannelSwitching(topModule, profilingResults);


  // [STEP 6] Calculate Steady State Handshake channel switching

  // [Step 7] Propagate handshake switching to the entire dataflow graph

  // [Step 8] Dump the switching estimation results

}

//===----------------------------------------------------------------------===//
//
// Information Extraction
//
//===----------------------------------------------------------------------===//

void SwitchingEstimationPass::extractALUNodesInOrder(handshake::FuncOp& funcOp) {
  for (Operation& op : funcOp.getOps()) {
    // TODO: Maybe we can get rid of the name extraction process after revising the data channel estimation method, 18/09/2025
    // Get the handshake.name attribute
    std::string opName = op.getAttrOfType<StringAttr>("handshake.name").str();
    std::string opType = removeDigits(opName);

    if (NAME_SENSE_LIST.find(opType) != NAME_SENSE_LIST.end()) {
      // If the operation type is found in the name sense list, add it to the ALU node list
      switchingInfo.staticinfo.traceOpNames.push_back(opName);
    }
  }
}

LogicalResult SwitchingEstimationPass::parseCFDFCInfo(handshake::FuncOp& funcOp, 
          CFDFCAnalysis &cfdfcAnalysis) {
  // Get all ALU names in the selected FuncOp
  extractALUNodesInOrder(funcOp);

  // Get the CFDFC info from the analysis result
  LLVM_DEBUG(llvm::dbgs() << "[DEBUG] [Step 0] Extracting CFDFC info\n");

  unsigned cfdfcIndex = 0;

  // Iterate over cfdfcs in the function
  for (auto& cfdfc : cfdfcAnalysis.mapFuncOpToCFDFCs[funcOp]) {
    // Note: here cfdfc must be a reference to the objects in the vector,
    // otherwise the copy that "&cfdfc" points to will immediately goes out of
    // scope after the loop.
    switchingInfo.staticinfo.cfdfcInfoMap[cfdfcIndex] = &cfdfc;

    //! Testing
    LLVM_DEBUG(llvm::dbgs() << "[DEBUG] [Step 0] Processing CFDFC index: " << cfdfcIndex << "\n");

    // Extract the backedge info
    for (auto& be : cfdfc.backedges) {
      // Get the src and dst BB indices
      Operation *defOp = be.getDefiningOp();
      std::optional<unsigned> srcBB = getLogicBB(defOp);
      std::optional<unsigned> dstBB = getLogicBB(*be.getUsers().begin());

      auto bbPair = std::make_pair(srcBB.value(), dstBB.value());

      // If not in the global backedge list, add it
      if (std::find(switchingInfo.staticinfo.backEdges.begin(), switchingInfo.staticinfo.backEdges.end(), bbPair) == switchingInfo.staticinfo.backEdges.end()) {
        switchingInfo.staticinfo.backEdges.push_back(bbPair);
      }

      // If not in the CFDFC backedge list, add it
      if (!contains(switchingInfo.staticinfo.backEdgeToCFDFC, bbPair)) {
        std::vector<unsigned> tmpBEToCfdfc{cfdfcIndex};
        switchingInfo.staticinfo.backEdgeToCFDFC[bbPair] = tmpBEToCfdfc;

        //! Testing
        LLVM_DEBUG(llvm::dbgs() << "[DEBUG] [Step 0]\tAdded new BE Pair: (" << srcBB.value() << ", " << dstBB.value() << ") with CFDFC index: " << cfdfcIndex << "\n");
      } else {
        if (std::find(switchingInfo.staticinfo.backEdgeToCFDFC[bbPair].begin(), switchingInfo.staticinfo.backEdgeToCFDFC[bbPair].end(), cfdfcIndex) == switchingInfo.staticinfo.backEdgeToCFDFC[bbPair].end()) {
          // If the CFDFC index is not already in the list for this backedge pair, add it
          switchingInfo.staticinfo.backEdgeToCFDFC[bbPair].push_back(cfdfcIndex);

          //! Testing
          LLVM_DEBUG(llvm::dbgs() << "[DEBUG] [Step 0]\tUpdated BE Pair: (" << bbPair.first << ", " << bbPair.second << ") with CFDFC index: " << cfdfcIndex << "\n");
        }
      }
    }

    // Get the throughput
    auto throughput = cfdfc.throughput;
    float_t II = 1.0 / throughput;

    switchingInfo.staticinfo.cfdfcThroughput[cfdfcIndex] = throughput;
    switchingInfo.staticinfo.cfdfcIIs[cfdfcIndex] = II;

    // Get the BBs in the CFDFC
    std::vector<unsigned> bbVector(cfdfc.cycle.begin(), cfdfc.cycle.end());
    switchingInfo.staticinfo.segToBBs[std::to_string(cfdfcIndex)] = bbVector;

    //! Testing
    LLVM_DEBUG(llvm::dbgs() << "[DEBUG] [Step 0]\tCFDFC index: " << cfdfcIndex << ", Throughput: " << throughput << ", II: " << II << "\n");
    LLVM_DEBUG(llvm::dbgs() << "[DEBUG] [Step 0]\tBBs in CFDFC: ");
    for (auto bb : bbVector) {
      LLVM_DEBUG(llvm::dbgs() << bb << " ");
    }
    LLVM_DEBUG(llvm::dbgs() << "\n");

    // Update the cfdfc index
    cfdfcIndex++;
  }

  return success();
} 

void SwitchingEstimationPass::dumpSwitchingResults(const SwitchingInfo& switchInfo, StringRef dumpPath) {
  // TODO: Add support for calculating the number of cycles for switching activity
  std::error_code EC;
  llvm::raw_fd_ostream OS(dumpPath, EC);

  if (EC) {
    llvm::errs() << "[ERROR] Could not open output file " << dumpPath
                 << ": " << EC.message() << "\n";
    return;
  }

  // Define csv header
  // TODO: Redesign the csv format for the following power estimation.
  OS << "node, data, valid, ready\n";

  // Dump the switching results for each node in the dataflow graph
  auto graph = switchInfo.staticinfo.dataflowGraph;
  for (const auto &entry : graph->nodes) {
    // TODO: Add support for mem_controller
    const auto &nodeName = entry.first();
    if (contains(nodeName, "mem_controller")) {
      continue;
    }

    const auto *node = entry.second.get();
    OS << nodeName << "," << node->totalDataSwitching << ","
        << node->totalValidSwitching << "," << node->totalReadySwitching << "\n";
  }
}

//===----------------------------------------------------------------------===//
//
// Data Channel Switching Calculation
//
//===----------------------------------------------------------------------===//

void SwitchingEstimationPass::calDataChannelSwitching(mlir::ModuleOp& topModule, SCFProfilingResult &profileResults) {
  // [SS 0] Construct the execution map
  LLVM_DEBUG(llvm::dbgs() << "[DEBUG] [Step 5.0] Constructing the execution map\n");
  for (unsigned i = 0; i < profileResults.executedSegTrace.size(); i++) {
    std::string segLabel = profileResults.executedSegTrace[i];
    if (switchingInfo.data.firstExecutedIter.find(segLabel) == switchingInfo.data.firstExecutedIter.end()) {
      switchingInfo.data.firstExecutedIter[segLabel] = i;
    }
  }

  // [SS 1] Get the iteration index for the first execution of each segment
  LLVM_DEBUG(llvm::dbgs() << "[DEBUG] [Step 5.1] Get the BB Pair to Control Merge Output Map\n");
  mapBBPairToControlMerge(switchingInfo);

  //! Testing
  for (const auto& [pair1, cmVec]: switchingInfo.data.bbPairToCtrlMerge) {
    LLVM_DEBUG(llvm::dbgs() << "[DEBUG] \t(" << pair1.first << ", " << pair1.second << ") : \n");
    for (auto selPair: cmVec) {
      LLVM_DEBUG(llvm::dbgs() << "[DEBUG] \t\t[" << selPair.first << " " << selPair.second << "]\n");
    }
  }

  // [SS 2] Construct the list of all data source nodes from scf-level profiling 
  LLVM_DEBUG(llvm::dbgs() << "[DEBUG] [Step 5.2] Construct the list of all data source nodes from scf-level profiling\n");
  getDataBaseNodes(switchingInfo, profileResults);

  //! Testing
  for (auto& [segLabel, selDB]: switchingInfo.data.segToDataBaseVec) {
    llvm::dbgs() << "[DEBUG] \t[SEGMENT] " << segLabel << "\n";
    printDataBaseNodesTriple(selDB);
  }

  // [SS 3] Contruct the data source node info of mux, condbr and mem node
  LLVM_DEBUG(llvm::dbgs() << "[DEBUG] [Step 5.3] Contruct the data source node info of mux, condbr and mem node\n");
  switchingInfo.staticinfo.dataflowGraph->buildSrcMaps();

  //! Testing
  printMuxToSrcNodeMap(switchingInfo.staticinfo.dataflowGraph->muxToSrcNodeMap);
  printSrcNodeToMuxMap(switchingInfo.staticinfo.dataflowGraph->srcNodeToMuxMap);
  LLVM_DEBUG(llvm::dbgs() << "[DEBUG] \tcondBr Node to control src map: \n");
  for (const auto& [cbrNode, controlSrc]: switchingInfo.staticinfo.dataflowGraph->condBrToConSrcMap) {
    llvm::dbgs() << "[DEBUG] \t\t(" << cbrNode << ", " << controlSrc << ")\n";
  }

  // [SS 4] Update value for all data base nodes
  LLVM_DEBUG(llvm::dbgs() << "[DEBUG] [Step 5.4] Update value for all data base nodes\n");
  dataChannelBaseNodesValueUpdate(switchingInfo, profileResults);

  // [SS 5] Build succeeding node list for data base nodes in different segments
  LLVM_DEBUG(llvm::dbgs() << "[DEBUG] [Step 5.5] Build succeeding node list for data base nodes in different segments\n");
  buildSegmentSuccNodesList(switchingInfo, profileResults);

  // [SS 6] Get all glitching base node in each MG
  LLVM_DEBUG(llvm::dbgs() << "[DEBUG] [Step 5.6] Get all glitching base node in each MG\n");
  
}

