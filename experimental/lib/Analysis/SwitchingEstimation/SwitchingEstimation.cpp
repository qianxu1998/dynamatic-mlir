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

