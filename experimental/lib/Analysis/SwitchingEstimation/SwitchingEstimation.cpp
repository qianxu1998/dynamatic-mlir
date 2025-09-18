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
  // Define global storing structure for swithcing information
  //
  
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
    return;
  }

  auto performanceAnalysis = getCachedAnalysis<CFDFCAnalysis>();
  if (!performanceAnalysis.has_value()) {
    topModule->emitError("CFDFCAnalysis not available; "
        "run handshake-place-buffers (fpga20/fpl22/costaware/mapbuf) before switching-estimation.");
    LLVM_DEBUG(llvm::dbgs() << "[DEBUG] [Step 0] Failed to get CFDFC analysis\n");
    return signalPassFailure();
  }

  // Get Corresponding performance analysis per funcOp
  auto analysis = performanceAnalysis.value();
  for (handshake::FuncOp funcOp: topModule.getOps<handshake::FuncOp>()) {
    LLVM_DEBUG(llvm::dbgs() << "[DEBUG] [Step 0] Processing function: " << funcOp.getName() << "\n");

    SmallVector<CFDFC *> cfdfcPtrs;

    for (auto &cfdfc: analysis.get().mapFuncOpToCFDFCs[funcOp]) {
      cfdfcPtrs.push_back(&cfdfc);


      // TODO: Update the buffer occupancy. 15/09/2025
    }
  }


}
