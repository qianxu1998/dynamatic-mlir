//===- HandShakeChannelCal.cpp - Estimate Swithicng Activities ------*- C++ -*-===//
//
// This file declares all functions used for data channel switching calculation
//
//===----------------------------------------------------------------------===//

#include "experimental/Transforms/Switching/HandShakeChannelCal.h"
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

void extractBufferInfo(SwitchingInfo &switchInfo, std::string selMG, bool debug) {
  // Get the corresponding graph 
  auto selAdjGraph = switchInfo.segToAdjGraphMap[selMG];

  // Get needed information
  double_t selMgThroughput = switchInfo.cfdfcThroughput[std::stoi(selMG)];
  // TODO: Validate the following rounding
  float rawII = 1.0f / selMgThroughput;
  unsigned selMGII = static_cast<unsigned>(std::round(rawII));
  std::string baseNode = selAdjGraph->baseNode;

  //! Testing
  if (debug) {
    llvm::dbgs() << "[MG " << selMG << "]\n";
    llvm::dbgs() << "[MG INFO] Throughput: " << selMgThroughput << "\n";
    llvm::dbgs() << "[MG INFO] II: " << selMGII << "\n";
    llvm::dbgs() << "[MG INFO] Base Node: " << baseNode << "\n";
  }

  // Create the "universe" set: set of all cycle indices from 0..selCfdfcII-1
  std::set<unsigned> selMGUniverseSet;
  for (unsigned i = 0; i < selMGII; ++i) {
    selMGUniverseSet.insert(i);
  }

  // Analyze timing info for each of the buffers
  for (const auto& selBuffName: selAdjGraph->orderedNodeName) {
    if (selBuffName.find("buffer") != std::string::npos) {
      if (debug) {
        llvm::dbgs() << "\t[" << selBuffName << "]\n";
      }

      // Get the buffer node
      auto selBuffNode = dyn_cast<BufferNode>(selAdjGraph->nodes[selBuffName].get());

      // Get the longest path
      LongestPathResult tmpLongPath = selLongestPath(switchInfo, selBuffName, selMG);

      if (debug) {
        llvm::dbgs() << "\t\tStarting node: " << tmpLongPath.selStartNode << "\n";
        llvm::dbgs() << "\t\tPath Latency: " << tmpLongPath.maxLatency << "\n";
        llvm::dbgs() << "\t\tPath Last second buffer: " << tmpLongPath.lastSecondBuffer << "\n";
        llvm::dbgs() << "\t\tStart Node Shift: " << selAdjGraph->startBaseNodeShiftMap[tmpLongPath.selStartNode] << "\n\n";
      }

      // Step 1: Calculate D. Get the steady state buffer starting point
      // fin_start_point = (path_latency % sel_cfdfc_II)
      //                 + cfdfc_tim_start_nodes[cfdfcIndex][pathStartNode]
      //                 + cfdfc_tim_start_nodes[cfdfcIndex][baseNode]
      unsigned pathLatencyMod = tmpLongPath.maxLatency % selMGII;
      unsigned finStartPoint = pathLatencyMod
        + static_cast<unsigned>(selAdjGraph->startBaseNodeShiftMap[tmpLongPath.selStartNode])
        + static_cast<unsigned>(selAdjGraph->startBaseNodeShiftMap[baseNode]);

      if (debug) {
        llvm::dbgs() << "\t\tpathLatencyMod: " << pathLatencyMod << "\n";
        llvm::dbgs() << "\t\tD : " << finStartPoint
                  << " regarding the start of " << baseNode << "\n";
      }

    }
  }

}

