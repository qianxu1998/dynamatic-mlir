//===- SwitchingEstimation.cpp - Switching estimation ------------*- C++
//-*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// This file implements the --switching-estimation pass.
//===----------------------------------------------------------------------===//

#include "experimental/Analysis/SwitchingEstimation/SwitchingEstimation.h"
#include "experimental/Analysis/SwitchingEstimation/DataChannelCal.h"
#include "experimental/Analysis/SwitchingEstimation/HandShakeChannelCal.h"
#include "experimental/Analysis/SwitchingEstimation/ProfilingAnalyzer.h"
#include "experimental/Analysis/SwitchingEstimation/SwitchingSupport.h"
#include "experimental/Analysis/SwitchingEstimation/utils.h"

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
#include "llvm/Support/Path.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>

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

  void runOnOperation() override;

  //
  // Define global storing structure for switching information
  //
  SwitchingInfo switchingInfo;

  //
  //  Information Extraction Related Functions
  //
  // This function extract all CFDFC related information from the CFDFC analysis
  LogicalResult parseCFDFCInfo(handshake::FuncOp &funcOp,
                               CFDFCAnalysis &cfdfcAnalysis);

  // This function extracts and store the names of the ALU nodes in order
  void extractALUNodesInOrder(handshake::FuncOp &funcOp);

  // Dump the switching information
  void dumpSwitchingResults(const SwitchingInfo &switchInfo,
                            StringRef dumpPath);
  // Dump propagated data-channel values/toggles for debug.
  void dumpDataChannelDetails(const SwitchingInfo &switchInfo,
                              StringRef dumpPath);
  // Dump per-MG steady-state handshake channel estimations for debug.
  void dumpPerMGHandshakeDetails(const SwitchingInfo &switchInfo,
                                 StringRef dumpPath);

  //
  //  DataChannel Switching Calculation
  //
  // This function calculates the data channel switching number based on the
  // profiling results
  void calDataChannelSwitching(mlir::ModuleOp &topModule,
                               SCFProfilingResult &profileResults);

  //
  //  Handshake Channel Switching Calculation
  //
  void computeSteadyStateHandshakeSwitching(mlir::ModuleOp &topModule,
                                            SCFProfilingResult &profileResults);

  // This function calculates the handshake channel switching for the entire
  // circuit simulation
  void computeTotalHandshakeSwitching(mlir::ModuleOp &topModule,
                                      SCFProfilingResult &profileResults);
};

namespace {

/// Build a sibling debug-dump file path from the main CSV dump path.
/// Example:
///   /tmp/switching_estimation.csv + "_data_channels"
///   -> /tmp/switching_estimation_data_channels.txt
static std::string deriveSiblingDumpPath(StringRef basePath, StringRef suffix) {
  if (basePath.empty())
    return "";

  llvm::SmallString<256> out(basePath);
  llvm::sys::path::remove_filename(out);

  llvm::SmallString<128> name(llvm::sys::path::stem(basePath));
  name += suffix;
  name += ".txt";

  llvm::sys::path::append(out, name);
  return std::string(out.str());
}

/// Render the active-cycle set as "{0, 3, 7}".
static std::string formatActiveCycles(const IISet &set) {
  std::string rendered;
  llvm::raw_string_ostream os(rendered);
  os << "{";
  bool first = true;
  for (unsigned i = 0, e = set.size(); i < e; ++i) {
    if (!set.test(i))
      continue;
    if (!first)
      os << ", ";
    os << i;
    first = false;
  }
  if (first)
    os << "-";
  os << "}";
  return os.str();
}

/// Render the active-cycle bitmask in index order [0..II-1], e.g. "1010".
static std::string formatCycleMask(const IISet &set) {
  std::string rendered;
  llvm::raw_string_ostream os(rendered);
  for (unsigned i = 0, e = set.size(); i < e; ++i)
    os << (set.test(i) ? '1' : '0');
  return os.str();
}

} // namespace

void SwitchingEstimationPass::runOnOperation() {
  llvm::dbgs() << "[DEBUG] Running switching estimation pass\n";
  // Read Component Timing Models
  TimingDatabase timingDB;
  if (failed(TimingDatabase::readFromJSON(timingModels, timingDB)))
    return signalPassFailure();

  //=======================================================================//
  // Step 0: Get the CFDFC info
  //=======================================================================//
  llvm::dbgs() << "[DEBUG] [Step 0] Getting CFDFC info\n";
  auto topModule = dyn_cast<ModuleOp>(getOperation());
  if (failed(verifyIRMaterialized(topModule))) {
    topModule->emitError() << "The module is not fully materialized!";
    return signalPassFailure();
  }

  auto performanceAnalysis = getCachedAnalysis<CFDFCAnalysis>();
  if (!performanceAnalysis.has_value()) {
    topModule->emitError(
        "CFDFCAnalysis not available; "
        "run handshake-place-buffers (fpga20/fpl22/costaware/mapbuf) before "
        "switching-estimation.");
    llvm::dbgs() << "[DEBUG] [Step 0] Failed to get CFDFC analysis\n";
    return signalPassFailure();
  }

  // [STEP 0] Extract CFDFC info
  auto analysis = performanceAnalysis.value();
  for (handshake::FuncOp funcOp : topModule.getOps<handshake::FuncOp>()) {
    llvm::dbgs() << "[DEBUG] [Step 0] Processing function: " << funcOp.getName()
                 << "\n";

    if (failed(parseCFDFCInfo(funcOp, analysis))) {
      topModule->emitError()
          << "Failed to parse CFDFC info for switching estimation";
      return signalPassFailure();
    }
  }

  //=======================================================================//
  // [STEP 1] Parse the SCF level profiling results
  //=======================================================================//
  llvm::dbgs() << "[DEBUG] [Step 1] Parsing SCF profiling results\n";
  llvm::dbgs() << "[DEBUG] Data trace file path: " << dataTrace << "\n";
  SCFProfilingResult profilingResults(dataTrace, switchingInfo);

  //=======================================================================//
  // [STEP 2] Build Adjacency graph for each CFDFC
  //=======================================================================//
  std::vector<std::pair<std::string, std::string>> allBackedges;
  llvm::dbgs() << "[DEBUG] [Step 2] Building adjacency graph for each CFDFC\n";

  for (auto [mgIndex, selCFDFC] : switchingInfo.staticInfo.cfdfcInfoMap) {
    llvm::dbgs() << "[DEBUG] [Step 2] \tProcessing CFDFC index: " << mgIndex
                 << "\n";
    auto adj = std::make_shared<AdjGraph>(
        selCFDFC, timingDB, switchingInfo.staticInfo.cfdfcIIs[mgIndex], mgIndex,
        targetPeriod);
    switchingInfo.staticInfo.segToGraph.insert_or_assign(
        std::to_string(mgIndex), adj);

    // Update the global backedge list
    for (const auto &selPair : adj->backedges) {
      allBackedges.push_back(selPair);
    }
  }

  //=======================================================================//
  // [STEP 3] Build the graph for the entire dataflow graph (Final storage
  // structure)
  //=======================================================================//
  llvm::dbgs()
      << "[DEBUG] [Step 3] Building the graph for the entire dataflow graph\n";
  // [STEP 3.1] First store the information of the backedges in the circuit
  for (const auto &[selSegLabel, segBBList] :
       switchingInfo.staticInfo.segToBBs) {
    if (contains(selSegLabel, "S")) {
      switchingInfo.segInvalidBackedgesMap[selSegLabel] = allBackedges;
    } else if (contains(selSegLabel, "E")) {
      switchingInfo.segInvalidBackedgesMap[selSegLabel] = allBackedges;
    } else if (contains(selSegLabel, "T")) {
      auto sucMg = switchingInfo.staticInfo.transToSucMGMap[selSegLabel];
      std::vector<std::pair<std::string, std::string>> tmpInvalidBackedges;
      for (const auto &[selCFDFCIndex, selGraph] :
           switchingInfo.staticInfo.segToGraph) {
        if (selCFDFCIndex != sucMg) {
          for (const auto &selPair : selGraph->backedges) {
            tmpInvalidBackedges.push_back(selPair);
          }
        }
      }
      switchingInfo.segInvalidBackedgesMap[selSegLabel] =
          std::move(tmpInvalidBackedges);
    } else {
      // MG segments keep only their own backedges valid; all others must be
      // filtered out during segment-local DFS traversals.
      std::vector<std::pair<std::string, std::string>> tmpInvalidBackedges;
      for (const auto &[selCFDFCIndex, selGraph] :
           switchingInfo.staticInfo.segToGraph) {
        if (selCFDFCIndex != selSegLabel) {
          for (const auto &selPair : selGraph->backedges)
            tmpInvalidBackedges.push_back(selPair);
        }
      }
      switchingInfo.segInvalidBackedgesMap[selSegLabel] =
          std::move(tmpInvalidBackedges);
    }
  }

  //! Testing
  // llvm::dbgs() << "[DEBUG] \tInvalid Backedge list map:\n";
  // for (const auto &[segLabel, edgeList] :
  //      switchingInfo.segInvalidBackedgesMap) {
  //   llvm::dbgs() << "[DEBUG] \t\tsegLabel: " << segLabel << " :
  //   "; for (const auto &selPair : edgeList) {
  //     llvm::dbgs()
  //                << "(" << selPair.first << ", " << selPair.second << "), ";
  //   }
  //   llvm::dbgs() << "\n";
  // }

  // [STEP 3.2] Create the storing structure for the entire dataflow graph
  for (handshake::FuncOp funcOp : topModule.getOps<handshake::FuncOp>()) {
    // Contruct the Adj graph for the entire dataflow circuit
    switchingInfo.staticInfo.dataflowGraph = std::make_shared<AdjGraph>(
        timingDB, switchingInfo.staticInfo.cfdfcIIs[0], funcOp, allBackedges,
        targetPeriod);
  }

  //=======================================================================//
  // [STEP 4] Determining the Global start time and shifting for each CFDFC
  //=======================================================================//
  llvm::dbgs() << "[DEBUG] [Step 4] Determining the global start "
                  "time and shifting for each CFDFC\n";
  for (auto &[mgIndex, selCFDFC] : switchingInfo.staticInfo.segToGraph) {
    // [STEP 4.1] Determine the latest start time for each cfdfc
    selCFDFC->obtainNodeGlobalOrder();

    // [STEP 4.2] Determine the shifting for each start node in the cfdfc
    selCFDFC->computeStartNodeShifts();

    //! Testing
    llvm::dbgs() << "[DEBUG] [Step 4] CFDFC index: " << mgIndex << "\n";
    llvm::dbgs() << "[DEBUG]    Base Node: " << selCFDFC->baseNode << "\n";
    for (const auto &[selNode, selShift] : selCFDFC->startBaseNodeShiftMap) {
      llvm::dbgs() << "[DEBUG]    Start Node: " << selNode
                   << " with shift: " << selShift
                   << "; Cycle Time: " << selCFDFC->cycleTimeMap[selNode]
                   << "\n";
    }
  }

  //=======================================================================//
  // [STEP 5] Calculate Data channel switching
  //=======================================================================//
  llvm::dbgs() << "[DEBUG] [Step 5] Calculating data channel switching\n";
  calDataChannelSwitching(topModule, profilingResults);
  return;
  //=======================================================================//
  // [STEP 6] Calculate Steady State Handshake channel switching
  //=======================================================================//
  llvm::dbgs() << "[DEBUG] [Step 6] Calculating steady state "
                  "handshake channel switching\n";
  computeSteadyStateHandshakeSwitching(topModule, profilingResults);

  //=======================================================================//
  // [Step 7] Propagate handshake switching to the entire dataflow graph
  //=======================================================================//
  llvm::dbgs() << "[DEBUG] [Step 7] Propagating handshake switching "
                  "to the entire dataflow graph\n";
  computeTotalHandshakeSwitching(topModule, profilingResults);

  //=======================================================================//
  // [Step 8] Dump the switching estimation results
  //=======================================================================//
  // TODO: Need to refine the dumping format, we are counting the output ports
  // now, need to record the input ports switching of each node as well
  llvm::dbgs() << "[DEBUG] [Step 8] Dumping the switching estimation results\n";
  llvm::dbgs() << "[DEBUG] Dump file path: " << dumpFile << "\n";
  dumpSwitchingResults(switchingInfo, dumpFile);

  // Extra debug dumps:
  //  1) data-channel propagation details (per channel/per bit/per value)
  //  2) per-MG steady-state handshake channel details (valid/ready + IISets)
  const std::string dataChannelDump =
      deriveSiblingDumpPath(dumpFile, "_data_channels");
  const std::string mgHandshakeDump =
      deriveSiblingDumpPath(dumpFile, "_mg_handshake");

  llvm::dbgs() << "[DEBUG] Data-channel detail dump path: " << dataChannelDump
               << "\n";
  llvm::dbgs() << "[DEBUG] Per-MG handshake detail dump path: "
               << mgHandshakeDump << "\n";
  dumpDataChannelDetails(switchingInfo, dataChannelDump);
  dumpPerMGHandshakeDetails(switchingInfo, mgHandshakeDump);
}

//===----------------------------------------------------------------------===//
//
// Information Extraction
//
//===----------------------------------------------------------------------===//

void SwitchingEstimationPass::extractALUNodesInOrder(
    handshake::FuncOp &funcOp) {
  for (Operation &op : funcOp.getOps()) {
    // TODO: Maybe we can get rid of the name extraction process after revising
    // the data channel estimation method, 18/09/2025 Get the handshake.name
    // attribute
    std::string opName = op.getAttrOfType<StringAttr>("handshake.name").str();
    std::string opType = removeDigits(opName);

    if (NAME_SENSE_LIST.find(opType) != NAME_SENSE_LIST.end()) {
      // If the operation type is found in the name sense list, add it to the
      // ALU node list
      switchingInfo.staticInfo.traceOpNames.push_back(opName);
    }
  }
}

LogicalResult
SwitchingEstimationPass::parseCFDFCInfo(handshake::FuncOp &funcOp,
                                        CFDFCAnalysis &cfdfcAnalysis) {
  // Get all ALU names in the selected FuncOp
  extractALUNodesInOrder(funcOp);

  // Get the CFDFC info from the analysis result
  llvm::dbgs() << "[DEBUG] [Step 0] Extracting CFDFC info\n";

  unsigned cfdfcIndex = 0;

  // Iterate over cfdfcs in the function
  for (auto &cfdfc : cfdfcAnalysis.mapFuncOpToCFDFCs[funcOp]) {
    // Note: here cfdfc must be a reference to the objects in the vector,
    // otherwise the copy that "&cfdfc" points to will immediately goes out of
    // scope after the loop.
    switchingInfo.staticInfo.cfdfcInfoMap[cfdfcIndex] = &cfdfc;

    //! Testing
    llvm::dbgs() << "[DEBUG] [Step 0] Processing CFDFC index: " << cfdfcIndex
                 << "\n";

    // Extract the backedge info
    for (auto &be : cfdfc.backedges) {
      // Get the src and dst BB indices
      Operation *defOp = be.getDefiningOp();
      std::optional<unsigned> srcBB = getLogicBB(defOp);
      std::optional<unsigned> dstBB = getLogicBB(*be.getUsers().begin());

      auto bbPair = std::make_pair(srcBB.value(), dstBB.value());

      // If not in the global backedge list, add it
      if (std::find(switchingInfo.staticInfo.backEdges.begin(),
                    switchingInfo.staticInfo.backEdges.end(),
                    bbPair) == switchingInfo.staticInfo.backEdges.end()) {
        switchingInfo.staticInfo.backEdges.push_back(bbPair);
      }

      // If not in the CFDFC backedge list, add it
      if (!contains(switchingInfo.staticInfo.backEdgeToCFDFC, bbPair)) {
        std::vector<unsigned> tmpBEToCfdfc{cfdfcIndex};
        switchingInfo.staticInfo.backEdgeToCFDFC[bbPair] = tmpBEToCfdfc;

        //! Testing
        llvm::dbgs() << "[DEBUG] [Step 0]\tAdded new BE Pair: ("
                     << srcBB.value() << ", " << dstBB.value()
                     << ") with CFDFC index: " << cfdfcIndex << "\n";
      } else {
        if (std::find(switchingInfo.staticInfo.backEdgeToCFDFC[bbPair].begin(),
                      switchingInfo.staticInfo.backEdgeToCFDFC[bbPair].end(),
                      cfdfcIndex) ==
            switchingInfo.staticInfo.backEdgeToCFDFC[bbPair].end()) {
          // If the CFDFC index is not already in the list for this backedge
          // pair, add it
          switchingInfo.staticInfo.backEdgeToCFDFC[bbPair].push_back(
              cfdfcIndex);

          //! Testing
          llvm::dbgs() << "[DEBUG] [Step 0]\tUpdated BE Pair: (" << bbPair.first
                       << ", " << bbPair.second
                       << ") with CFDFC index: " << cfdfcIndex << "\n";
        }
      }
    }

    // Get the throughput
    auto throughput = cfdfc.throughput;
    float_t II = 1.0 / throughput;

    switchingInfo.staticInfo.cfdfcThroughput[cfdfcIndex] = throughput;
    switchingInfo.staticInfo.cfdfcIIs[cfdfcIndex] = II;

    // Get the BBs in the CFDFC
    std::vector<unsigned> bbVector(cfdfc.cycle.begin(), cfdfc.cycle.end());
    switchingInfo.staticInfo.segToBBs[std::to_string(cfdfcIndex)] = bbVector;

    //! Testing
    llvm::dbgs() << "[DEBUG] [Step 0]\tCFDFC index: " << cfdfcIndex
                 << ", Throughput: " << throughput << ", II: " << II << "\n";
    llvm::dbgs() << "[DEBUG] [Step 0]\tBBs in CFDFC: ";
    for (auto bb : bbVector) {
      llvm::dbgs() << bb << " ";
    }
    llvm::dbgs() << "\n";

    // Update the cfdfc index
    cfdfcIndex++;
  }

  return success();
}

void SwitchingEstimationPass::dumpSwitchingResults(
    const SwitchingInfo &switchInfo, StringRef dumpPath) {
  // TODO: Add support for calculating the number of cycles for switching
  // activity
  std::error_code EC;
  llvm::raw_fd_ostream OS(dumpPath, EC);

  if (EC) {
    llvm::errs() << "[ERROR] Could not open output file " << dumpPath << ": "
                 << EC.message() << "\n";
    return;
  }

  // Define csv header
  // TODO: Redesign the csv format for the following power estimation.
  OS << "node, data, valid, ready\n";

  // Dump the switching results for each node in the dataflow graph
  auto graph = switchInfo.staticInfo.dataflowGraph;
  const char *readyTraceEnv = std::getenv("SWITCH_EST_READY_TRACE_NODE");
  auto shouldTraceReadyNode = [&](llvm::StringRef nodeName) {
    if (!readyTraceEnv || !*readyTraceEnv)
      return false;
    llvm::SmallVector<llvm::StringRef, 8> patterns;
    llvm::StringRef(readyTraceEnv).split(patterns, ',', -1, false);
    for (llvm::StringRef pattern : patterns) {
      pattern = pattern.trim();
      if (!pattern.empty() && nodeName.contains(pattern))
        return true;
    }
    return false;
  };
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

void SwitchingEstimationPass::dumpDataChannelDetails(
    const SwitchingInfo &switchInfo, StringRef dumpPath) {
  std::error_code EC;
  llvm::raw_fd_ostream OS(dumpPath, EC);

  if (EC) {
    llvm::errs() << "[ERROR] Could not open data-channel debug dump file "
                 << dumpPath << ": " << EC.message() << "\n";
    return;
  }

  OS << "# Switching Estimation Debug Dump: Data-Channel Propagation\n";
  OS << "# Includes per-node/per-channel values, total switches, and per-bit "
        "toggles.\n\n";

  auto graph = switchInfo.staticInfo.dataflowGraph;
  if (!graph) {
    OS << "(no dataflow graph available)\n";
    return;
  }

  for (const auto &nodeName : graph->orderedNodeName) {
    auto it = graph->nodes.find(nodeName);
    if (it == graph->nodes.end())
      continue;

    const auto *node = it->second.get();
    if (contains(nodeName, "mem_controller"))
      continue;

    OS << "================================================================\n";
    OS << "Node: " << nodeName << "\n";
    OS << "  total_data_switching: " << node->totalDataSwitching << "\n";
    OS << "  data_channel_count: " << node->dataOut.size() << "\n";

    if (node->dataOut.empty()) {
      OS << "  (no propagated data channels)\n\n";
      continue;
    }

    for (const auto &[channelName, values] : node->dataOut) {
      unsigned width = 0;
      if (auto widthIt = node->sucsDataWidthMap.find(channelName);
          widthIt != node->sucsDataWidthMap.end())
        width = widthIt->second;

      unsigned switches = 0;
      if (auto swIt = node->dataSwitches.find(channelName);
          swIt != node->dataSwitches.end())
        switches = swIt->second;

      OS << "  Channel: " << channelName << "\n";
      OS << "    width: " << width << "\n";
      OS << "    switches: " << switches << "\n";

      OS << "    values (sequence):\n";
      if (values.empty()) {
        OS << "      (none)\n";
      } else {
        for (unsigned idx = 0, e = values.size(); idx < e; ++idx) {
          int value = values[idx];
          if (value == -1)
            OS << "      [" << idx << "] X(-1)\n";
          else
            OS << "      [" << idx << "] " << value << "\n";
        }
      }

      OS << "    per_bit_toggles:\n";
      auto bitToggleIt = node->perChannelToggle.find(channelName);
      if (bitToggleIt == node->perChannelToggle.end() ||
          bitToggleIt->second.empty()) {
        OS << "      (none)\n";
      } else {
        for (const auto &[bitIdx, toggleNum] : bitToggleIt->second)
          OS << "      bit[" << bitIdx << "] = " << toggleNum << "\n";
      }
    }

    OS << "\n";
  }
}

void SwitchingEstimationPass::dumpPerMGHandshakeDetails(
    const SwitchingInfo &switchInfo, StringRef dumpPath) {
  std::error_code EC;
  llvm::raw_fd_ostream OS(dumpPath, EC);

  if (EC) {
    llvm::errs() << "[ERROR] Could not open per-MG handshake debug dump file "
                 << dumpPath << ": " << EC.message() << "\n";
    return;
  }

  OS << "# Switching Estimation Debug Dump: Per-MG Handshake Estimation\n";
  OS << "# Includes per-node valid/ready switching and active II sets.\n\n";

  std::vector<std::string> mgLabels;
  mgLabels.reserve(switchInfo.staticInfo.segToGraph.size());
  for (const auto &entry : switchInfo.staticInfo.segToGraph)
    mgLabels.push_back(entry.first().str());

  std::sort(mgLabels.begin(), mgLabels.end(),
            [](const std::string &lhs, const std::string &rhs) {
              unsigned lhsNum = 0, rhsNum = 0;
              bool lhsIsNum = !lhs.empty() &&
                              !llvm::StringRef(lhs).getAsInteger(10, lhsNum);
              bool rhsIsNum = !rhs.empty() &&
                              !llvm::StringRef(rhs).getAsInteger(10, rhsNum);
              if (lhsIsNum && rhsIsNum)
                return lhsNum < rhsNum;
              if (lhsIsNum != rhsIsNum)
                return lhsIsNum;
              return lhs < rhs;
            });

  for (const auto &mgLabel : mgLabels) {
    auto mgIt = switchInfo.staticInfo.segToGraph.find(mgLabel);
    if (mgIt == switchInfo.staticInfo.segToGraph.end())
      continue;
    auto mgGraph = mgIt->second;

    unsigned mgIndex = 0;
    double throughput = 0.0;
    if (!llvm::StringRef(mgLabel).getAsInteger(10, mgIndex)) {
      if (auto tpIt = switchInfo.staticInfo.cfdfcThroughput.find(mgIndex);
          tpIt != switchInfo.staticInfo.cfdfcThroughput.end())
        throughput = tpIt->second;
    }

    OS << "================================================================\n";
    OS << "MG: " << mgLabel << "\n";
    OS << "  II: " << mgGraph->cfdfcII << "\n";
    OS << "  Throughput: " << throughput << "\n";
    OS << "  Node Count: " << mgGraph->nodes.size() << "\n\n";

    for (const auto &nodeName : mgGraph->orderedNodeName) {
      auto nodeIt = mgGraph->nodes.find(nodeName);
      if (nodeIt == mgGraph->nodes.end())
        continue;
      const auto *node = nodeIt->second.get();

      OS << "  Node: " << nodeName << "\n";
      OS << "    total_valid_switching: " << node->totalValidSwitching << "\n";
      OS << "    total_ready_switching: " << node->totalReadySwitching << "\n";

      OS << "    valid_channels:\n";
      if (node->validSignal.empty()) {
        OS << "      (none)\n";
      } else {
        for (const auto &[sucName, switchNum] : node->validSignal) {
          OS << "      " << sucName << " : " << switchNum;
          auto setIt = node->setV.find(sucName);
          if (setIt != node->setV.end()) {
            OS << " ; active_cycles=" << formatActiveCycles(setIt->second)
               << " ; mask=" << formatCycleMask(setIt->second);
          } else {
            OS << " ; active_cycles={missing}";
          }
          OS << "\n";
        }
      }

      OS << "    ready_channels:\n";
      if (node->readySignal.empty()) {
        OS << "      (none)\n";
      } else {
        for (const auto &[preName, switchNum] : node->readySignal) {
          OS << "      " << preName << " : " << switchNum;
          auto setIt = node->setR.find(preName);
          if (setIt != node->setR.end()) {
            OS << " ; active_cycles=" << formatActiveCycles(setIt->second)
               << " ; mask=" << formatCycleMask(setIt->second);
          } else {
            OS << " ; active_cycles={missing}";
          }
          OS << "\n";
        }
      }
      OS << "\n";
    }
  }
}

//===----------------------------------------------------------------------===//
//
// Data Channel Switching Calculation
//
//===----------------------------------------------------------------------===//

void SwitchingEstimationPass::calDataChannelSwitching(
    mlir::ModuleOp &topModule, SCFProfilingResult &profileResults) {
  // [SS 0] Construct the execution map

  llvm::dbgs() << "[DEBUG] [Step 5.0] Constructing the execution map\n";
  switchingInfo.dataInfo.executedSegmentTrace =
      profileResults.executedSegmentTrace;
  switchingInfo.dataInfo.segmentToExecutionIndices.clear();
  for (unsigned i = 0; i < profileResults.executedSegmentTrace.size(); i++) {
    std::string segLabel = profileResults.executedSegmentTrace[i];
    switchingInfo.dataInfo.segmentToExecutionIndices[segLabel].push_back(i);
    if (switchingInfo.dataInfo.segmentToFirstExecutionIter.find(segLabel) ==
        switchingInfo.dataInfo.segmentToFirstExecutionIter.end()) {
      switchingInfo.dataInfo.segmentToFirstExecutionIter[segLabel] = i;
    }
  }

  // [SS 1] Get the iteration index for the first execution of each segment

  llvm::dbgs()
      << "[DEBUG] [Step 5.1] Get the BB Pair to Control Merge Output Map\n";
  mapBBPairToControlMerge(switchingInfo);

  //! Testing
  for (const auto &[pair1, cmVec] :
       switchingInfo.dataInfo.bbPairToControlMergeOutputs) {
    llvm::dbgs() << "[DEBUG] \t(" << pair1.first << ", " << pair1.second
                 << ") : \n";
    for (auto selPair : cmVec) {
      llvm::dbgs() << "[DEBUG] \t\t[" << selPair.first << " " << selPair.second
                   << "]\n";
    }
  }

  // [SS 2] Construct the list of all data source nodes from scf-level profiling
  llvm::dbgs() << "[DEBUG] [Step 5.2] Construct the list of all "
                  "data source nodes from scf-level profiling\n";
  getDataBaseNodes(switchingInfo, profileResults);

  //! Testing
  for (auto &[segLabel, selDB] :
       switchingInfo.dataInfo.segmentToDataSourceNodes) {
    llvm::dbgs() << "[DEBUG] \t[SEGMENT] " << segLabel << "\n";
    printSegmentDataSourceNodes(selDB);
  }

  // [SS 3] Contruct the data source node info of mux, condbr and mem node
  llvm::dbgs() << "[DEBUG] [Step 5.3] Contruct the data source node "
                  "info of mux, condbr and mem node and opaque buffers\n";
  switchingInfo.staticInfo.dataflowGraph->buildSrcMaps();

  //! Testing
  printMuxToSrcNodeMap(switchingInfo.staticInfo.dataflowGraph->muxToSrcNodeMap);
  printSrcNodeToMuxMap(switchingInfo.staticInfo.dataflowGraph->srcNodeToMuxMap);
  llvm::dbgs() << "[DEBUG] \tcondBr Node to control src map: \n";
  for (const auto &[cbrNode, controlSrc] :
       switchingInfo.staticInfo.dataflowGraph->condBrToConSrcMap) {
    llvm::dbgs() << "[DEBUG] \t\t(" << cbrNode << ", " << controlSrc << ")\n";
  }

  // [SS 4] Update value for all data base nodes
  llvm::dbgs() << "[DEBUG] [Step 5.4] Update value for all data base nodes\n";
  dataChannelBaseNodesValueUpdate(switchingInfo, profileResults);

  return;
  // [SS 5] Build succeeding node list for data base nodes in different segments
  llvm::dbgs() << "[DEBUG] [Step 5.5] Build succeeding node list "
                  "for data base nodes in different segments\n";
  buildSegmentSuccNodesList(switchingInfo, profileResults);

  // [SS 6] Get all glitching base node in each MG
  llvm::dbgs() << "[DEBUG] [Step 5.6] Get all glitching base node in each MG\n";
  dataGlitchNodeSearch(switchingInfo, profileResults);

  // [SS 7] Update all glitching value for data base nodes in the dataflow
  // circuit
  llvm::dbgs() << "[DEBUG] [Step 5.7] Update all glitching value "
                  "for data base nodes in the dataflow circuit\n";
  dataBaseNodeGlitchUpdate(switchingInfo, profileResults, false);

  // [SS 8] Propagate all the data base value
  llvm::dbgs() << "[DEBUG] [Step 5.8] Propagate all the data base value\n";
  dfgDataChannelPropagate(switchingInfo, profileResults, false);

  // [SS 9] Calculate the data channel switching for each node in the dataflow
  // graph
  llvm::dbgs() << "[DEBUG] [Step 5.9] Calculate the data channel "
                  "switching for each node in the dataflow graph\n";
  for (auto &entry : switchingInfo.staticInfo.dataflowGraph->nodes) {
    const auto &nodeName = entry.first();
    if (contains(nodeName, "mem_controller")) {
      continue;
    }

    auto *node = entry.second.get();
    node->totalDataSwitchingCounting(false);
    // llvm::dbgs() << "[DEBUG] \t[Node] " << nodeName << "\n";
    // llvm::dbgs() << "[DEBUG] \t\tData Switching: " <<
    // node->totalDataSwitching
    //              << "\n";
    // llvm::dbgs() << "[DEBUG] \t\tValid Switching: " <<
    // node->totalValidSwitching
    //              << "\n";
    // llvm::dbgs() << "[DEBUG] \t\tReady Switching: " <<
    // node->totalReadySwitching
    //              << "\n";
  }
}

//===----------------------------------------------------------------------===//
//
// Handshake Channel Switching
//
//===----------------------------------------------------------------------===//
void SwitchingEstimationPass::computeSteadyStateHandshakeSwitching(
    mlir::ModuleOp &topModule, SCFProfilingResult &profileResults) {
  // For each MG, we do the following two steps
  //  Step 1: Update buffer information
  //  Step 2: Calculate the steady state handshake channel switching

  // Step 1
  for (unsigned i = 0; i < switchingInfo.staticInfo.cfdfcThroughput.size();
       i++) {
    updateMGBufferSwitching(switchingInfo, std::to_string(i), false);
  }

  // Step 2
  for (unsigned i = 0; i < switchingInfo.staticInfo.cfdfcThroughput.size();
       i++) {
    mgHandshakeSwitchingCounting(switchingInfo, std::to_string(i), false);
  }

  // Debug: Print steady-state handshake values after calculation
  llvm::dbgs() << "[DEBUG] [STEP 6 RESULTS] Steady-state handshake values:\n";
  for (const auto &[mgIndex, mgGraph] : switchingInfo.staticInfo.segToGraph) {
    llvm::dbgs() << "[DEBUG] \tMG " << mgIndex << ":\n";
    for (const auto &[nodeName, nodePtr] : mgGraph->nodes) {
      llvm::dbgs() << "[DEBUG] \t\tNode " << nodeName
                   << ": V=" << nodePtr->totalValidSwitching
                   << " R=" << nodePtr->totalReadySwitching << "\n";
    }
  }
}

void SwitchingEstimationPass::computeTotalHandshakeSwitching(
    mlir::ModuleOp &topModule, SCFProfilingResult &profileResults) {
  /* -----------------------------------------------------------------------
     Assumptions
       • For nodes that belong to an MG segment we re-use the steady-state
         switching numbers that were computed previously.
       • If an MG is executed only once we apply the “buffer-only” rule for
         buffers and the steady-state rule for the other nodes.
       • For nodes that live in an S / E / T segment we assume that:
           – every Valid output toggles twice  (0→1→0)
           – every Ready input never toggles   (always 0)
     ----------------------------------------------------------------------*/

  //--------------------------------------------------------------------+
  // 1)  Iterate over the execution segments in sequential order
  //--------------------------------------------------------------------+
  for (auto it = profileResults.executionPhaseToSegmentExecCount.begin();
       it != profileResults.executionPhaseToSegmentExecCount.end(); ++it) {
    auto segIdx = it->first;
    auto segLabel = it->second.first;
    auto numExec = it->second.second;

    // ------------------------------------------------------------------
    // Fallback: some MG segments appear with an execution count of 0 in
    // executionPhaseToSegmentExecCount.  When that happens we derive the real
    // count directly from the execution trace so that handshake scaling
    // uses the correct multiplier.
    // ------------------------------------------------------------------
    if (numExec == 0) {
      numExec = static_cast<unsigned>(
          std::count(profileResults.executedSegmentTrace.begin(),
                     profileResults.executedSegmentTrace.end(), segLabel));
      llvm::dbgs() << "[DEBUG] \t[WARNING] Segment " << segLabel
                   << " has 0 executions in executionPhaseToSegmentExecCount. "
                   << "Using " << numExec
                   << " from executedSegmentTrace instead.\n";
    }

    llvm::dbgs() << "[DEBUG] \tSegIndex: " << segIdx
                 << "; MG_Label: " << segLabel << ", Num Exec: " << numExec
                 << "\n";

    //------------------------------------------------------------------+
    // 2-A)  SEGMENT TYPE :  “S”  or  “T”
    //------------------------------------------------------------------+
    // TODO: Define a separate data storing structure for active nodes in the
    // seg
    auto graph = switchingInfo.staticInfo.dataflowGraph;
    if (segLabel.find("S") != std::string::npos ||
        segLabel.find("T") != std::string::npos) {
      llvm::dbgs() << "[DEBUG] \t[SEGMENT] " << segLabel << "\n";
      llvm::dbgs() << "[DEBUG] \t\t[Type] S or T\n";

      // Traverse all active nodes in the segment
      for (const auto &nodeName : graph->orderedNodeName) {
        // Get the node
        AdjNode *node = graph->nodes[nodeName].get();

        auto segBBList = switchingInfo.staticInfo.segToBBs[segLabel];
        if (std::find(segBBList.begin(), segBBList.end(), node->bbindex) ==
            segBBList.end()) {
          continue;
        }
        // Calculate the number of valid switches
        unsigned numSucs = node->sucs.size();
        unsigned numValidSwitches = 2 * numSucs;

        // The node will be always ready
        unsigned numReadySwitches = 0;

        // Update the storing structure
        node->updateHandshakeChannelSwitching(numValidSwitches,
                                              numReadySwitches);

        // Update per channel switching information
        for (const auto &suc : node->sucs) {
          if (node->validSignal.find(suc) != node->validSignal.end()) {
            node->validSignal[suc] += 2;
          } else {
            node->validSignal[suc] = 2;
          }
        }
      }

      // Skip the rest of the steps
      continue;
    } else if (segLabel.find("E") != std::string::npos) {
      llvm::dbgs() << "[DEBUG] \t[SEGMENT] " << segLabel << "\n";
      // Get the previous segment
      std::string prevSegLabel = std::prev(it)->second.first;
      llvm::dbgs() << "[DEBUG] \t\t[Prev Segment] " << prevSegLabel << "\n";

      // Traverse all active nodes in the segment
      for (const auto &nodeName : graph->orderedNodeName) {
        // Get the node
        AdjNode *node = graph->nodes[nodeName].get();

        // This node is in segment E
        auto segBBList = switchingInfo.staticInfo.segToBBs[segLabel];
        if (std::find(segBBList.begin(), segBBList.end(), node->bbindex) ==
            segBBList.end()) {
          continue;
        }

        // Check wheter the node is in previous section or not
        //* Assumption: Seg E will only be following MG ?
        auto prevSegBBList = switchingInfo.staticInfo.segToBBs[prevSegLabel];
        if (containsValue(prevSegBBList, node->bbindex)) {
          // Get the node info in the previous segment
          AdjNode *prevNode = switchingInfo.staticInfo.segToGraph[prevSegLabel]
                                  ->nodes[nodeName]
                                  .get();
          // Node in the previous segment
          unsigned numValidSwitches = prevNode->totalValidSwitching;
          unsigned numSucs = prevNode->sucs.size();

          // if (numValidSwitches != (numSucs * 2)) {
          //   if (nodeName.find("constant") == std::string::npos &&
          //   nodeName.find("source") == std::string::npos) {
          //     // This is the ending segment transition 1 -> 0
          //     numValidSwitches += (numSucs * 2 - numValidSwitches) / 2;
          //   }
          // }

          // Always add the final 1→0 transition of the last token so that
          // each Valid line ends with the falling edge seen in the Python
          // reference implementation.
          if ((!contains(nodeName, "constant")) &&
              (!contains(nodeName, "source"))) {
            numValidSwitches += (numSucs * 2 - numValidSwitches) / 2;
          }

          unsigned numReadySwitches = prevNode->totalReadySwitching;

          // Check whether this is a load unit
          if (nodeName.find("load") != std::string::npos) {
            numValidSwitches *= 2;
            numReadySwitches *= 2;
          }

          // Update the perchannel information
          // Valid Channel
          for (const auto &suc : node->sucs) {
            unsigned selSucValidSwitching = 0;
            // Check whether this output is in the previous segment or not
            if (prevNode->validSignal.find(suc) !=
                prevNode->validSignal.end()) {
              selSucValidSwitching = prevNode->validSignal[suc];
            } else {
              selSucValidSwitching = 1;
            }

            // Update the valid signal
            if (node->validSignal.find(suc) != node->validSignal.end()) {
              node->validSignal[suc] += selSucValidSwitching;
            } else {
              node->validSignal[suc] = selSucValidSwitching;
            }
          }

          // Ready Channel
          for (const auto &pre : node->pres) {
            unsigned selPreReadySwitching = 0;
            // Check whether this input is in the previous segment or not
            if (prevNode->readySignal.find(pre) !=
                prevNode->readySignal.end()) {
              selPreReadySwitching = prevNode->readySignal[pre];
            } else {
              selPreReadySwitching = 0;
            }

            // Update the ready signal
            if (node->readySignal.find(pre) != node->readySignal.end()) {
              node->readySignal[pre] += selPreReadySwitching;
            } else {
              node->readySignal[pre] = selPreReadySwitching;
            }
          }

          // Update the storing structure
          node->updateHandshakeChannelSwitching(numValidSwitches,
                                                numReadySwitches);
        } else {
          // Get node type
          auto selNodeType = getNodeType(nodeName);
          unsigned numValidSwitches = 0;
          unsigned numReadySwitches = 0;
          if (selNodeType.find("cond_br") != std::string::npos) {
            numValidSwitches = 2;
            numReadySwitches = 4;
          } else if (JOIN_NODE.find(selNodeType) != JOIN_NODE.end()) {
            numValidSwitches = 2;
            numReadySwitches = 2;
          } else {
            unsigned numSucs = node->sucs.size();
            numValidSwitches = 2 * numSucs;
            numReadySwitches = 0;
          }

          // Update the storing structure
          node->updateHandshakeChannelSwitching(numValidSwitches,
                                                numReadySwitches);
        }
      }
    } else {
      llvm::dbgs() << "[DEBUG] \t[SEGMENT] " << segLabel << "\n";
      llvm::dbgs() << "[DEBUG] \t\t[Type] MG\n";
      // TODO: Update the logic here for the rest of the nodes

      // Traverse all active nodes in the segment
      for (const auto &nodeName :
           switchingInfo.staticInfo.segToGraph[segLabel]->orderedNodeName) {
        // Get the node
        AdjNode *graphNode = graph->nodes[nodeName].get();
        AdjNode *mgNode = switchingInfo.staticInfo.segToGraph[segLabel]
                              ->nodes[nodeName]
                              .get();

        // Get the number of switching
        unsigned numValidSwitches = numExec * mgNode->totalValidSwitching;
        unsigned numReadySwitches = numExec * mgNode->totalReadySwitching;

        // Update the per-channel information
        // Valid Channel
        for (const auto &suc : graphNode->sucs) {
          unsigned selSucValidSwitching = 0;
          // Check whether this output is in the previous segment or not
          if (mgNode->validSignal.find(suc) != mgNode->validSignal.end()) {
            selSucValidSwitching = numExec * mgNode->validSignal[suc];
          } else {
            selSucValidSwitching = 0;
          }

          // Update the valid signal
          if (graphNode->validSignal.find(suc) !=
              graphNode->validSignal.end()) {
            graphNode->validSignal[suc] += selSucValidSwitching;
          } else {
            graphNode->validSignal[suc] = selSucValidSwitching;
          }
        }

        // Ready Channel
        for (const auto &pre : graphNode->pres) {
          unsigned selPreReadySwitching = 0;
          // Check whether this input is in the previous segment or not
          if (mgNode->readySignal.find(pre) != mgNode->readySignal.end()) {
            selPreReadySwitching = numExec * mgNode->readySignal[pre];
          } else {
            selPreReadySwitching = 0;
          }

          // Update the ready signal
          if (graphNode->readySignal.find(pre) !=
              graphNode->readySignal.end()) {
            graphNode->readySignal[pre] += selPreReadySwitching;
          } else {
            graphNode->readySignal[pre] = selPreReadySwitching;
          }
        }

        // Check whether this is a load unit
        if (nodeName.find("load") != std::string::npos) {
          numValidSwitches *= 2;
          numReadySwitches *= 2;
        }

        // Update the storing structure
        graphNode->updateHandshakeChannelSwitching(numValidSwitches,
                                                   numReadySwitches);
      }
    }
  }
}
