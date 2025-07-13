//===- SwitchingEstimation.cpp - Estimate Swithicng Activities ------*- C++ -*-===//
//
// Implements the switching estimation pass for all untis in the generated
// dataflow circuit
//
//===----------------------------------------------------------------------===//

#include "experimental/Transforms/Switching/SwitchingEstimation.h"
#include "experimental/Transforms/Switching/SwitchingSupport.h"
#include "experimental/Transforms/Switching/DFSKernel.h"
#include "experimental/Transforms/Switching/DataChannelPropagation.h"

#include "experimental/Transforms/Switching/DataGlitches.h"
#include "experimental/Transforms/Switching/utils.h"

#include "experimental/Transforms/Switching/ProfilingAnalyzer.h"
#include "experimental/Transforms/Switching/DataChannelCal.h"
#include "experimental/Transforms/Switching/HandShakeChannelCal.h"
#include "experimental/Transforms/Switching/HandShakeBufferAnalysis.h"

#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Dialect/Handshake/HandshakeAttributes.h"
#include "dynamatic/Support/DynamaticPass.h"
#include "dynamatic/Support/CFG.h"
#include "dynamatic/Support/LLVM.h"
#include "dynamatic/Support/Logging.h"
#include "dynamatic/Support/Backedge.h"
#include "dynamatic/Support/TimingModels.h"
#include "dynamatic/Support/Attribute.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/IR/Value.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Parallel.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace dynamatic;
using namespace dynamatic::handshake;
using namespace dynamatic::buffer;
using namespace dynamatic::experimental;
using namespace dynamatic::experimental::switching;

namespace {
// Define Switching Estimation pass driver
struct SwitchingEstimationPass
    : public dynamatic::experimental::switching::impl::SwitchingEstimationBase<
        SwitchingEstimationPass> {

  SwitchingEstimationPass(StringRef dataTrace,

                          StringRef frequencies,
                          StringRef timingModels) {
    this->dataTrace = dataTrace.str();
    this->frequencies = frequencies.str();
    this->timingModels = timingModels.str();
  }

  // Main Interface
void runDynamaticPass() override;

  // 
  //  Define global storing structure
  //
  SwitchingInfo switchInfo;

  HandshakeInfo handshakeInfo;

  // 
  //  Information Extraction Related Functions
  //
  // This function extract the backedge archs from all archs in the dataflow graph
  llvm::SmallVector<std::pair<unsigned, unsigned>> extractBackedges(llvm::SmallVector<experimental::ArchBB> archs);

  // This function extract all CFDFCs from the bbList attribute and the corresponding II from CFDFCThroughputAttr
  // and store all the information in an instance of SwitchingInfo
  LogicalResult extractAllCFDFCs(mlir::ModuleOp& topModule);

  // Extract all op names of the alus in order
  void extractHandshakeOpNames(handshake::FuncOp& topFunc);

  // 
  //  DataChannel Switching Calculation
  //
  // Function calculates the number of switches in the data channels
  void calDataChannelSwitching(mlir::ModuleOp& topModule, SCFProfilingResult &profileResults);
  void dumpSwitchingResults(const SwitchingInfo &info,
      llvm::StringRef outPath = "estimation_cpp.csv");
  // 
  //  Handshake Channel Switching Calculation
  //
  void computeSteadyStateHandshakeSwitching(mlir::ModuleOp& topModule, SCFProfilingResult &profileResults);

  // This function calculates the handshake channel switching for the entire circuit simulation
  void computeTotalHandshakeSwitching(mlir::ModuleOp& topModule, SCFProfilingResult &profileResults);
};
} // namespace 

void SwitchingEstimationPass::runDynamaticPass() {
  // Read Component Latencies from the provided database.
  TimingDatabase timingDB(&getContext());
  if (failed(TimingDatabase::readFromJSON(timingModels, timingDB)))
    signalPassFailure();

  // Step 0 : Build the top-level CFDFC storing structure (abstract)
  // Get MLIR funcop, for each example, only 1 funcop can exist
  mlir::ModuleOp topModule = getOperation();
  if (failed(extractAllCFDFCs(topModule))) {
    llvm::errs() << "[ERROR] Extraction of CFDFCs failed\n";
  }

  // Step 1: Parse the SCF level profiling results
  llvm::dbgs() << "[DEBUG] [Step 1] Parsing Profiling Results\n";
  // llvm::dbgs() << "[DEBUG] \tBBList Log file: " << bbList << "\n";
  llvm::dbgs() << "[DEBUG] \tData Profiling Log file: " << dataTrace << "\n";
  SCFProfilingResult profilingResults(dataTrace, switchInfo);

  // Step 2: Build Adjacency graph for each CFDFC
  std::vector<std::pair<std::string, std::string>> allBackedges;
  llvm::dbgs() << "[DEBUG] [Step 2] Build Adjacency Graph for Each Segment\n";
  bool debug=1;
  for (const auto& [mgIndex, mgInstance]: switchInfo.staticinfo.cfdfcs) {
    llvm::dbgs() << "[DEBUG] \tMG : " << mgIndex << "\n";
    // AdjGraph a(mgInstance, timingDB, switchInfo.staticinfo.cfdfcIIs[mgIndex], mgIndex);


    auto adj = std::make_shared<AdjGraph>(mgInstance, timingDB, switchInfo.staticinfo.cfdfcIIs[mgIndex], mgIndex,debug);
    // AdjGraph tmpAdjGraph(mgInstance, timingDB, switchInfo.staticinfo.cfdfcIIs[mgIndex], mgIndex);
    switchInfo.staticinfo.segToGraph.insert_or_assign(std::to_string(mgIndex), adj);

    // Update the backedge list
    for (const auto& selPair : adj->backedges) {
      allBackedges.push_back(selPair);
    }
  }

// AdjGraph>(mgInstance, timingDB, switchInfo.staticinfo.cfdfcIIs[mgIndex], mgIndex)
  // Step 3: Build the graph for the entire dataflow graph
  llvm::dbgs() << "[DEBUG] [STEP 3] Construct the Adj Graph for the entire DFG and sort the multiplexers\n";
  /// Step 3.1: First store the information of the backedges in the circuit
  for (const auto& [selSegLabel, segBBList] : switchInfo.staticinfo.segToBBs) {
    if (contains(selSegLabel,"S")) {
      switchInfo.segInvalidBackedgesMap[selSegLabel] = allBackedges;
    } else if (contains(selSegLabel,"E")) {
      switchInfo.segInvalidBackedgesMap[selSegLabel] = allBackedges;
    } else if (contains(selSegLabel,"T")) {
      auto sucMg = switchInfo.staticinfo.transToSucMGMap[selSegLabel];
      std::vector<std::pair<std::string, std::string>> tmpInvalidBackedges; 
      for (const auto& [selCFDFCIndex, selGraph]: switchInfo.staticinfo.segToGraph) {
        if (selCFDFCIndex != sucMg) {
          for (const auto& selPair: selGraph->backedges) {
            tmpInvalidBackedges.push_back(selPair);
          }
        }
      }
    }
  }

  //! Testing
  llvm::dbgs() << "[DEBUG] \tInvalue Backedge list map:\n";
  for (const auto& [segLabel, edgeList] : switchInfo.segInvalidBackedgesMap) {
    llvm::dbgs() << "[DEBUG] \t\tsegLabel: " << segLabel << " : ";
    for (const auto& selPair: edgeList) {
      llvm::dbgs() << "(" << selPair.first << ", " << selPair.second << "), ";
    }
    llvm::dbgs() << "\n";
  }
  
  // Step 3.2: Create the storing structure for the entire dataflow graph
  for (handshake::FuncOp funcOp : topModule.getOps<handshake::FuncOp>()) {
    // Contruct the Adj graph for the entire dataflow circuit
    switchInfo.staticinfo.dataflowGraph = std::make_shared<AdjGraph>(timingDB, switchInfo.staticinfo.cfdfcIIs[0], funcOp, allBackedges,debug);
  }


llvm::dbgs() << "[DEBUG] [STEP 4] Determining the Global start time and shifting for each MG\n";

  // auto &graphMap = switchInfo.staticinfo.segToGraph;

  // // Collect pointers to AdjGraph
  // std::vector<std::shared_ptr<AdjGraph>> graphs;
  // graphs.reserve(graphMap.size());
  // for (auto &entry : graphMap)
  //   graphs.push_back(entry.second);
  
  // // Parallel processing
  // llvm::parallelForEach(graphs, [](const auto &selGraph) {
  //   selGraph->obtainNodeGlobalOrder();
  //   selGraph->computeStartNodeShifts();
  // });

  for (auto& [mgIndex, selGraph]: switchInfo.staticinfo.segToGraph) {
    // Step 4.1: Determining the latest start time for each node cfdfc
    selGraph->obtainNodeGlobalOrder();
    // selGraph->obtainNodeGlobalOrderold();
    
    //! Testing
    // llvm::dbgs() << "[DEBUG] \t\t Global order calculated"<<(++a)<<"\n"; 
    // Step 4.2: Check the shifting between different start node within a graph
    selGraph->computeStartNodeShifts();
    // selGraph->computeStartNodeShiftsold();

    
  }

  // Step 5: Calculate switches in data channel 
  llvm::dbgs() << "[DEBUG] [STEP 5] Calculate Data Channel Switching\n";
  calDataChannelSwitching(topModule, profilingResults);

  // Step 6: Calculate switches in handshake channels
  llvm::dbgs() << "[DEBUG] [STEP 6] Calculate Handshake Channel Switching\n";
  computeSteadyStateHandshakeSwitching(topModule, profilingResults);

  // Step 7: Calculate switches in handshake channels for the entire simualtion
  llvm::dbgs() << "[DEBUG] [STEP 7] Calculate Handshake Channel Switching for the entire simulation\n";

  computeTotalHandshakeSwitching(topModule, profilingResults);

    
  // Step 8: Dump node switching results for post‑processing
  dumpSwitchingResults(switchInfo);
}


//===----------------------------------------------------------------------===//
// Dump Switching Results to CSV (node,data,valid,ready)
//===----------------------------------------------------------------------===//
void SwitchingEstimationPass::dumpSwitchingResults(const SwitchingInfo &info,
                                                   llvm::StringRef outPath) {
  std::error_code EC;
  llvm::raw_fd_ostream OS(outPath, EC);
  if (EC) {
    llvm::errs() << "[ERROR] Could not open output file " << outPath
                 << ": " << EC.message() << "\n";
    return;
  }

  // CSV header
  OS << "node,data,valid,ready\n";

  auto graph = info.staticinfo.dataflowGraph;
  for (const auto &entry : graph->nodes) {
    const auto &nodeName = entry.first();
    if (contains(nodeName,"mem_controller")){
      continue;}

    const auto *node = entry.second.get();
    OS << nodeName << ',' << node->totalDataSwitching << ','
       << node->totalValidSwitching << ',' << node->totalReadySwitching
       << '\n';
  }
}

//===----------------------------------------------------------------------===//
//
// Data Channel Switching
//
//===----------------------------------------------------------------------===//
void SwitchingEstimationPass::calDataChannelSwitching(mlir::ModuleOp& topModule, SCFProfilingResult &profileResults) {
  // Step 0: Construct the execution map
  llvm::dbgs() << "[DEBUG]  [SS0] Construct the segLable to IterIdx Map\n";
  for (unsigned i = 0; i < profileResults.executedSegTrace.size(); i++) {
    std::string segLabel = profileResults.executedSegTrace[i];
    if (switchInfo.data.firstExecutedIter.find(segLabel) == switchInfo.data.firstExecutedIter.end()) {
      switchInfo.data.firstExecutedIter[segLabel] = i;
    }
  }

  // Step 1: Get the iteration index for the frist execution of each segment
  llvm::dbgs() << "[DEBUG]  [SS1] Get the BB Pair to Control Merge Output Map\n";
  mapBBPairToControlMerge(switchInfo);

  //! Testing
  for (const auto& [pair1, cmVec]: switchInfo.data.bbPairToCtrlMerge) {
    llvm::dbgs() << "[DEBUG] \t(" << pair1.first << ", " << pair1.second << ") : \n";
    for (auto selPair: cmVec) {
      llvm::dbgs() << "[DEBUG] \t\t[" << selPair.first << " " << selPair.second << "]\n";
    }
  }

  // Step 2: Construct the list of all data source nodes from scf-level profiling 
  llvm::dbgs() << "[DEBUG]  [SS2] Get all the database nodes in each segments\n";
  getDataBaseNodes(switchInfo, profileResults);
  //! Testing
  for (auto& [segLabel, selDB]: switchInfo.data.segToDataBaseVec) {
    llvm::dbgs() << "[DEBUG] \t[SEGMENT] " << segLabel << "\n";
    printDataBaseNodesTriple(selDB);
  }

  // Step 3: Contruct the data source node info of mux, condbr and mem node
  llvm::dbgs() << "[DEBUG]  [SS3] Construct the data source node storing structure for different nodes in the dataflow graph\n";
  // TODO: Sometimes the source node of a mux node is from the block argumnet, need to add new nodes in the grpah.
  switchInfo.staticinfo.dataflowGraph->builSrcMaps();

  //! Testing
  printMuxToSrcNodeMap(switchInfo.staticinfo.dataflowGraph->muxToSrcNodeMap);
  printSrcNodeToMuxMap(switchInfo.staticinfo.dataflowGraph->srcNodeToMuxMap);
  llvm::dbgs() << "[DEBUG] \tcondBr Node to control src map: \n";
  for (const auto& [cbrNode, controlSrc]: switchInfo.staticinfo.dataflowGraph->condBrToConSrcMap) {
    llvm::dbgs() << "[DEBUG] \t\t(" << cbrNode << ", " << controlSrc << ")\n";
  }

  // Step 4: Update value for all data base nodes
  llvm::dbgs() << "[DEBUG]  [SS4] Update the value for data base nodes\n";
  dataChannelBaseNodesValueUpdate(switchInfo, profileResults);

  // Step 5: Build succeeding node list for data base nodes in different segments
  llvm::dbgs() << "[DEBUG]  [SS5] Build succeeding node list\n";
  buildSegmentSuccNodesList(switchInfo, profileResults);

  // Step 6: Get all glitching base node in each MG
  llvm::dbgs() << "[DEBUG]  [SS6] Find all glitching nodes\n";
  dataGlitchNodeSearch(switchInfo, profileResults);

  // Step 7: Update all glitching value for data base nodes in the dataflow circuit
  llvm::dbgs() << "[DEBUG]  [SS7] Calculate all glitching values\n";
  dataBaseNodeGlitchUpdate(switchInfo, profileResults, false);

  // Step 8: Propagate all the data base value
  llvm::dbgs() << "[DEBUG]  [SS8] Final data channel value updates\n";
  dfgDataChannelPropagate(switchInfo, profileResults, false);
  llvm::dbgs() << "[DEBUG]  [SS9] Calculate final data switching counts\n";
  for (const auto &entry : switchInfo.staticinfo.dataflowGraph->nodes) {
      const auto &nodeName = entry.first();
      if (nodeName.find("mem_controller") != std::string::npos) {
          continue;
      }
      auto *node = entry.second.get();
      node->totalDataSwitchingCounting(false);  // Calculate final switching counts
  }
}

//===----------------------------------------------------------------------===//
//
// Handshake Channel Switching
//
//===----------------------------------------------------------------------===//
void SwitchingEstimationPass::computeSteadyStateHandshakeSwitching(mlir::ModuleOp& topModule, SCFProfilingResult &profileResults) {
  // For each MG, we do the following two steps
  //  Step 1: Update buffer information
  //  Step 2: Calculate the steady state handhshake channel switching

  // Step 1
  bool debugbuffer{1};
  for (unsigned i = 0; i < switchInfo.staticinfo.cfdfcThroughput.size(); i++) {
    updateMGBufferSwitching(switchInfo, std::to_string(i), debugbuffer);
  }

  // Step 2
  bool debughscount{1};
  for (unsigned i = 0; i < switchInfo.staticinfo.cfdfcThroughput.size(); i++) {
    mgHandshakeSwitchingCounting(switchInfo,handshakeInfo, std::to_string(i), debughscount);
  }
  
  // Debug: Print steady-state handshake values after calculation
  llvm::dbgs() << "[DEBUG] [STEP 6 RESULTS] Steady-state handshake values:\n";
  for (const auto& [mgIndex, mgGraph] : switchInfo.staticinfo.segToGraph) {
    llvm::dbgs() << "[DEBUG] \tMG " << mgIndex << ":\n";
    for (const auto& [nodeName, nodePtr] : mgGraph->nodes) {
      llvm::dbgs() << "[DEBUG] \t\tNode " << nodeName << ": V=" << nodePtr->totalValidSwitching 
                   << " R=" << nodePtr->totalReadySwitching << "\n";
    }
  }
}

void SwitchingEstimationPass::computeTotalHandshakeSwitching(mlir::ModuleOp& topModule, SCFProfilingResult &profileResults) {
  /* -----------------------------------------------------------------------
     Assumptions
       • For nodes that belong to an MG segment we re-use the steady-state
         switching numbers that were computed previously.
       • If an MG is executed only once we apply the "buffer-only" rule for
         buffers and the steady-state rule for the other nodes.
       • For nodes that live in an S / E / T segment we assume that:
           – every Valid output toggles twice  (0→1→0)
           – every Ready input never toggles   (always 0)
     ----------------------------------------------------------------------*/

  //--------------------------------------------------------------------+
  // 1)  Iterate over the execution segments in sequential order
  //--------------------------------------------------------------------+
  for (auto it = profileResults.execPhaseToSegExecNumMap.begin();
       it != profileResults.execPhaseToSegExecNumMap.end(); ++it) {

    auto segIdx   = it->first;
    auto segLabel = it->second.first;
    unsigned numExec = it->second.second;

    // ------------------------------------------------------------------
    // Fallback: some MG segments appear with an execution count of 0 in
    // execPhaseToSegExecNumMap.  When that happens we derive the real
    // count directly from the execution trace so that handshake scaling
    // uses the correct multiplier.
    // ------------------------------------------------------------------
    if (numExec == 0) {
      numExec = static_cast<unsigned>(
          std::count(profileResults.executedSegTrace.begin(),
                     profileResults.executedSegTrace.end(),
                     segLabel));
    }

    llvm::dbgs() << "[DEBUG] \tSegIndex: " << segIdx
                 << "; MG_Label: "      << segLabel
                 << ", Num Exec: "      << numExec << "\n";
    
    //------------------------------------------------------------------+
    // 2-A)  SEGMENT TYPE :  "S"  or  "T"
    //------------------------------------------------------------------+
    // TODO: Define a separate data storing structure for active nodes in the seg
    if (contains(segLabel,"S") || contains(segLabel,"T")) {
      llvm::dbgs() << "[DEBUG] \t[SEGMENT] " << segLabel << "\n";
      llvm::dbgs() << "[DEBUG] \t\t[Type] S or T\n";
      
      // Traverse all active nodes in the segment
      // for (const auto& nodeName: switchInfo.staticinfo.dataflowGraph->orderedNodeName) {
      auto graph= switchInfo.staticinfo.dataflowGraph;
      for(auto* node :   graph->nodePtrs){
        // Get the node
        // AdjNode *node = switchInfo.staticinfo.dataflowGraph->nodes[nodeName].get();
        
        // No BB-index filtering for S/T segments – every active node counts.
        // Calculate the number of valid switches
        unsigned numSucs = node->sucs.size();
        unsigned numValidSwitches = 2 * numSucs;

        // The node will be always ready
        unsigned numReadySwitches = 0;

        // Update the storing structure
        node->updateHandshakeChannelSwitching(numValidSwitches, numReadySwitches);

        // Update per-channel maps
        for (const auto &suc : node->sucs) {
          node->validSignal[suc] += 2;       // (create-if-absent already handled)
        }

        // NOW fold maps → scalar totals
        node->totalHandshakeSwitchingUpdate();
      }

      // Skip the rest of the steps
      continue;
    } else if (contains(segLabel,"E")) {
      llvm::dbgs() << "[DEBUG] \t[SEGMENT] " << segLabel << "\n";
      // Get the previous segment
      std::string prevSegLabel = std::prev(it)->second.first;
      llvm::dbgs() << "[DEBUG] \t\t[Prev Segment] " << prevSegLabel << "\n";

      // Traverse all active nodes in the segment
      for (const auto& nodeName: switchInfo.staticinfo.dataflowGraph->orderedNodeName) {
        // Get the node
        AdjNode *node = switchInfo.staticinfo.dataflowGraph->nodes[nodeName].get();
        
        // This node is in segment E
        auto segBBList = switchInfo.staticinfo.segToBBs[segLabel];
        if (!containsValue(segBBList, node->bbindex)) {
          continue;
        }

        // Check wheter the node is in previous section or not
        //* Assumption: Seg E will only be following MG ?
        auto prevSegBBList = switchInfo.staticinfo.segToBBs[prevSegLabel];
        if (containsValue(prevSegBBList, node->bbindex)) {
          // Get the node info in the previous segment
          AdjNode *prevNode = switchInfo.staticinfo.segToGraph[prevSegLabel]->nodes[nodeName].get();
          // Node in the previous segment
          unsigned numValidSwitches = prevNode->totalValidSwitching;
          unsigned numSucs = prevNode->sucs.size();

          // Always add the final 1→0 transition of the last token so that
          // each Valid line ends with the falling edge seen in the Python
          // reference implementation.
          if ((!contains(nodeName,"constant")) && (!contains(nodeName,"source"))) {
            numValidSwitches += (numSucs * 2 - numValidSwitches) / 2;
          }

          unsigned numReadySwitches = prevNode->totalReadySwitching;

          // Check whether this is a load unit
          if (contains(nodeName,"load")) {
            numValidSwitches *= 2;
            numReadySwitches *= 2;
          }

          // Update the perchannel information
          // Valid Channel
          for (const auto& suc: node->sucs) {
            unsigned selSucValidSwitching = 0;
            // Check whether this output is in the previous segment or not
            if (prevNode->validSignal.find(suc) != prevNode->validSignal.end()) {
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
          for (const auto & pre: node->pres) {
            unsigned selPreReadySwitching = 0;
            // Check whether this input is in the previous segment or not
            if (prevNode->readySignal.find(pre) != prevNode->readySignal.end()) {
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
          node->updateHandshakeChannelSwitching(numValidSwitches, numReadySwitches);
          node->totalHandshakeSwitchingUpdate();
        } else {
          // Get node type
          auto selNodeType = getNodeType(nodeName);
          unsigned numValidSwitches = 0;
          unsigned numReadySwitches = 0;
          if (contains(selNodeType,"cond_br")) {
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
          node->updateHandshakeChannelSwitching(numValidSwitches, numReadySwitches);
          node->totalHandshakeSwitchingUpdate();
        }
        
      }
    } else {
      llvm::dbgs() << "[DEBUG] \t[SEGMENT] " << segLabel << "\n";
      llvm::dbgs() << "[DEBUG] \t\t[Type] MG\n";
      // TODO: Update the logic here for the rest of the nodes
      
      // Traverse all active nodes in the segment
      for (const auto & nodeName: switchInfo.staticinfo.segToGraph[segLabel]->orderedNodeName) {
        // Get the node
        AdjNode *graphNode = switchInfo.staticinfo.dataflowGraph->nodes[nodeName].get();
        AdjNode *mgNode = switchInfo.staticinfo.segToGraph[segLabel]->nodes[nodeName].get();
        

        
        // Get the number of switching
        unsigned numValidSwitches = numExec * mgNode->totalValidSwitching;
        unsigned numReadySwitches = numExec * mgNode->totalReadySwitching;

        // Update the per-channel information
        // Valid Channel
        for (const auto& suc: graphNode->sucs) {
          unsigned selSucValidSwitching = 0;
          // Check whether this output is in the previous segment or not
          if (mgNode->validSignal.find(suc) != mgNode->validSignal.end()) {
            selSucValidSwitching = numExec * mgNode->validSignal[suc];
          } else {
            selSucValidSwitching = 0;
          }

          // Update the valid signal
          if (graphNode->validSignal.find(suc) != graphNode->validSignal.end()) {
            graphNode->validSignal[suc] += selSucValidSwitching;
          } else {
            graphNode->validSignal[suc] = selSucValidSwitching;
          }
        }

        // Ready Channel
        for (const auto & pre: graphNode->pres) {
          unsigned selPreReadySwitching = 0;
          // Check whether this input is in the previous segment or not
          if (mgNode->readySignal.find(pre) != mgNode->readySignal.end()) {
            selPreReadySwitching = numExec * mgNode->readySignal[pre];
          } else {
            selPreReadySwitching = 0;
          }

          // Update the ready signal
          if (graphNode->readySignal.find(pre) != graphNode->readySignal.end()) {
            graphNode->readySignal[pre] += selPreReadySwitching;
          } else {
            graphNode->readySignal[pre] = selPreReadySwitching;
          }
        }
        
        // Check whether this is a load unit
        if (contains(nodeName,"load")) {
          numValidSwitches *= 2;
          numReadySwitches *= 2;
        }

        // Update the storing structure
        graphNode->updateHandshakeChannelSwitching(numValidSwitches, numReadySwitches);
        graphNode->totalHandshakeSwitchingUpdate();
      }
    }
  } 

// -----------------------------------------------------------------------------------------------------------------
for (auto &[name, n] : switchInfo.staticinfo.dataflowGraph->nodes)
  n->totalHandshakeSwitchingUpdate();   // idempotent
}

//===----------------------------------------------------------------------===//
//
// Information Extraction
//
//===----------------------------------------------------------------------===//

void SwitchingEstimationPass::extractHandshakeOpNames(handshake::FuncOp& topFunc) {
  for (Operation& op : topFunc.getOps()) {
    // Get the handshake.name attribute
    // TODO: Make the retrieving of name attribute more natural
    std::string opName = op.getAttrOfType<StringAttr>("handshake.name").str();
    std::string opType = removeDigits(opName);

    if (NAME_SENSE_LIST.find(opType) != NAME_SENSE_LIST.end()) {
      switchInfo.staticinfo.funcOpNames.push_back(opName);
    }
  }
}

llvm::SmallVector<std::pair<unsigned, unsigned>> SwitchingEstimationPass::extractBackedges(llvm::SmallVector<experimental::ArchBB> archs) {
  llvm::SmallVector<std::pair<unsigned, unsigned>> backEdgeList;

  // Iterate through the archs list
  for (auto &selArch: archs) {
    if (selArch.isBackEdge) {
      backEdgeList.push_back(std::make_pair(selArch.srcBB, selArch.dstBB));
    }
  }

  return backEdgeList;
}

LogicalResult SwitchingEstimationPass::extractAllCFDFCs(mlir::ModuleOp& topModule) {
  for (handshake::FuncOp funcOp : topModule.getOps<handshake::FuncOp>()) {
    llvm::dbgs() << "[DEBUG] Entering Func: " << funcOp.getName() << "\n";

    // Get all ALU names in the selected FuncOp
    extractHandshakeOpNames(funcOp);

    // Get the CFDFC throughput and bb list from attributes
    llvm::dbgs() << "[DEBUG] [Step 0] Extracting CFDFCs\n";
    llvm::dbgs() << "[DEBUG] \tParsing CSV file: " << frequencies << "\n";

    llvm::SmallVector<experimental::ArchBB> archs; // Store all archs from the csv file

    // Read the CSV containing arch information (number of transitions between
    // pairs of basic blocks) from disk.
    if (failed(StdProfiler::readCSV(frequencies, archs))) {
      llvm::errs() << "[ERROR] Failed to read frequency profiling information from CSV\n";
      exit(-1);
    }

    // Get all Backedges
    switchInfo.staticinfo.backEdges = extractBackedges(archs);

    // Extract all needed attributes
    DictionaryAttr throughputAttr = getDialectAttr<handshake::CFDFCThroughputAttr>(funcOp).getThroughputMap();
    DictionaryAttr bbListAttr = getDialectAttr<handshake::CFDFCToBBListAttr>(funcOp).getCfdfcMap();

    // CONSTRUCT CFDFCS FROM THE BBLIST ATTRIBUTES
    for (auto &cfdfcBBListPair: bbListAttr) {
      llvm::SmallVector<experimental::ArchBB> tmpArchs;
      std::string cfdfcIndex = cfdfcBBListPair.getName().str();

      // Get the corresponding II
      float_t cfdfcII = 0;
      auto IIValueAttr = throughputAttr.get(cfdfcBBListPair.getName());
      if (IIValueAttr) {
        // Attribtue was successfully retrieved
        if (auto IIValue = IIValueAttr.dyn_cast<mlir::FloatAttr>()) {
          cfdfcII = 1.0 / IIValue.getValueAsDouble();

          switchInfo.staticinfo.cfdfcIIs[std::stoul(cfdfcIndex)] = cfdfcII;
          switchInfo.staticinfo.cfdfcThroughput[std::stoul(cfdfcIndex)] = IIValue.getValueAsDouble();
        }
      }

      // Debug
      llvm::dbgs() << "[DEBUG] \t[CFDFC] " << cfdfcIndex << "\n";

      mlir::ArrayAttr bbList = llvm::dyn_cast<mlir::ArrayAttr>(cfdfcBBListPair.getValue());
      auto iter = bbList.begin();
      unsigned prevBBId = (*iter++).cast<IntegerAttr>().getUInt();
      unsigned curBBId = prevBBId;
      unsigned startBBId = prevBBId;

      // If more than 1 BB in the CFDFC
      if (bbList.size() > 1) {
        for (; iter != bbList.end(); iter++) {
          curBBId = (*iter).cast<IntegerAttr>().getUInt();

          // Check whether this is a backedge
          if (std::find(switchInfo.staticinfo.backEdges.begin(), switchInfo.staticinfo.backEdges.end(), std::make_pair(prevBBId, curBBId)) != switchInfo.staticinfo.backEdges.end()) {
            tmpArchs.push_back(experimental::ArchBB(prevBBId, curBBId, 0, true));

            switchInfo.insertBE(prevBBId, curBBId, cfdfcIndex);

            // Debug
            llvm::dbgs() << "[DEBUG] \t[Backedge] Arch: (" << prevBBId << ", " << curBBId << ") Found in CFDFC: " << cfdfcIndex << "\n";
          } else {
            tmpArchs.push_back(experimental::ArchBB(prevBBId, curBBId, 0, false));

            // Debug
            llvm::dbgs() << "[DEBUG] \t[Edge] Arch: (" << prevBBId << ", " << curBBId << ") Found in CFDFC: " << cfdfcIndex << "\n";
          }

          prevBBId = curBBId;
        }

        // Check the loop back edge
        if (std::find(switchInfo.staticinfo.backEdges.begin(), switchInfo.staticinfo.backEdges.end(), std::make_pair(prevBBId, startBBId)) != switchInfo.staticinfo.backEdges.end()) {
          tmpArchs.push_back(experimental::ArchBB(prevBBId, startBBId, 0, true));

          switchInfo.insertBE(prevBBId, startBBId, cfdfcIndex);

          // Debug
          llvm::dbgs() << "[DEBUG] \t[Backedge] Arch: (" << prevBBId << ", " << startBBId << ") Found in CFDFC: " << cfdfcIndex << "\n";
        } else {
          tmpArchs.push_back(experimental::ArchBB(prevBBId, startBBId, 0, false));

          // Debug
          llvm::dbgs() << "[DEBUG] \t[Edge] Arch: (" << prevBBId << ", " << startBBId << ") Found in CFDFC: " << cfdfcIndex << "\n";
        }
      } else {
        if (std::find(switchInfo.staticinfo.backEdges.begin(), switchInfo.staticinfo.backEdges.end(), std::make_pair(prevBBId, curBBId)) != switchInfo.staticinfo.backEdges.end()) {
          tmpArchs.push_back(experimental::ArchBB(prevBBId, curBBId, 0, true));

          switchInfo.insertBE(prevBBId, curBBId, cfdfcIndex);

          // Debug
          llvm::dbgs() << "[DEBUG] \t[Backedge] Arch: (" << prevBBId << ", " << curBBId << ") Found in CFDFC: " << cfdfcIndex << "\n";
        } else {
          tmpArchs.push_back(experimental::ArchBB(prevBBId, curBBId, 0, false));

          // Debug
          llvm::dbgs() << "[DEBUG] \t[Edge] Arch: (" << prevBBId << ", " << curBBId << ") Found in CFDFC: " << cfdfcIndex << "\n";
        }
      }
      
      // Construct the archset
      buffer::ArchSet archSet;
      for (auto &arch: tmpArchs) {
        archSet.insert(&arch);
      }

      // Since we don't care about the execution number of each CFDFC, it's set to 0
      buffer::CFDFC tmpMG(funcOp, archSet, 0);
      switchInfo.staticinfo.cfdfcs.insert_or_assign(std::stoul(cfdfcIndex), tmpMG);

      // Insert to the segment map as well
      std::vector<unsigned> newVector(tmpMG.cycle.begin(), tmpMG.cycle.end());
      switchInfo.staticinfo.segToBBs[cfdfcIndex] = newVector;
    }
  }
  
  return success();
}


namespace dynamatic {
namespace experimental {
namespace switching {

// Return a unique pointer for the switching estimation pass
std::unique_ptr<dynamatic::DynamaticPass>
createSwitchingEstimation(StringRef dataTrace,  StringRef frequencies, StringRef timingModels) {
  return std::make_unique<SwitchingEstimationPass>(dataTrace, frequencies, timingModels);
}

} // namespace switching
} // namespace experimental
} // namespace dynamatic
