//===- SwitchingEstimation.cpp - Estimate Swithicng Activities ------*- C++ -*-===//
//
// Implements the switching estimation pass for all untis in the generated
// dataflow circuit
//
//===----------------------------------------------------------------------===//

#include "experimental/Transforms/Switching/SwitchingEstimation.h"
#include "experimental/Transforms/Switching/SwitchingSupport.h"
#include "experimental/Transforms/Switching/NodeId.h"
#include "experimental/Transforms/Switching/NameTable.h"

#include "experimental/Transforms/Switching/ProfilingAnalyzer.h"
#include "experimental/Transforms/Switching/DataChannelCal.h"
#include "experimental/Transforms/Switching/HandShakeChannelCal.h"
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
                          StringRef bbList,
                          StringRef frequencies,
                          StringRef timingModels) {
    this->dataTrace = dataTrace.str();
    this->bbList = bbList.str();
    this->frequencies = frequencies.str();
    this->timingModels = timingModels.str();
  }

  // Main Interface
  void runDynamaticPass() override;

  // 
  //  Define global storing structure
  //
  SwitchingInfo switchInfo;

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

  // 
  //  Handshake Channel Switching Calculation
  //
  void calHSChannelSwitchingSteady(mlir::ModuleOp& topModule, SCFProfilingResult &profileResults);

  // This function calculates the handshake channel switching for the entire circuit simulation
  void countHSChannelSwitchingOverall(mlir::ModuleOp& topModule, SCFProfilingResult &profileResults);
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
  llvm::dbgs() << "[DEBUG] \tBBList Log file: " << bbList << "\n";
  llvm::dbgs() << "[DEBUG] \tData Profiling Log file: " << dataTrace << "\n";
  SCFProfilingResult profilingResults(dataTrace, bbList, switchInfo);// works at scf level, converts to HS

  // Step 2: Build Adjacency graph for each CFDFC
  std::vector<std::pair<std::string, std::string>> allBackedges;
  llvm::dbgs() << "[DEBUG] [Step 2] Build Adjacency Graph for Each Segment\n";
  
  for (const auto& [mgIndex, mgInstance]: switchInfo.cfdfcs) {
    llvm::dbgs() << "[DEBUG] \tMG : " << mgIndex << "\n";
    // AdjGraph a(mgInstance, timingDB, switchInfo.cfdfcIIs[mgIndex], mgIndex);


    auto adj = std::make_shared<AdjGraph>(mgInstance, timingDB, switchInfo.cfdfcIIs[mgIndex], mgIndex);
    // AdjGraph tmpAdjGraph(mgInstance, timingDB, switchInfo.cfdfcIIs[mgIndex], mgIndex);
    switchInfo.segToAdjGraphMap.insert_or_assign(std::to_string(mgIndex), adj);
  
    // Update the backedge list
    for (const auto& selPair : adj->backedges) {
      allBackedges.push_back(selPair);
    }
  }
// AdjGraph>(mgInstance, timingDB, switchInfo.cfdfcIIs[mgIndex], mgIndex)
  // Step 3: Build the graph for the entire dataflow graph
  llvm::dbgs() << "[DEBUG] [STEP 3] Construct the Adj Graph for the entire DFG\n";
  /// Step 3.1: First store the information of the backedges in the circuit
  for (const auto& [selSegLabel, segBBList] : switchInfo.segToBBListMap) {
    if (selSegLabel.find("S") != std::string::npos) {
      switchInfo.segInvalidBackedgesMap[selSegLabel] = allBackedges;
    } else if (selSegLabel.find("E") != std::string::npos) {
      switchInfo.segInvalidBackedgesMap[selSegLabel] = allBackedges;
    } else if (selSegLabel.find("T") != std::string::npos) {
      auto sucMg = switchInfo.transToSucMGMap[selSegLabel];
      std::vector<std::pair<std::string, std::string>> tmpInvalidBackedges; 
      for (const auto& [selCFDFCIndex, selGraph]: switchInfo.segToAdjGraphMap) {
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
    switchInfo.dataflowGraph = std::make_shared<AdjGraph>(timingDB, switchInfo.cfdfcIIs[0], funcOp, allBackedges);
  }

  llvm::dbgs() << "[DEBUG] [STEP 4] Determining the Global start time and shifting for each MG\n";
  for (auto& [mgIndex, selGraph]: switchInfo.segToAdjGraphMap) {
    // Step 4.1: Determining the latest start time for each node cfdfc
    selGraph->obtainNodeGlobalOrder();

    //! Testing
    // llvm::dbgs() << "[DEBUG] \t\t Global order calculated\n"; 

    // Step 4.2: Check the shifting between different start node within a graph
    selGraph->analyzeStartNodeShifting();
  }

  // Step 5: Calculate switches in data channel 
  llvm::dbgs() << "[DEBUG] [STEP 5] Calculate Data Channel Switching\n";
  calDataChannelSwitching(topModule, profilingResults);

  // Step 6: Calculate switches in handshake channels
  llvm::dbgs() << "[DEBUG] [STEP 6] Calculate Handshake Channel Switching\n";
  calHSChannelSwitchingSteady(topModule, profilingResults);

  // Step 7: Calculate switches in handshake channels for the entire simualtion
  llvm::dbgs() << "[DEBUG] [STEP 7] Calculate Handshake Channel Switching for the entire simulation\n";
  countHSChannelSwitchingOverall(topModule, profilingResults);
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
    if (switchInfo.segToExecutedIter.find(segLabel) == switchInfo.segToExecutedIter.end()) {
      switchInfo.segToExecutedIter[segLabel] = i;
    }
  }

  // Step 1: Get the iteration index for the frist execution of each segment
  llvm::dbgs() << "[DEBUG]  [SS1] Get the BB Pair to Control Merge Output Map\n";
  constructBBPairToCMResMap(switchInfo);

  //! Testing
  for (const auto& [pair1, cmVec]: switchInfo.bbPairToCMResultMap) {
    llvm::dbgs() << "[DEBUG] \t(" << pair1.first << ", " << pair1.second << ") : \n";
    for (auto selPair: cmVec) {
      llvm::dbgs() << "[DEBUG] \t\t[" << selPair.first << " " << selPair.second << "]\n";
    }
  }

  // Step 2: Construct the list of all data source nodes from scf-level profiling 
  llvm::dbgs() << "[DEBUG]  [SS2] Get all the database nodes in each segments\n";
  getDataBaseNodes(switchInfo, profileResults);
  //! Testing
  for (auto& [segLabel, selDB]: switchInfo.segToDataBaseVecMap) {
    llvm::dbgs() << "[DEBUG] \t[SEGMENT] " << segLabel << "\n";
    printDataBaseNodesTriple(selDB);
  }

  // Step 3: Contruct the data source node info of mux, condbr and mem node
  llvm::dbgs() << "[DEBUG]  [SS3] Construct the data source node storing structure for different nodes in the dataflow graph\n";
  // TODO: Sometimes the source node of a mux node is from the block argumnet, need to add new nodes in the grpah.
  switchInfo.dataflowGraph->buildMuxSrcMap();
  switchInfo.dataflowGraph->buildCondandStoreSrcMap();

  //! Testing
  printMuxToSrcNodeMap(switchInfo.dataflowGraph->muxToSrcNodeMap);
  printSrcNodeToMuxMap(switchInfo.dataflowGraph->srcNodeToMuxMap);
  llvm::dbgs() << "[DEBUG] \tcondBr Node to control src map: \n";
  for (const auto& [cbrNode, controlSrc]: switchInfo.dataflowGraph->condBrToConSrcMap) {
    llvm::dbgs() << "[DEBUG] \t\t(" << cbrNode << ", " << controlSrc << ")\n";
  }

  // Step 4: Update value for all data base nodes
  llvm::dbgs() << "[DEBUG]  [SS4] Update the value for data base nodes\n";
  dataChannelBaseNodesValueUpdate(switchInfo, profileResults);

  // Step 5: Build succeeding node list for data base nodes in different segments
  llvm::dbgs() << "[DEBUG]  [SS5] Build succeeding node list\n";
  conSegSuccNodesList(switchInfo, profileResults);

  // Step 6: Get all glitching base node in each MG
  llvm::dbgs() << "[DEBUG]  [SS6] Find all glitching nodes\n";
  dataGlitchNodeSearch(switchInfo, profileResults);

  // Step 7: Update all glitching value for data base nodes in the dataflow circuit
  llvm::dbgs() << "[DEBUG]  [SS7] Calculate all glitching values\n";
  dataBaseNodeGlitchUpdate(switchInfo, profileResults, false);

  // Step 8: Propagate all the data base value
  llvm::dbgs() << "[DEBUG]  [SS8] Final data channel value updates\n";
  dfgDataChannelPropagate(switchInfo, profileResults, false);
}

//===----------------------------------------------------------------------===//
//
// Handshake Channel Switching
//
//===----------------------------------------------------------------------===//
void SwitchingEstimationPass::calHSChannelSwitchingSteady(mlir::ModuleOp& topModule, SCFProfilingResult &profileResults) {
  // For each MG, we do the following two steps
  //  Step 1: Update buffer information
  //  Step 2: Calculate the steady state handhshake channel switching

  // Step 1
  for (unsigned i = 0; i < switchInfo.cfdfcThroughput.size(); i++) {
    extractBufferInfo(switchInfo, std::to_string(i), true);
  }

  // Step 2
  for (unsigned i = 0; i < switchInfo.cfdfcThroughput.size(); i++) {
    mgHandshakeSwitchingCounting(switchInfo, std::to_string(i), true);
  }
}

void SwitchingEstimationPass::countHSChannelSwitchingOverall(mlir::ModuleOp& topModule, SCFProfilingResult &profileResults) {
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
  for (auto it = profileResults.execPhaseToSegExecNumMap.begin(); it != profileResults.execPhaseToSegExecNumMap.end(); ++it) {
    auto segIdx = it->first;
    auto segLabel = it->second.first;
    auto numExec = it->second.second;
    llvm::dbgs() << "[DEBUG] \tSegIndex: " << segIdx << "; MG_Label: " << segLabel << ", Num Exec: " << numExec << "\n";
    
    //------------------------------------------------------------------+
    // 2-A)  SEGMENT TYPE :  “S”  or  “T”
    //------------------------------------------------------------------+
    // TODO: Define a separate data storing structure for active nodes in the seg
    if (segLabel.find("S") != std::string::npos || segLabel.find("T") != std::string::npos) {
      llvm::dbgs() << "[DEBUG] \t[SEGMENT] " << segLabel << "\n";
      llvm::dbgs() << "[DEBUG] \t\t[Type] S or T\n";

      // Traverse all active nodes in the segment
      for (const auto& nodeName: switchInfo.dataflowGraph->orderedNodeName) {
        // Get the node
        AdjNode *node = switchInfo.dataflowGraph->nodes[nodeName].get();
        
        auto segBBList = switchInfo.segToBBListMap[segLabel];
        if (std::find(segBBList.begin(), segBBList.end(), node->bbindex) == segBBList.end()) {
          continue;
        }
        // Calculate the number of valid switches
        unsigned numSucs = node->sucs.size();
        unsigned numValidSwitches = 2 * numSucs;

        // The node will be always ready
        unsigned numReadySwitches = 0;

        // Update the storing structure
        node->updateHandshakeChannelSwitching(numValidSwitches, numReadySwitches);

        // Update per channel switching information
        for (const auto& suc: node->sucs) {
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
      for (const auto& nodeName: switchInfo.dataflowGraph->orderedNodeName) {
        // Get the node
        AdjNode *node = switchInfo.dataflowGraph->nodes[nodeName].get();
        
        // This node is in segment E
        auto segBBList = switchInfo.segToBBListMap[segLabel];
        if (std::find(segBBList.begin(), segBBList.end(), node->bbindex) == segBBList.end()) {
          continue;
        }

        // Check wheter the node is in previous section or not
        //* Assumption: Seg E will only be following MG ?
        auto prevSegBBList = switchInfo.segToBBListMap[prevSegLabel];
        if (std::find(prevSegBBList.begin(), prevSegBBList.end(), node->bbindex) != prevSegBBList.end()) {
          // Get the node info in the previous segment
          AdjNode *prevNode = switchInfo.segToAdjGraphMap[prevSegLabel]->nodes[nodeName].get();
          // Node in the previous segment
          unsigned numValidSwitches = prevNode->totalValidSwitching;
          unsigned numSucs = prevNode->sucs.size();

          if (numValidSwitches != (numSucs * 2)) {
            if (nodeName.find("constant") == std::string::npos && nodeName.find("source") == std::string::npos) {
              // This is the ending segment transition 1 -> 0
              numValidSwitches += (numSucs * 2 - numValidSwitches) / 2;
            }
          }

          unsigned numReadySwitches = prevNode->totalReadySwitching;

          // Check whether this is a load unit
          if (nodeName.find("load") != std::string::npos) {
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
          node->updateHandshakeChannelSwitching(numValidSwitches, numReadySwitches);
        }
        
      }
    } else {
      llvm::dbgs() << "[DEBUG] \t[SEGMENT] " << segLabel << "\n";
      llvm::dbgs() << "[DEBUG] \t\t[Type] MG\n";
      // TODO: Update the logic here for the rest of the nodes
      
      // Traverse all active nodes in the segment
      for (const auto & nodeName: switchInfo.segToAdjGraphMap[segLabel]->orderedNodeName) {
        // Get the node
        AdjNode *graphNode = switchInfo.dataflowGraph->nodes[nodeName].get();
        AdjNode *mgNode = switchInfo.segToAdjGraphMap[segLabel]->nodes[nodeName].get();
        
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
        if (nodeName.find("load") != std::string::npos) {
          numValidSwitches *= 2;
          numReadySwitches *= 2;
        }

        // Update the storing structure
        graphNode->updateHandshakeChannelSwitching(numValidSwitches, numReadySwitches);
      }
    }
  } 
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
      switchInfo.funcOpNames.push_back(opName);
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
    switchInfo.backEdges = extractBackedges(archs);

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

          switchInfo.cfdfcIIs[std::stoul(cfdfcIndex)] = cfdfcII;
          switchInfo.cfdfcThroughput[std::stoul(cfdfcIndex)] = IIValue.getValueAsDouble();
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
          if (std::find(switchInfo.backEdges.begin(), switchInfo.backEdges.end(), std::make_pair(prevBBId, curBBId)) != switchInfo.backEdges.end()) {
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
        if (std::find(switchInfo.backEdges.begin(), switchInfo.backEdges.end(), std::make_pair(prevBBId, startBBId)) != switchInfo.backEdges.end()) {
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
        if (std::find(switchInfo.backEdges.begin(), switchInfo.backEdges.end(), std::make_pair(prevBBId, curBBId)) != switchInfo.backEdges.end()) {
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
      switchInfo.cfdfcs.insert_or_assign(std::stoul(cfdfcIndex), tmpMG);

      // Insert to the segment map as well
      std::vector<unsigned> newVector(tmpMG.cycle.begin(), tmpMG.cycle.end());
      switchInfo.segToBBListMap[cfdfcIndex] = newVector;
    }
  }
  
  return success();
}


namespace dynamatic {
namespace experimental {
namespace switching {

// Return a unique pointer for the switching estimation pass
std::unique_ptr<dynamatic::DynamaticPass>
createSwitchingEstimation(StringRef dataTrace, StringRef bbList, StringRef frequencies, StringRef timingModels) {
  return std::make_unique<SwitchingEstimationPass>(dataTrace, bbList, frequencies, timingModels);
}

} // namespace switching
} // namespace experimental
} // namespace dynamatic
