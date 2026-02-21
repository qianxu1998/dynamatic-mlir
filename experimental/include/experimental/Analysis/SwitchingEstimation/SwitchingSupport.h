//===- SwitchingSupport.h - Switching estimation -------------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares supporting data structures for the switching estimation
// pass
//
//===----------------------------------------------------------------------===//

#ifndef EXPERIMENTAL_ANALYSIS_SWITCHINGESTIMATION_SWITCHINGSUPPORT_H
#define EXPERIMENTAL_ANALYSIS_SWITCHINGESTIMATION_SWITCHINGSUPPORT_H

#include "dynamatic/Dialect/Handshake/HandshakeAttributes.h"
#include "dynamatic/Support/TimingModels.h"
#include "dynamatic/Transforms/BufferPlacement/CFDFC.h"
#include "experimental/Analysis/SwitchingEstimation/GraphModel.h"
#include "mlir/IR/Attributes.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/TypeSwitch.h"

#include "llvm/Support/Debug.h"
#include <algorithm>
#include <cctype>
#include <cmath>
#include <optional>
#include <regex>
#include <set>
#include <string>
#include <typeinfo>
#include <unordered_set>
#include <vector>

using namespace llvm;
using namespace mlir;
using namespace dynamatic;
using namespace dynamatic::handshake;

// Declarations
class AdjNode;
class AdjGraph;
class Path;
class DataBase;

// Per-segment partition of source nodes used by data-channel propagation.
// Example:
//   all     = {"addi_0", "mux_1", "control_merge_0"}
//   control = {"mux_1", "control_merge_0"}
//   data    = {"addi_0"}
struct SegmentDataSourceNodes {
  std::vector<std::string> all;
  std::vector<std::string> control;
  std::vector<std::string> data;
};

// Struct storing the list of mux and control merge nodes in each seg
struct SegmentControlNodes {
  std::vector<std::string> muxNodes;
  std::vector<std::string> controlMergeNodes;
};

struct NodeGlitchInfo {
  std::string srcNode;
  int steadyTime;
  bool buffered;
};

// Helper datatype for switching estimation. Aggregates all useful information
// for the switching estimation process
struct StaticInfo {
  //
  //  Internal Storing Variables
  //
  // All backedge BB pairs in the dataflow circuit
  SmallVector<std::pair<unsigned, unsigned>> backEdges;
  // Map from CFDFC index to the CFDFC info stroing class
  std::map<unsigned, CFDFC *> cfdfcInfoMap;
  // Map from Backedge pair to the list of CFDFC lable vector
  // i.e. {(1, 1) : [1]}
  std::map<std::pair<unsigned, unsigned>, std::vector<unsigned>>
      backEdgeToCFDFC;
  // Map from Transaction Segment label (including cfdfc index) to the list of
  // BBs
  StringMap<std::vector<unsigned>> segToBBs;
  // Map from Transaction Segment label to the successing MG label
  // Map from Segment Label (CFDFC and temporal transaction sections that are
  // not MGs)
  StringMap<std::string> transToSucMGMap;
  // Map storing the subgraph of different segments in the dataflow circuit
  StringMap<std::shared_ptr<AdjGraph>> segToGraph;
  // Set of nodes that are expected to appear during scf profiling
  // Used to retrieve the date from SCF level profiling
  std::vector<std::string> traceOpNames;
  // dataflow graph
  // Below has to be a shared pointer, otherwise need to override the
  // clonePass() implementation in MLIR
  std::shared_ptr<AdjGraph> dataflowGraph;
  // Map from CFDFC index to the corresponding II values
  llvm::DenseMap<unsigned, float_t> cfdfcIIs;

  // Map from CFDFC index to the corresponding throughput
  llvm::DenseMap<unsigned, double_t> cfdfcThroughput;
};

// Runtime storage for the data-channel estimation pipeline.
struct DataInfo {
  // Segment label -> data/control source-node buckets.
  StringMap<SegmentDataSourceNodes> segmentToDataSourceNodes;

  // Segment label -> node name -> glitch descriptors.
  StringMap<std::map<std::string, std::vector<NodeGlitchInfo>>> glitches;
  // Segment label -> topologically ordered ALU nodes.
  StringMap<std::vector<std::string>> segmentToOrderedAluNodes;
  // Segment label -> mux/control_merge node lists.
  StringMap<SegmentControlNodes> segmentControlNodes;
  // Segment label -> first execution index in the profiling trace.
  StringMap<unsigned> segmentToFirstExecutionIter;
  // Node name -> persistent propagation state.
  StringMap<std::shared_ptr<DataBase>> nodeToDataState;
  // (preBB, curBB) -> [(control_merge_node, selected_input_port)].
  std::map<std::pair<unsigned, unsigned>,
           std::vector<std::pair<std::string, int>>>
      bbPairToControlMergeOutputs;
  // Segment label -> topologically ordered data-source nodes.
  StringMap<std::vector<std::string>> segmentToOrderedDataSourceNodes;
  // Global topological order over mux nodes.
  std::vector<std::string> orderedMuxNodes;
  // Full execution-segment trace indexed by global iteration.
  std::vector<std::string> executedSegmentTrace;
  // Segment label -> all global iteration indices where that segment executes.
  StringMap<std::vector<unsigned>> segmentToExecutionIndices;
};

struct SwitchingInfo {
  //  Variables for static information about the dataflow circuit
  StaticInfo staticInfo;
  DataInfo dataInfo;
  //  HandshakeInfo hs; not sure wherher to put it
  // This function insert (backedge pair, mgLabel) to the backEdgeToCFDFC
  void insertBE(unsigned srcBB, unsigned dstBB, StringRef mgLabel);
  // Map from segLabel to the corresponding backedge pair
  StringMap<std::pair<unsigned, unsigned>> segToBackedgePairMap; // TODO delete
  // Map from segment label to the vector of invalid backedges
  StringMap<std::vector<std::pair<std::string, std::string>>>
      segInvalidBackedgesMap;

  //
  //  Variables for handshake channel switching calculation
  //
  // Vector storing the list of load units influenced by transparent buffers in
  // each MG, with ascending order for different MG
  std::vector<std::vector<std::string>> mgInfluencedLoadUnits;
};

//===----------------------------------------------------------------------===//
//
// Unique Sets for the parsing process
//
//===----------------------------------------------------------------------===//
// Define the constant name sensitive list used for parsing the profiling
// results
// TODO: Add support for more node types
const std::set<std::string> NAME_SENSE_LIST = {
    "muli",     "addi",     "subi",      "ori",  "andi",  "cmpi", "mc_load",
    "mc_store", "lsq_load", "lsq_store", "load", "store", "shli", "shrsi"};

// Define the set of node types that potentially have glitches
const std::unordered_set<std::string> GLITCH_NODE = {
    "addi",  "subi",  "muli", "addf", "subf", "mulf",
    "divui", "divsi", "divf", "ori",  "andi"};

// Define all join type like nodes
static const std::unordered_set<std::string> JOIN_NODE = {
    "cmpi",  "addi",  "subi", "muli", "shli",
    "shrsi", "shrui", "ori",  "andi", "divui"};

//===----------------------------------------------------------------------===//
//
// Class for storing data-channel values
//
//===----------------------------------------------------------------------===//
// A small sruct for storing a single "(value, iterIndex)" pair
struct IterationValue {
  int value;
  unsigned iterIndex;
};

// Per-segment successor metadata used by propagation and glitch handling.
struct SegmentSuccessorInfo {
  // Successors reached on the original (non-glitch) value path.
  std::vector<std::string> original;
  // Successors reached by glitch propagation.
  std::vector<std::string> glitch;
  // The following two vectors are specific to control_merge handling.
  std::vector<std::string> control;
  std::vector<std::string> data;
  // Successor node -> effective propagated data width.
  std::map<std::string, unsigned> dataWidthMap;
};

// A helper structure to return the longest path result.
// This is used during glitch node selection
// TODO: Remove this during refinement
struct LongestPathResult {
  std::string selStartNode;
  unsigned maxLatency;
  std::string lastSecondBuffer;
};

// Class used to store information for finished nodes needed for data
// propagation The instances of this class shall be stored globally, as this
// will be used for the update of all segments
class DataBase {
public:
  DataBase(const std::string &node);

  virtual ~DataBase() = default;

  void printDetail();

  //
  //  Internal Storing Variables
  //
  // Node name
  std::string nodeName;

  // Key: iteration_index -> single (value, iteration) pair
  // TODO: remove the redudant index information
  std::map<unsigned, IterationValue> originalDataOut;

  // Map from iter_index to value Vec with glitch values
  std::map<unsigned, std::vector<int>> glitchDataOutByIter;

  // Map from segindex to succeeding node storing structure
  std::map<std::string, SegmentSuccessorInfo> segmentSuccessorInfoMap;

  // For control merge node, we need to store the controlDataOut info as well
  std::map<unsigned, IterationValue> controlDataOut;

  // Support LLVM node casting
  enum class NodeKind { DataBaseKind, CMergeDataKind };

  // By default, an DataBase has kind = DataBaseKind
  virtual NodeKind getKind() const { return NodeKind::DataBaseKind; }

  // "classof " used by llvm casting
  static bool classof(const DataBase *node) {
    return node->getKind() == NodeKind::DataBaseKind;
  }

  //
  unsigned lastUpdateIndex;
  bool skipControlCal = false;
  // Last valid seg label, used for glitch calculation
  std::string lastValidSeg = "";
};

class CMergeData : public DataBase {
public:
  CMergeData(const std::string &nodeName) : DataBase(nodeName) {}

  void printDetail();

  // This function gets the desired control dataout from the control_dataout
  // dict During MG transitions, the control dataout value doesn't exist, we
  // directly give it a 0
  int getControlOutput(unsigned selIter);

  // LLVM casting support
  NodeKind getKind() const override { return NodeKind::CMergeDataKind; }

  // "classof" needed for dyn_cast
  static bool classof(const DataBase *node) {
    return node->getKind() == NodeKind::CMergeDataKind;
  }

  //
  //  Internal Storing Variables
  //
  // ControlMerge has one extra output channel: control dataout.
  std::map<unsigned, IterationValue> controlDataOut;

  // Per CFDFC storing structure
  // Format:
  // {"mg_label": {"control": control_dataout_node_list; "data":
  // data_channel_node_list}}
  std::map<std::string, SegmentSuccessorInfo> cmergeSegmentSuccessorInfoMap;

  // Vector to store the control glitch value
  std::vector<int> controlGlitchValues;
};

//===----------------------------------------------------------------------===//
//
// Helper Functions
//
//===----------------------------------------------------------------------===//
// Get the operation name
std::string getHandshakeNodeName(mlir::Value &selRes);

// The following function prints the backEdgeToCFDFCMap
void printBEToCFDFCMap(const std::map<std::pair<unsigned, unsigned>,
                                      std::vector<unsigned>> &selMap);

// This function prints the Segment ID to BBlist map
void printSegToBBListMap(
    const std::map<std::string, mlir::SetVector<unsigned>> &selMap);

// Helper function: extracts the initial alphabetic portion from a node name.
std::string getNodeType(const std::string &nodeName);

// This function prints all values in a vector
template <typename T>
inline void printVector(const T &selVec) {
  int counter = 0;

  dbgs() << "[DEBUG] Vector Contents: ";

  for (auto &selVal : selVec) {
    dbgs() << "[" << counter << "] : " << selVal << "; ";

    counter++;
  }

  dbgs() << ";\n";
}

// This function remove the digits in the given string and keep the rest
std::string removeDigits(const std::string &inStr);

// Get unsigned number from a float
unsigned getUnsigned(float_t inputValue);

// This function prints the node succ list info
void printSegmentSuccessorInfo(const SegmentSuccessorInfo &info);

// Function to print a vector of strings (mainStack)
void printMainStack(const std::vector<std::string> &mainStack);

// Function to print a vector of vector of strings (adjStack)
void printAdjStack(const std::vector<std::vector<std::string>> &adjStack);

#endif // EXPERIMENTAL_ANALYSIS_SWITCHINGESTIMATION_SWITCHINGSUPPORT_H
