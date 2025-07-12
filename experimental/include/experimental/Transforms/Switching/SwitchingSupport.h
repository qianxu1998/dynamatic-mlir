//===- SwitchingSupport.h - Switching Estimation -----*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares supporting data structures for the swithcing estimation pass
//
//===----------------------------------------------------------------------===//

#ifndef EXPERIMENTAL_TRANSFORMS_SWITCHING_SUPPORT_H
#define EXPERIMENTAL_TRANSFORMS_SWITCHING_SUPPORT_H

#include "dynamatic/Transforms/BufferPlacement/CFDFC.h"
#include "mlir/IR/Attributes.h"
#include "dynamatic/Dialect/Handshake/HandshakeAttributes.h"
#include "experimental/Support/StdProfiler.h"
#include "dynamatic/Support/TimingModels.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringMap.h"

#include "llvm/Support/Debug.h"
#include "experimental/Transforms/Switching/NodeInfo.h"
#include <unordered_set>

#include <algorithm>
#include <cmath>
#include <vector>
#include <set>
#include <regex>
#include <string>
#include <cctype>
#include <typeinfo>
#include <optional>



using namespace mlir;
using namespace dynamatic;
using namespace dynamatic::handshake;

// Declaration
class AdjNode;
class AdjGraph;
class Path;
class DataBase;

// Struct used to store all the data source nodes in differernt segments in the dfg
// We have this kind of definiton as we obtain the profiling results from scf level
// If we have handshake level simulator one day, we can get rid of the entire
// data channel estimation process.
struct DataBaseNodesTriple {
  std::vector<std::string> all;
  std::vector<std::string> control;
  std::vector<std::string> data;
};

// Struct storing the list of mux and control merge nodes in each seg
struct muxCMNodesList {
  std::vector<std::string> muxNodeList;
  std::vector<std::string> cmNodeList;
};

// New struct for glitch info.
struct NodeGlitchInfo {
  std::string srcNode;
  int steadyTime;
  bool buffered;
};
enum class SegmentKind:uint8_t{
  Prologue,SteadyState,Epilogue,Transition
};

// Helper datatype for switching estimation. Aggregates all useful information
// for the swithicng estimation process


struct StaticInfo{
    // 
    //  Internal Storing Variables
    //
      // All backedge BB pairs in the dataflow circuit
    llvm::SmallVector<std::pair<unsigned, unsigned>> backEdges;
    // Map from Backedge pair to the list of CFDFC lable vector
    // i.e. {(1, 1) : [1]}
    std::map<std::pair<unsigned, unsigned>, std::vector<unsigned>> backEdgeToCFDFC;
    std::map<std::pair<unsigned, unsigned>, std::vector<unsigned>> backEdgeToCFDFCfast;
    llvm::StringMap<std::vector<unsigned>> segToBBs;
    // Map from Transaction Segment label to the successing MG label
      // Map from Segment Label (CFDFC and temporal transaction sections that are not MGs)
      llvm::StringMap< std::string> transToSucMGMap;
    // Map storing the subgraph of different segments in the dataflow circuit
    llvm::StringMap< std::shared_ptr<AdjGraph>> segToGraph;
    // Set of names of the ALUs in the Handshake level mlir file
    // Used to retrieve the date from SCF level profiling
    std::vector<std::string> funcOpNames;
      // dataflow graph
    // Below has to be a shared pointer, otherwise need to override the clonePass() implementation in MLIR
    std::shared_ptr<AdjGraph> dataflowGraph;
    // Map from CFDFC index to the corresponding II values
    llvm::DenseMap<unsigned, float_t> cfdfcIIs;
    
    // Map from CFDFC index to the corresponding throughput
    llvm::DenseMap<unsigned, double_t> cfdfcThroughput;
    // Map from CFDFC index to the CFDFC info stroing class
    // Only contain the number of edges, backedges etc.
    // No info about unit delay, node neighbors etc.
    std::map<unsigned, buffer::CFDFC> cfdfcs;
};
// Helper struct for switching estimation handshake pass
struct HandshakeInfo{
  // Vector storing the list of load units influenced by transparent buffers in each MG,
  // with ascending order for different MG
  std::vector<std::vector<std::string>> mgInfluencedLoadUnits;
};
// helper struct to store relevant info for data channel pass
struct DataInfo{
  // Map from segment index to dataBaseNode struct
  llvm::StringMap< DataBaseNodesTriple> segToDataBaseVec;

  llvm::StringMap< std::map<std::string, std::vector<NodeGlitchInfo>>> glitches;
  // Map from seg label to ordered ALU nodes
  llvm::StringMap<std::vector<std::string>> segToOrderedALUNodes;
  // Map from seg label to ordered mux and control merge node list
  llvm::StringMap< muxCMNodesList> controlNodes;
  // Map stroing the first iteration index that the seg is executed
  llvm::StringMap<unsigned> firstExecutedIter;
    // Map from node name to DataBase class
    llvm::StringMap<std::shared_ptr<DataBase>> dfgBaseNodeValue;
  // Map from pair of BB sequence to the corresponding control_merge output
  // Format: {(preBB, curBB) : [(control_merge_node, output_value)]}
  std::map<std::pair<unsigned, unsigned>, std::vector<std::pair<std::string, int>>> bbPairToCtrlMerge;
  // Map from seg label to ordered Data base nodes
  llvm::StringMap<std::vector<std::string>> segToOrderedDataBaseNodes;
  // Mux Nodes topologically ordered
  std::vector<std::string> orderedMuxNodes;
};

struct SwitchingInfo {
   StaticInfo    staticinfo;
   DataInfo data;
  //  HandshakeInfo hs; not sure wherher to put it
    // This function insert (backedge pair, mgLabel) to the backEdgeToCFDFC
  void insertBE(unsigned srcBB, unsigned dstBB, StringRef mgLabel);
  // Map from segLabel to the corresponding backedge pair
  llvm::StringMap< std::pair<unsigned, unsigned>> segToBackedgePairMap;// TODO delete
  // Map from segment label to the vector of invalid backedges
  llvm::StringMap< std::vector<std::pair<std::string, std::string>>> segInvalidBackedgesMap;
  // 
  //  Variables for handshake channel switching calculation
  //
};





// Class used to construct the per segment (MG & one-time execution segment)
// The class defined following is purely for path fining and preserve the structure
// of each subgraph.
/*
  Base Class used to store information related to node in the dataflow circuit,
        which will be used to calculate the switching activity in steady state.
            - Handshake channels
            - Data channels
        
        ! Handshake Signal Calculation:
            All child classes of this base class shall have it's own implementation of
                - cal_valid_switching
                - cal_ready_switching
                - cal_valid_set
                - cal_ready_set
        
            For the above functions, we have following assumptions:
                - Num switches must be >= 0.
                - For num switches, we use -1 to represent invalid value
                - For active set, we use None to represesnt invalid value
        
        ! Data Channel Calculation:
            For each node, we only store output data to its successors, like a directed chain list.
            Two possible situations for a given node:
                - It has only one (value, #iter) pair, like constant node. As the value never change
                - It has multiple (value, #iter) pair
            Thus, during dataout calculation, we need to check the len of the input data
            *If the node has multiple input data only two situations: 
                1) lengths they match with each other 
                2) one input has only 1 pair and the other has multiple.
            For the second case, the outputdata will have list length equal to the longer input data
*/




//===----------------------------------------------------------------------===//
//
// Path class, used to store additional information in the found path
//
//===----------------------------------------------------------------------===//
class Path {
public:
  // Constructors
  Path() = default;
  Path(const std::vector<std::string> &nodes) : nodeList(nodes) {}

  // Backedge management
  void add_backedge(const std::pair<std::string, std::string> &be) {
    contain_backedge = true;
    backedges.push_back(be);
  }

  // Latency
  void set_latency(unsigned lat) { latency = lat; }

  // Printing
  void printDetail() {
    llvm::dbgs() << "[DEBUG] \t[Path] \n[DEBUG] \t\t[ ";
    for (const auto& selNode: nodeList) {
      llvm::dbgs() << selNode << ", ";
    }
    llvm::dbgs() << "]\n";

    //
    llvm::dbgs() << "[DEBUG] \t\tBackedges: ";
    for (const auto& selPair: backedges) {
      llvm::dbgs() << "( " << selPair.first << ", " << selPair.second << " ); ";
    }
    llvm::dbgs() << "\n";
    
    //
    llvm::dbgs() << "[DEBUG] \t\tPath Latency: " << latency << "\n";
  }

  std::vector<std::string> nodeList;
  std::vector<std::pair<std::string, std::string>> backedges;
  bool contain_backedge = false;
  unsigned latency = 0;
};

//===----------------------------------------------------------------------===//
//
// Unique Sets for the parsing process
//
//===----------------------------------------------------------------------===//
// Define the constant name sensitive list used for parsing the profiling results
// As the name of the same operation in scf level IR and the final handshake IR
// is different, we need to map the scf level op to the handshake mlir file.
// TODO: Add support for more node types
const std::set<std::string> NAME_SENSE_LIST = {
  "muli",
  "addi",
  "subi",
  "ori",
  "andi",
  "cmpi",
  "mc_load",
  "mc_store",
  "lsq_load",
  "lsq_store",
  "load",
  "store",
  "shli",
  "shrsi"
};

// Define the set of node types that potentially have glitches
const std::unordered_set<std::string> GLITCH_NODE = {
  "addi", "subi", "muli", "addf", "subf",
  "mulf", "divui", "divsi", "divf", "ori", "andi"
};

// Define all join type like nodes
static const std::unordered_set<std::string> JOIN_NODE = {
  "cmpi",
  "addi",
  "subi",
  "muli",
  "shli",
  "shrsi",
  "shrui",
  "ori",
  "andi",
  "divui"
};

//===----------------------------------------------------------------------===//
//
// Class for storing data channel values
//
//===----------------------------------------------------------------------===//
// A small sruct for storing a single "(value, iterIndex)" pair
struct ValueIter {
  int value;
  unsigned iterIndex;
};

// Struct storing the list of nodes used for data value updates
struct MgNodeInfo {
  // "original" => vector of strings
  std::vector<std::string> original;
  // "glitch" => vector of strings
  std::vector<std::string> glitch;
  // Following two vectors are specific for control_merge nodes
  std::vector<std::string> control;
  std::vector<std::string> data;
  // "datawidth" => map from string to unsigned
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

// Class used to store information for the finished node that's needed for data propagation
// The instances of this class shall be stored globally, as this will be used for the update of all segments
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
  std::map<unsigned, ValueIter> originalDataOut;
  
  // Map from iter_index to value Vec with glitch values
  std::map<unsigned, std::vector<int>> oriGlitchDataOut;

  // Map from segindex to succeeding node storing structure
  llvm::StringMap< MgNodeInfo> segSucNodeMap;

  // For control merge node, we need to store the controlDataOut info as well
  std::map<unsigned, ValueIter> controlDataOut;

  // Support LLVM node casting
  enum class NodeKind {DataBaseKind, CMergeDataKind};

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

class CMergeData: public DataBase {
public:
  CMergeData(const std::string &nodeName): DataBase(nodeName) {}

  void printDetail();

  // This function gets the desired control dataout from the control_dataout dict
  // During MG transitions, the control dataout value doesn't exist, we directly give it a 0
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
  // ControlMerge node has one more data out port: control dataout
  std::map<unsigned, ValueIter> controlDataOut;

  // Per CFDFC storing structure
  // Format:
  // {"mg_label": {"control": control_dataout_node_list; "data": data_channel_node_list}}
  std::map<std::string, MgNodeInfo> mgSucNodeDict;

  // Vector to store the control glitch value
  std::vector<int> controlGlitchVec;
};



//===----------------------------------------------------------------------===//
//
// Helper Functions
//
//===----------------------------------------------------------------------===//
// Get the operation name
std::string getHandshakeNodeName(mlir::Value &selRes);

// The following function prints the backEdgeToCFDFC 
void printBEToCFDFCMap(const std::map<std::pair<unsigned, unsigned>, std::vector<unsigned>>& selMap);

// This function prints the Segment ID to BBlist map
void printSegToBBListMap(const std::map<std::string, mlir::SetVector<unsigned>>& selMap);

// Helper function: extracts the initial alphabetic portion from a node name.
std::string getNodeType(const std::string &nodeName);


// This function remove the digits in the given string and keep the rest
std::string removeDigits(const std::string& inStr);


// Get unsigned number from a float
unsigned getUnsigned(float_t inputValue);

// This function prints the node succ list info
void printMgNodeInfo(const MgNodeInfo &info);

// Function to print a vector of strings (mainStack)
void printMainStack(const std::vector<std::string>& mainStack); 

// Function to print a vector of vector of strings (adjStack)
void printAdjStack(const std::vector<std::vector<std::string>>& adjStack); 

template<AdjNode::NodeKind K>
inline bool isNode(const AdjNode * node){
  if (!node) return false;
  auto kind = node->getKind();
  return kind==K;
}


#endif // EXPERIMENTAL_TRANSFORMS_SWITCHING_SUPPORT_H
