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
#include "llvm/Support/Debug.h"

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

// Struct used to store all the data source nodes in differernt segments in the dfg
// We have this kind of definiton as we obtain the profiling results from scf level
// If we have handshake level simulator one day, we can get rid of the entire
// data channel estimation process.
struct DataBaseNodesTriple {
  std::vector<std::string> all;
  std::vector<std::string> control;
  std::vector<std::string> data;
};

// Helper datatype for switching estimation. Aggregates all useful information
// for the swithicng estimation process
struct SwitchingInfo {
  // This function insert (backedge pair, mgLabel) to the backEdgeToCFDFCMap
  void insertBE(unsigned srcBB, unsigned dstBB, StringRef mgLabel);

  // 
  //  Internal Storing Variables
  //
  // All backedge BB pairs in the dataflow circuit
  llvm::SmallVector<std::pair<unsigned, unsigned>> backEdges;
  // Map from CFDFC index to the corresponding II values
  std::unordered_map<unsigned, float_t> cfdfcIIs;
  // Map from CFDFC index to the CFDFC info stroing class
  // Only contain the number of edges, backedges etc.
  // No info about unit delay, node neighbors etc.
  std::map<unsigned, buffer::CFDFC> cfdfcs;
  // Set of names of the ALUs in the Handshake level mlir file
  // Used to retrieve the date from SCF level profiling
  std::vector<std::string> funcOpNames;

  // Map from Backedge pair to the list of CFDFC lable vector
  // i.e. {(1, 1) : [1]}
  std::map<std::pair<unsigned, unsigned>, std::vector<unsigned>> backEdgeToCFDFCMap;
  // Map from Segment Label (CFDFC and temporal transaction sections that are not MGs)
  std::map<std::string, std::vector<unsigned>> segToBBListMap;
  // Map from Transaction Segment label to the successing MG label
  std::map<std::string, std::string> transToSucMGMap;
  // Map storing the subgraph of different segments in the dataflow circuit
  std::map<std::string, std::shared_ptr<AdjGraph>> segToAdjGraphMap;
  // Map from segment label to the vector of invalid backedges
  std::map<std::string, std::vector<std::pair<std::string, std::string>>> segInvalidBackedgesMap;
  // dataflow graph
  // Below has to be a shared pointer, otherwise need to override the clonePass() implementation in MLIR
  std::shared_ptr<AdjGraph> dataflowGraph;

  // 
  //  Variables for data channel switching calculation
  //
  // Map from pair of BB sequence to the corresponding control_merge output
  // Format: {(preBB, curBB) : [(control_merge_node, output_value)]}
  std::map<std::pair<unsigned, unsigned>, std::vector<std::pair<std::string, unsigned>>> bbPairToCMResultMap;
  // Map from segment index to dataBaseNode struct
  std::map<std::string, DataBaseNodesTriple> segToDataBaseVecMap;
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
class AdjNode {
public:
  // Define the virtual destructor
  virtual ~AdjNode() = default;

  // Constructer
  AdjNode(mlir::Operation* selOp, 
          const std::vector<std::string>& predecessors, const std::vector<std::string>& successors, 
          const std::map<std::string, unsigned>& sucDataWidthMap, const unsigned& latency);

  // This funciton checks whether the handshake channel swiching information updating is finished or not
  bool handshakeUpdateFinished();

  // This function checks the handshake switching number of the node itself is updated or not
  bool handshakeSwitchingChecking();

  // This function prints all information of the node
  virtual void printDetail();

  // This function prints details of handshake switching
  void printHandshakeSwitching();

  // This function will calculate the total valid and ready swithcing for the node
  void totalHandshakeSwitchingUpdate();

  // This function calculate the number of switches in the dataout channel
  void updateDataoutChannel(int inputData);

  // Add extra handshake channel switches
  void updateHandshakeChannelSwitching(unsigned validChannelSwitching, unsigned readyChannelSwitching);

  // This function calculates the total number of switches of all data out channels
  //  Note: We treat "X" as invalid data and will count it only once
  void totalDataSwitchingCounting(bool mapped);

  // This function will return the vector of 1's position in the number's binary format
  std::vector<unsigned> getPositionList(int number);

  //
  void printDataChannelSwitching();
  void printPerDataChannelToggleNumber();
  void printPerHandshakeChannelToggleNumber();

  // Support LLVM node casting
  // Following enum class is needed for isa<> and dyn_cast<>
  enum class NodeKind {AdjNodeKind, BufferNodeKind, JoinNodeKind, PassNodeKind, CmpiNodeKind, AddiNodeKind,
                        SubiNodeKind, MuliNodeKind, ExtsiNodeKind, DLoadNodeKind, DStoreNodeKind, MergeNodeKind,
                        CMergeNodeKind, ForkNodeKind, CBrNodeKind, ShliNodeKind, ShrsiNodeKind, ShruiNodeKind, MuxNodeKind, TrunciNodeKind,
                        ExtuiNodeKind, ConstantNodeKind, OriNodeKind, AndiNodeKind, SourceNodeKind, EndNodeKind, SinkNodeKind, StartNodeKind};

  // By default, an AdjNode has kind = AdjNodeKind
  virtual NodeKind getKind() const { return NodeKind::AdjNodeKind; }

  // "classof" used by llvm casting
  static bool classof(const AdjNode *node) {
    // We claim "yes" if node->getKind() == AdjNodeKind
    // Usually it's trivial for the base class
    return node->getKind() == NodeKind::AdjNodeKind;
  }

  // 
  //  Internal Storing Variables
  //
  unsigned nodeLatency = 0;       // Used to store the latency of the chosen node
  mlir::Operation* op;            // Pointer to the operation in the mlir file
  // Memory controller is exclueded from the pres and sucs
  std::vector<std::string> pres;  // Vector storing the predecessors of the node in the segemnt
  std::vector<std::string> sucs;  // Vector storing the successors of the node in the segement

  // Handshake Signal Switching
  std::map<std::string, unsigned> validSignal;    // Map used to store number of switching of the node's valid signals (per channel, e.x., {"node_name" : 2})
  std::map<std::string, unsigned> readySignal;    // Map used to store number of switching of the node's ready signals (per channel, e.x., {"node_name" : 2})
  // TODO: Need to improve the way the setR/V is stored
  std::map<std::string, std::set<unsigned>> setV; // Set used to store the active range of the corresponding valid signal in different channels
  std::map<std::string, std::set<unsigned>> setR; // Set used to store active range of the corresponding ready signal in different channels

  // Data channel Switching
  std::map<std::string, unsigned> sucsDataWidthMap; // Map from succeeding node name to the channel width
  std::map<std::string, std::map<unsigned, unsigned>> perChannelToggle; // toggle count dict per bitwidth
  std::map<std::string, std::vector<int>> dataOut;  // Map from succeeding node name to the corresponding outptu channel dict
  
  // Overall signal switching status
  std::map<std::string, unsigned> dataSwitches;  // Map storing the number of data switches for different data channels
  std::map<std::string, unsigned> handshakeSwitches;          // Map storing the number of switches in different handshake channels

  unsigned totalValidSwitching = 0;
  unsigned totalReadySwitching = 0;
  unsigned totalDataSwitching = 0;

  // Update Flag
  bool handshakeUpdateFlag = false;
};

// Seg subgraph that stores all the nodes of a segment in the dataflow circuit
class AdjGraph {
public:
  AdjGraph(const buffer::CFDFC& cfdfc, const TimingDatabase& timingDB, 
            const unsigned &II, const unsigned &mgIndex);
  
  // Create an AdjGraph from the funcop
  AdjGraph(const TimingDatabase& timingDB, const unsigned &II, 
            handshake::FuncOp funcOp,
            std::vector<std::pair<std::string, std::string>> &allBackedges);

  // This function insert a new element to the given map
  void insertToSurroundingList(std::map<std::string, std::vector<std::string>>& selMap, 
                                std::string& key, std::string& value);

  // This function create the corresponding storing structure for a node based on the mlir op type
  // and returns a unique pointer to it.
  std::shared_ptr<AdjNode> createNodeFromOperation(mlir::Operation *op,
                                                    std::vector<std::string> &pres, std::vector<std::string> &sucs,
                                                    unsigned &nodeLatency);

  // The following function calculates the path latency based on all the information in the MG
  unsigned calPathLatency(const Path &selPath, bool useGlobalOrder);

  // Find all the paths between the specified src and dst node within the graph
  std::vector<Path> findPaths(const std::string &srcNode, const std::string &dstNode,
                              bool noStartingNode, bool useGlobalOrder);
  
  // This function conducts backtracking in the dfg starting from the specified node.
  // The search will stop: (1) No node left; (2)Base node encountered
  std::string graphBacktrack(std::string srcNode, std::unordered_set<std::string> &baseNodeSet);

  //===----------------------------------------------------------------------===//
  // Data Channel Switching Calculation Functions
  //===----------------------------------------------------------------------===//
  // The following function calculates the global order of the units and cycle times in the AdjGraph
  // the results will be stored in graphGlobalOrder
  void obtainNodeGlobalOrder();
  // During handshake and path latency analysis, we use those start nodes as the base points for analysis
  // However, they may not be active at the same time, so we need to take the shifting into account.
  void analyzeStartNodeShifting();
  // This function constructs the src map for all mux nodes in the dfg
  // While building the src dict, this function also update the mapping from src node to the corresponding mux node with the following format
  void buildMuxSrcMap();
  // This funciton finds the cond_src node for all cond_br nodes and data/address src for store nodes in the dfg
  void buildCondandStoreSrcMap();

  //===----------------------------------------------------------------------===//
  // Data Channel Switching Calculation Variables
  //===----------------------------------------------------------------------===//
  // Data Base nodes used for data propagation through data channels (from scf level profiling)
  std::unordered_set<std::string> profileBaseNodes;
  // Data + Control base nodes
  std::unordered_set<std::string> allDataBaseNode;
  // mux node to data source map
  // Format: {"mux_node_name" : {"control" : control_src_node_name, 0 : src_node_name_0, 1 : src_node_name_1}}
  std::map<std::string, std::map<std::string, std::string>> muxToSrcNodeMap;
  // Source node to vector of mux and port pair map
  // Format: {"src_node_name" : [(mux_node_name, corresponding_input_port_id)]}.
  std::map<std::string, std::vector<std::pair<std::string, unsigned>>> srcNodeToMuxMap;
  // Control_merge to mux node map
  // The control port of the mux node is always connecting to a control_merge node
  std::map<std::string, std::vector<std::string>> cmToMuxMap;
  // CondBr to control source node map
  std::map<std::string, std::string> condBrToConSrcMap;

  // 
  //  Internal Storing Variables
  //
  unsigned cfdfcIndex = 0;                                    // Variable storing the corresponding cfdfc index
  // TODO: Check the rounding of the II
  unsigned cfdfcII = 0;                                       // The II of the cfdfc
  std::string baseNode;                                       // The node that serves as the base point for path calculation
  std::vector<std::string> segStartNodes;                     // Vector storing all starting nodes in the segment
  std::map<std::string, std::shared_ptr<AdjNode>> nodes;      // Map from unit name to the corresponding node storing structure
  std::vector<std::pair<std::string, std::string>> backedges; // Vector storing all backedges in the Adjacency graph;
  // Map storing the global order (actual start time) of different nodes in the graph
  // the format is like : {node_name : (start_node, delay)}
  std::map<std::string, std::pair<std::string, unsigned>> graphGlobalOrder;
  // Map storing the maximum cycle time from different start node in the graph
  // This will be used to analyze the shifting between start node
  std::map<std::string, unsigned> cycleTimeMap;
  // Map storing the shifting between different different start node and the base node
  std::map<std::string, int> startBaseNodeShiftMap;
  // List storing the name of nodes in the graph in program order
  std::vector<std::string> orderedNodeName;
};

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

//===----------------------------------------------------------------------===//
//
// Helper Functions
//
//===----------------------------------------------------------------------===//
// Get the operation name
std::string getHandshakeNodeName(mlir::Value &selRes);

// The following function prints the backEdgeToCFDFCMap 
void printBEToCFDFCMap(const std::map<std::pair<unsigned, unsigned>, std::vector<unsigned>>& selMap);

// This function prints the Segment ID to BBlist map
void printSegToBBListMap(const std::map<std::string, mlir::SetVector<unsigned>>& selMap);

// This function prints all values in a vector
template <typename T>
inline void printVector(const T& selVec) {
  int counter = 0;

  llvm::dbgs() << "[DEBUG] Vector Contents: "; 

  for (auto& selVal : selVec) {
    llvm::dbgs() << "[" << counter << "] : " << selVal << "; ";

    counter++;
  }

  llvm::dbgs() << ";\n";
}

// This function remove the digits in the given string and keep the rest
std::string removeDigits(const std::string& inStr);

// This function split a given string into a vector based on the delimiter
std::vector<std::string> split(const std::string &s, const std::string& delimiter);

// This function removes the starting and ending empty space
std::string strip(const std::string &inputStr, const std::string &toRemove);


#endif // EXPERIMENTAL_TRANSFORMS_SWITCHING_SUPPORT_H
