//===- GraphModel.h - Switching estimation -------------*- C++ -*-===//
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
#pragma once
#ifndef EXPERIMENTAL_ANALYSIS_SWITCHINGESTIMATION_GRAPHMODEL_H
#define EXPERIMENTAL_ANALYSIS_SWITCHINGESTIMATION_GRAPHMODEL_H

#include "dynamatic/Support/LLVM.h"
#include "dynamatic/Support/TimingModels.h"
#include "dynamatic/Transforms/BufferPlacement/CFDFC.h"
#include "experimental/Analysis/SwitchingEstimation/Debug.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/Debug.h"
#include <map>
#include <set>
#include <string>
#include <unordered_set>
#include <vector>

#define DEBUG_TYPE "switching-estimation"

using IISet = llvm::SmallBitVector;

using namespace llvm;
using namespace mlir;
using namespace dynamatic;
using namespace dynamatic::buffer;

//===----------------------------------------------------------------------===//
//
// Path class, used to store information related to path analysis
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
    using ::dynamatic::experimental::SwitchingDebugCategory;
    using ::dynamatic::experimental::switchingDebugStream;
    LLVM_DEBUG(switchingDebugStream(SwitchingDebugCategory::Graph)
               << "[DEBUG] \t[Path] \n[DEBUG] \t\t[ ");
    for (const auto &selNode : nodeList) {
      LLVM_DEBUG(
          switchingDebugStream(SwitchingDebugCategory::Graph) << selNode << ", ");
    }
    switchingDebugStream(SwitchingDebugCategory::Graph) << "]\n";

    //
    LLVM_DEBUG(switchingDebugStream(SwitchingDebugCategory::Graph)
               << "[DEBUG] \t\tBackedges: ");
    for (const auto &selPair : backedges) {
      LLVM_DEBUG(switchingDebugStream(SwitchingDebugCategory::Graph)
                 << "( " << selPair.first << ", " << selPair.second
                 << " ); ");
    }
    LLVM_DEBUG(switchingDebugStream(SwitchingDebugCategory::Graph) << "\n");

    //
    LLVM_DEBUG(switchingDebugStream(SwitchingDebugCategory::Graph)
               << "[DEBUG] \t\tPath Latency: " << latency << "\n");
  }

  std::vector<std::string> nodeList;
  std::vector<std::pair<std::string, std::string>> backedges;
  bool contain_backedge = false;
  unsigned latency = 0;
};

// Class used to construct the per segment (MG & one-time execution segment)
// The class defined following is purely for path fining and preserve the
// structure of each subgraph.
/*
  Base Class used to store information related to node in the dataflow circuit,
        which will be used to calculate the switching activity in steady state.
            - Handshake channels
            - Data channels

        ! Handshake Signal Calculation:
            All child classes of this base class shall have it's own
  implementation of
                - cal_valid_switching
                - cal_ready_switching
                - cal_valid_set
                - cal_ready_set

            For the above functions, we have following assumptions:
                - Num switches must be >= 0.
                - For num switches, we use -1 to represent invalid value
                - For active set, we use None to represesnt invalid value

        ! Data Channel Calculation:
            For each node, we only store output data to its successors, like a
  directed chain list. Two possible situations for a given node:
                - It has only one (value, #iter) pair, like constant node. As
  the value never change
                - It has multiple (value, #iter) pair
            Thus, during dataout calculation, we need to check the len of the
  input data *If the node has multiple input data only two situations: 1)
  lengths they match with each other 2) one input has only 1 pair and the other
  has multiple. For the second case, the outputdata will have list length equal
  to the longer input data
*/
class AdjNode {
public:
  // Define the virtual destructor
  virtual ~AdjNode() = default;

  // Constructer
  AdjNode(mlir::Operation *selOp, const std::vector<std::string> &predecessors,
          const std::vector<std::string> &successors,
          const std::map<std::string, unsigned> &sucDataWidthMap,
          const unsigned &latency, const unsigned &bbIndex);

  // This funciton checks whether the handshake channel swiching information
  // updating is finished or not
  bool handshakeUpdateFinished();

  // This function checks the handshake switching number of the node itself is
  // updated or not
  bool handshakeSwitchingChecking();

  // This function prints all information of the node
  virtual void printNodeDetails();

  // This function will calculate the total valid and ready switching for the
  // node
  void totalHandshakeSwitchingCounting();

  // This function calculate the number of switches in the dataout channel
  void updateDataoutChannel(int inputData);

  // Add extra handshake channel switches
  void updateHandshakeChannelSwitching(unsigned validChannelSwitching,
                                       unsigned readyChannelSwitching);

  // This function calculates the total number of switches of all data out
  // channels
  //  Note: We treat "X" as invalid data and will count it only once
  void totalDataSwitchingCounting(bool mapped);

  // This function will return the vector of 1's position in the number's binary
  // format
  std::vector<unsigned> getPositionList(int number);

  // This function prints details of handshake switching
  void printHandshakeSwitching();

  // This function prints details of data channel switching
  void printDataChannelSwitching();

  // This function prints the data-channel per bit toggle number
  void printPerDataChannelPerBitToggleNumber();

  // This function prints the per-handshake-channel toggle number
  void printPerHandshakeChannelToggleNumber();

  // Support LLVM node casting
  // Following enum class is needed for isa<> and dyn_cast<>
  enum class NodeKind {
    AdjNodeKind,
    BufferNodeKind,
    JoinNodeKind,
    PassNodeKind,
    CmpiNodeKind,
    CmpfNodeKind,
    AddiNodeKind,
    AddfNodeKind,
    SubiNodeKind,
    MuliNodeKind,
    MulfNodeKind,
    ExtsiNodeKind,
    DLoadNodeKind,
    DStoreNodeKind,
    MergeNodeKind,
    SelectNodeKind,
    CMergeNodeKind,
    ForkNodeKind,
    CBrNodeKind,
    ShliNodeKind,
    ShrsiNodeKind,
    ShruiNodeKind,
    MuxNodeKind,
    TrunciNodeKind,
    ExtuiNodeKind,
    ConstantNodeKind,
    OriNodeKind,
    AndiNodeKind,
    SourceNodeKind,
    EndNodeKind,
    SinkNodeKind,
    StartNodeKind
  };

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
  unsigned nodeLatency = 0; // Used to store the latency of the chosen node
  mlir::Operation *op;      // Pointer to the operation in the mlir file
  unsigned bbindex;         // BB index for the node
  // Memory controller is exclueded from the pres and sucs
  std::vector<std::string>
      pres; // Vector storing the predecessors of the node in the segemnt
  std::vector<std::string>
      sucs; // Vector storing the successors of the node in the segement
  std::vector<AdjNode *> sucsPtrs; // ← p

  // Handshake Signal Switching
  std::map<std::string, unsigned>
      validSignal; // Map used to store number of switching of the node's valid
                   // signals (per channel, e.x., {"node_name" : 2})
  std::map<std::string, unsigned>
      readySignal; // Map used to store number of switching of the node's ready
                   // signals (per channel, e.x., {"node_name" : 2})
  // TODO: Need to improve the way the setR/V is stored
  std::map<std::string, IISet>
      setV; // Set used to store the active range of the corresponding valid
            // signal in different channels
  std::map<std::string, IISet>
      setR; // Set used to store active range of the corresponding ready signal
            // in different channels

  // Data channel Switching
  std::map<std::string, unsigned>
      sucsDataWidthMap; // Map from succeeding node name to the channel width
  std::map<std::string, std::map<unsigned, unsigned>>
      perChannelToggle; // toggle count dict per bitwidth
  std::map<std::string, std::vector<int>>
      dataOut; // Map from succeeding node name to the corresponding output
               // channel dict

  // Overall signal switching status
  std::map<std::string, unsigned>
      dataSwitches; // Map storing the number of data switches for different
                    // data channels
  std::map<std::string, unsigned>
      handshakeSwitches; // Map storing the number of switches in different
                         // handshake channels

  unsigned totalValidSwitching = 0;
  unsigned totalReadySwitching = 0;
  unsigned totalDataSwitching = 0;

  // Update Flag
  bool handshakeUpdateFlag = false;
};

//===----------------------------------------------------------------------===//
//
// Seg subgraph that stores all the nodes of a segment in the dataflow circuit
//
//===----------------------------------------------------------------------===//

class AdjGraph {
public:
  AdjGraph(CFDFC *cfdfc, const TimingDatabase &timingDB, const unsigned &II,
           const unsigned &mgIndex, const double &targetPeriod);

  // Create an AdjGraph from the funcop
  AdjGraph(const TimingDatabase &timingDB, const unsigned &II,
           handshake::FuncOp funcOp,
           std::vector<std::pair<std::string, std::string>> &allBackedges,
           const double &targetPeriod);

  // This function insert a new element to the given map
  void insertToSurroundingList(
      std::map<std::string, std::vector<std::string>> &selMap, std::string &key,
      std::string &value);

  // This function create the corresponding storing structure for a node based
  // on the mlir op type and returns a unique pointer to it.
  std::shared_ptr<AdjNode>
  createNodeFromOperation(Operation *op, std::vector<std::string> &pres,
                          std::vector<std::string> &sucs, unsigned &nodeLatency,
                          unsigned &bbIndex);

  // The following function calculates the path latency based on all the
  // information in the MG
  unsigned calPathLatency(const Path &selPath, bool useGlobalOrder);

  // Find all the paths between the specified src and dst node within the graph
  std::vector<Path> findPaths(const std::string &srcNode,
                              const std::string &dstNode, bool noStartingNode,
                              bool useGlobalOrder);

  // This function conducts backtracking in the dfg starting from the specified
  // node. The search will stop: (1) No node left; (2)Base node encountered
  std::string graphBacktrack(std::string srcNode,
                             std::unordered_set<std::string> &baseNodeSet);

  // This function conducts backtracking in the MG from the specified node,
  // the search will stop when there is no node left and return all buffer
  // nodes.
  std::vector<std::string> mgBacktrackBuffer(std::string srcNode);

  //===----------------------------------------------------------------------===//
  // Data Channel Switching Calculation Functions
  //===----------------------------------------------------------------------===//
  // The following function calculates the global order of the units and cycle
  // times in the AdjGraph the results will be stored in graphGlobalOrder
  void obtainNodeGlobalOrder();

  // During handshake and path latency analysis, we use those start nodes as the
  // base points for analysis However, they may not be active at the same time,
  // so we need to take the shifting into account.
  void computeStartNodeShifts();

  // This function constructs the src map for all mux nodes in the dfg
  // While building the src dict, this function also update the mapping from src
  // node to the corresponding mux node with the following format This function
  // finds the cond_src node for all cond_br nodes and data/address src for
  // store nodes in the dfg This function constructs the source map for all mux
  // ,cond_br and Dstore nodes in the dfg
  void buildSrcMaps();

  //===----------------------------------------------------------------------===//
  // Data Channel Switching Calculation Variables
  //===----------------------------------------------------------------------===//
  // Data Base nodes used for data propagation through data channels (from scf
  // level profiling)
  std::unordered_set<std::string> profileBaseNodes;
  // Data + Control base nodes
  std::unordered_set<std::string> allDataBaseNode;
  // mux node to data source map
  // Format: {"mux_node_name" : {"control" : control_src_node_name, 0 :
  // src_node_name_0, 1 : src_node_name_1}}
  llvm::StringMap<std::map<std::string, std::string>> muxToSrcNodeMap;
  // Source node to vector of mux and port pair map
  // Format: {"src_node_name" : [(mux_node_name, corresponding_input_port_id)]}.
  llvm::StringMap<std::vector<std::pair<std::string, unsigned>>>
      srcNodeToMuxMap;
  // Control_merge to mux node map
  // The control port of the mux node is always connecting to a control_merge
  // node
  llvm::StringMap<std::vector<std::string>> cmToMuxMap;
  // CondBr to control source node map
  llvm::StringMap<std::string> condBrToConSrcMap;
  // Map from cond_br node to the directly connected buffer nodes and the
  // corresponding port idx, if exists Format: {"cond_br_node" : [(buffer_name,
  // port_idx), ], }
  llvm::StringMap<std::vector<std::pair<std::string, unsigned>>>
      condBrToBufferMap;
  // Map from data source node to opaque buffer nodes
  llvm::StringMap<std::string> dataSrcToOpaqueBufferMap;

  //===----------------------------------------------------------------------===//
  // Path Caching Structures
  //===----------------------------------------------------------------------===//
  struct PathKey {
    std::string srcNode;
    std::string dstNode;
    bool noStartingNode;
    bool useGlobalOrder;

    // Constructor
    PathKey(std::string srcNode, std::string dstNode, bool noStartingNode,
            bool useGlobalOrder)
        : srcNode(srcNode), dstNode(dstNode), noStartingNode(noStartingNode),
          useGlobalOrder(useGlobalOrder) {}

    // Operator overload
    bool operator==(const PathKey &o) const {
      return srcNode == o.srcNode && dstNode == o.dstNode &&
             noStartingNode == o.noStartingNode &&
             useGlobalOrder == o.useGlobalOrder;
    }
  };

  struct PathKeyHasher {
    std::size_t operator()(PathKey const &p) const noexcept {
      std::size_t h1 = std::hash<std::string>{}(p.srcNode);
      std::size_t h2 = std::hash<std::string>{}(p.dstNode);
      std::size_t h3 = std::hash<bool>{}(p.noStartingNode);
      std::size_t h4 = std::hash<bool>{}(p.useGlobalOrder);
      return (((h1 * 31) ^ (h2 * 17)) >> 1) ^ ((h3 << 1) | (h4 << 2));
    }
  };

  // cache for longest paths from each entry node to all other nodes
  std::unordered_map<PathKey, std::pair<unsigned, std::string>, PathKeyHasher>
      maxLatencyCache;

  // Helper function to get the max latency path from the cache
  std::pair<unsigned, std::string> getMaxLatency(const std::string &srcNode,
                                                 const std::string &dstNode,
                                                 bool noStartingNode,
                                                 bool useGlobalOrder) {

    // Get the path key in this graph
    PathKey selKey(srcNode, dstNode, noStartingNode, useGlobalOrder);

    // Check whether the key exists in the cache
    auto it = maxLatencyCache.find(selKey);
    // [CASE 1] Cache hit
    if (it != maxLatencyCache.end()) {
      return it->second;
    }

    // [CASE 2] Cache miss, recalculate the max latency path
    std::vector<Path> tmpPaths =
        findPaths(srcNode, dstNode, noStartingNode, useGlobalOrder);
    unsigned maxLatency = 0;
    std::string bestSrcNode = "";
    for (const auto &selPath : tmpPaths) {
      auto tmpLatency = selPath.latency;
      if (tmpLatency >= maxLatency) {
        maxLatency = tmpLatency;
        bestSrcNode = srcNode;
      }
    }

    // Update the cache
    maxLatencyCache[selKey] = std::make_pair(maxLatency, bestSrcNode);
    return std::make_pair(maxLatency, bestSrcNode);
  }

  //
  //  Internal Storing Variables
  //

  CFDFC *cfdfcPrt = nullptr; // Pointer to the corresponding cfdfc structure
  unsigned cfdfcIndex = 0;   // Variable storing the corresponding cfdfc index
  // TODO: Check the rounding of the II
  unsigned cfdfcII = 0; // The II of the cfdfc
  std::string
      baseNode; // The node that serves as the base point for path calculation
  std::vector<std::string>
      segStartNodes; // Vector storing all starting nodes in the segment
  llvm::StringMap<std::shared_ptr<AdjNode>>
      nodes; // Map from unit name to the corresponding node storing structure
  std::vector<std::pair<std::string, std::string>>
      backedges; // Vector storing all backedges in the Adjacency graph;
  // Map storing the global order (actual start time) of different nodes in the
  // graph the format is like : {node_name : (start_node, delay)}
  llvm::StringMap<std::pair<std::string, unsigned>> graphGlobalOrder;
  // Map storing the maximum cycle time from different start node in the graph
  // This will be used to analyze the shifting between start node
  llvm::StringMap<unsigned> cycleTimeMap;
  // Map storing the shifting between different different start node and the
  // base node
  llvm::StringMap<int> startBaseNodeShiftMap;
  // List storing the name of nodes in the graph in program order
  std::vector<std::string> orderedNodeName;
};

#endif // EXPERIMENTAL_ANALYSIS_SWITCHINGESTIMATION_GRAPHMODEL_H
