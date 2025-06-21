//===- GraphModel.h - Switching Estimation -----*- C++ -*-===//
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
#pragma once
#ifndef GRAPHMODEL_HPP
#define GRAPHMODEL_HPP

#include <string>
#include <unordered_set>

#include "dynamatic/Support/LLVM.h"
#include "dynamatic/Support/TimingModels.h"
// #include "experimental/Transforms/Switching/NodeInfo.h"
#include "dynamatic/Transforms/BufferPlacement/CFDFC.h"
#include "experimental/Transforms/Switching/SwitchingSupport.h"
// #include "experimental/Transforms/Switching/SwitchingSupport.h"



using namespace mlir;
using namespace dynamatic;
using namespace dynamatic::handshake;
class AdjGraph;
class AdjNode;
class Path;

// Seg subgraph that stores all the nodes of a segment in the dataflow circuit
class AdjGraph {
public:
  AdjGraph();
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
                                                    unsigned &nodeLatency, unsigned &bbIndex);

  // The following function calculates the path latency based on all the information in the MG
  unsigned calPathLatency(const Path &selPath, bool useGlobalOrder);
  void mergeFrom(const AdjGraph &other);
  llvm::StringMap<llvm::StringMap<std::string>> predecessorFrom;

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
  // Map from cond_br node to the directly connected buffer nodes and the corresponding port idx, if exists
  // Format: {"cond_br_node" : [(buffer_name, port_idx), ], }
  std::map<std::string, std::vector<std::pair<std::string, unsigned>>> condBrToBufferMap;

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





  struct Pathkey{
    std::string src;
    std::string dst;
    bool nostart,useglobal;

    //constructor
    Pathkey(std::string s, std::string d, bool n, bool g)
    : src(s), dst(d), nostart(n), useglobal(g) {}

    //operator overload
    bool operator==(Pathkey const &o) const {
      return src == o.src
          && dst == o.dst
          && nostart == o.nostart
          && useglobal == o.useglobal;
    }
  };
  struct PathkeyHasher{

    size_t operator ()(Pathkey const &k ) const noexcept{
    auto h1 = std::hash<std::string>()(k.src);
    auto h2 = std::hash<std::string>()(k.dst);
    auto h3 = std::hash<bool>()(k.nostart);
    auto h4 = std::hash<bool>()(k.useglobal);
    return ( (( h1*31) ^ (h2*17)  )>>1) ^ ((h3<<1) | (h4<<2));
  }
  };

  // cache for longest paths from each entry node to all other nodes 
  std::unordered_map<Pathkey , std::pair<unsigned,std::string> ,PathkeyHasher> maxLatencycache;

  std::pair<unsigned,std::string> getMaxLatency(const std::string &srcNode,
    const std::string &dstNode,
    bool noStartingNode,
    bool useGlobalOrder) {

    Pathkey key(srcNode,dstNode,noStartingNode,useGlobalOrder);

    auto it=maxLatencycache.find(key);
    //Case 1 : cache hit
      if(it!=maxLatencycache.end()){
        llvm::dbgs() <<"step4 found\n";
        return    it->second;// return cached latency and node name
      }

          // Case 2: cache miss
          llvm::dbgs() << "step4 longest path data channel "<< srcNode<<" "<< dstNode <<"\n";

      std::vector<Path> tmppaths=  findPaths(srcNode,dstNode,noStartingNode,useGlobalOrder);
      unsigned maxLatency{0};
      std::string bestSrcNode{""};
      for (const auto &selpath:tmppaths){
        auto tmpPathLat=selpath.latency;
        if (tmpPathLat >= maxLatency) {
          maxLatency = tmpPathLat;
          bestSrcNode = srcNode;
      }
    }
    maxLatencycache[key] = std::make_pair(maxLatency,bestSrcNode);
    return  {maxLatency,bestSrcNode};
  }


  // // Compute the longest paths from a given entry node to all other nodes
  // void computeLongestPathsFromEntry(const std::string &entry);
  
  // // Get the longest path latency from entry to node (computes if not cached)
  // unsigned getLongestPathLatency(const std::string &entry, const std::string &node);






  
};


#endif