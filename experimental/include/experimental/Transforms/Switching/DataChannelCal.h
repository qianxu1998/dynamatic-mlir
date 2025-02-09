//===- DataChannelCal.h - Switching Estimation -----*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares all functions used for data channel switching calculation
//
//===----------------------------------------------------------------------===//

#ifndef EXPERIMENTAL_TRANSFORMS_DATACHANNEL_SWITCHING_H
#define EXPERIMENTAL_TRANSFORMS_DATACHANNEL_SWITCHING_H

#include "experimental/Transforms/Switching/SwitchingSupport.h"
#include "experimental/Transforms/Switching/ProfilingAnalyzer.h"
#include "experimental/Transforms/Switching/ExecModel.h"
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

// This function builds the map from BB pair to the corresponding control_merge output
// Format: {(preBB, curBB) : (control_merge_node, output_value)}
// The results will be stored in a SwitchingInfo instance
void constructBBPairToCMResMap(SwitchingInfo &switchInfo);

// This function finds all base nodes in the dataflow circuit and will contruct two level storing
// We will have 4 types of MG in a given dfg:
//     - "S" : MG contain starting BBs of the program
//     - "Tn": The nth transition section, will have the same invalid backedges as the successing MG (e.x. "T1", "T2")
//     - "n" : Actual CFDFC in the dataflow circuit (e.x. 1, 2, )
//     - "E" : MG contain ending BBs of the program 

// Two types of base nodes will be extracted for each of the MG:
//     - Control base nodes:
//         -- Control_Merge Nodes
//         -- Mux nodes
//     - Data path nodes:
//         -- All units from the mapped unit list
//         -- Start Nodes
void getDataBaseNodes(SwitchingInfo &switchInfo, SCFProfilingResult &profileResults);



//===----------------------------------------------------------------------===//
//
// Class for storing data channel values
//
//===----------------------------------------------------------------------===//
// A small sruct for storing a single "(value, iterIndex)" pair
struct ValueIter {
  int value;
  int iterIndex;
};

// Struct storing the list of nodes used for 
struct MgNodeInfo {
  // "original" => vector of strings
  std::vector<std::string> original;
  // "glitch" => vector of strings
  std::vector<std::string> glitch;
  // "datawidth" => map from string to unsigned
  std::map<std::string, unsigned> dataWidthMap;
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
  std::map<unsigned, ValueIter> originalDataOut;
  
  // Map from iter_index to value Vec with glitch values
  std::map<unsigned, std::vector<int>> oriGlitchDataOut;

  // Map from segindex to succeeding node storing structure
  std::map<std::string, MgNodeInfo> segSucNodeMap;

  // For control merge node, we need to store the controlDataOut info as well
  std::map<unsigned, ValueIter> controlDataOut;

  //
  unsigned lastUpdateIndex;
  bool skipControlCal;
};

//===----------------------------------------------------------------------===//
//
// Helper function for debuging
//
//===----------------------------------------------------------------------===//
// Function to print the contents of a DataBaseNodesTriple
void printDataBaseNodesTriple(DataBaseNodesTriple dbnt);

// 1) Print the muxToSrcNodeMap
// Format: {"mux_node_name" : {"control" : ctrlSrcName, "0" : srcName0, "1" : srcName1}}
void printMuxToSrcNodeMap(const std::map<std::string, std::map<std::string, std::string>> &muxToSrcNodeMap);

// 2) Print the srcNodeToMuxMap
// Format: {"src_node_name" : [ (mux_node_name, portId), ... ]}
void printSrcNodeToMuxMap(const std::map<std::string, std::vector<std::pair<std::string, unsigned>>> &srcNodeToMuxMap);

#endif // EXPERIMENTAL_TRANSFORMS_DATACHANNEL_SWITCHING_H