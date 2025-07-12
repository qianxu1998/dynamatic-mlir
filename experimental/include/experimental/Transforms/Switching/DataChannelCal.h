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

#include "dynamatic/Dialect/Handshake/HandshakeAttributes.h"
#include "dynamatic/Support/TimingModels.h"
#include "dynamatic/Transforms/BufferPlacement/CFDFC.h"
#include "experimental/Support/StdProfiler.h"
#include "experimental/Transforms/Switching/SwitchingNodeModels/SwitchingNodeModels.h"
#include "experimental/Transforms/Switching/GraphModel.h"
#include "experimental/Transforms/Switching/ProfilingAnalyzer.h"
#include "experimental/Transforms/Switching/SwitchingSupport.h"
#include "experimental/Transforms/Switching/utils.h"

#include "mlir/IR/Attributes.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include "llvm/ADT/StringRef.h"

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

using namespace mlir;
using namespace dynamatic;
using namespace dynamatic::handshake;

// This function builds the map from BB pair to the corresponding control_merge
// output Format: {(preBB, curBB) : (control_merge_node, output_value)} The
// results will be stored in a SwitchingInfo instance
void mapBBPairToControlMerge(SwitchingInfo &switchInfo);

// This function finds all base nodes in the dataflow circuit and will contruct
// two level storing We will have 4 types of MG in a given dfg:
//     - "S" : MG contain starting BBs of the program
//     - "Tn": The nth transition section, will have the same invalid backedges
//     as the successing MG (e.x. "T1", "T2")
//     - "n" : Actual CFDFC in the dataflow circuit (e.x. 1, 2, )
//     - "E" : MG contain ending BBs of the program

// Two types of base nodes will be extracted for each of the MG:
//     - Control base nodes:
//         -- Control_Merge Nodes
//         -- Mux nodes
//     - Data path nodes:
//         -- All units from the mapped unit list
//         -- Start Nodes
void getDataBaseNodes(SwitchingInfo &switchInfo,
                      SCFProfilingResult &profileResults);

// This function updates the ori_data for all base nodes in the dfg
// we update 1) database nodes  2) controlmerge nodes 3) influenced multiplexer nodes
void dataChannelBaseNodesValueUpdate(SwitchingInfo &switchInfo,
                                     SCFProfilingResult &profileResults);

// This function constructs the succeeding node list for different segments (not
// only MGs)
void buildSegmentSuccNodesList(SwitchingInfo &switchInfo,
                         SCFProfilingResult &profileResults);

// This function returns the execution iter index based on the given bb_index
unsigned getExecutionIter(unsigned bbIndex, unsigned curBB,
                          SCFProfilingResult &profileResults);




//===----------------------------------------------------------------------===//
//
// Functions for finding the data source node in different segments
//
//===----------------------------------------------------------------------===//
// This function builds the succlist for control merge nodes
std::vector<std::string>
segCtrlMergeSuccSearch(SwitchingInfo &switchInfo, std::string startNode,
                      std::vector<std::string> &excludingList,
                      llvm::StringRef segLabel);

// This fucntion builds the glitch succlist for the control merge node
std::vector<std::string>
segCtrlMergeGlitchSuccSearch(SwitchingInfo &switchInfo, std::string startNode,
                            std::vector<std::string> &excludingList,
                            llvm::StringRef segLabel);



// This function builds the succlist for general nodes
MgNodeInfo segGeneralSuccSearch(SwitchingInfo &switchInfo,
  std::string startNode, llvm::StringRef segLabel);

//===----------------------------------------------------------------------===//
//
// Helper function for debuging
//
//===----------------------------------------------------------------------===//
// Function to print the contents of a DataBaseNodesTriple
void printDataBaseNodesTriple(DataBaseNodesTriple dbnt);

// 1) Print the muxToSrcNodeMap
// Format: {"mux_node_name" : {"control" : ctrlSrcName, "0" : srcName0, "1" :
// srcName1}}
void printMuxToSrcNodeMap(
    const llvm::StringMap<  std::map<std::string, std::string>>
        &muxToSrcNodeMap);

// 2) Print the srcNodeToMuxMap
// Format: {"src_node_name" : [ (mux_node_name, portId), ... ]}
void printSrcNodeToMuxMap(
    const llvm::StringMap< std::vector<std::pair<std::string, unsigned>>>
        &srcNodeToMuxMap);





        








#endif // EXPERIMENTAL_TRANSFORMS_DATACHANNEL_SWITCHING_H
