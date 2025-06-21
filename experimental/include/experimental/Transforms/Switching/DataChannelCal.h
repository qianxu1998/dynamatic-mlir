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
#include "experimental/Transforms/Switching/GraphModel.h"
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

// This funcitonn updates the ori_data for all base nodes in the dfg
void dataChannelBaseNodesValueUpdate(SwitchingInfo &switchInfo, SCFProfilingResult &profileResults);

// This function constructs the succeeding node list for different segments (not only MGs)
void conSegSuccNodesList(SwitchingInfo &switchInfo, SCFProfilingResult &profileResults);

// This function returns the execution iter index based on the given bb_index
unsigned getExecutionIter(unsigned bbIndex, unsigned curBB, SCFProfilingResult &profileResults);


// This function propagates values from base nodes to all other nodes in the dfg circuit based on the execution trace
// *Special treatments needed for LOAD and SOTORE nodes
//  Load: 
//    Output 1: to mem_con with the address value; Output 2: the actual data output
//  COND_BR:
//    Need to keep track of the last valid update (the input data is not -1)
void dfgDataChannelPropagate(SwitchingInfo &switchInfo, SCFProfilingResult &profileResults, bool debug);


// This function returns the data src node of the specified mux node in the selected execution iteration
std::string getMuxDataSrc(SwitchingInfo &switchInfo, std::string selMuxNode, unsigned selIter);

// This function will reduce the input integer value to the target bit width
int reduceBits(int value, unsigned targetBitWidth);

// This fucntion gets the cond_value of for the cond_br node for data channel update
int getCondBrNodeCondValue(SwitchingInfo &switchInfo, unsigned iterIndex, std::string nodeName);

//===----------------------------------------------------------------------===//
//
// Functions for finding the data source node in different segments
//
//===----------------------------------------------------------------------===//
// This function builds the succlist for control merge nodes
std::vector<std::string> segConMergeSuccSearch(SwitchingInfo &switchInfo, std::string startNode, 
                                                std::vector<std::string> &excludingList, std::string segLabel);

// This fucntion builds the glitch succlist for the control merge node
std::vector<std::string> segConMergeGlitchSuccSearch(SwitchingInfo &switchInfo, std::string startNode, 
                                                      std::vector<std::string> &excludingList, std::string segLabel);

// This function builds the succlist for general nodes
MgNodeInfo segGeneralSuccSearch(SwitchingInfo &switchInfo, std::string startNode, std::string segLabel);

// This function finds the actual source of the specified start_node in the selected MG
// TODO: Merge the following function with graphBacktrack
std::string segNodeDataSrcSearch(SwitchingInfo &switchInfo, std::string startNode, AdjGraph *selGraph);

// This function finds the src node of the specified node in the last segment(E)
std::string segENodeSrcSearch(SwitchingInfo &switchInfo, std::string nodeName, SCFProfilingResult &profileResults);

// This function find the address src of the specifed mem_load unit, in segment "S" and "T"
std::string memAddrSrcSearch(SwitchingInfo &switchInfo, SCFProfilingResult &profileResults, std::string nodeName, std::string selSeg, unsigned iterIdx);

//  This function returns the longest path form the start point(s) of the specified cfdfc to the desired node
// TODO: Merge the finding with the analyzeglobalorder function.
LongestPathResult selLongestPath(SwitchingInfo &switchInfo, std::string dstNode, std::string mgLabel);

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
