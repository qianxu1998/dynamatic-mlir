#pragma once
#ifndef DATAGLITCHES_HPP
#define DATAGLITCHES_HPP
#include "experimental/Transforms/Switching/ProfilingAnalyzer.h"
#include "experimental/Transforms/Switching/SwitchingSupport.h"


// This function finds the actual source of the specified start_node in the
// selected MG
// TODO: Merge the following function with graphBacktrack


std::string segNodeDataSrcSearch(SwitchingInfo &switchInfo,
  std::string startNode,
  AdjGraph *selGraph) ;
// This function find all glitching nodes within different MGs, "S", "E", and "T" segments will be ignored
void dataGlitchNodeSearch(SwitchingInfo &switchInfo, SCFProfilingResult &profileResults);

// This function calculates a single glitching value based on the given input operands and node type
int calculateGlitchValue(SwitchingInfo &switchInfo, 
  const std::string& nodeName, 
  const std::vector<NodeGlitchInfo>& glitchInfoVec,
  unsigned iterIdx);

// This function update the glitching value for data base nodes
void dataBaseNodeGlitchUpdate(SwitchingInfo &switchInfo, SCFProfilingResult &profileResults, bool debug);


//  This function returns the longest path form the start point(s) of the
//  specified cfdfc to the desired node
// TODO: Merge the finding with the analyzeglobalorder function.
LongestPathResult selLongestPath(SwitchingInfo &switchInfo, std::string dstNode,
  std::string mgLabel);
  LongestPathResult selLongestPathnew(SwitchingInfo &switchInfo, const std::string& dstNode,
    const std::string& mgLabel);
// This function returns the data src node of the specified mux node in the
// selected execution iteration
std::string getMuxDataSrc(SwitchingInfo &switchInfo, std::string selMuxNode,
  unsigned selIter);

#endif