#ifndef DATACHANNELCAL_HPP
#define DATACHANNELCAL_HPP
#include "experimental/Transforms/Switching/SwitchingSupport.h"
#include "experimental/Transforms/Switching/ProfilingAnalyzer.h"


// This function propagates values from base nodes to all other nodes in the dfg
// circuit based on the execution trace *Special treatments needed for LOAD and
// SOTORE nodes
//  Load:
//    Output 1: to mem_con with the address value; Output 2: the actual data
//    output
//  COND_BR:
//    Need to keep track of the last valid update (the input data is not -1)
void dfgDataChannelPropagate(SwitchingInfo &switchInfo,
  SCFProfilingResult &profileResults, bool debug);
// This function will reduce the input integer value to the target bit width
int reduceBits(int value, unsigned targetBitWidth);

// This fucntion gets the cond_value of for the cond_br node for data channel
// update
int getCondBrNodeCondValue(SwitchingInfo &switchInfo, unsigned iterIndex,
  std::string nodeName);

// This function finds the src node of the specified node in the last segment(E)
std::string segENodeSrcSearch(SwitchingInfo &switchInfo, const std::string& nodeName,
SCFProfilingResult &profileResults);

// This function find the address src of the specifed mem_load unit, in segment
// "S" and "T"
std::string memAddrSrcSearch(SwitchingInfo &switchInfo,
  SCFProfilingResult &profileResults,
  const std::string& nodeName, const std::string& selSeg,
  unsigned iterIdx);


#endif