#ifndef DATACHANNELCAL_HPP
#define DATACHANNELCAL_HPP
#include "experimental/Transforms/Switching/SwitchingSupport.h"


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


#endif