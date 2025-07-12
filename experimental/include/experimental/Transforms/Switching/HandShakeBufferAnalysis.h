

#ifndef EXPERIMENTAL_TRANSFORMS_HANDSHAKE_BUFFER_H
#define EXPERIMENTAL_TRANSFORMS_HANDSHAKE_BUFFER_H
#include "experimental/Transforms/Switching/SwitchingSupport.h"

using namespace mlir;
using namespace dynamatic;
using namespace dynamatic::handshake;
// This function calculates the valid start time of different buffers in the specified cfdfc
// With this information, we can determine the Active time range of the ready and valid signal
// of different buffers.
// All info will be stored in the corresponding structure in the corresponding buffer node
void updateMGBufferSwitching(SwitchingInfo &switchInfo, std::string selMG, bool debug);


#endif