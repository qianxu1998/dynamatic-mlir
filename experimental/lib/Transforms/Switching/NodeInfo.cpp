//===- NodeInfo.cpp - Estimate Swithicng Activities ------*- C++ -*-===//
//
// This file declares all functions used for the Node Class
//
//===----------------------------------------------------------------------===//

#include "experimental/Transforms/Switching/NodeInfo.h"
#include "llvm/Support/Debug.h"
#include "experimental/Support/StdProfiler.h"
#include "dynamatic/Transforms/BufferPlacement/CFDFC.h"
#include "mlir/IR/Attributes.h"
#include "dynamatic/Dialect/Handshake/HandshakeAttributes.h"


//===----------------------------------------------------------------------===//
//
// Definitions of AdjNode
//
//===----------------------------------------------------------------------===//
AdjNode::AdjNode(mlir::Operation *selOp,
  const std::vector<std::string> &predecessors,
  const std::vector<std::string> &successors,
  const std::map<std::string, unsigned> &sucDataWidthMap,
  const unsigned &latency, const unsigned &bbIndex) {
// Initialize all variables
this->op = selOp;
this->pres = predecessors;
this->sucs = successors;
this->sucsDataWidthMap = sucDataWidthMap;
this->nodeLatency = latency;
this->bbindex = bbIndex;

// Initialize toggle count map
for (auto &[key, value] : sucsDataWidthMap) {
std::map<unsigned, unsigned> tmpMap;

for (unsigned i = 0; i < value; i++) {
tmpMap[i] = 0;
}

perChannelToggle[key] = tmpMap;
}
}

bool AdjNode::handshakeUpdateFinished() {
if (validSignal.size() == sucs.size()) {
if (readySignal.size() == pres.size()) {
if (setR.size() == pres.size()) {
if (setV.size() == sucs.size()) {
return true;
}
}
}
}

return false;
}

bool AdjNode::handshakeSwitchingChecking() {
if (validSignal.size() == sucs.size()) {
if (readySignal.size() == pres.size()) {
return true;
}
}

return false;
}

void AdjNode::totalHandshakeSwitchingUpdate() {
for (const auto &[s, selValue] : validSignal) {
totalValidSwitching += selValue;
}

for (const auto &[p, selValue] : readySignal) {
totalReadySwitching += selValue;
}
}

void AdjNode::updateDataoutChannel(int inputData) {
// For each successor, we do xor for the data init
for (const auto &suc : sucs) {
int diff = 0;

// Check whether the key is in dataOut or not
if (dataOut.find(suc) != dataOut.end()) {
auto &hist = dataOut[suc];
diff = hist.back() ^ inputData;
hist.push_back(inputData);
} else {
diff = inputData;
std::vector<int> tmpVec = {inputData};
dataOut[suc] = tmpVec;
}

// Update the per_channel count
auto positions = getPositionList(diff);

for (const auto &selPos : positions) {
if (perChannelToggle[suc].find(selPos) != perChannelToggle[suc].end()) {
perChannelToggle[suc][selPos] += 1;
}
}
}
}

void AdjNode::updateHandshakeChannelSwitching(unsigned validChannelSwitching,
                               unsigned readyChannelSwitching) {
totalValidSwitching += validChannelSwitching;
totalReadySwitching += readyChannelSwitching;

// Update the handshake status as well
handshakeUpdateFlag = true;
}

void AdjNode::totalDataSwitchingCounting(bool mapped) {
  // Reset total before recalculating
  totalDataSwitching = 0;
  
  for (const auto &[suc, valueVec] : dataOut) {
    // Define tmp storing structure
    unsigned numSwitches = 0;

    // If this suc is one of the units at scf level
    int lastInput = mapped && valueVec[0] == -1 ? 1 : valueVec[0];

    // The corresponding output channel maynot have dataout
    // Check the validity of the data channel value
    for (unsigned i = 0; i < valueVec.size(); i++) {
      int curVal = (valueVec[i] == -1) ? 1 : valueVec[i];
      int diff = lastInput ^ curVal;

      // Count bits
      unsigned bitCount = 0;
      while (diff) {
        bitCount += (diff & 1);
        diff >>= 1;
      }
      numSwitches += bitCount;
      lastInput = curVal;
    }

    // Store the total number of channel switches
    dataSwitches[suc] = numSwitches;
    totalDataSwitching += numSwitches;
  }
}

std::vector<unsigned> AdjNode::getPositionList(int number) {
  std::vector<unsigned> positions;
  unsigned idx = 0;
  int tmp = number;
  while (tmp) {
    if (tmp & 1)
      positions.push_back(idx);
    tmp >>= 1;
    idx++;
}

return positions;
}

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



// Define all the printing functions to faciliate debug
void AdjNode::printNodeDetails() {
llvm::dbgs()
<< "[DEBUG] "
"\t=============================================================\n";
llvm::dbgs() << "[DEBUG] \t[Node Info Start]\n";

//
llvm::dbgs() << "[DEBUG] \t\tLatency: " << nodeLatency << ";\n";
llvm::dbgs() << "[DEBUG] \t\tPredecessors: [";
for (const auto &p : pres) {
llvm::dbgs() << p << " ";
}
llvm::dbgs() << "]\n[DEBUG] \t\tSuccessors: [";
for (const auto &s : sucs) {
llvm::dbgs() << s << " ";
}
llvm::dbgs() << "]\n";

llvm::dbgs() << "[DEBUG] \t\tSuccessor Channel DataWidth: \n";
for (const auto &[s, width] : sucsDataWidthMap) {
llvm::dbgs() << "[DEBUG] \t\t\tNode: " << s << ", Data_width: " << width
  << "\n";
}

llvm::dbgs() << "[DEBUG] \t\tSuccessor Channel Data Value\n";
for (const auto &[s, valueVec] : dataOut) {
llvm::dbgs() << "[DEBUG] \t\t\tNode: " << s << ", Output Value Vector: \n";
printVector(valueVec);
llvm::dbgs() << "\n";
}

llvm::dbgs() << "[DEBUG] \t\tValid Channel Switching: \n";
for (const auto &[s, numSwitches] : validSignal) {
llvm::dbgs() << "[DEBUG] \t\t\tNode: " << s
  << ", Data_width: " << numSwitches << "\n";
}

llvm::dbgs() << "[DEBUG] \t\tReady Channel Switching: \n";
for (const auto &[s, numSwitches] : readySignal) {
llvm::dbgs() << "[DEBUG] \t\t\tNode: " << s
  << ", Data_width: " << numSwitches << "\n";
}

llvm::dbgs() << "[DEBUG] \t\tValid Active Range: \n";
for (const auto &[s, valueVec] : setV) {
llvm::dbgs() << "[DEBUG] \t\t\tNode: " << s << " ; Active Range: [";
for (const auto &selValue : valueVec.set_bits() ) {
llvm::dbgs() << selValue << " ";
}
llvm::dbgs() << "]\n";
}

llvm::dbgs() << "[DEBUG] \t\tReady Active Range: \n";
for (const auto &[s, valueVec] : setR) {
llvm::dbgs() << "[DEBUG] \t\t\tNode: " << s << " ; Active Range: [";
for (const auto &selValue : valueVec.set_bits()) {
llvm::dbgs() << selValue << " ";
}
llvm::dbgs() << "]\n";
}
}

void AdjNode::printHandshakeSwitching() {
llvm::dbgs() << "[DEBUG] \t\tTotal Valid Switching Number: "
<< totalValidSwitching << "\n";
llvm::dbgs() << "[DEBUG] \t\tTotal Ready Switching Number: "
<< totalReadySwitching << "\n";

llvm::dbgs() << "[DEBUG] \t\tValid Channel Switching: \n";
for (const auto &[s, numSwitches] : validSignal) {
llvm::dbgs() << "[DEBUG] \t\t\tNode: " << s
  << ", Data_width: " << numSwitches << "\n";
}

llvm::dbgs() << "[DEBUG] \t\tReady Channel Switching: \n";
for (const auto &[s, numSwitches] : readySignal) {
llvm::dbgs() << "[DEBUG] [DEBUG] \t\t\t\tNode: " << s
  << ", Data_width: " << numSwitches << "\n";
}

llvm::dbgs() << "[DEBUG] \t\tValid Active Range: \n";
for (const auto &[s, valueVec] : setV) {
llvm::dbgs() << "[DEBUG] \t\t\tNode: " << s << " ; Active Range: [";
for (const auto &selValue : valueVec.set_bits()) {
llvm::dbgs() << selValue << " ";
}
llvm::dbgs() << "]\n";
}

llvm::dbgs() << "[DEBUG] \t\tReady Active Range: \n";
for (const auto &[s, valueVec] : setR) {
llvm::dbgs() << "[DEBUG] \t\t\tNode: " << s << " ; Active Range: [";
for (const auto &selValue : valueVec.set_bits()) {
llvm::dbgs() << selValue << " ";
}
llvm::dbgs() << "]\n";
}
}

void AdjNode::printDataChannelSwitching() {
llvm::dbgs() << "[DEBUG] \t\tTotal Number of Data Channel Switches: "
<< totalDataSwitching << "\n";

// Print per channel data switches
for (const auto &[suc, value] : dataSwitches) {
llvm::dbgs() << "[DEBUG] \t\t\tChannel: " << suc << ": " << value << "\n";
}
}

void AdjNode::printPerDataChannelToggleNumber() {
llvm::dbgs() << "[DEBUG] \t\tData Channel Per BIT Switches: \n";

for (const auto &[suc, valueVec] : perChannelToggle) {
llvm::dbgs() << "[DEBUG] \t\t Node: " << suc << "\n";

for (const auto &[selBit, value] : valueVec) {
llvm::dbgs() << "[DEBUG] \t\t\tBit " << selBit << ": " << value << "\n";
}
}
}

void AdjNode::printPerHandshakeChannelToggleNumber() {
llvm::dbgs() << "[DEBUG] \t\t[VALID CHANNEL]\n";
for (const auto &[suc, validSwitch] : validSignal) {
llvm::dbgs() << "[DEBUG] \t\t\tNode: " << suc
  << ", Valid Switching: " << validSwitch << "\n";
}

llvm::dbgs() << "[DEBUG] \t\t[READY CHANNEL]\n";
for (const auto &[pre, readySwitching] : readySignal) {
llvm::dbgs() << "[DEBUG] \t\t\tNode: " << pre
  << ", Ready Switching: " << readySwitching << "\n";
}
}

