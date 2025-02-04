//===- ExecModel.cpp - Estimate Swithicng Activities ------*- C++ -*-===//
//
// Implements all the execution models for different types of nodes
//
//===----------------------------------------------------------------------===//

#include "experimental/Transforms/Switching/ExecModel.h"

// Helper: Build a full set {0, 1, ..., II-1}
static std::set<unsigned> makeFullSet(unsigned II) {
  std::set<unsigned> fullSet;
  for (unsigned i = 0; i < II; ++i)
    fullSet.insert(i);
  return fullSet;
}

//===----------------------------------------------------------------------===//
//
// Buffer Node
//
//===----------------------------------------------------------------------===//
BufferNode::BufferNode(mlir::Operation *op,
                       const std::vector<std::string> &predecessors,
                       const std::vector<std::string> &successors,
                       const std::map<std::string, unsigned> &sucDataWidthMap,
                       const unsigned &latency)
    : AdjNode(op, predecessors, successors, sucDataWidthMap, latency),
      START(0), occupancy(0.0), numSlots(0), transparent(false) {}


void BufferNode::calValidSwitching(const std::string &sucNodeName, unsigned II) {
  if (setV.find(sucNodeName) != setV.end()) {
    if (setV[sucNodeName].size() >= II) {
      validSignal[sucNodeName] = 0;
    } else {
      validSignal[sucNodeName] = 2;
    }
  }
}

void BufferNode::calReadySwitching(const std::string &preNodeName, unsigned II) {
  if (setR.find(preNodeName) != setR.end()) {
    if (setR[preNodeName].size() >= II) {
      readySignal[preNodeName] = 0;
    } else {
      readySignal[preNodeName] = 2;
    }
  }
}

void BufferNode::calValidSet(const std::string &sucNodeName, std::set<unsigned> &inSetV) {
  setV[sucNodeName] = inSetV;
}

void BufferNode::calReadySet(const std::string &preNodeName, std::set<unsigned> &inSetR) {
  setR[preNodeName] = inSetR;
}

void BufferNode::printDetail() {
  // Call base class's printDetail()
  AdjNode::printDetail();

  llvm::dbgs() << "[DEBUG] \t\tSTART: " << START <<";\n";
  llvm::dbgs() << "[DEBUG] \t\tOccupancy: " << occupancy <<";\n";
  llvm::dbgs() << "[DEBUG] \t\tNumSlots: " << numSlots <<";\n";
  llvm::dbgs() << "[DEBUG] \t\ttransparent: " << transparent <<";\n";
}

//===----------------------------------------------------------------------===//
//
// Join Node
//
//===----------------------------------------------------------------------===//
void JoinNode::calValidSwitching(const std::string &sucNodeName,
                                 unsigned &numValid0,
                                 unsigned &numValid1,
                                 unsigned &II) {
  if (numValid0 > 0 || numValid1 > 0) {
    validSignal[sucNodeName] = 2;
  } else if (numValid0 == 0 && numValid1 == 0) {
    validSignal[sucNodeName] = 0;
  } else if (II == 1) {
    validSignal[sucNodeName] = 0;
  } else {
    validSignal[sucNodeName] = 2;
  }
}

void JoinNode::calValidSet(const std::string &sucNodeName,
                           unsigned &nodeStartTime,
                           unsigned &II) {
  // Check if validSignal contains sucNodeName.
  if (validSignal.find(sucNodeName) != validSignal.end()) {
    if (validSignal[sucNodeName] == 0 || II == 1) {
      setV[sucNodeName] = makeFullSet(II);
    } else {
      std::set<unsigned> tmp;
      tmp.insert(nodeStartTime);
      setV[sucNodeName] = tmp;
    }
  }
}

void JoinNode::calReadySwitching(const std::string &preNodeName,
                                 unsigned &numValid,
                                 unsigned &numReady) {
  if (numValid == 0 && numReady == 0) {
    readySignal[preNodeName] = 0;
  } else if (numValid > 0 || numReady > 0) {
    readySignal[preNodeName] = 2;
  }
}

void JoinNode::calReadySet(const std::string &preNodeName,
                           const std::set<unsigned> *setValid,
                           const std::set<unsigned> *setReady,
                           unsigned &nodeStartTime,
                           unsigned &II) {
  if (setValid != nullptr && setReady != nullptr) {
    std::set<unsigned> finalSet;
    std::set_intersection(setValid->begin(), setValid->end(),
                          setReady->begin(), setReady->end(),
                          std::inserter(finalSet, finalSet.begin()));
    setR[preNodeName] = finalSet;
  } else if (readySignal.find(preNodeName) != readySignal.end()) {
    if (readySignal[preNodeName] == 0) {
      setR[preNodeName] = makeFullSet(II);
    } else if (readySignal[preNodeName] > 0) {
      std::set<unsigned> tmp;
      tmp.insert(nodeStartTime);
      setR[preNodeName] = tmp;
    }
  }
}

void JoinNode::setReadySet(const std::string &preNodeName) {
  // If setVMap is not empty, take the first available set and copy it to setRMap.
  if (!setV.empty()) {
    auto it = setV.begin();
    setR[preNodeName] = it->second;
  }
}

//===----------------------------------------------------------------------===//
//
// Pass Node
//
//===----------------------------------------------------------------------===//
void PassNode::calValidSwitching(const std::string &sucNodeName,
                                 unsigned &numValid) {
  if (numValid == 0)
    validSignal[sucNodeName] = 0;
  else
    validSignal[sucNodeName] = 2;
}

void PassNode::calValidSet(const std::string &sucNodeName,
                           unsigned &nodeStartTime,
                           unsigned &II) {
  if (validSignal.find(sucNodeName) != validSignal.end()) {
    if (validSignal[sucNodeName] == 0) {
      setV[sucNodeName] = makeFullSet(II);
    } else {
      std::set<unsigned> tmp;
      tmp.insert(nodeStartTime);
      setV[sucNodeName] = tmp;
    }
  }
}

void PassNode::calReadySwitching(const std::string &preNodeName,
                                 unsigned &numReady) {
  if (numReady == 0)
    readySignal[preNodeName] = 0;
  else
    readySignal[preNodeName] = 2;
}

void PassNode::calReadySet(const std::string &preNodeName,
                           const std::set<unsigned> *setReady,
                           unsigned &nodeStartTime,
                           unsigned &II) {
  if (setReady != nullptr) {
    setR[preNodeName] = *setReady;
  } else if (readySignal.find(preNodeName) != readySignal.end()) {
    if (readySignal[preNodeName] == 0) {
      setR[preNodeName] = makeFullSet(II);
    } else if (readySignal[preNodeName] > 0) {
      std::set<unsigned> tmp;
      tmp.insert(nodeStartTime);
      setR[preNodeName] = tmp;
    } else if (!setV.empty()) {
      // Fallback: copy the set from the first key in setVMap if it is not the full set.
      auto it = setV.begin();
      std::set<unsigned> fullSet;
      for (unsigned i = 0; i < II; ++i)
        fullSet.insert(i);
      if (it->second != fullSet)
        setR[preNodeName] = it->second;
    }
  }
}

void PassNode::setReadySwitching(const std::string &preNodeName,
                                 unsigned &numReadySwitches) {
  readySignal[preNodeName] = numReadySwitches;
}

void PassNode::setReadySet(const std::string &preNodeName,
                           const std::set<unsigned> &setReady) {
  setR[preNodeName] = setReady;
}

//===----------------------------------------------------------------------===//
//
// Load Node
//
//===----------------------------------------------------------------------===//
DLoadNode::DLoadNode(mlir::Operation *op,
                     const std::vector<std::string> &predecessors,
                     const std::vector<std::string> &successors,
                     const std::map<std::string, unsigned> &sucDataWidthMap,
                     const unsigned &latency)
  : AdjNode(op, predecessors, successors, sucDataWidthMap, latency) {
  // Get the LoadOp
  auto loadOp = dyn_cast<handshake::LoadOp>(op);
  auto dataOutRes = loadOp.getDataResult();
  auto addrOutRes = loadOp.getAddressResult();
  auto addrInRes = loadOp.getAddressInput();

  dataOutNodeName = getHandshakeNodeName(dataOutRes);
  addressOutNodeName = getHandshakeNodeName(addrOutRes);
  addressInNodeName = addrInRes.getDefiningOp()->getAttrOfType<mlir::StringAttr>("handshake.name").getValue().str();
}

void DLoadNode::calValidSwitching(const std::string &sucNodeName, unsigned &numValid) {
  if (numValid == 0)
    validSignal[sucNodeName] = 0;
  else
    validSignal[sucNodeName] = 2;
}

void DLoadNode::calValidSet(const std::string &sucNodeName, const std::set<unsigned> *inSetV, unsigned &II) {
  if (validSignal.find(sucNodeName) != validSignal.end()) {
    if (validSignal[sucNodeName] == 0) {
      setV[sucNodeName] = makeFullSet(II);
    } else if (inSetV != nullptr) {
      std::set<unsigned> finalSet;
      for (unsigned v : *inSetV)
        finalSet.insert((v + nodeLatency) % II);
      setV[sucNodeName] = finalSet;
    }
  }
}

void DLoadNode::calReadySwitching(const std::string &preNodeName, unsigned &numReady) {
  if (numReady == 0)
    readySignal[preNodeName] = 0;
  else
    readySignal[preNodeName] = 2;
}

void DLoadNode::calReadySet(const std::string &preNodeName, const std::set<unsigned> *inSetR, unsigned &nodeStartTime, unsigned &II) {
  if (inSetR != nullptr) {
    setR[preNodeName] = *inSetR;
  } else if (readySignal.find(preNodeName) != readySignal.end()) {
    if (readySignal[preNodeName] == 0) {
      setR[preNodeName] = makeFullSet(II);
    } else if (readySignal[preNodeName] > 0) {
      std::set<unsigned> tmp;
      tmp.insert(nodeStartTime);
      setR[preNodeName] = tmp;
    }
  }
}

void DLoadNode::updateDataout(int &inputData, int addressValue) {
  // --- Update data output channel ---
  int dataDiff = 0;
  if (dataOut.find(dataOutNodeName) != dataOut.end() && !dataOut[dataOutNodeName].empty()) {
    dataDiff = dataOut[dataOutNodeName].back() ^ inputData;
    dataOut[dataOutNodeName].push_back(inputData);
  } else {
    dataDiff = inputData;
    dataOut[dataOutNodeName] = { inputData };
  }
  // Update per-channel toggle counts (using getPositionList() from AdjNode)
  auto dataPositions = getPositionList(dataDiff);
  for (unsigned pos : dataPositions) {
    if (perChannelToggle[dataOutNodeName].find(pos) != perChannelToggle[dataOutNodeName].end())
      perChannelToggle[dataOutNodeName][pos]++;
  }

  // --- Update address output channel ---
  int addressDiff = 0;
  if (dataOut.find(addressOutNodeName) != dataOut.end() && !dataOut[addressOutNodeName].empty()) {
    if (addressValue == -10)
      addressValue = dataOut[addressOutNodeName].back();
    addressDiff = dataOut[addressOutNodeName].back() ^ addressValue;
    dataOut[addressOutNodeName].push_back(addressValue);
  } else {
    addressDiff = addressValue;
    dataOut[addressOutNodeName] = { addressValue };
  }
  auto addrPositions = getPositionList(addressDiff);
  for (unsigned pos : addrPositions) {
    if (perChannelToggle[addressOutNodeName].find(pos) != perChannelToggle[addressOutNodeName].end())
      perChannelToggle[addressOutNodeName][pos]++;
  }
}

void DLoadNode::printDetail() {
  // Call base class's printDetail()
  AdjNode::printDetail();

  llvm::dbgs() << "[DEBUG] \t\tAddress input node: " << addressInNodeName <<";\n";
  llvm::dbgs() << "[DEBUG] \t\tAddress output node: " << addressOutNodeName <<";\n";
  llvm::dbgs() << "[DEBUG] \t\tData output node: " << dataOutNodeName <<";\n";
}

//===----------------------------------------------------------------------===//
//
// Store Node
//
//===----------------------------------------------------------------------===//
// TODO: Seperate MC_store and LSQ_store
DStoreNode::DStoreNode(mlir::Operation *op,
                       const std::vector<std::string> &predecessors,
                       const std::vector<std::string> &successors,
                       const std::map<std::string, unsigned> &sucDataWidthMap,
                       unsigned latency)
    : AdjNode(op, predecessors, successors, sucDataWidthMap, latency) {
  // Add the data channel to the mem_controller
  unsigned dataWidth = 0;
  for (const auto& selPair : sucDataWidthMap) {
    dataWidth = selPair.second;
    break;
  }
  sucsDataWidthMap["mem_data"] = dataWidth;
  
  // Get the StoreOp
  auto storeOp = dyn_cast<handshake::StoreOp>(op);
  auto dataInValue = storeOp.getDataInput();
  auto addrInNodeValue = storeOp.getAddressInput();

  dataInNode = dataInValue.getDefiningOp()->getAttrOfType<mlir::StringAttr>("handshake.name").getValue().str();
  addressInNode = addrInNodeValue.getDefiningOp()->getAttrOfType<mlir::StringAttr>("handshake.name").getValue().str();
  
  // The "source" nodes for these inputs are to be set externally.
  dataInSrcNode = "";
  addressInSrcNode = "";
}

void DStoreNode::calValidSwitching(unsigned numValid1, unsigned numValid2) {
  // For the memory controller channel, use key "mc".
  if (numValid1 > 0 || numValid2 > 0)
    validSignal["mem"] = 4;
  else if (numValid1 == 0 && numValid2 == 0)
    validSignal["mem"] = 0;
  else
    validSignal["mem"] = 4;
}

void DStoreNode::calValidSet(const std::set<unsigned> *setV0, const std::set<unsigned> *setV1, unsigned II) {
  // TODO: Need to differentiate between addrToMem and dataToMem handshake channels
  if (setV0 != nullptr && setV1 != nullptr) {
    std::set<unsigned> tmp;
    std::set_intersection(setV0->begin(), setV0->end(),
                          setV1->begin(), setV1->end(),
                          std::inserter(tmp, tmp.begin()));
    std::set<unsigned> finalSet;
    for (unsigned val : tmp)
      finalSet.insert((val + nodeLatency) % II);
    setV["mem"] = finalSet;
  } else if (validSignal.find("mem") != validSignal.end() && validSignal["mem"] == 0) {
    setV["mem"] = makeFullSet(II);
  }
}

void DStoreNode::calReadySwitching(const std::string &preNodeName, unsigned numValid1, unsigned numValid2) {
  // TODO: May need to change the following modeling as the implementation changed
  if (numValid1 > 0 || numValid2 > 0)
    readySignal[preNodeName] = 2;
  else if (numValid1 == 0 && numValid2 == 0)
    readySignal[preNodeName] = 0;
  else
    readySignal[preNodeName] = 2;
}

void DStoreNode::calReadySet(const std::string &preNodeName, const std::set<unsigned> *setV0, const std::set<unsigned> *setV1, unsigned II) {
  // TODO: Validate the following estimation based on new implementations
  if (setV0 != nullptr && setV1 != nullptr) {
    std::set<unsigned> tmp;
    std::set_intersection(setV0->begin(), setV0->end(),
                          setV1->begin(), setV1->end(),
                          std::inserter(tmp, tmp.begin()));
    std::set<unsigned> finalSet;
    for (unsigned val : tmp)
      finalSet.insert((val + nodeLatency) % II);
    // Here we store the result into setV for preNodeName.
    setV[preNodeName] = finalSet;
  } else if (validSignal.find(preNodeName) != validSignal.end() && validSignal[preNodeName] == 0) {
    setV[preNodeName] = makeFullSet(II);
  }
}

void DStoreNode::updateDataout(int inputData, const std::string &srcInputNode) {
  // The method assumes that dataInSrcNode and addressInSrcNode have been set.
  assert(!dataInSrcNode.empty());
  assert(!addressInSrcNode.empty());
  
  int valueDiff = 0;
  if (srcInputNode == addressInSrcNode) {
    // Update the address output channel (use key "address_out")
    if (dataOut.find("address_out") != dataOut.end() && !dataOut["address_out"].empty()) {
      valueDiff = dataOut["address_out"].back() ^ inputData;
      dataOut["address_out"].push_back(inputData);
    } else {
      valueDiff = inputData;
      dataOut["address_out"] = { inputData };
    }
  } else if (srcInputNode == dataInSrcNode) {
    // Update the actual data output channel (use key "data_out")
    if (dataOut.find("data_out") != dataOut.end() && !dataOut["data_out"].empty()) {
      valueDiff = dataOut["data_out"].back() ^ inputData;
      dataOut["data_out"].push_back(inputData);
    } else {
      valueDiff = inputData;
      dataOut["data_out"] = { inputData };
    }
  }
  // (Optional: update perChannelToggle accordingly.)
}

void DStoreNode::printDetail() {
  // Call the base class printDetail for common node info.
  AdjNode::printDetail();

  llvm::dbgs() << "[DEBUG] \t\tAddress input node: " << addressInNode << "\n";
  llvm::dbgs() << "[DEBUG] \t\tSrc address input node: " << addressInSrcNode << "\n";
  llvm::dbgs() << "[DEBUG] \t\tData input node: " << dataInNode << "\n";
  llvm::dbgs() << "[DEBUG] \t\tSrc data input node: " << dataInSrcNode << "\n";
}

bool DStoreNode::handshakeSwitchingChecking() {
  // For example, if the number of ready channels equals the number of predecessors.
  return (readySignal.size() == pres.size());
}

//===----------------------------------------------------------------------===//
//
// Merge Node
//
//===----------------------------------------------------------------------===//

void MergeNode::calValidSwitching(const std::string &sucNodeName,
                                  const std::vector<std::set<unsigned>> &setVList,
                                  const std::vector<unsigned> &numVList,
                                  unsigned II) {
  if (numVList.size() == 1) {
    if (numVList[0] == 0)
      validSignal[sucNodeName] = 0;
    else if (numVList[0] > 0)
      validSignal[sucNodeName] = 2;
  } else {
    if (numVList[0] == 0)
      validSignal[sucNodeName] = 0;
    else if (numVList[1] == 0)
      validSignal[sucNodeName] = 0;
    else if (setVList.size() >= 2) {
      // Check that both input sets are provided (we assume a non-empty set represents a valid set)
      if (!setVList[0].empty() && !setVList[1].empty()) {
        // Compute the union of setVList[0] and setVList[1]
        std::set<unsigned> unionSet;
        std::set_union(setVList[0].begin(), setVList[0].end(),
                       setVList[1].begin(), setVList[1].end(),
                       std::inserter(unionSet, unionSet.begin()));
        // Build the full set: {0, 1, ..., II-1}
        std::set<unsigned> fullSet;
        for (unsigned i = 0; i < II; ++i)
          fullSet.insert(i);
        if (unionSet == fullSet)
          validSignal[sucNodeName] = 0;
      } else if (numVList[0] > 0 && numVList[1] > 0) {
        validSignal[sucNodeName] = 2;
      }
    }
  }
}

void MergeNode::calValidSet(const std::string &sucNodeName,
                            const std::vector<std::set<unsigned>> &setVList,
                            unsigned II) {
  if (setVList.size() == 1) {
    // One input channel
    if (validSignal.find(sucNodeName) != validSignal.end() &&
        validSignal[sucNodeName] == 0) {
      setV[sucNodeName] = makeFullSet(II);
    } else if (!setVList[0].empty()) {
      setV[sucNodeName] = setVList[0];
    }
  } else {
    if (validSignal.find(sucNodeName) != validSignal.end() &&
        validSignal[sucNodeName] == 0) {
      setV[sucNodeName] = makeFullSet(II);
    } else if (setVList.size() >= 2 && !setVList[0].empty() && !setVList[1].empty()) {
      // Compute the union: in C++ we use std::set_union.
      std::set<unsigned> unionSet;
      std::set_union(setVList[0].begin(), setVList[0].end(),
                     setVList[1].begin(), setVList[1].end(),
                     std::inserter(unionSet, unionSet.begin()));
      setV[sucNodeName] = unionSet;
    }
  }
}

void MergeNode::calReadySwitching(const std::string &preNodeName,
                                  unsigned numReady) {
  if (numReady == 0)
    readySignal[preNodeName] = 0;
  else if (numReady > 0)
    readySignal[preNodeName] = 2;
}

void MergeNode::calReadySet(const std::string &preNodeName,
                            const std::set<unsigned> *setRPtr,
                            unsigned II) {
  if (setRPtr != nullptr) {
    setR[preNodeName] = *setRPtr;
  } else if (readySignal.find(preNodeName) != readySignal.end()) {
    if (readySignal[preNodeName] == 0) {
      setR[preNodeName] = makeFullSet(II);
    }
  }
}

//===----------------------------------------------------------------------===//
//
// CMerge Node
//
//===----------------------------------------------------------------------===//
CMergeNode::CMergeNode(mlir::Operation *op,
                       const std::vector<std::string> &predecessors,
                       const std::vector<std::string> &successors,
                       const std::map<std::string, unsigned> &sucDataWidthMap,
                       unsigned latency)
  : AdjNode(op, predecessors, successors, sucDataWidthMap, latency) {

  // Determine the control and data channel successor names.
  // We assume that the mapping sucDataWidthMap associates each successor name with a port number.
  conSucNodeName = "";
  dataSucNodeName = "";
  for (const auto &pair : sucDataWidthMap) {
    if (pair.second == 1) {
      conSucNodeName = pair.first;
      break;
    }
  }

  // For control merge node, we don't have actual data output
  for (const auto& selNode : successors) {
    if (selNode != conSucNodeName) dataSucNodeName = selNode;
  }
}

void CMergeNode::calValidSwitching(const std::string &sucNodeName, unsigned II) {
  if (II == 1)
    validSignal[sucNodeName] = 0;
  else
    validSignal[sucNodeName] = 2;
}

void CMergeNode::calValidSet(const std::string &sucNodeName, unsigned nodeStartTime, unsigned II) {
  if (validSignal[sucNodeName] == 0) {
    setV[sucNodeName] = makeFullSet(II);
  } else {
    std::set<unsigned> tmp;
    tmp.insert(nodeStartTime);
    setV[sucNodeName] = tmp;
  }
}

void CMergeNode::calReadySwitching(const std::string &preNodeName) {
  readySignal[preNodeName] = 0;
}

void CMergeNode::calReadySet(const std::string &preNodeName, unsigned II) {
  setR[preNodeName] = makeFullSet(II);
}

void CMergeNode::calDataout(int inputValue) {
  // Update the control channel:
  if (dataOut.find(conSucNodeName) == dataOut.end() || dataOut[conSucNodeName].empty()) {
    dataOut[conSucNodeName] = { inputValue };
  } else {
    dataOut[conSucNodeName].push_back(inputValue);
  }
  // For the data channel, simply set its output to [-1]
  dataOut[dataSucNodeName] = { -1 };
}

//
// updateDataout:
//   For each successor in the node’s successor list, if it is the condition channel,
//   update using an XOR difference between the last value and the new input and update 
//   per-channel toggle counts; otherwise, update the data channel with -1.
//
void CMergeNode::updateDataout(int inputData) {
  int valueDiff = 0;
  // Iterate over the successor names (we assume 'sucs' is available from AdjNode as a vector<string>)
  for (const std::string &suc : sucs) {
    // Check if there is already data for this successor.
    if (dataOut.find(suc) != dataOut.end() && !dataOut[suc].empty()) {
      if (suc == conSucNodeName) {
        valueDiff = dataOut[suc].back() ^ inputData;
        dataOut[suc].push_back(inputData);
        // Get positions of set bits in the difference (using the inherited getPositionList)
        std::vector<unsigned> posList = getPositionList(valueDiff);
        for (unsigned pos : posList) {
          if (perChannelToggle[suc].find(pos) != perChannelToggle[suc].end())
            perChannelToggle[suc][pos]++;
        }
      } else {
        dataOut[suc].push_back(-1);
      }
    } else {
      if (suc == conSucNodeName) {
        valueDiff = inputData;
        dataOut[suc] = { inputData };
        std::vector<unsigned> posList = getPositionList(valueDiff);
        for (unsigned pos : posList) {
          if (perChannelToggle[suc].find(pos) != perChannelToggle[suc].end())
            perChannelToggle[suc][pos]++;
        }
      } else {
        dataOut[suc] = { -1 };
      }
    }
  }
}

void CMergeNode::printDetail() {
  AdjNode::printDetail();

  llvm::dbgs() << "[DEBUG] \t\tCon_suc_node_name: " << conSucNodeName << "\n";
  llvm::dbgs() << "[DEBUG] \t\tData_suc_node_name: " << dataSucNodeName << "\n";
}

//===----------------------------------------------------------------------===//
//
// Lazy fork Node
//
//===----------------------------------------------------------------------===//


//===----------------------------------------------------------------------===//
//
// Fork Node
//
//===----------------------------------------------------------------------===//

void ForkNode::calValidSwitching(const std::string &sucNodeName,
                                 unsigned numValid,
                                 const std::map<std::string, std::set<unsigned>> &setRDict,
                                 const std::map<std::string, unsigned> &numReadyDict,
                                 unsigned sucNodeStart,
                                 unsigned nodeSteadyStart,
                                 unsigned II) {
  // If any valid switching occurs at the input, the fork must toggle.
  if (numValid > 0) {
    validSignal[sucNodeName] = 2;
    return;
  }

  // Case 1: Check if all ready counts are 0.
  bool allReadyZero = true;
  for (const auto &entry : numReadyDict) {
    if (entry.second != 0) {
      allReadyZero = false;
      break;
    }
  }
  if (allReadyZero) {
    validSignal[sucNodeName] = 0;
    return;
  }

  // Case 2: Check the active ready set for sucNodeName.
  unsigned selStartPoint = 0;
  auto it = setRDict.find(sucNodeName);
  if (it != setRDict.end()) {
    const std::set<unsigned> &s = it->second;
    if (!s.empty()) {
      if (s.size() == 1)
        selStartPoint = *s.begin();
      else {
        validSignal[sucNodeName] = 2;
        return;
      }
    } else {
      selStartPoint = 0;
    }
  } else {
    selStartPoint = 0;
  }

  bool flag = true;
  bool existFlag = true;
  // For each entry in setRDict (other than sucNodeName), check if selStartPoint is contained.
  for (const auto &entry : setRDict) {
    if (entry.first != sucNodeName) {
      const std::set<unsigned> &s = entry.second;
      // In our design, an empty set is equivalent to None.
      if (s.empty()) {
        flag = false;
        existFlag = false;
      } else if (s.find(selStartPoint) == s.end()) {
        flag = false;
      }
    }
  }

  if (flag && existFlag) {
    validSignal[sucNodeName] = 0;
    return;
  } else {
    if (existFlag) {
      // Case 3: Compute desired cycle time.
      unsigned desiredCycleTime = (nodeSteadyStart != 0) ? (nodeSteadyStart - 1) : (II - 1);
      if (selStartPoint == 0)
        validSignal[sucNodeName] = 0;
      else
        validSignal[sucNodeName] = 2;
    }
  }
}

void ForkNode::calValidSet(const std::string &sucNodeName,
                           unsigned nodeStartTime,
                           unsigned numValid,
                           const std::map<std::string, std::set<unsigned>> &setRDict,
                           unsigned II) {
  if (validSignal.find(sucNodeName) != validSignal.end()) {
    if (validSignal[sucNodeName] == 0) {
      setV[sucNodeName] = makeFullSet(II);
    } else {
      if (numValid > 0) {
        std::set<unsigned> tmp;
        tmp.insert(nodeStartTime);
        setV[sucNodeName] = tmp;
      } else {
        auto it = setRDict.find(sucNodeName);
        if (it != setRDict.end())
          setV[sucNodeName] = it->second;
      }
    }
  }
}

void ForkNode::calReadySwitching(const std::string &preNodeName,
                                 const std::vector<unsigned> &numReadyList) {
  // If any value in numReadyList is greater than 0, set ready signal to 2 and return.
  for (unsigned value : numReadyList) {
    if (value > 0) {
      readySignal[preNodeName] = 2;
      return;
    }
  }
  // Otherwise, if all values are 0, set ready signal to 0.
  readySignal[preNodeName] = 0;
}

void ForkNode::calReadySet(const std::string &preNodeName,
                           const std::map<std::string, std::set<unsigned>> &setRDict,
                           unsigned II) {
  if (readySignal.find(preNodeName) != readySignal.end()) {
    if (readySignal[preNodeName] == 0) {
      setR[preNodeName] = makeFullSet(II);
    } else {
      // Check existence of all nReady sets for successors.
      bool exist = true;
      std::set<unsigned> unionSet;
      // Start with an empty union.
      for (const std::string &key : sucs) {
        auto it = setRDict.find(key);
        if (it != setRDict.end()) {
          const std::set<unsigned> &s = it->second;
          if (s.empty()) {
            exist = false;
            unionSet.clear(); // intersection with an empty set remains empty
          } else {
            // Compute the union: unionSet = unionSet ∪ s.
            std::set<unsigned> temp;
            std::set_union(unionSet.begin(), unionSet.end(),
                           s.begin(), s.end(),
                           std::inserter(temp, temp.begin()));
            unionSet = temp;
          }
        }
      }
      if (exist) {
        std::set<unsigned> fullSet;
        for (unsigned i = 0; i < II; ++i)
          fullSet.insert(i);
        if (unionSet == fullSet)
          setR[preNodeName] = {0};
        else {
          // Retrieve the last element of unionSet.
          unsigned last = 0;
          if (!unionSet.empty())
            last = *unionSet.rbegin();
          setR[preNodeName] = {last};
        }
      }
    }
  }
}

//===----------------------------------------------------------------------===//
//
// CBr Node
//
//===----------------------------------------------------------------------===//
CBrNode::CBrNode(mlir::Operation *op,
                 const std::vector<std::string> &predecessors,
                 const std::vector<std::string> &successors,
                 const std::map<std::string, unsigned> &sucDataWidthMap,
                 unsigned latency)
    : AdjNode(op, predecessors, successors, sucDataWidthMap, latency),
      lastValidDataValue(-1)
{
  // Get the ConditionalBranch node
  auto CBrOp = dyn_cast<handshake::ConditionalBranchOp>(op);
  auto condPreOperand = CBrOp.getConditionOperand();
  auto dataPreOperand = CBrOp.getDataOperand();
  auto trueSucOperand = CBrOp.getTrueResult();
  auto falseSucOperand = CBrOp.getFalseResult();

  condPreNodeName = condPreOperand.getDefiningOp()->getAttrOfType<mlir::StringAttr>("handshake.name").getValue().str();
  dataPreNodeName = dataPreOperand.getDefiningOp()->getAttrOfType<mlir::StringAttr>("handshake.name").getValue().str();
  trueSucNodeName = getHandshakeNodeName(trueSucOperand);
  falseSucNodeName = getHandshakeNodeName(falseSucOperand);

  // Initialize per_channel_dataout for channels 0 and 1.
  // In MLIR implementation 
  // True result port : 0
  // False result port : 1
  outChannelNameToIndexMap[trueSucNodeName] = 0;
  outChannelNameToIndexMap[falseSucNodeName] = 1;
  per_channel_dataout[0] = std::vector<int>();
  per_channel_dataout[1] = std::vector<int>();
}

//
// calValidSwitching
//
// If the condition value equals the port number of the successor (from outChannelNameToIndexMap),
// then that channel is not selected so valid signal is 0. Otherwise, if either input valid count is > 0,
// valid signal is 2; if both are 0, valid signal is 0.
//
void CBrNode::calValidSwitching(const std::string &sucNodeName,
                                unsigned condValue,
                                unsigned numValid0,
                                unsigned numValid1) {
  // TODO: Check the polarity of the cond selection signal
  if (condValue == outChannelNameToIndexMap[sucNodeName])
    validSignal[sucNodeName] = 0;
  else if (numValid0 > 0 || numValid1 > 0)
    validSignal[sucNodeName] = 2;
  else if (numValid0 == 0 && numValid1 == 0)
    validSignal[sucNodeName] = 0;
}

void CBrNode::calValidSet(const std::string &sucNodeName,
                          unsigned nodeStartTime,
                          unsigned condValue,
                          unsigned II) {
  if (validSignal.find(sucNodeName) != validSignal.end()) {
    if (validSignal[sucNodeName] == 0 || II == 1) {
      setV[sucNodeName] = makeFullSet(II);
    } else {
      if (condValue == outChannelNameToIndexMap[sucNodeName])
        setV[sucNodeName].clear(); // assign empty set
      else {
        std::set<unsigned> tmp;
        tmp.insert(nodeStartTime);
        setV[sucNodeName] = tmp;
      }
    }
  }
}

void CBrNode::calReadySwitching(const std::string &preNodeName,
                                unsigned numValid,
                                unsigned numReady) {
  if (numValid == 0 && numReady == 0)
    readySignal[preNodeName] = 0;
  else if (numValid > 0 || numReady > 0)
    readySignal[preNodeName] = 2;
}

void CBrNode::calReadySet(const std::string &preNodeName,
                          const std::set<unsigned> *setValid,
                          const std::set<unsigned> *setReady,
                          unsigned II) {
  if (setReady != nullptr && setValid != nullptr) {
    std::set<unsigned> finalSet;
    std::set_intersection(setValid->begin(), setValid->end(),
                          setReady->begin(), setReady->end(),
                          std::inserter(finalSet, finalSet.begin()));
    setR[preNodeName] = finalSet;
  } else if (readySignal.find(preNodeName) != readySignal.end() &&
             readySignal[preNodeName] == 0) {
    setR[preNodeName] = makeFullSet(II);
  }
}

void CBrNode::setReadySet(const std::string &preNodeName, unsigned II) {
  std::string selSucName = "";
  for (const auto &entry : validSignal) {
    if (entry.second != 0) {
      selSucName = entry.first;
      break;
    }
  }
  if (!selSucName.empty())
    setR[preNodeName] = setV[selSucName];
  else {
    setR[preNodeName] = makeFullSet(II);
  }
}

void CBrNode::updateDataout(int inputData, unsigned condValue) {
  if (inputData == -1 && lastValidDataValue != -1)
    inputData = lastValidDataValue;
  else
    lastValidDataValue = inputData;

  int diffValue = 0;
  for (const std::string &suc : sucs) {
    if (dataOut.find(suc) != dataOut.end() && !dataOut[suc].empty()) {
      diffValue = dataOut[suc].back() ^ inputData;
      dataOut[suc].push_back(inputData);
    } else {
      diffValue = inputData;
      dataOut[suc] = { inputData };
    }
    std::vector<unsigned> posList = getPositionList(diffValue);
    for (unsigned pos : posList) {
      if (perChannelToggle[suc].find(pos) != perChannelToggle[suc].end())
        perChannelToggle[suc][pos]++;
    }
  }
  // TODO: Check this assignment strategy
  // Update per_channel_dataout for channel (1 - condValue)
  int targetChannel = 1 - static_cast<int>(condValue);
  per_channel_dataout[targetChannel].push_back(inputData);
}

void CBrNode::printDetail() {
  AdjNode::printDetail();

  llvm::dbgs() << "[DEBUG] \t\tCond_pre_node_name: " << condPreNodeName << "\n";
  llvm::dbgs() << "[DEBUG] \t\tData_pre_node_name: " << dataPreNodeName << "\n";
  llvm::dbgs() << "[DEBUG] \t\tTrue_suc_node_name: " << trueSucNodeName << "\n";
  llvm::dbgs() << "[DEBUG] \t\tFalse_suc_node_name: " << falseSucNodeName << "\n";
  llvm::dbgs() << "[DEBUG] \t\tPer_channel_dataout:\n";
  for (const auto &entry : per_channel_dataout) {
    llvm::dbgs() << "[DEBUG] \t\t\tChannel " << entry.first << ": ";
    for (int v : entry.second)
      llvm::dbgs() << v << " ";
    llvm::dbgs() << "\n";
  }
}

//===----------------------------------------------------------------------===//
//
// Mux Node
//
//===----------------------------------------------------------------------===//
// TODO: Check the port selection condition
// Assumption it selects the port with cond_value = (port_id - 1)
MuxNode::MuxNode(mlir::Operation *op,
                 const std::vector<std::string> &predecessors,
                 const std::vector<std::string> &successors,
                 const std::map<std::string, unsigned> &sucDataWidthMap,
                 unsigned latency)
    : AdjNode(op,
              predecessors,
              successors,
              sucDataWidthMap,
              latency)
{
  // Get the operator
  auto muxOp = dyn_cast<handshake::MuxOp>(op);
  auto condPreValue = muxOp.getSelectOperand();
  
  conPreNodeName = condPreValue.getDefiningOp()->getAttrOfType<mlir::StringAttr>("handshake.name").getValue().str();

  // Construct the port map
  auto opOperands = muxOp.getOperands();
  for (unsigned i = 0; i < opOperands.size(); i++) {
    std::string nodeName = "inArgument";
    if (opOperands[i].getDefiningOp()) {
      nodeName = opOperands[i].getDefiningOp()->getAttrOfType<mlir::StringAttr>("handshake.name").getValue().str();
    } else if (auto blockArg = opOperands[i].dyn_cast<mlir::BlockArgument>()){
      // This operand is a block argument
      unsigned argNumber = blockArg.getArgNumber();

      auto funcOp = blockArg.getOwner()->getParentOp();
      if (auto handshakeFunc = dyn_cast<handshake::FuncOp>(funcOp)) {
        // Check whether the argName attributes present
        auto argNameAttr = handshakeFunc->getAttrOfType<mlir::ArrayAttr>("argNames");
        if (!argNameAttr) {
          llvm::errs() << "ERROR: No argNames attributes found on the function!\n";
        } else {
          if (argNumber < argNameAttr.size()) {
            auto strAttr = dyn_cast<mlir::StringAttr>(argNameAttr.getValue()[argNumber]);
            nodeName = strAttr.getValue().str();
          }
        }
      }

    }
    preNameToPortIdxMap[nodeName] = i;
  }
}

void MuxNode::calValidSwitching(const std::string &sucNodeName, unsigned II) {
  if (II == 1)
    validSignal[sucNodeName] = 0;
  else
    validSignal[sucNodeName] = 2;
}

void MuxNode::calValidSet(const std::string &sucNodeName,
                          unsigned nodeStartTime,
                          unsigned II) {
  if (validSignal.find(sucNodeName) != validSignal.end() && validSignal[sucNodeName] == 0) {
    setV[sucNodeName] = makeFullSet(II);
  } else {
    std::set<unsigned> tmp;
    tmp.insert(nodeStartTime);
    setV[sucNodeName] = tmp;
  }
}

void MuxNode::calReadySwitching(const std::string &preNodeName,
                                unsigned condValue,
                                unsigned num_v,
                                const std::set<unsigned> *setV0,
                                const std::set<unsigned> *setVSelect,
                                const std::set<unsigned> *setReady,
                                unsigned II) {
  std::set<unsigned> uSet = makeFullSet(II);
  if (preNodeName == conPreNodeName) {
    // Case 1: this channel is used for the condition signal.
    if (setVSelect != nullptr && setReady != nullptr) {
      if (*setVSelect == uSet && *setReady == uSet) {
        readySignal[preNodeName] = 0;
        return;
      }
    }
    if (setV0 != nullptr && setVSelect != nullptr && setReady != nullptr) {
      // Compute intersection: (setVSelect ∩ setR)
      std::set<unsigned> inter;
      std::set_intersection(setVSelect->begin(), setVSelect->end(),
                            setReady->begin(), setReady->end(),
                            std::inserter(inter, inter.begin()));
      // Check if setV0 is a subset of the intersection.
      bool subset = std::all_of(setV0->begin(), setV0->end(), [&](unsigned x) {
        return inter.find(x) != inter.end();
      });
      readySignal[preNodeName] = subset ? 0 : 2;
    }
  } else {
    // Case 2: data channel.
    // TODO: Check the port assignment methods
    unsigned prePort = (preNodeName == conPreNodeName) ? 1 : 2;
    if (condValue != (prePort - 1)) {
      if (num_v == 0)
        readySignal[preNodeName] = 0;
      else
        readySignal[preNodeName] = 2;
    } else {
      if (setV0 != nullptr && setReady != nullptr) {
        if (*setReady == uSet && *setV0 == uSet) {
          readySignal[preNodeName] = 0;
          return;
        }
      }
      if (setV0 != nullptr && setVSelect != nullptr && setReady != nullptr) {
        std::set<unsigned> inter;
        std::set_intersection(setV0->begin(), setV0->end(),
                              setReady->begin(), setReady->end(),
                              std::inserter(inter, inter.begin()));
        bool subset = std::all_of(setVSelect->begin(), setVSelect->end(), [&](unsigned x) {
          return inter.find(x) != inter.end();
        });
        readySignal[preNodeName] = subset ? 0 : 2;
      }
    }
  }
}

void MuxNode::calReadySet(const std::string &preNodeName,
                          const std::set<unsigned> *setCond,
                          const std::set<unsigned> *setValid,
                          const std::set<unsigned> *setReady,
                          unsigned II) {
  if (readySignal.find(preNodeName) != readySignal.end()) {
    if (readySignal[preNodeName] == 0) {
      setR[preNodeName] = makeFullSet(II);
    } else {
      if (setCond != nullptr && setValid != nullptr && setReady != nullptr) {
        std::set<unsigned> temp;
        // Compute setCond ∩ setV.
        std::set_intersection(setCond->begin(), setCond->end(),
                              setValid->begin(), setValid->end(),
                              std::inserter(temp, temp.begin()));
        std::set<unsigned> finalSet;
        std::set_intersection(temp.begin(), temp.end(),
                            setReady->begin(), setReady->end(),
                            std::inserter(finalSet, finalSet.begin()));
        setR[preNodeName] = finalSet;
      }
      
    }
  }
}

void MuxNode::printDetail() {
  AdjNode::printDetail();

  llvm::dbgs() << "[DEBUG] \t\tCon_pre_node_name: " << conPreNodeName << "\n";
  llvm::dbgs() << "[DEBUG] \t\tInput Port Mapping: \n";
  for (const auto& [selName, portIdx] : preNameToPortIdxMap) {
    llvm::dbgs() << "[DEBUG] \t\t\tNode Name: " << selName << "; Port Index: " << portIdx << ";\n";
  } 
}

//===----------------------------------------------------------------------===//
//
// Start Node
//
//===----------------------------------------------------------------------===//
StartNode::StartNode(mlir::Operation *op,
                     const std::vector<std::string> &predecessors,
                     const std::vector<std::string> &successors,
                     const std::map<std::string, unsigned> &sucDataWidthDict,
                     unsigned latency)
    : AdjNode(op, predecessors, successors, sucDataWidthDict, latency) {
  // No extra initialization needed.
}

void StartNode::calValidSwitching(const std::string &sucNodeName) {
  // In the Start node, set the valid signal for the given successor to 2.
  validSignal[sucNodeName] = 2;
}

void StartNode::calValidSet() {
  // No additional valid-set computation for Start node.
}

void StartNode::calReadySwitching() {
  // No ready switching computation for Start node.
}

void StartNode::calReadySet() {
  // No ready set computation for Start node.
}

