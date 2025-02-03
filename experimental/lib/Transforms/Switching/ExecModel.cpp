//===- ExecModel.cpp - Estimate Swithicng Activities ------*- C++ -*-===//
//
// Implements all the execution models for different types of nodes
//
//===----------------------------------------------------------------------===//

#include "experimental/Transforms/Switching/ExecModel.h"

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
      std::set<unsigned> fullSet;
      for (unsigned i = 0; i < II; ++i)
        fullSet.insert(i);
      setV[sucNodeName] = fullSet;
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
      std::set<unsigned> fullSet;
      for (unsigned i = 0; i < II; ++i)
        fullSet.insert(i);
      setR[preNodeName] = fullSet;
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
      std::set<unsigned> fullSet;
      for (unsigned i = 0; i < II; ++i)
        fullSet.insert(i);
      setV[sucNodeName] = fullSet;
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
      std::set<unsigned> fullSet;
      for (unsigned i = 0; i < II; ++i)
        fullSet.insert(i);
      setR[preNodeName] = fullSet;
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

  dataOutNodeName = getHandshakeNodeName(dataOutRes);
  addressOutNodeName = getHandshakeNodeName(addrOutRes);
  addressInNodeName = predecessors[0];
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
      std::set<unsigned> fullSet;
      for (unsigned i = 0; i < II; ++i)
        fullSet.insert(i);
      setV[sucNodeName] = fullSet;
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
      std::set<unsigned> fullSet;
      for (unsigned i = 0; i < II; ++i)
        fullSet.insert(i);
      setR[preNodeName] = fullSet;
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
  // Assume that the predecessor list is ordered:
  // index 0 is the data input, index 1 is the address input.
  if (!pres.empty())
    dataInNode = pres[0];
  if (pres.size() > 1)
    addressInNode = pres[1];
  // The "source" nodes for these inputs are to be set externally.
  dataInSrcNode = "";
  addressInSrcNode = "";
}

void DStoreNode::calValidSwitching(unsigned numValid1, unsigned numValid2) {
  // For the memory controller channel, use key "mc".
  if (numValid1 > 0 || numValid2 > 0)
    validSignal["mc"] = 4;
  else if (numValid1 == 0 && numValid2 == 0)
    validSignal["mc"] = 0;
  else
    validSignal["mc"] = 4;
}

void DStoreNode::calValidSet(const std::set<unsigned> *setV0, const std::set<unsigned> *setV1, unsigned II) {
  if (setV0 != nullptr && setV1 != nullptr) {
    std::set<unsigned> tmp;
    std::set_intersection(setV0->begin(), setV0->end(),
                          setV1->begin(), setV1->end(),
                          std::inserter(tmp, tmp.begin()));
    std::set<unsigned> finalSet;
    for (unsigned val : tmp)
      finalSet.insert((val + nodeLatency) % II);
    setV["mc"] = finalSet;
  } else if (validSignal.find("mc") != validSignal.end() && validSignal["mc"] == 0) {
    std::set<unsigned> fullSet;
    for (unsigned i = 0; i < II; ++i)
      fullSet.insert(i);
    setV["mc"] = fullSet;
  }
}

void DStoreNode::calReadySwitching(const std::string &preNodeName, unsigned numValid1, unsigned numValid2) {
  if (numValid1 > 0 || numValid2 > 0)
    readySignal[preNodeName] = 2;
  else if (numValid1 == 0 && numValid2 == 0)
    readySignal[preNodeName] = 0;
  else
    readySignal[preNodeName] = 2;
}

void DStoreNode::calReadySet(const std::string &preNodeName, const std::set<unsigned> *setV0, const std::set<unsigned> *setV1, unsigned II) {
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
    std::set<unsigned> fullSet;
    for (unsigned i = 0; i < II; ++i)
      fullSet.insert(i);
    setV[preNodeName] = fullSet;
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

  llvm::dbgs() << "Address input node: " << addressInNode << "\n";
  llvm::dbgs() << "Src address input node: " << addressInSrcNode << "\n";
  llvm::dbgs() << "Data input node: " << dataInNode << "\n";
  llvm::dbgs() << "Src data input node: " << dataInSrcNode << "\n";
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
//
// calValidSwitching
//
// Python version:
//   def cal_valid_switching(self, suc_node_name, set_v_list: list, num_v_list: list, II):
//       if (len(num_v_list) == 1):
//           if (num_v_list[0] == 0):
//               self.valid_signal[suc_node_name] = 0
//           elif (num_v_list[0] > 0):
//               self.valid_signal[suc_node_name] = 2
//       else:
//           if (num_v_list[0] == 0):
//               self.valid_signal[suc_node_name] = 0
//           elif (num_v_list[1] == 0):
//               self.valid_signal[suc_node_name] = 0
//           elif (set_v_list[0] != None and set_v_list[1] != None):
//               if ((set_v_list[0] | set_v_list[1]) == set(range(II))):
//                   self.valid_signal[suc_node_name] = 0
//           elif (num_v_list[0] > 0 and num_v_list[1] > 0):
//               self.valid_signal[suc_node_name] = 2 
//
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
        else if (numVList[0] > 0 && numVList[1] > 0)
          validSignal[sucNodeName] = 2;
      } else if (numVList[0] > 0 && numVList[1] > 0) {
        validSignal[sucNodeName] = 2;
      }
    }
  }
}

//
// calValidSet
//
// Python version:
//   def cal_valid_set(self, suc_node_name, set_v_list: list, II):
//       if (len(set_v_list) == 1):
//           if (suc_node_name in self.valid_signal.keys()):
//               if (self.valid_signal[suc_node_name] == 0):
//                   self.set_v[suc_node_name] = set(range(II))
//           elif (set_v_list[0] != None):
//               self.set_v[suc_node_name] = set_v_list[0]
//       else:
//           if (suc_node_name in self.valid_signal.keys()):
//               if (self.valid_signal[suc_node_name] == 0):
//                   self.set_v[suc_node_name] = set(range(II))
//           elif (set_v_list[0] != None and set_v_list[1] != None):
//               self.set_v[suc_node_name] = set_v_list[0] | set_v_list[1]
//
void MergeNode::calValidSet(const std::string &sucNodeName,
                            const std::vector<std::set<unsigned>> &setVList,
                            unsigned II) {
  if (setVList.size() == 1) {
    // One input channel
    if (validSignal.find(sucNodeName) != validSignal.end() &&
        validSignal[sucNodeName] == 0) {
      std::set<unsigned> fullSet;
      for (unsigned i = 0; i < II; ++i)
        fullSet.insert(i);
      setV[sucNodeName] = fullSet;
    } else if (!setVList[0].empty()) {
      setV[sucNodeName] = setVList[0];
    }
  } else {
    if (validSignal.find(sucNodeName) != validSignal.end() &&
        validSignal[sucNodeName] == 0) {
      std::set<unsigned> fullSet;
      for (unsigned i = 0; i < II; ++i)
        fullSet.insert(i);
      setV[sucNodeName] = fullSet;
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

//
// calReadySwitching
//
// Python version:
//   def cal_ready_switching(self, pre_node_name, num_ready):
//       if (num_ready == 0):
//           self.ready_signal[pre_node_name] = 0
//       elif (num_ready > 0):
//           self.ready_signal[pre_node_name] = 2
//
void MergeNode::calReadySwitching(const std::string &preNodeName,
                                  unsigned numReady) {
  if (numReady == 0)
    readySignal[preNodeName] = 0;
  else if (numReady > 0)
    readySignal[preNodeName] = 2;
}

//
// calReadySet
//
// Python version:
//   def cal_ready_set(self, pre_node_name, set_r, II):
//       if (set_r != None):
//           self.set_r[pre_node_name] = set_r
//       elif (pre_node_name in self.ready_signal.keys()):
//           if (self.ready_signal[pre_node_name] == 0):
//               self.set_r[pre_node_name] = set(range(II))
//
void MergeNode::calReadySet(const std::string &preNodeName,
                            const std::set<unsigned> *setRPtr,
                            unsigned II) {
  if (setRPtr != nullptr) {
    setR[preNodeName] = *setRPtr;
  } else if (readySignal.find(preNodeName) != readySignal.end()) {
    if (readySignal[preNodeName] == 0) {
      std::set<unsigned> fullSet;
      for (unsigned i = 0; i < II; ++i)
        fullSet.insert(i);
      setR[preNodeName] = fullSet;
    }
  }
}

//===----------------------------------------------------------------------===//
//
// CMerge Node
//
//===----------------------------------------------------------------------===//


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



//===----------------------------------------------------------------------===//
//
// CBr Node
//
//===----------------------------------------------------------------------===//


//===----------------------------------------------------------------------===//
//
// Shli Node
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
//
// Shrsi Node
//
//===----------------------------------------------------------------------===//


//===----------------------------------------------------------------------===//
//
// Shrui Node
//
//===----------------------------------------------------------------------===//


//===----------------------------------------------------------------------===//
//
// Mux Node
//
//===----------------------------------------------------------------------===//


//===----------------------------------------------------------------------===//
//
// Trunci Node
//
//===----------------------------------------------------------------------===//


//===----------------------------------------------------------------------===//
//
// Extui Node
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
//
// Constant Node
//
//===----------------------------------------------------------------------===//


//===----------------------------------------------------------------------===//
//
// Ori Node
//
//===----------------------------------------------------------------------===//


//===----------------------------------------------------------------------===//
//
// Andi Node
//
//===----------------------------------------------------------------------===//


//===----------------------------------------------------------------------===//
//
// Source Node
//
//===----------------------------------------------------------------------===//


//===----------------------------------------------------------------------===//
//
// Start Node
//
//===----------------------------------------------------------------------===//


