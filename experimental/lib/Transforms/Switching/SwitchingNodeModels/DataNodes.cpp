#include "experimental/Transforms/Switching/SwitchingNodeModels/DataNodes.h"

//===----------------------------------------------------------------------===//
//
// Buffer Node
//
//===----------------------------------------------------------------------===//
BufferNode::BufferNode(mlir::Operation *op,
                       const std::vector<std::string> &predecessors,
                       const std::vector<std::string> &successors,
                       const std::map<std::string, unsigned> &sucDataWidthMap,
                       const unsigned &latency, const unsigned &bbIndex)
    : AdjNode(op, predecessors, successors, sucDataWidthMap, latency, bbIndex),
      START(0), occupancy(0.0), numSlots(0), transparent(false) {}

void BufferNode::calValidSet(const std::string &sucNodeName,
                             IISet &inSetV) {
  setV[sucNodeName] = inSetV;
}

void BufferNode::calReadySet(const std::string &preNodeName,
                             const IISet &inSetR) {
  setR[preNodeName] = inSetR;
}

void BufferNode::printNodeDetails() {
  // Call base class's printNodeDetails()
  AdjNode::printNodeDetails();

  llvm::dbgs() << "[DEBUG] \t\tSTART: " << START << ";\n";
  llvm::dbgs() << "[DEBUG] \t\tOccupancy: " << occupancy << ";\n";
  llvm::dbgs() << "[DEBUG] \t\tNumSlots: " << numSlots << ";\n";
  llvm::dbgs() << "[DEBUG] \t\ttransparent: " << transparent << ";\n";
}

//===----------------------------------------------------------------------===//
//
// Pass Node
//
//===----------------------------------------------------------------------===//


void PassNode::calValidSet(const std::string &sucNodeName,
                           unsigned &nodeStartTime, unsigned &II) {
  if (validSignal.find(sucNodeName) != validSignal.end()) {
    if (validSignal[sucNodeName] == 0) {
      setV[sucNodeName] = IISet(II,true);
    } else {
      IISet tmp(II,false);
      tmp.set(nodeStartTime);
      setV[sucNodeName] = std::move(tmp);
    }
  }
}



void PassNode::calReadySet(const std::string &preNodeName,
                           const IISet &setReady,
                           unsigned &nodeStartTime, unsigned &II) {
  if (setReady.any()) {
    setR[preNodeName] = setReady;
  } else if (readySignal.find(preNodeName) != readySignal.end()) {
    if (readySignal[preNodeName] == 0) {
      setR[preNodeName] = IISet(II,true);
    } else if (readySignal[preNodeName] > 0) {
      IISet tmp(II,false);
      tmp.set(nodeStartTime);
      setR[preNodeName] = std::move(tmp);
    } else if (!setV.empty()) {
      // Fallback: copy the set from the first key in setVMap if it is not the
      // full set.
      auto it = setV.begin();
      IISet fullSet(II,true);
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
                           const IISet &setReady) {
  setR[preNodeName] = setReady;
}

//===---------------------e-------------------------------------------------===//
//
// Load Node
//
//===----------------------------------------------------------------------===//
DLoadNode::DLoadNode(mlir::Operation *op,
                     const std::vector<std::string> &predecessors,
                     const std::vector<std::string> &successors,
                     const std::map<std::string, unsigned> &sucDataWidthMap,
                     const unsigned &latency, const unsigned &bbIndex)
    : AdjNode(op, predecessors, successors, sucDataWidthMap, latency, bbIndex) {
  // Get the LoadOp
  auto loadOp = dyn_cast<handshake::LoadOp>(op);
  auto dataOutRes = loadOp.getDataResult();
  auto addrOutRes = loadOp.getAddressResult();
  auto addrInRes = loadOp.getAddressInput();

  dataOutNodeName = getHandshakeNodeName(dataOutRes);
  addressOutNodeName = getHandshakeNodeName(addrOutRes);
  addressInNodeName = addrInRes.getDefiningOp()
                          ->getAttrOfType<mlir::StringAttr>("handshake.name")
                          .getValue()
                          .str();
}






void DLoadNode::calReadySet(const std::string &preNodeName,
                            const IISet& inSetR,
                            unsigned &nodeStartTime, unsigned &II) {
  if (inSetR.any()) {
    setR[preNodeName] = inSetR;
  } else if (readySignal.find(preNodeName) != readySignal.end()) {
    if (readySignal[preNodeName] == 0) {
      setR[preNodeName] = fullSet(II);
    } else if (readySignal[preNodeName] > 0) {
      IISet tmp(II,true);
      tmp.set(nodeStartTime);
      setR[preNodeName] =std::move(tmp);
    }
  }
}

void DLoadNode::updateDataout(int &inputData, int addressValue) {
  // --- Update data output channel ---
  int dataDiff = 0;
  if (dataOut.find(dataOutNodeName) != dataOut.end() &&
      !dataOut[dataOutNodeName].empty()) {
    dataDiff = dataOut[dataOutNodeName].back() ^ inputData;
    dataOut[dataOutNodeName].push_back(inputData);
  } else {
    dataDiff = inputData;
    dataOut[dataOutNodeName] = {inputData};
  }
  // Update per-channel toggle counts (using getPositionList() from AdjNode)
  auto dataPositions = getPositionList(dataDiff);
  for (unsigned pos : dataPositions) {
    if (perChannelToggle[dataOutNodeName].find(pos) !=
        perChannelToggle[dataOutNodeName].end())
      perChannelToggle[dataOutNodeName][pos]++;
  }

  // --- Update address output channel ---
  int addressDiff = 0;
  if (dataOut.find(addressOutNodeName) != dataOut.end() &&
      !dataOut[addressOutNodeName].empty()) {
    if (addressValue == -10)
      addressValue = dataOut[addressOutNodeName].back();
    addressDiff = dataOut[addressOutNodeName].back() ^ addressValue;
    dataOut[addressOutNodeName].push_back(addressValue);
  } else {
    addressDiff = addressValue;
    dataOut[addressOutNodeName] = {addressValue};
  }
  auto addrPositions = getPositionList(addressDiff);
  for (unsigned pos : addrPositions) {
    if (perChannelToggle[addressOutNodeName].find(pos) !=
        perChannelToggle[addressOutNodeName].end())
      perChannelToggle[addressOutNodeName][pos]++;
  }
}

void DLoadNode::printNodeDetails() {
  // Call base class's printNodeDetails()
  AdjNode::printNodeDetails();

  llvm::dbgs() << "[DEBUG] \t\tAddress input node: " << addressInNodeName
               << ";\n";
  llvm::dbgs() << "[DEBUG] \t\tAddress output node: " << addressOutNodeName
               << ";\n";
  llvm::dbgs() << "[DEBUG] \t\tData output node: " << dataOutNodeName << ";\n";
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
                       unsigned latency, const unsigned &bbIndex)
    : AdjNode(op, predecessors, successors, sucDataWidthMap, latency, bbIndex) {
  // Add the data channel to the mem_controller
  unsigned dataWidth = 0;
  for (const auto &selPair : sucDataWidthMap) {
    dataWidth = selPair.second;
    break;
  }
  sucsDataWidthMap["mem_data"] = dataWidth;

  // Get the StoreOp
  auto storeOp = dyn_cast<handshake::StoreOp>(op);
  auto dataInValue = storeOp.getDataInput();
  auto addrInNodeValue = storeOp.getAddressInput();

  dataInNode = dataInValue.getDefiningOp()
                   ->getAttrOfType<mlir::StringAttr>("handshake.name")
                   .getValue()
                   .str();
  addressInNode = addrInNodeValue.getDefiningOp()
                      ->getAttrOfType<mlir::StringAttr>("handshake.name")
                      .getValue()
                      .str();

  // The "source" nodes for these inputs are to be set externally.
  dataInSrcNode = "";
  addressInSrcNode = "";
}


void DStoreNode::calValidSet(const IISet& setV0,
                             const IISet& setV1, unsigned II) {
  // TODO: Need to differentiate between addrToMem and dataToMem handshake
  // channels
  if (setV0.any() && setV1.any()) {
    IISet v0v1intersect = setV0;
    v0v1intersect &= setV1 ;
    IISet finalSet(II,false) ;
    for (unsigned bitidx : v0v1intersect.set_bits())
      finalSet.set((bitidx + nodeLatency) % II);
    setV["mem"] = finalSet;
  } else if (validSignal.find("mem") != validSignal.end() &&
             validSignal["mem"] == 0) {
    setV["mem"] = IISet(II,true);
  }
}

void DStoreNode::calReadySet(const std::string &preNodeName,
                             const IISet & setV0,
                             const IISet& setV1, unsigned II) {
  // TODO: Validate the following estimation based on new implementations
  if (setV0.any() && setV1.any()) {// shouldnt we wtirt ins set r instead of v?
    IISet tmp = intersection_(tmp,setV0);
    IISet finalSet(II,false);
    for (unsigned i :tmp.set_bits())
      finalSet.set((i + nodeLatency) % II);
    // Here we store the result into setV for preNodeName.
    setV[preNodeName] = finalSet;
  } else if (validSignal.find(preNodeName) != validSignal.end() &&
             validSignal[preNodeName] == 0) {
    setV[preNodeName] = IISet(II,true);
  }
}

void DStoreNode::updateDataout(int inputData, const std::string &srcInputNode) {
  // The method assumes that dataInSrcNode and addressInSrcNode have been set.
  assert(!dataInSrcNode.empty());
  assert(!addressInSrcNode.empty());

  [[maybe_unused]] int valueDiff = 0;
  if (srcInputNode == addressInSrcNode) {
    // Update the address output channel (use key "address_out")
    if (dataOut.find("address_out") != dataOut.end() &&
        !dataOut["address_out"].empty()) {
      valueDiff = dataOut["address_out"].back() ^ inputData;
      dataOut["address_out"].push_back(inputData);
    } else {
      valueDiff = inputData;
      dataOut["address_out"] = {inputData};
    }
  } else if (srcInputNode == dataInSrcNode) {
    // Update the actual data output channel (use key "data_out")
    if (dataOut.find("data_out") != dataOut.end() &&
        !dataOut["data_out"].empty()) {
      valueDiff = dataOut["data_out"].back() ^ inputData;
      dataOut["data_out"].push_back(inputData);
    } else {
      valueDiff = inputData;
      dataOut["data_out"] = {inputData};
    }
  }
  // (Optional: update perChannelToggle accordingly.)
}

void DStoreNode::printNodeDetails() {
  // Call the base class printNodeDetails for common node info.
  AdjNode::printNodeDetails();

  llvm::dbgs() << "[DEBUG] \t\tAddress input node: " << addressInNode << "\n";
  llvm::dbgs() << "[DEBUG] \t\tSrc address input node: " << addressInSrcNode
               << "\n";
  llvm::dbgs() << "[DEBUG] \t\tData input node: " << dataInNode << "\n";
  llvm::dbgs() << "[DEBUG] \t\tSrc data input node: " << dataInSrcNode << "\n";
}

bool DStoreNode::handshakeSwitchingChecking() {
  // For example, if the number of ready channels equals the number of
  // predecessors.
  return (readySignal.size() == pres.size());
}