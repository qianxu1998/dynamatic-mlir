#include "experimental/Transforms/Switching/SwitchingNodeModels/ControlNodes.h"

//===----------------------------------------------------------------------===//
//
// Merge Node
//
//===----------------------------------------------------------------------===//

void MergeNode::calValidSwitching(
    const std::string &sucNodeName,
    const std::vector<IISet *> &setVList,
    const std::vector<int> &numVList, unsigned II) {
  if (numVList.size() == 1) {
      validSignal[sucNodeName]= (numVList[0] == 0) ? 0 : 2;
  } else {
    if (numVList[0] == 0)
      validSignal[sucNodeName] = 0;
    else if (numVList[1] == 0)
      validSignal[sucNodeName] = 0;
    else if (setVList[0] && setVList[1]) {
      // Case 2: the union active range of the two inputs covers the entire II
      IISet   unionSet= * setVList[0];
      unionSet |= *setVList[1];
      // Build the full set: {0, 1, ..., II-1}
      IISet fullSet(II,true);
      if (unionSet == fullSet)
        validSignal[sucNodeName] = 0;
    } else if (numVList[0] > 0 && numVList[1] > 0) {
      validSignal[sucNodeName] = 2;
    }
  }
}

void MergeNode::calValidSet(const std::string &sucNodeName,
                            const std::vector<IISet *> &setVList,
                            unsigned II) {
  if (setVList.size() == 1) {
    // One input channel
    if (validSignal.find(sucNodeName) != validSignal.end() &&
        validSignal[sucNodeName] == 0) {
      setV[sucNodeName] = IISet(II,true);
    } else if (setVList[0]) {
      setV[sucNodeName] = *setVList[0];
    }
  } else {
    if (validSignal.find(sucNodeName) != validSignal.end() &&
        validSignal[sucNodeName] == 0) {
      setV[sucNodeName] = IISet(II,true);
    } else if (setVList.size() >= 2 && setVList[0] && setVList[1]) {
      // Compute the union: in C++ we use std::set_union.
      IISet  unionSet = * setVList[0];
      unionSet |=  * setVList[1];
      setV[sucNodeName] = std::move(unionSet);
    }
  }
}


void MergeNode::calReadySet(const std::string &preNodeName,
                            const IISet& setRPtr, unsigned II) {
  if (setRPtr.any() ) {
    setR[preNodeName] = setRPtr;
  } else if (readySignal.find(preNodeName) != readySignal.end()) {
    if (readySignal[preNodeName] == 0) {
      setR[preNodeName] = IISet(II,true);
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
                       unsigned latency, const unsigned &bbIndex)
    : AdjNode(op, predecessors, successors, sucDataWidthMap, latency, bbIndex) {

  // Determine the control and data channel successor names.
  // We assume that the mapping sucDataWidthMap associates each successor name
  // with a port number.
  conSucNodeName = "";
  dataSucNodeName = "";
  for (const auto &pair : sucDataWidthMap) {
    if (pair.second == 1) {
      conSucNodeName = pair.first;
      break;
    }
  }

  // For control merge node, we don't have actual data output
  for (const auto &selNode : successors) {
    if (selNode != conSucNodeName)
      dataSucNodeName = selNode;
  }
}
void CMergeNode::calDataout(int inputValue) {
  // Update the control channel:
  if (dataOut.find(conSucNodeName) == dataOut.end() ||
      dataOut[conSucNodeName].empty()) {
    dataOut[conSucNodeName] = {inputValue};
  } else {
    dataOut[conSucNodeName].push_back(inputValue);
  }
  // For the data channel, simply set its output to [-1]
  dataOut[dataSucNodeName] = {-1};
}

//
// updateDataout:
//   For each successor in the node’s successor list, if it is the condition
//   channel, update using an XOR difference between the last value and the new
//   input and update per-channel toggle counts; otherwise, update the data
//   channel with -1.
//
void CMergeNode::updateDataout(int inputData) {
  int valueDiff = 0;
  // Iterate over the successor names (we assume 'sucs' is available from
  // AdjNode as a vector<string>)
  for (const std::string &suc : sucs) {
    // Check if there is already data for this successor.
    if (dataOut.find(suc) != dataOut.end() && !dataOut[suc].empty()) {
      if (suc == conSucNodeName) {
        valueDiff = dataOut[suc].back() ^ inputData;
        dataOut[suc].push_back(inputData);
        // Get positions of set bits in the difference (using the inherited
        // getPositionList)
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
        dataOut[suc] = {inputData};
        std::vector<unsigned> posList = getPositionList(valueDiff);
        for (unsigned pos : posList) {
          if (perChannelToggle[suc].find(pos) != perChannelToggle[suc].end())
            perChannelToggle[suc][pos]++;
        }
      } else {
        dataOut[suc] = {-1};
      }
    }
  }
}

void CMergeNode::printNodeDetails() {
  AdjNode::printNodeDetails();

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

void ForkNode::calValidSwitching(
    const std::string &sucNodeName, int numValid,
    llvm::DenseMap<StringRef, IISet *> &setRDict,
    const llvm::DenseMap<StringRef, int> &numReadyDict,
    unsigned sucNodeStart, unsigned nodeSteadyStart, unsigned II) {
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
    IISet *s = it->second;
    if (s && s->any()) {
      if (s->size() == 1)
        selStartPoint = s->find_first();
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
  // For each entry in setRDict (other than sucNodeName), check if selStartPoint
  // is contained.
  for (auto &entry : setRDict) {
    if (entry.first != sucNodeName) {
      //! Testing
      llvm::dbgs() << "selSuc " << nodeSteadyStart << "\n";
      IISet *s = entry.second;
      // In our design, an empty set is equivalent to None.
      if (!s) {
        // If the corresponding setR is not empty
        flag = false;
      } else if (!s->test(selStartPoint)) {
        flag = false;
      }
    }
  }

  //! Testing
  llvm::dbgs() << "Node Steady Start: " << nodeSteadyStart << "\n";
  llvm::dbgs() << "Sel Start Point: " << selStartPoint << "\n";
  llvm::dbgs() << "Exist Flag: " << existFlag << "\n";

  if (flag && existFlag) {
    validSignal[sucNodeName] = 0;
    return;
  } else {
    //! Testing
    llvm::dbgs() << "Hit Valid Switching Calculation Case III\n";
    if (existFlag) {
      // Case 3: Compute desired cycle time.
      // TODO: Verify the following desired criteria
      [[maybe_unused]] unsigned desiredCycleTime =
          (nodeSteadyStart != 0) ? (nodeSteadyStart - 1) : (II - 1);
      if (selStartPoint == 0)
        validSignal[sucNodeName] = 0;
      else
        validSignal[sucNodeName] = 2;
    }
  }
}

void ForkNode::calValidSet(
    const std::string &sucNodeName, unsigned nodeStartTime, int numValid,
    llvm::DenseMap<StringRef, IISet *> &setRDict,

    unsigned II) {
  if (validSignal.find(sucNodeName) != validSignal.end()) {
    if (validSignal[sucNodeName] == 0) {
      setV[sucNodeName] = IISet(II,true);
    } else {
      if (numValid > 0) {
        IISet tmp(II,false);
        tmp.set(nodeStartTime);
        setV[sucNodeName] = std::move(tmp);
      } else {
        auto it = setRDict.find(sucNodeName);
        if (it != setRDict.end() && it->second)
            setV[sucNodeName] = *(it->second);
      }
    }
  }
}

void ForkNode::calReadySwitching(const std::string &preNodeName,
                                 const std::vector<int> &numReadyList) {
  // If any value in numReadyList is greater than 0, set ready signal to 2 and
  // return.
  bool tmpFlag = true;
  for (int value : numReadyList) {
    if (value > 0) {
      readySignal[preNodeName] = 2;
      return;
    }

    if (value != 0)
      tmpFlag = false;
  }
  // Otherwise, if all values are 0, set ready signal to 0.
  if (tmpFlag)
    readySignal[preNodeName] = 0;
}

void ForkNode::calReadySet(
    const std::string &preNodeName,
    const    llvm::DenseMap<StringRef, IISet *> &setRDict,

    unsigned II) {


  if (readySignal.find(preNodeName) != readySignal.end()) {
    if (readySignal[preNodeName] == 0) {
      setR[preNodeName] = IISet(II, true/* all 1's */);
    } else {
      // Check existence of all nReady sets for successors.
      bool exist = true;
      IISet unionSet(II,false/*all 0's*/);
      // Start with an empty union.
      for (const std::string &key : sucs) {
        auto it = setRDict.find(key);
        if (it != setRDict.end()) {
          IISet *s = it->second;
          if (!s) {
            exist = false;
            unionSet.reset(); // intersection with an empty set remains empty
          } else {
            // Compute the union: unionSet = unionSet ∪ s.
            unionSet |=  *s;
          }
        }
      }
      if (exist) {
         IISet fullMask(II,true/*all 1's*/);

        if (unionSet == fullMask){
          IISet tmp (II,true/*all 1's*/);
          tmp.set(0);
          setR[preNodeName] = std::move(tmp);
        }
        else {
          // Retrieve the last element of unionSet.
          unsigned last = 0;
          for (unsigned bitidx : unionSet.set_bits()){
          // if (!unionSet.empty())
            last = bitidx;
          }
          IISet tmp (II,false/*all 1's*/);
          tmp.set(last);
          setR[preNodeName] = std::move(tmp);
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
                 unsigned latency, const unsigned &bbIndex)
    : AdjNode(op, predecessors, successors, sucDataWidthMap, latency, bbIndex),
      lastValidDataValue(-1) {
  // Get the ConditionalBranch node
  auto CBrOp = dyn_cast<handshake::ConditionalBranchOp>(op);
  auto condPreOperand = CBrOp.getConditionOperand();
  auto dataPreOperand = CBrOp.getDataOperand();
  auto trueSucOperand = CBrOp.getTrueResult();
  auto falseSucOperand = CBrOp.getFalseResult();

  condPreNodeName = condPreOperand.getDefiningOp()
                        ->getAttrOfType<mlir::StringAttr>("handshake.name")
                        .getValue()
                        .str();
  dataPreNodeName = dataPreOperand.getDefiningOp()
                        ->getAttrOfType<mlir::StringAttr>("handshake.name")
                        .getValue()
                        .str();
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
// If the condition value equals the port number of the successor (from
// outChannelNameToIndexMap), then that channel is not selected so valid signal
// is 0. Otherwise, if either input valid count is > 0, valid signal is 2; if
// both are 0, valid signal is 0.
//
void CBrNode::calValidSwitching(const std::string &sucNodeName,
                                unsigned condValue, int numValid0,
                                int numValid1) {
  // TODO: Check the polarity of the cond selection signal
  if (condValue == outChannelNameToIndexMap[sucNodeName])
    validSignal[sucNodeName] = 0;
  else if (numValid0 > 0 || numValid1 > 0)
    validSignal[sucNodeName] = 2;
  else if (numValid0 == 0 && numValid1 == 0)
    validSignal[sucNodeName] = 0;
}

void CBrNode::calValidSet(const std::string &sucNodeName,
                          unsigned nodeStartTime, unsigned condValue,
                          unsigned II) {
  if (validSignal.find(sucNodeName) != validSignal.end()) {
    if (validSignal[sucNodeName] == 0 || II == 1) {
      setV[sucNodeName] = IISet(II,true /* all 1's*/);
    } else {
      if (condValue == outChannelNameToIndexMap[sucNodeName])
        setV[sucNodeName].reset(); // assign empty set
      else {
        IISet tmp(II,false /* all 0's*/);;
        tmp.set(nodeStartTime);
        setV[sucNodeName] = std::move(tmp);
      }
    }
  }
}

void CBrNode::calReadySwitching(const std::string &preNodeName, int numValid,
                                int numReady) {
  if (numValid == 0 && numReady == 0)
    readySignal[preNodeName] = 0;
  else if (numValid > 0 || numReady > 0)
    readySignal[preNodeName] = 2;
}

void CBrNode::calReadySet(const std::string &preNodeName,
                          const IISet& setValid,
                          const IISet& setReady, unsigned II) {
  if (setReady.any() && setValid.any()) {
    IISet finalSet=setValid;
    finalSet &= setReady;
    setR[preNodeName] = std::move(finalSet);
  } else if (readySignal.find(preNodeName) != readySignal.end() &&
             readySignal[preNodeName] == 0) {
    setR[preNodeName] = IISet(II,/* all 1's*/ true);
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
    setR[preNodeName] = IISet(II,true/* all 1's*/ );;
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
      dataOut[suc] = {inputData};
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

void CBrNode::printNodeDetails() {
  AdjNode::printNodeDetails();

  llvm::dbgs() << "[DEBUG] \t\tCond_pre_node_name: " << condPreNodeName << "\n";
  llvm::dbgs() << "[DEBUG] \t\tData_pre_node_name: " << dataPreNodeName << "\n";
  llvm::dbgs() << "[DEBUG] \t\tTrue_suc_node_name: " << trueSucNodeName << "\n";
  llvm::dbgs() << "[DEBUG] \t\tFalse_suc_node_name: " << falseSucNodeName
               << "\n";
  llvm::dbgs() << "[DEBUG] \t\tOut Channel Name to Index Map:\n";
  for (const auto &[outName, portIdx] : outChannelNameToIndexMap) {
    llvm::dbgs() << "[DEBUG] \t\t\tOutput Node: " << outName
                 << "; Port Idx: " << portIdx << "\n";
  }
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
                 unsigned latency, const unsigned &bbIndex)
    : AdjNode(op, predecessors, successors, sucDataWidthMap, latency, bbIndex) {
  // Get the operator
  auto muxOp = dyn_cast<handshake::MuxOp>(op);
  auto condPreValue = muxOp.getSelectOperand();

  conPreNodeName = condPreValue.getDefiningOp()
                       ->getAttrOfType<mlir::StringAttr>("handshake.name")
                       .getValue()
                       .str();

  // Construct the port map
  auto opOperands = muxOp.getOperands();
  for (unsigned i = 0; i < opOperands.size(); i++) {
    std::string nodeName = "inArgument";
    if (opOperands[i].getDefiningOp()) {
      nodeName = opOperands[i]
                     .getDefiningOp()
                     ->getAttrOfType<mlir::StringAttr>("handshake.name")
                     .getValue()
                     .str();
    } else if (auto blockArg = opOperands[i].dyn_cast<mlir::BlockArgument>()) {
      // This operand is a block argument
      unsigned argNumber = blockArg.getArgNumber();

      auto funcOp = blockArg.getOwner()->getParentOp();
      if (auto handshakeFunc = dyn_cast<handshake::FuncOp>(funcOp)) {
        // Check whether the argName attributes present
        auto argNameAttr =
            handshakeFunc->getAttrOfType<mlir::ArrayAttr>("argNames");
        if (!argNameAttr) {
          llvm::errs()
              << "ERROR: No argNames attributes found on the function!\n";
        } else {
          if (argNumber < argNameAttr.size()) {
            auto strAttr =
                dyn_cast<mlir::StringAttr>(argNameAttr.getValue()[argNumber]);
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
                          unsigned nodeStartTime, unsigned II) {
  if (validSignal.find(sucNodeName) != validSignal.end() &&
      validSignal[sucNodeName] == 0) {
    setV[sucNodeName] = IISet(II,true/*all 1's*/);
  } else {
    IISet tmp(II,false/* all 0's*/);
    tmp.set(nodeStartTime);
    setV[sucNodeName] = std::move(tmp);
  }
}

void MuxNode::calReadySwitching(const std::string &preNodeName,
                                unsigned condValue, int num_v,
                                const IISet& setV0,
                                const  IISet& setVSelect,
                                const  IISet& setReady,
                                unsigned II) {
  IISet uSet = IISet(II,true);
  if (preNodeName == conPreNodeName) {
    // Case 1: this channel is used for the condition signal.
    if (setVSelect.any() && setReady.any()) {
      if (setVSelect == uSet && setReady == uSet) {
        readySignal[preNodeName] = 0;
        return;
      }
    }
    if (setV0.any()  && setVSelect.any() && setReady.any()) {
      // Compute intersection: (setVSelect ∩ setReady)
      IISet inter=setVSelect;
      inter &= setReady;
      // Check if setV0 is a subset of the intersection.
      IISet tmp_setV0 = setV0;
      tmp_setV0 &= inter;
      bool subset = (tmp_setV0 == setV0);

      readySignal[preNodeName] = subset ? 0 : 2;
    }
  } else {
    // Case 2: data channel.
    // TODO: Check the port assignment methods
    unsigned prePort = preNameToPortIdxMap[preNodeName];
    if (condValue != (prePort - 1)) {
      if (num_v == 0) readySignal[preNodeName] = 0;
      else if (num_v > 0) readySignal[preNodeName] = 2;
    } else {
      if (setV0.any() && setReady.any()) {
        if (setReady == uSet && setV0 == uSet) {
          readySignal[preNodeName] = 0;
          return;
        }
      }
      if (setV0.any() && setVSelect.any() && setReady.any()) {
        IISet inter=setV0;
        inter &=setReady;
          // Check if setVSelect is a subset of setV0.
        IISet tmp_setVSelect = setVSelect;
        tmp_setVSelect &= inter;
         bool subset = (tmp_setVSelect == setVSelect );
        readySignal[preNodeName] = subset ? 0 : 2;
      }
    }
  }
}

void MuxNode::calReadySet(const std::string &preNodeName,
                          const IISet& setCond,
                          const IISet& setValid,
                          const IISet& setReady, unsigned II) {
  if (readySignal.find(preNodeName) != readySignal.end()) {
    if (readySignal[preNodeName] == 0) {
      setR[preNodeName] = IISet(II,true /* all 1's*/);
    } else {
      if (setCond.any() && setValid.any() && setReady.any()) {
        // Compute setCond ∩ setV.
        IISet inter=setCond;
        inter &= setValid;
        IISet finalSet=inter;
        //Compute setCond ∩ setV ∩  setReady.
        inter &=setReady;
        setR[preNodeName] = std::move(finalSet);
      }
    }
  }
}

void MuxNode::printNodeDetails() {
  AdjNode::printNodeDetails();

  llvm::dbgs() << "[DEBUG] \t\tCon_pre_node_name: " << conPreNodeName << "\n";
  llvm::dbgs() << "[DEBUG] \t\tInput Port Mapping: \n";
  for (const auto &[selName, portIdx] : preNameToPortIdxMap) {
    llvm::dbgs() << "[DEBUG] \t\t\tNode Name: " << selName
                 << "; Port Index: " << portIdx << ";\n";
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
                     unsigned latency, const unsigned &bbIndex)
    : AdjNode(op, predecessors, successors, sucDataWidthDict, latency,
              bbIndex) {
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
