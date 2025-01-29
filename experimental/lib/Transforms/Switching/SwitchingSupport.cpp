//===- SwitchingSupport.cpp - Estimate Swithicng Activities ------*- C++ -*-===//
//
// Implements the supporting datastructures for the switching estimation
// pass
//
//===----------------------------------------------------------------------===//

#include "experimental/Transforms/Switching/SwitchingSupport.h"

using namespace mlir;
using namespace dynamatic;

/// Function declaration
static unsigned extractNodeLatency(mlir::Operation *op, TimingDatabase timingDB);

void SwitchingInfo::insertBE(unsigned srcBB, unsigned dstBB, StringRef mgLabel) {
  std::pair<unsigned, unsigned> BBPair = {srcBB, dstBB};

  // Check the existence of the backedge pair
  if (backEdgeToCFDFCMap.find(BBPair) != backEdgeToCFDFCMap.end()) {
    backEdgeToCFDFCMap[BBPair].push_back(static_cast<unsigned>(std::stoul(mgLabel.str())));
  } else {
    std::vector<unsigned> tmpVector{static_cast<unsigned>(std::stoul(mgLabel.str()))};
    backEdgeToCFDFCMap[std::make_pair(srcBB, dstBB)] = tmpVector;
  }
}

//===----------------------------------------------------------------------===//
//
// Definitions of AdjNode
//
//===----------------------------------------------------------------------===//
AdjNode::AdjNode(mlir::Operation* selOp, 
          const std::vector<std::string>& predecessors, const std::vector<std::string>& successors, 
          const std::map<std::string, unsigned>& sucDataWidthMap, const unsigned& latency) {
  // Initialize all variables
  this->op = selOp;
  this->pres = predecessors;
  this->sucs = successors;
  this->sucsDataWidthMap = sucDataWidthMap;
  this->nodeLatency = latency;

  // Initialize toggle count map
  for (auto& [key, value] : sucsDataWidthMap) {
    std::map<unsigned, unsigned> tmpMap;

    for (int i = 0; i < value; i++) {
      tmpMap[i] = 0;
    }

    perChannelToggle[key] = tmpMap;
  }
}

//===----------------------------------------------------------------------===//
//
// Definitions of AdjGraph
//
//===----------------------------------------------------------------------===//
// Initialize the whole adjacency graph for the selected segment
AdjGraph::AdjGraph(const buffer::CFDFC& cfdfc, const TimingDatabase& timingDB, unsigned II) {
  std::map<std::string, std::vector<std::string>> nodeToPresMap;
  std::map<std::string, std::vector<std::string>> nodeToSucsMap;

  // Construct the pres and sucs map for all units in the CFDFC
  for (const auto& selChannel: cfdfc.channels) {
    mlir::Operation* srcOp = selChannel.getDefiningOp();
    std::string srcName = srcOp->getAttrOfType<StringAttr>("handshake.name").str();

    for (const auto& dstOp: selChannel.getUsers()) {
      std::string dstName = dstOp->getAttrOfType<StringAttr>("handshake.name").str();
      
      // Insert to sucs
      insertToSurroundingList(nodeToSucsMap, srcName, dstName);

      // Insert the pres
      insertToSurroundingList(nodeToPresMap, dstName, srcName); 
    }
  }


  // Traverse all nodes in the MG
  for (auto& selNode: cfdfc.units) {
    std::string unitType = selNode->getName().getStringRef().str();
    std::string unitName = selNode->getAttrOfType<StringAttr>("handshake.name").str();
    bool isBuffer = false;

    // Get the unit latency
    unsigned nodeLatency = extractNodeLatency(selNode, timingDB);

    // Get the datawidth of all output channels
    //! Testing
    llvm::dbgs() << "[DEBUG] \tNode Name: " << unitName << "\n";
    llvm::dbgs() << "[DEBUG] \t\tSucs Node List: [";
    for (const auto& selSuc : nodeToSucsMap[unitName]) {
      llvm::dbgs() << selSuc << ", ";
    }
    llvm::dbgs() << "]\n";


    /// If this is a buffer node, get the number of slots
    if (isa<handshake::BufferOp>(selNode)) {
      auto params = selNode->getAttrOfType<DictionaryAttr>(RTL_PARAMETERS_ATTR_NAME);
      if (!params) {
        llvm::errs() << "[ERROR] " << RTL_PARAMETERS_ATTR_NAME << " is missing for op " << unitName << " in handshake_export.mlir\n";
        exit(-1);
      }
      
      // Retrieve the number of slots
      auto optSlots = params.getNamed(BufferOp::NUM_SLOTS_ATTR_NAME);
      unsigned numSlotsValue = 0;
      if (optSlots) {
        if (auto numSlots = dyn_cast<IntegerAttr>(optSlots->getValue())) {
          if (numSlots.getType().isUnsignedInteger())
            numSlotsValue = numSlots.getUInt();
        }
      }

      /// Debug
      
    }

  }

}

void AdjGraph::insertToSurroundingList(std::map<std::string, std::vector<std::string>>& selMap, 
                                std::string& key, std::string& value) {
  if (selMap.find(key) != selMap.end()) {
    selMap[key].push_back(value);
  } else {
    std::vector<std::string> tmpVec = {value};
    selMap[key] = tmpVec;
  }

}

//===----------------------------------------------------------------------===//
//
// Helper Functions
//
//===----------------------------------------------------------------------===//

void printBEToCFDFCMap(const std::map<std::pair<unsigned, unsigned>, std::vector<unsigned>>& selMap) {
  for (const auto& selPair : selMap) {
    const std::pair<unsigned, unsigned>& key = selPair.first;
    const std::vector<unsigned> mgList = selPair.second;

    llvm::dbgs() << "[DEBUG] \tBackEdge Pair: (" << key.first << ", " << key.second <<") : [";

    for (const auto& selMG: mgList) {
      llvm::dbgs() << selMG << ", ";
    }

    llvm::dbgs() << "]\n";
  }
}

void printSegToBBListMap(const std::map<std::string, mlir::SetVector<unsigned>>& selMap) {
  for (const auto& selPair: selMap) {
    const std::string segLabel = selPair.first;
    const mlir::SetVector<unsigned> BBList = selPair.second;

    llvm::dbgs() << "[DEBUG] \tSeg Label: " << segLabel <<" : [";

    for (const auto& selBB: BBList) {
      llvm::dbgs() << selBB << ", ";
    }

    llvm::dbgs() << "]\n";
  }
}

std::string removeDigits(const std::string& inStr) {
  std::regex digitsRegex("\\d");

  std::string outStr = std::regex_replace(inStr, digitsRegex, "");

  return outStr;
}

std::vector<std::string> split(const std::string& s, const std::string& delimiter) {
  std::vector<std::string> tokens;
  std::regex re(delimiter);
  std::sregex_token_iterator it(s.begin(), s.end(), re, -1);
  std::sregex_token_iterator end;

  for (; it != end; ++it) {
    tokens.push_back(it->str());
  }

  return tokens;
}

std::string strip(const std::string &inputStr, const std::string &toRemove) {
  // Trim leading and trailing whitespace
  size_t start = inputStr.find_first_not_of(" \t\n\r\f\v");
  if (start == std::string::npos) {
      return "";  // Return an empty string if there are only whitespaces
  }
  size_t end = inputStr.find_last_not_of(" \t\n\r\f\v");
  std::string stripped = inputStr.substr(start, end - start + 1);

  if (toRemove == "") {
    return stripped;
  }

  // Remove all occurrences of the specified substring 'toRemove'
  size_t pos = stripped.find(toRemove);
  while (pos != std::string::npos) {
      stripped.erase(pos, toRemove.length());
      pos = stripped.find(toRemove, pos);
  }

  return stripped;
}

/// Extracts the latency for each operation
/// This is done in 3 ways:
/// 1. If the operation is in the timingDB, the latency is extracted from the
/// timingDB
/// 2. If the operation is a buffer operation, the latency is extracted from the
/// timing attribute
/// 3. If the operation is neither, then its latency is set to 0
static unsigned extractNodeLatency(mlir::Operation *op, TimingDatabase timingDB) {
  double latency = 0;

  if (!failed(timingDB.getLatency(op, SignalType::DATA, latency))) {
    return latency;
  }

  if (isa<handshake::BufferOp>(op)) {
    auto params = op->getAttrOfType<DictionaryAttr>(RTL_PARAMETERS_ATTR_NAME);
    if (!params)
      return 0;

    auto optTiming = params.getNamed(handshake::BufferOp::TIMING_ATTR_NAME);
    if (!optTiming)
      return 0;

    if (auto timing = dyn_cast<handshake::TimingAttr>(optTiming->getValue())) {
      handshake::TimingInfo info = timing.getInfo();
      return info.getLatency(SignalType::DATA).value_or(0);
    }
  }

  return 0;
}
