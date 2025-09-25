//===- SwitchingSupport.cpp - Switching estimation ------------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Implements the supporting datastructures for the switching estimation pass.
//===----------------------------------------------------------------------===//

#include "experimental/Analysis/SwitchingEstimation/SwitchingSupport.h"
#include "dynamatic/Dialect/Handshake/HandshakeAttributes.h"
#include "dynamatic/Dialect/Handshake/HandshakeDialect.h"
#include "dynamatic/Dialect/Handshake/HandshakeInterfaces.h"
#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Dialect/Handshake/MemoryInterfaces.h"
#include "dynamatic/Support/Attribute.h"
#include "dynamatic/Support/Backedge.h"
#include "dynamatic/Support/CFG.h"
#include "dynamatic/Support/DynamaticPass.h"
#include "dynamatic/Support/LLVM.h"
#include "dynamatic/Support/Logging.h"
#include "dynamatic/Support/TimingModels.h"
#include "dynamatic/Transforms/BufferPlacement/CFDFC.h"
#include "experimental/Analysis/SwitchingEstimation/utils.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Path.h"

using namespace llvm;
using namespace mlir;
using namespace dynamatic;
using namespace dynamatic::handshake;
using namespace dynamatic::buffer;

// Function definition for SwitchingInfo
void SwitchingInfo::insertBE(unsigned srcBB, unsigned dstBB,
                             StringRef mgLabel) {
  std::pair<unsigned, unsigned> BBPair = {srcBB, dstBB};

  // Update the seg label to Backedge pair list
  segToBackedgePairMap[mgLabel.str()] = BBPair;

  // Check the existence of the backedge pair
  if (contains(staticinfo.backEdgeToCFDFC, BBPair)) {
    staticinfo.backEdgeToCFDFC[BBPair].push_back(
        static_cast<unsigned>(std::stoul(mgLabel.str())));
  } else {
    std::vector<unsigned> tmpVector{
        static_cast<unsigned>(std::stoul(mgLabel.str()))};
    staticinfo.backEdgeToCFDFC[std::make_pair(srcBB, dstBB)] = tmpVector;
  }
}

//===----------------------------------------------------------------------===//
//
// Helper Functions
//
//===----------------------------------------------------------------------===//

std::string getHandshakeNodeName(mlir::Value &selRes) {
  for (mlir::Operation *user : selRes.getUsers()) {
    // Try to get the successor's name attribute.
    if (auto nameAttr =
            user->getAttrOfType<mlir::StringAttr>("handshake.name")) {
      return nameAttr.getValue().str();
    }
  }
}

void printBEToCFDFCMap(const std::map<std::pair<unsigned, unsigned>,
                                      std::vector<unsigned>> &selMap) {
  for (const auto &selPair : selMap) {
    const std::pair<unsigned, unsigned> &key = selPair.first;
    const std::vector<unsigned> mgList = selPair.second;

    LLVM_DEBUG(llvm::dbgs() << "[DEBUG] \tBackEdge Pair: (" << key.first << ", "
                            << key.second << ") : [");

    for (const auto &selMG : mgList) {
      LLVM_DEBUG(llvm::dbgs() << selMG << ", ");
    }

    LLVM_DEBUG(llvm::dbgs() << "]\n");
  }
}

void printSegToBBListMap(
    const std::map<std::string, mlir::SetVector<unsigned>> &selMap) {
  for (const auto &selPair : selMap) {
    const std::string segLabel = selPair.first;
    const mlir::SetVector<unsigned> BBList = selPair.second;

    LLVM_DEBUG(llvm::dbgs() << "[DEBUG] \tSeg Label: " << segLabel << " : [");

    for (const auto &selBB : BBList) {
      LLVM_DEBUG(llvm::dbgs() << selBB << ", ");
    }

    LLVM_DEBUG(llvm::dbgs() << "]\n");
  }
}

std::string removeDigits(const std::string &inStr) {
  std::regex digitsRegex("\\d");

  std::string outStr = std::regex_replace(inStr, digitsRegex, "");

  return outStr;
}

unsigned getUnsigned(float_t inputValue) {
  // 1) Apply floor, which returns a double
  double floored = std::floor(static_cast<double>(inputValue));

  // 2) Cast to unsigned
  unsigned result = static_cast<unsigned>(floored);

  return result;
}

std::string getNodeType(const std::string &nodeName) {
  size_t pos = 0;
  while (pos < nodeName.size() && std::isalpha(nodeName[pos])) {
    ++pos;
  }
  return nodeName.substr(0, pos);
}
