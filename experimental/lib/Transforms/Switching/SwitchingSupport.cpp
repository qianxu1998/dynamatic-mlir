//===- SwitchingSupport.cpp - Estimate Swithicng Activities ------*- C++
//-*-===//
//
// Implements the supporting datastructures for the switching estimation
// pass
//
//===----------------------------------------------------------------------===//

#include "experimental/Transforms/Switching/SwitchingSupport.h"
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
#include "experimental/Transforms/Switching/ExecModel.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Path.h"

using namespace mlir;
using namespace dynamatic;




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
};

void printBEToCFDFCMap(const std::map<std::pair<unsigned, unsigned>,
                                      std::vector<unsigned>> &selMap) {
  for (const auto &selPair : selMap) {
    const std::pair<unsigned, unsigned> &key = selPair.first;
    const std::vector<unsigned> mgList = selPair.second;

    llvm::dbgs() << "[DEBUG] \tBackEdge Pair: (" << key.first << ", "
                 << key.second << ") : [";

    for (const auto &selMG : mgList) {
      llvm::dbgs() << selMG << ", ";
    }

    llvm::dbgs() << "]\n";
  }
}

void printSegToBBListMap(
    const std::map<std::string, mlir::SetVector<unsigned>> &selMap) {
  for (const auto &selPair : selMap) {
    const std::string segLabel = selPair.first;
    const mlir::SetVector<unsigned> BBList = selPair.second;

    llvm::dbgs() << "[DEBUG] \tSeg Label: " << segLabel << " : [";

    for (const auto &selBB : BBList) {
      llvm::dbgs() << selBB << ", ";
    }

    llvm::dbgs() << "]\n";
  }
}

void printMgNodeInfo(const MgNodeInfo &info) {
  llvm::dbgs() << "MgNodeInfo contents:\n";

  // Print "original"
  llvm::dbgs() << "  original: [";
  for (size_t i = 0; i < info.original.size(); ++i) {
    if (i > 0)
      llvm::dbgs() << ", ";
    llvm::dbgs() << info.original[i];
  }
  llvm::dbgs() << "]\n";

  // Print "glitch"
  llvm::dbgs() << "  glitch: [";
  for (size_t i = 0; i < info.glitch.size(); ++i) {
    if (i > 0)
      llvm::dbgs() << ", ";
    llvm::dbgs() << info.glitch[i];
  }
  llvm::dbgs() << "]\n";

  // Print "control"
  llvm::dbgs() << "  control: [";
  for (size_t i = 0; i < info.control.size(); ++i) {
    if (i > 0)
      llvm::dbgs() << ", ";
    llvm::dbgs() << info.control[i];
  }
  llvm::dbgs() << "]\n";

  // Print "data"
  llvm::dbgs() << "  data: [";
  for (size_t i = 0; i < info.data.size(); ++i) {
    if (i > 0)
      llvm::dbgs() << ", ";
    llvm::dbgs() << info.data[i];
  }
  llvm::dbgs() << "]\n";

  // Print "dataWidthMap"
  llvm::dbgs() << "  dataWidthMap:\n";
  for (const auto &pair : info.dataWidthMap) {
    llvm::dbgs() << "    \"" << pair.first << "\" => " << pair.second << "\n";
  }
  llvm::dbgs() << "\n";
}

// Helper function: extracts the initial alphabetic portion from a node name.
std::string getNodeType(const std::string &nodeName) {
  size_t pos = 0;
  while (pos < nodeName.size() && std::isalpha(nodeName[pos])) {
    ++pos;
  }
  return nodeName.substr(0, pos);
}

// Function to print a vector of strings (mainStack)
void printMainStack(const std::vector<std::string> &mainStack) {
  llvm::dbgs() << "Main Stack: [ ";
  for (const auto &elem : mainStack) {
    llvm::dbgs() << elem << " ";
  }
  llvm::dbgs() << "]" << "\n";
}

// Function to print a vector of vector of strings (adjStack)
void printAdjStack(const std::vector<std::vector<std::string>> &adjStack) {
  llvm::dbgs() << "Adjacency Stack:" << "\n";
  for (size_t i = 0; i < adjStack.size(); ++i) {
    llvm::dbgs() << "  Level " << i << ": [ ";
    for (const auto &elem : adjStack[i]) {
      llvm::dbgs() << elem << " ";
    }
    llvm::dbgs() << "]" << "\n";
  }
}

std::string removeDigits(const std::string &inStr) {
  std::regex digitsRegex("\\d");

  std::string outStr = std::regex_replace(inStr, digitsRegex, "");

  return outStr;
}

std::vector<std::string> split(const std::string &s,
                               const std::string &delimiter) {
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
    return ""; // Return an empty string if there are only whitespaces
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

unsigned getUnsigned(float_t inputValue) {
  // 1) apply floor, which returns a double
  double floored = std::floor(static_cast<double>(inputValue));

  // 2) cast to unsigned
  unsigned result = static_cast<unsigned>(floored);

  return result;
}

