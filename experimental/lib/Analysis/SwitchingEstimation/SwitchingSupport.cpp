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
#include "experimental/Analysis/SwitchingEstimation/Debug.h"
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
using namespace dynamatic::experimental;
using namespace dynamatic::handshake;
using namespace dynamatic::buffer;

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

  return "";
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
