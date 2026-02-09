//===- HandshakeInsertGates.cpp - Insert gates in Handshake IR -*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the pass that inserts `handshake.gate` operations on
// outputs of arithmetic operations.
//
//===----------------------------------------------------------------------===//

#include "experimental/Transforms/HandshakeInsertGates.h"
#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Support/CFG.h"

using namespace mlir;
using namespace dynamatic;
using namespace dynamatic::experimental;
using namespace dynamatic::experimental::gating;

namespace {

struct HandshakeInsertGatesPass
    : public dynamatic::experimental::gating::impl::
          HandshakeInsertGatesBase<HandshakeInsertGatesPass> {

  void runDynamaticPass() override {
    ModuleOp modOp = getOperation();
    OpBuilder builder(&getContext());

    for (handshake::FuncOp funcOp : modOp.getOps<handshake::FuncOp>())
      insertGatesInFunc(funcOp, builder);
  }

private:
  static void insertGatesInFunc(handshake::FuncOp funcOp, OpBuilder &builder) {
    SmallVector<OpOperand *> usesToGate;

    funcOp.walk([&](Operation *op) {
      auto arithOp = dyn_cast<handshake::ArithOpInterface>(op);
      if (!arithOp || isa<handshake::ConstantOp>(op))
        return;

      for (OpResult result : op->getResults()) {
        for (OpOperand &use : result.getUses()) {
          // Avoid repeatedly wrapping channels already consumed by a gate.
          if (isa<handshake::GateOp>(use.getOwner()))
            continue;
          usesToGate.push_back(&use);
        }
      }
    });

    for (OpOperand *use : usesToGate) {
      Operation *user = use->getOwner();
      Value channel = use->get();

      builder.setInsertionPoint(user);
      handshake::GateOp gateOp =
          builder.create<handshake::GateOp>(channel.getLoc(), channel);
      inheritBBFromValue(channel, gateOp);
      use->set(gateOp.getResult());
    }
  }
};

} // namespace

std::unique_ptr<dynamatic::DynamaticPass>
dynamatic::experimental::gating::createHandshakeInsertGates() {
  return std::make_unique<HandshakeInsertGatesPass>();
}
