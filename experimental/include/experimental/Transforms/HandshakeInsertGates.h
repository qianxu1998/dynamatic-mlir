//===- HandshakeInsertGates.h - Insert gates in Handshake IR ---*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the --handshake-insert-gates pass.
//
//===----------------------------------------------------------------------===//

#ifndef EXPERIMENTAL_TRANSFORMS_HANDSHAKEINSERTGATES_H
#define EXPERIMENTAL_TRANSFORMS_HANDSHAKEINSERTGATES_H

#include "dynamatic/Support/DynamaticPass.h"

namespace dynamatic {
namespace experimental {
namespace gating {

std::unique_ptr<dynamatic::DynamaticPass> createHandshakeInsertGates();

#define GEN_PASS_DECL_HANDSHAKEINSERTGATES
#define GEN_PASS_DEF_HANDSHAKEINSERTGATES
#include "experimental/Transforms/Passes.h.inc"

} // namespace gating
} // namespace experimental
} // namespace dynamatic

#endif // EXPERIMENTAL_TRANSFORMS_HANDSHAKEINSERTGATES_H
