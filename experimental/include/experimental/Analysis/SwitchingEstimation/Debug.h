//===- Debug.h - Switching estimation debug controls -----------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Centralized debug configuration for the switching-estimation analysis.
//
//===----------------------------------------------------------------------===//

#ifndef EXPERIMENTAL_ANALYSIS_SWITCHINGESTIMATION_DEBUG_H
#define EXPERIMENTAL_ANALYSIS_SWITCHINGESTIMATION_DEBUG_H

#include "dynamatic/Support/LLVM.h"

#include <cstdint>

namespace dynamatic {
namespace experimental {

enum class SwitchingDebugCategory : uint64_t {
  Pipeline = 1ULL << 0,
  CFDFC = 1ULL << 1,
  Profiling = 1ULL << 2,
  Data = 1ULL << 3,
  Mux = 1ULL << 4,
  Handshake = 1ULL << 5,
  Node = 1ULL << 6,
  Graph = 1ULL << 7,
  Dump = 1ULL << 8,
};

/// Configure runtime debug behavior for the current switching-estimation run.
/// `categories` is a comma-separated list (case-insensitive) containing:
///   pipeline, cfdfc, profiling, data, mux, handshake, node, graph, dump, all
void configureSwitchingDebug(bool enabled, llvm::StringRef categories,
                             bool dumpDataChannels, bool dumpMGHandshake);

/// Returns true if `category` is currently enabled.
bool isSwitchingDebugEnabled(SwitchingDebugCategory category);

/// Stream helper that routes to `errs()` when enabled, otherwise `nulls()`.
llvm::raw_ostream &switchingDebugStream(SwitchingDebugCategory category);

/// Debug dump toggles (independent from regular console debug output).
bool shouldDumpDataChannelDetails();
bool shouldDumpMGHandshakeDetails();

} // namespace experimental
} // namespace dynamatic

#endif // EXPERIMENTAL_ANALYSIS_SWITCHINGESTIMATION_DEBUG_H
