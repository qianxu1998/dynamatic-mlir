//===- Debug.cpp - Switching estimation debug controls ---------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "experimental/Analysis/SwitchingEstimation/Debug.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

namespace dynamatic {
namespace experimental {
namespace {

struct SwitchingDebugConfig {
  bool enabled = false;
  uint64_t mask = 0;
  bool dumpDataChannels = false;
  bool dumpMGHandshake = false;
};

SwitchingDebugConfig config;

constexpr uint64_t categoryMask(SwitchingDebugCategory category) {
  return static_cast<uint64_t>(category);
}

constexpr uint64_t allCategoryMask() {
  return categoryMask(SwitchingDebugCategory::Pipeline) |
         categoryMask(SwitchingDebugCategory::CFDFC) |
         categoryMask(SwitchingDebugCategory::Profiling) |
         categoryMask(SwitchingDebugCategory::Data) |
         categoryMask(SwitchingDebugCategory::Mux) |
         categoryMask(SwitchingDebugCategory::Handshake) |
         categoryMask(SwitchingDebugCategory::Node) |
         categoryMask(SwitchingDebugCategory::Graph) |
         categoryMask(SwitchingDebugCategory::Dump);
}

static void enableCategoryByName(StringRef token, uint64_t &mask) {
  SmallString<32> lowered(token.trim());
  for (char &c : lowered)
    c = llvm::toLower(c);
  const StringRef name = lowered.str().trim();
  if (name.empty())
    return;

  if (name == "all") {
    mask = allCategoryMask();
    return;
  }
  if (name == "pipeline" || name == "steps") {
    mask |= categoryMask(SwitchingDebugCategory::Pipeline);
    return;
  }
  if (name == "cfdfc") {
    mask |= categoryMask(SwitchingDebugCategory::CFDFC);
    return;
  }
  if (name == "profiling") {
    mask |= categoryMask(SwitchingDebugCategory::Profiling);
    return;
  }
  if (name == "data") {
    mask |= categoryMask(SwitchingDebugCategory::Data);
    return;
  }
  if (name == "mux") {
    mask |= categoryMask(SwitchingDebugCategory::Mux);
    return;
  }
  if (name == "handshake") {
    mask |= categoryMask(SwitchingDebugCategory::Handshake);
    return;
  }
  if (name == "node" || name == "nodes") {
    mask |= categoryMask(SwitchingDebugCategory::Node);
    return;
  }
  if (name == "graph") {
    mask |= categoryMask(SwitchingDebugCategory::Graph);
    return;
  }
  if (name == "dump" || name == "dumps") {
    mask |= categoryMask(SwitchingDebugCategory::Dump);
    return;
  }

  llvm::errs() << "[WARNING] Unknown switching debug category '" << token
               << "'.\n";
}

static uint64_t parseCategories(StringRef categories) {
  if (categories.trim().empty())
    return categoryMask(SwitchingDebugCategory::Pipeline);

  uint64_t mask = 0;
  SmallVector<StringRef, 8> parts;
  categories.split(parts, ',', -1, false);
  for (StringRef token : parts)
    enableCategoryByName(token, mask);

  if (mask == 0)
    mask = categoryMask(SwitchingDebugCategory::Pipeline);
  return mask;
}

} // namespace

void configureSwitchingDebug(bool enabled, StringRef categories,
                             bool dumpDataChannels, bool dumpMGHandshake) {
  config.enabled = enabled;
  config.mask = enabled ? parseCategories(categories) : 0;
  config.dumpDataChannels = dumpDataChannels;
  config.dumpMGHandshake = dumpMGHandshake;

  if (enabled && isSwitchingDebugEnabled(SwitchingDebugCategory::Dump)) {
    config.dumpDataChannels = true;
    config.dumpMGHandshake = true;
  }
}

bool isSwitchingDebugEnabled(SwitchingDebugCategory category) {
  if (!config.enabled)
    return false;
  return (config.mask & categoryMask(category)) != 0;
}

raw_ostream &switchingDebugStream(SwitchingDebugCategory category) {
  if (isSwitchingDebugEnabled(category))
    return llvm::errs();
  return llvm::nulls();
}

bool shouldDumpDataChannelDetails() { return config.dumpDataChannels; }

bool shouldDumpMGHandshakeDetails() { return config.dumpMGHandshake; }

} // namespace experimental
} // namespace dynamatic
