//===- utils.h - Switching estimation -------------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares utility functions for the switching estimation pass
//
//===----------------------------------------------------------------------===//

#ifndef EXPERIMENTAL_ANALYSIS_SWITCHINGESTIMATION_UTILS_H
#define EXPERIMENTAL_ANALYSIS_SWITCHINGESTIMATION_UTILS_H
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include <map>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_set>

#include "experimental/Analysis/SwitchingEstimation/GraphModel.h"

inline bool contains(const std::string &s, std::string_view substring) {
  return s.find(substring) != std::string::npos;
}

inline bool contains(std::string_view s, std::string_view sub) {
  return s.find(sub) != std::string::npos;
}

inline bool contains(const std::map<std::string, std::shared_ptr<AdjGraph>> &m,
                     std::string const &key) {
  return m.count(key) != 0;
}

inline bool contains(const std::unordered_set<std::string> &m,
                     std::string const &key) {
  return m.count(key) != 0;
}

template <typename ValueT>
inline bool contains(const llvm::StringMap<ValueT> &m, llvm::StringRef key) {
  return m.find(key) != m.end();
}

template <typename Map>
inline bool contains(const Map &m, const typename Map::key_type &key) {
  return m.find(key) != m.end();
}

template <typename Container, typename T>
inline bool containsValue(const Container &c, const T &value) {
  return std::find(c.begin(), c.end(), value) != c.end();
}

#endif // EXPERIMENTAL_ANALYSIS_SWITCHINGESTIMATION_UTILS_H
