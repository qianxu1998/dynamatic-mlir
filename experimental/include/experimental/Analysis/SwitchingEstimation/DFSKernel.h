//===- DFSKernel.h - Switching estimation -------------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains helper functions for DFS kernels
//
//===----------------------------------------------------------------------===//

#ifndef EXPERIMENTAL_ANALYSIS_SWITCHINGESTIMATION_DFSKERNEL_H
#define EXPERIMENTAL_ANALYSIS_SWITCHINGESTIMATION_DFSKERNEL_H

#include "experimental/Analysis/SwitchingEstimation/SwitchingSupport.h"
#include "experimental/Analysis/SwitchingEstimation/NodeModels.h"
#include "experimental/Analysis/SwitchingEstimation/utils.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include <string>
#include <type_traits>
#include <unordered_set>
#include <vector>

using namespace llvm;
using namespace mlir;
using namespace dynamatic;
using namespace dynamatic::buffer;

// A helper function to safely call a function if the pointer is not null
template <typename Arg>
void safeCall(std::nullptr_t, Arg &&) {
  // Do nothing for nullptr
}

// A helper function to safely call a function if the pointer is not null
template <typename Fn, typename Arg>
void safeCall(Fn &&fn, Arg &&arg) {
  fn(std::forward<Arg>(arg));
}

template <typename SuccRangeGetter, typename PreVisitFn = std::nullptr_t,
          typename PostVisitFn = std::nullptr_t>
inline void genericDFS(llvm::ArrayRef<std::string> roots, // root nodes stack
                       SuccRangeGetter &&succOf, PreVisitFn &&pre = nullptr,
                       PostVisitFn &&post = nullptr) {
  using llvm::SmallVector;

  if (roots.empty()) // bounds check
    return;

  std::unordered_set<std::string> visited; // visited nodes
  SmallVector<std::string, 64> stack(roots.begin(),
                                     roots.end()); // dfs stack

  while (!stack.empty()) { // do dfs until stack empty
    auto n = stack.pop_back_val();// get the next node
    if (!visited.insert(n).second) // if the node is visited,skip
      continue;
    
    safeCall(pre, n); // process the current node

    for (auto successor : succOf(n))                // for each successor of the node n
      if (visited.find(successor) == visited.end()) // if successor not visited
        stack.push_back(successor);// add successor to stack

    safeCall(post, n); // post-visit callback
  }
}

template <typename SuccRangeGetter, typename PreVisitFn = std::nullptr_t,
          typename PostVisitFn = std::nullptr_t>
inline void genericDFSnoVisit(llvm::ArrayRef<std::string> roots, // root nodes stack
                       SuccRangeGetter &&succOf, PreVisitFn &&pre = nullptr,
                       PostVisitFn &&post = nullptr) {
  using llvm::SmallVector;

  if (roots.empty()) // bounds check
    return;

  SmallVector<std::string, 64> stack(roots.begin(),
                                     roots.end()); // dfs stack

  while (!stack.empty()) { // do dfs until stack empty
    auto n = stack.pop_back_val();// get the next node

    
    safeCall(pre, n); // process the current node

    for (auto successor : succOf(n))                // for each successor of the node n
        stack.push_back(successor);// add successor to stack
    safeCall(post, n); // post-visit callback
  }
}

template<typename SuccFn, typename PreFn, typename PostFn>
void dfsWithEvents(ArrayRef<std::string> roots,
                   SuccFn &&succOf,
                   PreFn  &&pre,
                   PostFn &&post) {
  enum class Evt { Enter, Exit };
  using Item = std::pair<std::string, Evt>;

  SmallVector<Item, 64> stack;
  for (auto &r : roots)
    stack.emplace_back(r, Evt::Enter);

  while (!stack.empty()) {
    auto [node, evt] = stack.pop_back_val();
    if (evt == Evt::Exit) {
      post(node);
    } else {
      pre(node);
      // schedule exit *after* all children
      stack.emplace_back(node, Evt::Exit);
      // push children in reverse so the first child is processed next
      auto children = succOf(node);
      for (auto it = children.rbegin(); it != children.rend(); ++it)
        stack.emplace_back(*it, Evt::Enter);
    }
  }
}

// returns true if node matches either exactly or as substring
inline bool searchInSuccessorLists( SwitchingInfo &si,
                                    const std::string &mappedNode,
                                    const std::string &nodeName,
                                    const std::string &segment,
                                    bool useSubstring = false ) {
  if (contains(nodeName, mappedNode)) return true;
  auto &m = si.data.dfgBaseNodeValue[mappedNode]->segSucNodeMap[segment];
  auto matches = [&](auto const &lst){
    return std::any_of(lst.begin(), lst.end(), [&](auto const &s){
      return useSubstring ? s.find(nodeName) != std::string::npos
                          : s == nodeName;
    });
  };
  return matches(m.control) || matches(m.data)
      || matches(m.glitch)  || matches(m.original);
}

// Check if a node is a DataBase node
inline bool isDataBase(const SwitchingInfo &SwitchInfo, const std::string &node) {
  return contains(SwitchInfo.staticinfo.dataflowGraph->allDataBaseNode,node);
}

// SKip conditional branch nodes' data port during DFS
inline bool skipCondBrPort(const SwitchingInfo &si,const std::string &prev,const std::string &cur) {
// only care about cond_br nodes
  if (!contains(cur,"cond_br")) 
    return false;

  auto *cbr = llvm::dyn_cast<CBrNode>(si.staticinfo.dataflowGraph->nodes[cur].get());
  if (!cbr) return false;

  // Python did:
  //  if pre_node == cond_pre_node_name:
  //     if data_pre_node_name != "":  // i.e. node *does* have a data port
  //        non_conditional_flag = False  → skip
  return (prev == cbr->condPreNodeName && !cbr->dataPreNodeName.empty());
}

inline bool crossesCondPort(const SwitchingInfo &SwitchInfo, const std::string &prev,
                            const std::string &cur) {
  // cond_br control port?
  if (contains(cur, "cond_br")) {
    auto *cbr = llvm::dyn_cast<CBrNode>(SwitchInfo.staticinfo.dataflowGraph->nodes[cur].get());
    return cbr && prev == cbr->condPreNodeName;
  }
  // mux control port?
  if (contains(cur, "mux")) {
    auto *mux = llvm::dyn_cast<MuxNode>(SwitchInfo.staticinfo.dataflowGraph->nodes[cur].get());
    return mux && prev == mux->conPreNodeName;
  }
  return false;
}

// Skip mux nodes' condition port during DFS
inline bool skipMuxPort(const SwitchingInfo &si,
                        const std::string &prev,
                        const std::string &cur) {
  if (cur.find("mux") == std::string::npos) 
    return false;

  auto *mux = llvm::dyn_cast<MuxNode>(si.staticinfo.dataflowGraph->nodes[cur].get());
  if (!mux) return false;

  // Python did:
  //  if pre_node == con_pre_node_name:
  //     non_conditional_flag = False  → skip
  return (prev == mux->conPreNodeName);
}

//helper functions for building the succeeding nodes to visit in the dfs kernel

// check if buffer is outside of the segment
inline bool bufferOutsideSeg(const SwitchingInfo &SwitchInfo, const std::string &n,
                             llvm::ArrayRef<unsigned> segBBs) {
  if (!(contains(n,"buffer"))) 
    return false;
  unsigned bb = SwitchInfo.staticinfo.dataflowGraph->nodes[n]->bbindex;
  return !llvm::is_contained(segBBs, bb);
}
// check if it's invalid backge, return true
inline bool isInvalidBackedge(const SwitchingInfo& SwitchInfo, const llvm::StringRef segLabel,
  const std::string& from, const std::string& to){
  auto it = SwitchInfo.segInvalidBackedgesMap.find(segLabel);// as workaroundfor const
  if(it==SwitchInfo.segInvalidBackedgesMap.end()){
    return false;
  }
const auto& selInvalidBEList=it->second;
  std::pair<std::string, std::string> edge = std::make_pair(from, to);
  return std::find(selInvalidBEList.begin(), selInvalidBEList.end(), edge ) != selInvalidBEList.end();
}
//check if it's in excluded list
inline bool isExcluded(const std::vector<std::string>& excludingList, const std::string &node){
  return std::find(excludingList.begin(), excludingList.end(), node) != excludingList.end();
}
inline bool isInPath(const std::unordered_set<std::string> &pathSet, const std::string &node) {
  return pathSet.find(node) != pathSet.end();
}

inline bool isEndNode(const std::string &node, const llvm::StringRef &segLabel) {
  return contains(node, "end") && segLabel != "E";
}

inline bool isMemController(const std::string &node) {
  return contains(node, "mem_controller");
}


#endif // EXPERIMENTAL_ANALYSIS_SWITCHINGESTIMATION_DFSKERNEL_H
