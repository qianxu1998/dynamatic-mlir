//===- NodeHelpers.h - Switching estimation -------------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares switching models for different unit types
//
//===----------------------------------------------------------------------===//
#pragma once
#ifndef EXPERIMENTAL_ANALYSIS_SWITCHINGESTIMATION_NODEMODELS_H
#define EXPERIMENTAL_ANALYSIS_SWITCHINGESTIMATION_NODEMODELS_H

#include "experimental/Analysis/SwitchingEstimation/SwitchingSupport.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/DenseMap.h"

#include <map>
#include <memory>
#include <string>
#include <vector>

#define DEBUG_TYPE "switching-estimation"
// Helper macro to define LLVM-style RTTI for node classes.
// Usage: DEFINE_NODE_KIND(ClassName, NodeKind::SomeEnum)
#define DEFINE_NODE_KIND(ClassName, KindEnum) \
  NodeKind getKind() const override { return KindEnum; } \
  static bool classof(const AdjNode *node) { \
    return node->getKind() == KindEnum; \
  }

using namespace llvm;
using namespace mlir;
using namespace dynamatic;
using namespace dynamatic::buffer;

//===----------------------------------------------------------------------===//
//
// Node Model Helpers
//
//===----------------------------------------------------------------------===//

// This function generates a full IISet with all bits set to 1
inline IISet fullSet(unsigned II) { return IISet(II, true); }

// This function generates a singleton IISet with only the given bit set to 1
inline IISet singleton(unsigned bit, unsigned II) {
  IISet s(II, false); s.set(bit); return s;
}

// This function generates the union of two IISets
inline IISet union_(const IISet &a, const IISet &b) { IISet x = a; x |= b; return x; }

// This function generates the intersection of two IISets
inline IISet intersection_(const IISet &a, const IISet &b) { IISet x = a; x &= b; return x; }

// A generic “create‑if‑absent then mutate” helper for per‑node handshake map
template <auto MapPtr, typename KeyT, typename Updater>
inline void updateHandshake(AdjNode *node, const KeyT &key, Updater &&updater) {
  auto &map = node->*MapPtr;
  auto it = map.try_emplace(key).first;
  updater(it->second);
}

// Set the valid switching count for a given key
template <typename KeyT>
inline void setReady(AdjNode *node, const KeyT &key, unsigned val) {
  updateHandshake<&AdjNode::readySignal>(node, key,
      [&](unsigned &slot){ slot = val; });
}

// Set the ready switching count for a given key
template <typename KeyT>
inline void setValid(AdjNode *node, const KeyT &key, unsigned val) {
  updateHandshake<&AdjNode::validSignal>(node, key,
      [&](unsigned &slot){ slot = val; });
}

// Set the valid switching count for a given key
template <typename KeyT>
inline void setVSet(AdjNode *node, const KeyT &key, IISet set) {
  updateHandshake<&AdjNode::setV>(node, key,
      [&](IISet &slot){ slot = std::move(set); });
}

// Set the read switching count for a given key
template <typename KeyT>
inline void setRSet(AdjNode *node, const KeyT &key, IISet set) {
  updateHandshake<&AdjNode::setR>(node, key,
      [&](IISet &slot){ slot = std::move(set); });
}

//===----------------------------------------------------------------------===//
//
// Declare switching model for data nodes
//
//===----------------------------------------------------------------------===//

//
// Model for Buffer Node
//
class BufferNode : public AdjNode {
public:
  // TODO: Further refine the handshake switching model based on the new buffer type
  // Constructor
  BufferNode(mlir::Operation *op, const std::vector<std::string> &predecessors,
              const std::vector<std::string> &successors,
              const std::map<std::string, unsigned> &sucDataWidthMap,
              const unsigned &latency, const unsigned &bbIndex);

  // Handshake Counting functions
  void calValidSwitching(const std::string &sucNodeName, unsigned II);
  void calReadySwitching(const std::string &preNodeName, unsigned II);

  void calValidSet(const std::string &sucNodeName, const IISet &inSetV);
  void calReadySet(const std::string &preNodeName, const IISet &inSetR);

  // Override printDetail function
  void printNodeDetails() override;
  DEFINE_NODE_KIND(BufferNode,  NodeKind::BufferNodeKind)  ;

  //
  /// Internal variable
  //
  unsigned START;
  float_t occupancy;
  unsigned numSlots;
  bool transparent;
  BufferType buffType;
};

//
// Model for Join Node
//
class JoinNode : public AdjNode {
public:
  using AdjNode::AdjNode; // Inherit the constructor

  // - calValidSwitching: if either numValid0 or numValid1 > 0, set valid signal to 2;
  // if both are 0, then set valid signal to 0; if II==1, then also set to 0; otherwise, set to 2.
  void calValidSwitching(const std::string &sucNodeName,
                        int &numValid0,
                        int &numValid1,
                        unsigned &II);

  // calValidSet: if the valid signal for sucNodeName is 0 (or II==1), then store the full set {0,...,II-1};
  // otherwise, store only {nodeStartTime}.
  void calValidSet(const std::string &sucNodeName,
                  unsigned &nodeStartTime,
                  unsigned &II);

  // calReadySwitching: if both numValid and numReady are 0, set ready signal to 0;
  // if either > 0, set it to 2.
  void calReadySwitching(const std::string &preNodeName,
                         int &numValid,
                         int &numReady);

  // calReadySet: if both set_valid and set_ready are provided (non-null),
  // then take their intersection; otherwise, if readySignal[preNodeName] is 0 then store the full set {0,...,II-1},
  // if >0 then store {nodeStartTime}.
  // (We pass the sets as pointers so that a null pointer represents “None”.)
  void calReadySet(const std::string &preNodeName,
                   const IISet *setValid,
                   const IISet *setReady,
                   unsigned &nodeStartTime,
                   unsigned &II);

  // setReadySet: as a convenience, take the first key from setVMap and copy its set into setRMap for preNodeName.
  void setReadySet(const std::string &preNodeName);

  DEFINE_NODE_KIND(JoinNode,  NodeKind::JoinNodeKind)  ;
};

//
// Model for Pass Node
//
class PassNode : public AdjNode {
public:
  using AdjNode::AdjNode; // Inherit the constructor

  // Handshake counting functions:
  // calValidSwitching: if numValid is 0, set valid signal to 0; else to 2.
  void calValidSwitching(const std::string &sucNodeName,
                         int &numValid);

  // calValidSet: if validSignal[sucNodeName] is 0, store full set {0,...,II-1}; else store {nodeStartTime}.
  void calValidSet(const std::string &sucNodeName,
                   unsigned &nodeStartTime,
                   unsigned &II);

  // calReadySwitching: if numReady is 0, set ready signal to 0; else to 2.
  void calReadySwitching(const std::string &preNodeName,
                         int &numReady);

  // calReadySet: if a set (setR) is provided, simply store it; else if readySignal[preNodeName] is 0, store full set; 
  // if >0, store {nodeStartTime}; otherwise, if setVMap is not empty and its full set is not equal to {0,...,II-1}, use that.
  void calReadySet(const std::string &preNodeName,
                   const IISet *setReady,
                   unsigned &nodeStartTime,
                   unsigned &II);

  // setReadySwitching: directly set readySignal.
  void setReadySwitching(const std::string &preNodeName,
                         unsigned &numReadySwitches);

  // setReadySet: store the provided set into setRMap.
  void setReadySet(const std::string &preNodeName,
                   const IISet &setReady);

  DEFINE_NODE_KIND(PassNode, NodeKind::PassNodeKind);
};

//
// Model for Cmpi Node: derived from JoinNode
//
class CmpiNode : public JoinNode {
public:
  using JoinNode::JoinNode; // Inherit the constructor
  DEFINE_NODE_KIND(CmpiNode, NodeKind::CmpiNodeKind);
};

//
// Model for Addi Node: derived from JoinNode
//
class AddiNode : public JoinNode {
public:
  using JoinNode::JoinNode; // Inherit the constructor
  DEFINE_NODE_KIND(AddiNode, NodeKind::AddiNodeKind);
};

//
// Model for Subi Node
//
class SubiNode : public JoinNode {
public:
  using JoinNode::JoinNode;  // inherit all of JoinNode’s constructors
  DEFINE_NODE_KIND(SubiNode,  NodeKind::SubiNodeKind);
};

//
// Model for Muli Node
//
class MuliNode : public JoinNode {
public:
  using JoinNode::JoinNode;  // inherit all of JoinNode’s constructors
  DEFINE_NODE_KIND(MuliNode,  NodeKind::MuliNodeKind);
};

//
// Model for Extsi Node
//
class ExtsiNode : public PassNode {
public:
  using PassNode::PassNode;  // inherit all of PassNode’s constructors
  DEFINE_NODE_KIND(ExtsiNode, NodeKind::ExtsiNodeKind);
};

//
// Model for Load Node
//
class DLoadNode : public AdjNode {
public:
  DLoadNode(mlir::Operation *op,
            const std::vector<std::string> &predecessors,
            const std::vector<std::string> &successors,
            const std::map<std::string, unsigned> &sucDataWidthMap,
            const unsigned &latency, const unsigned& bbIndex);

  // Handshake functions:
  // For valid switching: if num_valid == 0, valid signal is 0; else if > 0, valid signal is 2.
  void calValidSwitching(const std::string &sucNodeName, int &numValid);

  // For valid set: if valid signal is 0, use the full set {0,...,II-1};
  // else, if an input set is provided, shift it by the node latency.
  void calValidSet(const std::string &sucNodeName, const IISet *inSetV, unsigned &II);

  // For ready switching: if num_ready == 0, ready signal is 0; else ready signal is 2.
  void calReadySwitching(const std::string &preNodeName, int &numReady);

  // For ready set: for a load node, we assume the ready set is taken directly if provided;
  // otherwise, if readySignal is 0 then use full set, or else use {nodeStartTime}.
  void calReadySet(const std::string &preNodeName, const IISet *inSetR, 
                    unsigned &nodeStartTime, unsigned &II);

  // Update the data output channels.
  //   - For data output: if a previous value exists, compute diff (using XOR) and update toggle counts.
  //   - For address output: if the new address is -10, reuse the last valid value.
  void updateDataout(int &inputData, int addressValue);

  void printNodeDetails();

  DEFINE_NODE_KIND(DLoadNode, NodeKind::DLoadNodeKind);

  // Extra members for DLoad nodes
  std::string dataOutNodeName;    // Name for the data out channel
  std::string addressOutNodeName; // this should be the corresponding name of the mem_controller
  std::string addressInNodeName;  // Name of the address input node name
};

//
// Model for Store Node
// TODO: Separate MC_store and LSQ_store
class DStoreNode : public AdjNode {
public:
  // Constructor: forwards to AdjNode.
  DStoreNode(mlir::Operation *op,
             const std::vector<std::string> &predecessors,
             const std::vector<std::string> &successors,
             const std::map<std::string, unsigned> &sucDataWidthMap,
             unsigned latency, const unsigned& bbIndex);

  // Handshake functions for d_store.
  // For valid switching on the memory controller channel (key "mc").
  void calValidSwitching(int numValid1, int numValid2);
  void calValidSet(const IISet *setV0,
                  const IISet *setV1,
                  unsigned II);

  void calReadySwitching(const std::string &preNodeName,
                                 int numValid1,
                                 int numValid2);
  void calReadySet(const std::string &preNodeName,
                           const IISet *setV0,
                           const IISet *setV1,
                           unsigned II);

  // Update data output based on the source input node.
  void updateDataout(int inputData, const std::string &srcInputNode);

  void printNodeDetails();

  // Handshake checking (e.g. comparing size of readySignal with number of predecessors)
  bool handshakeSwitchingChecking();

  DEFINE_NODE_KIND(DStoreNode, NodeKind::DStoreNodeKind);

  // Extra members for DStoreNode.
  std::string dataInNode;
  std::string addressInNode;
  std::string dataInSrcNode;
  std::string addressInSrcNode;
};

//
// TODO: Add Model for Lazy fork node
//

//
// Model for fork Node
//
class ForkNode : public AdjNode {
public:
  using AdjNode::AdjNode; // Inherit the constructor

  // Handshake functions:
  // calValidSwitching:
  //   Parameters:
  //     - sucNodeName: key for a particular successor.
  //     - numValid: the valid count for the input channel.
  //     - setRDict: a mapping from successor names to their active ready sets.
  //     - numReadyDict: a mapping from successor names to their ready counts.
  //     - sucNodeStart: the starting cycle for this output channel.
  //     - nodeSteadyStart: the steady‐state start cycle of this node.
  //     - II: the initiation interval.
  void calValidSwitching(const std::string &sucNodeName,
                         int numValid,
                         std::unordered_map<std::string, IISet *> &setRDict,
                         const std::unordered_map<std::string, int> &numReadyDict,
                         unsigned sucNodeStart,
                         unsigned nodeSteadyStart,
                         unsigned II);

  // calValidSet:
  //   If the valid signal for sucNodeName is 0, assign the full set {0,...,II-1}.
  //   Otherwise, if numValid > 0, assign {nodeStartTime}; else, assign the ready set from setRDict.
  void calValidSet(const std::string &sucNodeName,
                   unsigned nodeStartTime,
                   int numValid,
                   std::unordered_map<std::string, IISet *> &setRDict,
                   unsigned II);

  // calReadySwitching:
  //   Takes a list of ready counts (one per channel) and if any value > 0 sets the ready signal to 2;
  //   otherwise, if all values are 0, sets it to 0.
  void calReadySwitching(const std::string &preNodeName,
                         const std::vector<int> &numReadyList);

  // calReadySet:
  //   If a ready set (setR) is provided (non-null), use it.
  //   Otherwise, if readySignal for the predecessor is 0, assign the full set.
  //   Else, compute the union of all ready sets from all successors and then assign:
  //     - If the union equals the full set {0,...,II-1}, assign {0};
  //     - Otherwise, assign the last element of the union.
  void calReadySet(const std::string &preNodeName,
                   const std::unordered_map<std::string, IISet *> &setRDict,
                   unsigned II);
  
  DEFINE_NODE_KIND(ForkNode, NodeKind::ForkNodeKind);
};

//
// Model for Shli Node
//
class ShliNode : public JoinNode {
public:
  using JoinNode::JoinNode; // Inherit the constructor
  DEFINE_NODE_KIND(ShliNode, NodeKind::ShliNodeKind);
};

//
// Model for Shrsi Node
//
class ShrsiNode : public JoinNode {
public:
  using JoinNode::JoinNode; // Inherit the constructor
  DEFINE_NODE_KIND(ShrsiNode, NodeKind::ShrsiNodeKind);
};

//
// Model for ShruiNode Node
//
class ShruiNode : public JoinNode {
public:
  using JoinNode::JoinNode; // Inherit the constructor
  DEFINE_NODE_KIND(ShruiNode, NodeKind::ShruiNodeKind);
};

//
// Model for Trunci Node
//
class TrunciNode : public PassNode {
public:
  using PassNode::PassNode; // Inherit the constructor
  DEFINE_NODE_KIND(TrunciNode, NodeKind::TrunciNodeKind);
};

//
// Model for Extui Node
//
class ExtuiNode : public PassNode {
public:
  using PassNode::PassNode; // Inherit the constructor
  DEFINE_NODE_KIND(ExtuiNode, NodeKind::ExtuiNodeKind);
};

//
// Model for ConstantNode Node
//
class ConstantNode : public PassNode {
public:
  using PassNode::PassNode; // Inherit the constructor
  DEFINE_NODE_KIND(ConstantNode, NodeKind::ConstantNodeKind);
};

//
// Model for OriNode 
//
class OriNode : public JoinNode {
public:
    using JoinNode::JoinNode;  // inherit all of JoinNode’s constructors
    DEFINE_NODE_KIND(OriNode,  NodeKind::OriNodeKind)  ;
};

//
// Model for Andi 
//
class AndiNode : public JoinNode {
public:
    using JoinNode::JoinNode;  // inherit all of JoinNode’s constructors
    DEFINE_NODE_KIND(AndiNode,  NodeKind::AndiNodeKind)  ;
};

//
// Model for Source Node
//
class SourceNode : public AdjNode {
public:
  SourceNode(mlir::Operation *op,
             const std::vector<std::string> &predecessors,
             const std::vector<std::string> &successors,
             const std::map<std::string, unsigned> &sucDataWidthMap,
             unsigned latency, const unsigned& bbIndex)
      : AdjNode(op, predecessors, successors, sucDataWidthMap, latency, bbIndex) {}

  void calValidSwitching(const std::string &sucNodeName) {
    validSignal[sucNodeName] = 0;
  }
  
  void calValidSet(const std::string &sucNodeName, unsigned II) {
    // Build full set {0, 1, ..., II-1}
    setV[sucNodeName] = fullSet(II);
  }
  
  void calReadySwitching(const std::string &preNodeName) {
    readySignal[preNodeName] = 0;
  }
  
  void calReadySet(const std::string &preNodeName, unsigned II) {
    setR[preNodeName] = fullSet(II);
  }

  DEFINE_NODE_KIND(SourceNode, NodeKind::SourceNodeKind);
};

//
// Model for End Node
//
class EndNode : public AdjNode {
public:
  EndNode(mlir::Operation *op,
           const std::vector<std::string> &predecessors,
           const std::vector<std::string> &successors,
           const std::map<std::string, unsigned> &sucDataWidthMap,
           unsigned latency, const unsigned& bbIndex)
      : AdjNode(op, predecessors, successors, sucDataWidthMap, latency, bbIndex) {}

  DEFINE_NODE_KIND(EndNode, NodeKind::EndNodeKind);
};



//===----------------------------------------------------------------------===//
//
// Declare switching model for control nodes
//
//===----------------------------------------------------------------------===//

//
// Model for MergeNode Node
//
class MergeNode : public AdjNode {
public:
  // Constructor: simply forward the parameters to the AdjNode constructor.
  MergeNode(mlir::Operation *op,
            const std::vector<std::string> &predecessors,
            const std::vector<std::string> &successors,
            const std::map<std::string, unsigned> &sucDataWidthMap,
            unsigned latency, const unsigned& bbIndex)
      : AdjNode(op, predecessors, successors, sucDataWidthMap, latency, bbIndex) {}

  // calValidSwitching:
  //   setVList: vector of active valid sets for each input channel.
  //   numVList: vector of valid signal switching counts for each input channel.
  //   II: initiation interval.
  void calValidSwitching(const std::string &sucNodeName,
                                 const std::vector<IISet *> &setVList,
                                 const std::vector<int> &numVList,
                                 unsigned II);

  // calValidSet:
  //   setVList: vector of active valid sets for each input channel.
  //   II: initiation interval.
  void calValidSet(const std::string &sucNodeName,
                           const std::vector<IISet *> &setVList,
                           unsigned II);

  // calReadySwitching:
  //   For ready switching we have a single number.
  void calReadySwitching(const std::string &preNodeName,
                                 int numReady);

  // calReadySet:
  //   setR is provided as a pointer (if not null, it is used).
  void calReadySet(const std::string &preNodeName,
                           const IISet &setR,
                           unsigned II);

  DEFINE_NODE_KIND(MergeNode, NodeKind::MergeNodeKind);
};

//
// Model for CMerge Node
//
class CMergeNode : public AdjNode {
public:
  // Constructor: Pass the basic information to the base class.
  // sucDataWidthMap is assumed to map a successor name to its port number.
  CMergeNode(mlir::Operation *op,
             const std::vector<std::string> &predecessors,
             const std::vector<std::string> &successors,
             const std::map<std::string, unsigned> &sucDataWidthMap,
             unsigned latency, const unsigned &bbIndex);

  // Handshake signal calculation functions
  void calValidSwitching(const std::string &sucNodeName, unsigned II);
  void calValidSet(const std::string &sucNodeName, unsigned nodeStartTime, unsigned II);
  
  void calReadySwitching(const std::string &preNodeName);
  void calReadySet(const std::string &preNodeName, unsigned II);

  // Data channel functions
  // calDataout: For control merge, the control (conditional) channel output equals the actual data input port index,
  // and the data channel output is set to an invalid value (-1).
  void calDataout(int inputValue);

  // updateDataout: for every successor in the node’s successor list,
  // if it is the condition channel, update using XOR-diff and update toggle counts;
  // otherwise, set its data to -1.
  void updateDataout(int inputData);

  // Print details: First call the base class version, then print the control-channel successor name.
  void printNodeDetails();

  DEFINE_NODE_KIND(CMergeNode, NodeKind::CMergeNodeKind);

  // Extra member variables specific to CMergeNode:
  std::string conSucNodeName;   // The name of the successor on the condition channel (port 1)
  std::string dataSucNodeName;  // The name of the successor on the data channel (all others)
};

//
// Model for CBr Node
//
class CBrNode : public AdjNode {
public:
  // Constructor with updated signature.
  CBrNode(mlir::Operation *op,
          const std::vector<std::string> &predecessors,
          const std::vector<std::string> &successors,
          const std::map<std::string, unsigned> &sucDataWidthMap,
          unsigned latency, const unsigned& bbIndex);

  // Handshake calculation functions:
  // calValidSwitching: Determines valid switching for a given successor based on the condition value and
  // input valid counts.
  void calValidSwitching(const std::string &sucNodeName,
                         unsigned condValue,
                         int numValid0,
                         int numValid1);

  // calValidSet: If valid signal for sucNodeName is 0 (or II==1), then the entire cycle range is active;
  // otherwise, if condValue matches the port number of sucNodeName then assign an empty set,
  // else assign a singleton set {nodeStartTime}.
  void calValidSet(const std::string &sucNodeName,
                   unsigned nodeStartTime,
                   unsigned condValue,
                   unsigned II);

  // calReadySwitching: For the given predecessor, if both numValid and numReady are 0 then ready is 0;
  // else ready is 2.
  void calReadySwitching(const std::string &preNodeName,
                         int numValid,
                         int numReady);

  // calReadySet: If a ready set is provided (non-null pointer), assign it.
  // Otherwise, if readySignal for the predecessor is 0, assign the full set {0,...,II-1}.
  void calReadySet(const std::string &preNodeName,
                   const IISet *setValid,
                   const IISet *setReady,
                   unsigned II);

  // setReadySet: For convenience, choose a successor with nonzero valid switching (if any) and copy its
  // valid set (from setV) to the ready set for the given predecessor; otherwise, assign full set.
  void setReadySet(const std::string &preNodeName, unsigned II);

  // updateDataout: Update the data output channels for all successors:
  // For each successor, if data already exists, compute an XOR difference with the new input and update toggle counts.
  // Also update the per-channel dataout for channel (1 - condValue).
  void updateDataout(int inputData, unsigned condValue);

  // Print details: call base class printDetail(), then print extra CBrNode info.
  void printNodeDetails();

  DEFINE_NODE_KIND(CBrNode, NodeKind::CBrNodeKind);

  // Extra members.
  std::string condPreNodeName;  // Condition predecessor name.
  std::string dataPreNodeName;  // Data predecessor name.
  std::string trueSucNodeName;  // Succeeding node when the condition evaluated to true
  std::string falseSucNodeName; // Succeeding node when the condition evaluated to false
  std::map<std::string, unsigned> outChannelNameToIndexMap;
  int lastValidDataValue;       // Last valid data value (initialized to -1).
  std::map<int, std::vector<int>> per_channel_dataout; // Map: channel -> vector of output data.
};

//
// Model for Mux Node
//
// MuxNode represents a mux operator in the CFDFC.
// - The condition signal selects the input port whose cond_value equals (port_id - 1).
// - It is assumed that in steady state, the condition input is always 0.
class MuxNode : public AdjNode {
public:
  // Constructor.
  //   - op: pointer to the MLIR operation.
  //   - predecessors: a mapping from predecessor names to their port numbers.
  //   - successors: a vector of successor names.
  //   - sucDataWidthMap: mapping from successor names to (data) port numbers.
  //   - latency: the node latency.
  MuxNode(mlir::Operation *op,
          const std::vector<std::string> &predecessors,
          const std::vector<std::string> &successors,
          const std::map<std::string, unsigned> &sucDataWidthMap,
          unsigned latency, const unsigned& bbIndex);

  // Handshake functions:
  // calValidSwitching: If II == 1, the valid signal is 0; otherwise, it is 2.
  void calValidSwitching(const std::string &sucNodeName, unsigned II);

  // calValidSet: If the valid signal for sucNodeName is 0 then assign full set {0,...,II-1};
  // otherwise, assign {nodeStartTime}.
  void calValidSet(const std::string &sucNodeName, unsigned nodeStartTime, unsigned II);

  // calReadySwitching:
  //   Parameters:
  //     - preNodeName: name of a predecessor.
  //     - condValue: the condition value at this node.
  //     - num_v: an unsigned representing the valid count.
  //     - setV0: pointer to a set (e.g., active cycles) for one input.
  //     - setVSelect: pointer to a set for the selected input.
  //     - setR: pointer to a set for ready.
  //     - II: initiation interval.
  void calReadySwitching(const std::string &preNodeName,
                         unsigned condValue,
                         int num_v,
                         const IISet *setV0,
                         const IISet *setVSelect,
                         const IISet *setR,
                         unsigned II);

  // calReadySet:
  //   Parameters:
  //     - preNodeName: name of a predecessor.
  //     - setCond: pointer to a set for the condition.
  //     - setV: pointer to a valid set.
  //     - setR: pointer to a ready set.
  //     - II: initiation interval.
  // If readySignal for preNodeName is 0, assign full set {0,...,II-1};
  // else assign the intersection: setCond ∩ setV ∩ setR.
  void calReadySet(const std::string &preNodeName,
                   const IISet *setCond,
                   const IISet *setValid,
                   const IISet *setReady,
                   unsigned II);

  // Print detail: call the base class printDetail() and then print extra info.
  void printNodeDetails();

  // LLVM Casting support
  NodeKind getKind() const override { return NodeKind::MuxNodeKind; }

  // "classof" needed for dyn_cast
  static bool classof(const AdjNode *node) {
    return node->getKind() == NodeKind::MuxNodeKind;
  }

  // Extra member: The name of the predecessor on the condition channel.
  std::string conPreNodeName;
  // Map from node name to the corresponding port index
  std::map<std::string, unsigned> preNameToPortIdxMap;
};

//
// TODO: Finish the model for the select node
//
class SelectNode : public MuxNode {
public:
  using MuxNode::MuxNode; // Inherit the constructor
  DEFINE_NODE_KIND(SelectNode, NodeKind::SelectNodeKind);

  // Extra member: The name of the predecessor on the condition channel.
  // std::string conPreNodeName;
  // Map from node name to the corresponding port index
  std::map<std::string, unsigned> preNameToPortIdxMap;
};

//
// Model for Sink Node
//
class SinkNode : public AdjNode {
public:
  using AdjNode::AdjNode; // Inherit the constructor

  // For SinkNode, valid switching and valid set are not defined by hardware;
  // They are intended to be provided via profiling.
  void calValidSwitching() {}
  void calValidSet() {}

  void calReadySwitching(const std::string &preNodeName) {
    // Sink node is always ready.
    readySignal[preNodeName] = 0;
  }
  
  void calReadySet(const std::string &preNodeName, unsigned II) {
    setR[preNodeName] = fullSet(II);
  }

  void calDataout() {}

  // LLVM Casting support
  NodeKind getKind() const override { return NodeKind::SinkNodeKind; }

  // "classof" needed for dyn_cast
  static bool classof(const AdjNode *node) {
    return node->getKind() == NodeKind::SinkNodeKind;
  }
};

//
// Model for Start Node
//
class StartNode : public AdjNode {
public:
  /// Constructor.
  StartNode(mlir::Operation *op,
            const std::vector<std::string> &predecessors,
            const std::vector<std::string> &successors,
            const std::map<std::string, unsigned> &sucDataWidthDict,
            unsigned latency, const unsigned& bbIndex);

  // Handshake functions:
  /// Set the valid switching value for a given successor to 2.
  void calValidSwitching(const std::string &sucNodeName);

  /// These methods are not implemented (pass).
  void calValidSet();
  void calReadySwitching();
  void calReadySet();

  DEFINE_NODE_KIND(StartNode, NodeKind::StartNodeKind);
};

//
// Model for Memory controllers
// TODO: Add concrete implementation for MemCon and LSQ nodes
class MemConNode : public AdjNode {
public:
  using AdjNode::AdjNode; // Inherit the constructor
};

class LSQNode : public AdjNode {
public:
  using AdjNode::AdjNode; // Inherit the constructor
};

#endif // EXPERIMENTAL_ANALYSIS_SWITCHINGESTIMATION_NODEMODELS_H
