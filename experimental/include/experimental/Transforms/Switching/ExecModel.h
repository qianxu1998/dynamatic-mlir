//===- ExecModel.h - Switching Estimation -----*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the execution model for all node types used in dynamatic
// during switching estimation
//
//===----------------------------------------------------------------------===//

#ifndef EXPERIMENTAL_TRANSFORMS_SWITCHING_EXECUTION_MODEL_H
#define EXPERIMENTAL_TRANSFORMS_SWITCHING_EXECUTION_MODEL_H

#include "experimental/Transforms/Switching/NodeInfo.h"
#include "experimental/Transforms/Switching/SwitchingSupport.h"

#include "mlir/IR/Operation.h"

#include <map>
#include <memory>
// #include <set>
#include <string>
#include <vector>
using IISet = llvm::SmallBitVector;
//===----------------------------------------------------------------------===//
//
// Model for Buffer Node
//
//===----------------------------------------------------------------------===//
class BufferNode : public AdjNode {
public:
  // Constructor
  BufferNode(mlir::Operation *op, const std::vector<std::string> &predecessors,
             const std::vector<std::string> &successors,
             const std::map<std::string, unsigned> &sucDataWidthMap,
             const unsigned &latency, const unsigned &bbIndex);

  // Handshake Counting functions
  void calValidSwitching(const std::string &sucNodeName, unsigned II);
  void calReadySwitching(const std::string &preNodeName, unsigned II);

  void calValidSet(const std::string &sucNodeName, IISet &inSetV);//  TODO1 change to const
  void calReadySet(const std::string &preNodeName,// TODO1 change to const
                   const IISet &inSetR); // TODO reference  TODO1 change to const
  // Override printDetail function
  void printDetail() override;

  // LLVM Casting support
  NodeKind getKind() const override { return NodeKind::BufferNodeKind; }

  // "classof" needed for dyn_cast
  static bool classof(const AdjNode *node) {
    return node->getKind() == NodeKind::BufferNodeKind;
  }

  //
  /// Internal variable
  //
  unsigned START;
  float_t occupancy;
  unsigned numSlots;
  bool transparent;
};

//===----------------------------------------------------------------------===//
//
// Model for Join Node
//
//===----------------------------------------------------------------------===//
class JoinNode : public AdjNode {
public:
  // Constructor: simply forwards to AdjNode
  JoinNode(mlir::Operation *op, const std::vector<std::string> &predecessors,
           const std::vector<std::string> &successors,
           const std::map<std::string, unsigned> &sucDataWidthMap,
           unsigned latency, const unsigned &bbIndex)
      : AdjNode(op, predecessors, successors, sucDataWidthMap, latency,
                bbIndex) {}

  // - calValidSwitching: if either numValid0 or numValid1 > 0, set valid signal
  // to 2;
  //   if both are 0, then set valid signal to 0; if II==1, then also set to 0;
  //   otherwise, set to 2.
  virtual void calValidSwitching(const std::string &sucNodeName, int &numValid0,
                                 int &numValid1, unsigned &II);

  // calValidSet: if the valid signal for sucNodeName is 0 (or II==1), then
  // store the full set {0,...,II-1}; otherwise, store only {nodeStartTime}.
  virtual void calValidSet(const std::string &sucNodeName,
                           unsigned &nodeStartTime, unsigned &II);

  // calReadySwitching: if both numValid and numReady are 0, set ready signal to
  // 0; if either > 0, set it to 2.
  virtual void calReadySwitching(const std::string &preNodeName, int &numValid,
                                 int &numReady);

  // calReadySet: if both set_valid and set_ready are provided (non-null),
  // then take their intersection; otherwise, if readySignal[preNodeName] is 0
  // then store the full set {0,...,II-1}, if >0 then store {nodeStartTime}. (We
  // pass the sets as bitvectors so that a negative  vector.any() represents “None”.)
  virtual void calReadySet(const std::string &preNodeName,
                           const IISet &setValid, const IISet &setReady,

                           unsigned &nodeStartTime, unsigned &II);

  // setReadySet: as a convenience, take the first key from setVMap and copy its
  // set into setRMap for preNodeName.
  void setReadySet(const std::string &preNodeName);

  // LLVM Casting support
  NodeKind getKind() const override { return NodeKind::JoinNodeKind; }

  // "classof" needed for dyn_cast
  static bool classof(const AdjNode *node) {
    return node->getKind() == NodeKind::JoinNodeKind;
  }
};

//===----------------------------------------------------------------------===//
//
// Model for Pass Node
//
//===----------------------------------------------------------------------===//
class PassNode : public AdjNode {
public:
  // Constructor: forwards to AdjNode
  PassNode(mlir::Operation *op, const std::vector<std::string> &predecessors,
           const std::vector<std::string> &successors,
           const std::map<std::string, unsigned> &sucDataWidthMap,
           unsigned latency, const unsigned &bbIndex)
      : AdjNode(op, predecessors, successors, sucDataWidthMap, latency,
                bbIndex) {}

  // Handshake counting functions:
  // calValidSwitching: if numValid is 0, set valid signal to 0; else to 2.
  virtual void calValidSwitching(const std::string &sucNodeName, int &numValid);

  // calValidSet: if validSignal[sucNodeName] is 0, store full set {0,...,II-1};
  // else store {nodeStartTime}.
  virtual void calValidSet(const std::string &sucNodeName,
                           unsigned &nodeStartTime, unsigned &II);

  // calReadySwitching: if numReady is 0, set ready signal to 0; else to 2.
  virtual void calReadySwitching(const std::string &preNodeName, int &numReady);

  // calReadySet: if a set (setR) is provided, simply store it; else if
  // readySignal[preNodeName] is 0, store full set; if >0, store
  // {nodeStartTime}; otherwise, if setVMap is not empty and its full set is not
  // equal to {0,...,II-1}, use that.
  virtual void calReadySet(const std::string &preNodeName,
                           const IISet &setReady, unsigned &nodeStartTime,
                           unsigned &II);

  // setReadySwitching: directly set readySignal.
  void setReadySwitching(const std::string &preNodeName,
                         unsigned &numReadySwitches);

  // setReadySet: store the provided set into setRMap.
  void setReadySet(const std::string &preNodeName, const IISet &setReady);
};

//===----------------------------------------------------------------------===//
//
// Model for Cmpi Node: derived from JoinNode
//
//===----------------------------------------------------------------------===//
class CmpiNode : public JoinNode {
public:
  CmpiNode(mlir::Operation *op, const std::vector<std::string> &predecessors,
           const std::vector<std::string> &successors,
           const std::map<std::string, unsigned> &sucDataWidthMap,
           unsigned latency, const unsigned &bbIndex)
      : JoinNode(op, predecessors, successors, sucDataWidthMap, latency,
                 bbIndex) {}

  void calValidSwitching(const std::string &sucNodeName, int &numValid0,
                         int &numValid1, unsigned &II) override {
    JoinNode::calValidSwitching(sucNodeName, numValid0, numValid1, II);
  }

  void calValidSet(const std::string &sucNodeName, unsigned &nodeStartTime,
                   unsigned &II) override {
    JoinNode::calValidSet(sucNodeName, nodeStartTime, II);
  }

  void calReadySwitching(const std::string &preNodeName, int &numValid,
                         int &numReady) override {
    JoinNode::calReadySwitching(preNodeName, numValid, numReady);
  }

  void calReadySet(const std::string &preNodeName, const IISet &setValid,
                   const IISet &setReady, unsigned &nodeStartTime,
                   unsigned &II) override {
    JoinNode::calReadySet(preNodeName, setValid, setReady, nodeStartTime, II);
  }

  // LLVM Casting support
  NodeKind getKind() const override { return NodeKind::CmpiNodeKind; }

  // "classof" needed for dyn_cast
  static bool classof(const AdjNode *node) {
    return node->getKind() == NodeKind::CmpiNodeKind;
  }
};

//===----------------------------------------------------------------------===//
//
// Model for Addi Node: derived from JoinNode
//
//===----------------------------------------------------------------------===//
class AddiNode : public JoinNode {
public:
  AddiNode(mlir::Operation *op, const std::vector<std::string> &predecessors,
           const std::vector<std::string> &successors,
           const std::map<std::string, unsigned> &sucDataWidthMap,
           unsigned latency, const unsigned &bbIndex)
      : JoinNode(op, predecessors, successors, sucDataWidthMap, latency,
                 bbIndex) {}

  void calValidSwitching(const std::string &sucNodeName, int &numValid0,
                         int &numValid1, unsigned &II) override {
    JoinNode::calValidSwitching(sucNodeName, numValid0, numValid1, II);
  }

  void calValidSet(const std::string &sucNodeName, unsigned &nodeStartTime,
                   unsigned &II) override {
    JoinNode::calValidSet(sucNodeName, nodeStartTime, II);
  }

  void calReadySwitching(const std::string &preNodeName, int &numValid,
                         int &numReady) override {
    JoinNode::calReadySwitching(preNodeName, numValid, numReady);
  }

  void calReadySet(const std::string &preNodeName,

                   const IISet &setValid, const IISet &setReady,
                   unsigned &nodeStartTime, unsigned &II) override {
    JoinNode::calReadySet(preNodeName, setValid, setReady, nodeStartTime, II);
  }

  // LLVM Casting support
  NodeKind getKind() const override { return NodeKind::AddiNodeKind; }

  // "classof" needed for dyn_cast
  static bool classof(const AdjNode *node) {
    return node->getKind() == NodeKind::AddiNodeKind;
  }
};

//===----------------------------------------------------------------------===//
//
// Model for Subi Node
//
//===----------------------------------------------------------------------===//
class SubiNode : public JoinNode {
public:
  SubiNode(mlir::Operation *op, const std::vector<std::string> &predecessors,
           const std::vector<std::string> &successors,
           const std::map<std::string, unsigned> &sucDataWidthMap,
           unsigned latency, const unsigned &bbIndex)
      : JoinNode(op, predecessors, successors, sucDataWidthMap, latency,
                 bbIndex) {}

  void calValidSwitching(const std::string &sucNodeName, int &numValid0,
                         int &numValid1, unsigned &II) override {
    JoinNode::calValidSwitching(sucNodeName, numValid0, numValid1, II);
  }

  void calValidSet(const std::string &sucNodeName, unsigned &nodeStartTime,
                   unsigned &II) override {
    JoinNode::calValidSet(sucNodeName, nodeStartTime, II);
  }

  void calReadySwitching(const std::string &preNodeName, int &numValid,
                         int &numReady) override {
    JoinNode::calReadySwitching(preNodeName, numValid, numReady);
  }

  void calReadySet(const std::string &preNodeName, const IISet &setValid,
                   const IISet &setReady, unsigned &nodeStartTime,
                   unsigned &II) override {
    JoinNode::calReadySet(preNodeName, setValid, setReady, nodeStartTime, II);
  }

  // LLVM Casting support
  NodeKind getKind() const override { return NodeKind::SubiNodeKind; }

  // "classof" needed for dyn_cast
  static bool classof(const AdjNode *node) {
    return node->getKind() == NodeKind::SubiNodeKind;
  }
};

//===----------------------------------------------------------------------===//
//
// Model for Muli Node
//
//===----------------------------------------------------------------------===//
class MuliNode : public JoinNode {
public:
  MuliNode(mlir::Operation *op, const std::vector<std::string> &predecessors,
           const std::vector<std::string> &successors,
           const std::map<std::string, unsigned> &sucDataWidthMap,
           unsigned latency, const unsigned &bbIndex)
      : JoinNode(op, predecessors, successors, sucDataWidthMap, latency,
                 bbIndex) {}

  void calValidSwitching(const std::string &sucNodeName, int &numValid0,
                         int &numValid1, unsigned &II) override {
    JoinNode::calValidSwitching(sucNodeName, numValid0, numValid1, II);
  }

  void calValidSet(const std::string &sucNodeName, unsigned &nodeStartTime,
                   unsigned &II) override {
    JoinNode::calValidSet(sucNodeName, nodeStartTime, II);
  }

  void calReadySwitching(const std::string &preNodeName, int &numValid,
                         int &numReady) override {
    JoinNode::calReadySwitching(preNodeName, numValid, numReady);
  }

  void calReadySet(const std::string &preNodeName, const IISet &setValid,
                   const IISet &setReady, unsigned &nodeStartTime,
                   unsigned &II) override {
    JoinNode::calReadySet(preNodeName, setValid, setReady, nodeStartTime, II);
  }

  // LLVM Casting support
  NodeKind getKind() const override { return NodeKind::MuliNodeKind; }

  // "classof" needed for dyn_cast
  static bool classof(const AdjNode *node) {
    return node->getKind() == NodeKind::MuliNodeKind;
  }
};

//===----------------------------------------------------------------------===//
//
// Model for Extsi Node
//
//===----------------------------------------------------------------------===//
class ExtsiNode : public PassNode {
public:
  ExtsiNode(mlir::Operation *op, const std::vector<std::string> &predecessors,
            const std::vector<std::string> &successors,
            const std::map<std::string, unsigned> &sucDataWidthMap,
            unsigned latency, const unsigned &bbIndex)
      : PassNode(op, predecessors, successors, sucDataWidthMap, latency,
                 bbIndex) {}

  void calValidSwitching(const std::string &sucNodeName,
                         int &numValid) override {
    PassNode::calValidSwitching(sucNodeName, numValid);
  }

  void calValidSet(const std::string &sucNodeName, unsigned &nodeStartTime,
                   unsigned &II) override {
    PassNode::calValidSet(sucNodeName, nodeStartTime, II);
  }

  void calReadySwitching(const std::string &preNodeName,
                         int &numReady) override {
    PassNode::calReadySwitching(preNodeName, numReady);
  }

  void calReadySet(const std::string &preNodeName,

                   const IISet &setR, unsigned &nodeStartTime,
                   unsigned &II) override {
    PassNode::calReadySet(preNodeName, setR, nodeStartTime, II);
  }

  // LLVM Casting support
  NodeKind getKind() const override { return NodeKind::ExtsiNodeKind; }

  // "classof" needed for dyn_cast
  static bool classof(const AdjNode *node) {
    return node->getKind() == NodeKind::ExtsiNodeKind;
  }
};

//===----------------------------------------------------------------------===//
//
// Model for Load Node
//
//===----------------------------------------------------------------------===//
class DLoadNode : public AdjNode {
public:
  DLoadNode(mlir::Operation *op, const std::vector<std::string> &predecessors,
            const std::vector<std::string> &successors,
            const std::map<std::string, unsigned> &sucDataWidthMap,
            const unsigned &latency, const unsigned &bbIndex);

  // Handshake functions:
  // For valid switching: if num_valid == 0, valid signal is 0; else if > 0,
  // valid signal is 2.
  void calValidSwitching(const std::string &sucNodeName, int &numValid);

  // For valid set: if valid signal is 0, use the full set {0,...,II-1};
  // else, if an input set is provided, shift it by the node latency.
  void calValidSet(const std::string &sucNodeName, const IISet &inSetV,
                   unsigned &II);

  // For ready switching: if num_ready == 0, ready signal is 0; else ready
  // signal is 2.
  void calReadySwitching(const std::string &preNodeName, int &numReady);

  // For ready set: for a load node, we assume the ready set is taken directly
  // if provided; otherwise, if readySignal is 0 then use full set, or else use
  // {nodeStartTime}.
  void calReadySet(const std::string &preNodeName, const IISet &inSetR,
                   unsigned &nodeStartTime, unsigned &II);

  // Update the data output channels.
  //   - For data output: if a previous value exists, compute diff (using XOR)
  //   and update toggle counts.
  //   - For address output: if the new address is -10, reuse the last valid
  //   value.
  void updateDataout(int &inputData, int addressValue);

  void printDetail() override;

  // LLVM Casting support
  NodeKind getKind() const override { return NodeKind::DLoadNodeKind; }

  // "classof" needed for dyn_cast
  static bool classof(const AdjNode *node) {
    return node->getKind() == NodeKind::DLoadNodeKind;
  }

  // Extra members for DLoad nodes
  std::string dataOutNodeName;    // Name for the data out channel
  std::string addressOutNodeName; // this should be the corresponding name of
                                  // the mem_controller
  std::string addressInNodeName;  // Name of the address input node name
};

//===----------------------------------------------------------------------===//
//
// Model for Store Node
//
//===----------------------------------------------------------------------===//
// TODO: Seperate MC_store and LSQ_store
class DStoreNode : public AdjNode {
public:
  // Constructor: forwards to AdjNode.
  DStoreNode(mlir::Operation *op, const std::vector<std::string> &predecessors,
             const std::vector<std::string> &successors,
             const std::map<std::string, unsigned> &sucDataWidthMap,
             unsigned latency, const unsigned &bbIndex);

  // Handshake functions for d_store.
  // For valid switching on the memory controller channel (key "mc").
  void calValidSwitching(int numValid1, int numValid2);
  void calValidSet(const IISet &setV0, const IISet &setV1, unsigned II);

  void calReadySwitching(const std::string &preNodeName, int numValid1,
                         int numValid2);
  void calReadySet(const std::string &preNodeName, const IISet &setV0,
                   const IISet &setV1, unsigned II);

  // Update data output based on the source input node.
  void updateDataout(int inputData, const std::string &srcInputNode);

  void printDetail() override;

  // Handshake checking (e.g. comparing size of readySignal with number of
  // predecessors)
  bool handshakeSwitchingChecking();

  // LLVM Casting support
  NodeKind getKind() const override { return NodeKind::DStoreNodeKind; }

  // "classof" needed for dyn_cast
  static bool classof(const AdjNode *node) {
    return node->getKind() == NodeKind::DStoreNodeKind;
  }

  // Extra members for DStoreNode.
  std::string dataInNode;
  std::string addressInNode;
  std::string dataInSrcNode;
  std::string addressInSrcNode;
};

//===----------------------------------------------------------------------===//
//
// Model for Merge Node
//
//===----------------------------------------------------------------------===//
// MergeNode is derived from AdjNode.
class MergeNode : public AdjNode {
public:
  // Constructor: simply forward the parameters to the AdjNode constructor.
  MergeNode(mlir::Operation *op, const std::vector<std::string> &predecessors,
            const std::vector<std::string> &successors,
            const std::map<std::string, unsigned> &sucDataWidthMap,
            unsigned latency, const unsigned &bbIndex)
      : AdjNode(op, predecessors, successors, sucDataWidthMap, latency,
                bbIndex) {}

  // calValidSwitching:
  //   setVList: vector of active valid sets for each input channel.
  //   numVList: vector of valid signal switching counts for each input channel.
  //   II: initiation interval.
  void calValidSwitching(const std::string &sucNodeName,
                         const std::vector<IISet *> &setVList,
                         const std::vector<int> &numVList, unsigned II);

  // calValidSet:
  //   setVList: vector of active valid sets for each input channel.
  //   II: initiation interval.
  void calValidSet(const std::string &sucNodeName,
                   const std::vector<IISet *> &setVList, unsigned II);

  // calReadySwitching:
  //   For ready switching we have a single number.
  void calReadySwitching(const std::string &preNodeName, int numReady);

  // calReadySet:
  //   setR is provided as a bitvector (if not empty with vec.any(), it is used).
  void calReadySet(const std::string &preNodeName,

                   const IISet &setR, unsigned II);

  // LLVM Casting support
  NodeKind getKind() const override { return NodeKind::MergeNodeKind; }

  // "classof" needed for dyn_cast
  static bool classof(const AdjNode *node) {
    return node->getKind() == NodeKind::MergeNodeKind;
  }
};

//===----------------------------------------------------------------------===//
//
// Model for CMerge Node
//
//===----------------------------------------------------------------------===//
class CMergeNode : public AdjNode {
public:
  // Constructor: Pass the basic information to the base class.
  // sucDataWidthMap is assumed to map a successor name to its port number.
  CMergeNode(mlir::Operation *op, const std::vector<std::string> &predecessors,
             const std::vector<std::string> &successors,
             const std::map<std::string, unsigned> &sucDataWidthMap,
             unsigned latency, const unsigned &bbIndex);

  // Handshake signal calculation functions
  void calValidSwitching(const std::string &sucNodeName, unsigned II);
  void calValidSet(const std::string &sucNodeName, unsigned nodeStartTime,
                   unsigned II);

  void calReadySwitching(const std::string &preNodeName);
  void calReadySet(const std::string &preNodeName, unsigned II);

  // Data channel functions
  // calDataout: For control merge, the control (conditional) channel output
  // equals the actual data input port index, and the data channel output is set
  // to an invalid value (-1).
  void calDataout(int inputValue);

  // updateDataout: for every successor in the node’s successor list,
  // if it is the condition channel, update using XOR-diff and update toggle
  // counts; otherwise, set its data to -1.
  void updateDataout(int inputData);

  // Print details: First call the base class version, then print the
  // control-channel successor name.
  void printDetail() override;

  // LLVM Casting support
  NodeKind getKind() const override { return NodeKind::CMergeNodeKind; }

  // "classof" needed for dyn_cast
  static bool classof(const AdjNode *node) {
    return node->getKind() == NodeKind::CMergeNodeKind;
  }

  // Extra member variables specific to CMergeNode:
  std::string conSucNodeName;  // The name of the successor on the condition
                               // channel (port 1)
  std::string dataSucNodeName; // The name of the successor on the data channel
                               // (all others)
};

//===----------------------------------------------------------------------===//
//
// Model for Lazy fork Node
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
//
// Model for Fork Node
//
//===----------------------------------------------------------------------===//
class ForkNode : public AdjNode {
public:
  // Constructor: simply forward arguments to the AdjNode constructor.
  ForkNode(mlir::Operation *op, const std::vector<std::string> &predecessors,
           const std::vector<std::string> &successors,
           const std::map<std::string, unsigned> &sucDataWidthMap,
           unsigned latency, const unsigned &bbIndex)
      : AdjNode(op, predecessors, successors, sucDataWidthMap, latency,
                bbIndex) {}

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
  void
  calValidSwitching(const std::string &sucNodeName, int numValid,
                    std::unordered_map<std::string, IISet *> &setRDict,
                    const std::unordered_map<std::string, int> &numReadyDict,
                    unsigned sucNodeStart, unsigned nodeSteadyStart,
                    unsigned II);

  // calValidSet:
  //   If the valid signal for sucNodeName is 0, assign the full set
  //   {0,...,II-1}. Otherwise, if numValid > 0, assign {nodeStartTime}; else,
  //   assign the ready set from setRDict.
  void calValidSet(const std::string &sucNodeName, unsigned nodeStartTime,
                   int numValid,
                   std::unordered_map<std::string, IISet *> &setRDict,
                   unsigned II);

  // calReadySwitching:
  //   Takes a list of ready counts (one per channel) and if any value > 0 sets
  //   the ready signal to 2; otherwise, if all values are 0, sets it to 0.
  void calReadySwitching(const std::string &preNodeName,
                         const std::vector<int> &numReadyList);

  // calReadySet:
  //   If a ready set (setR) is provided (non-null), use it.
  //   Otherwise, if readySignal for the predecessor is 0, assign the full set.
  //   Else, compute the union of all ready sets from all successors and then
  //   assign:
  //     - If the union equals the full set {0,...,II-1}, assign {0};
  //     - Otherwise, assign the last element of the union.
  void calReadySet(const std::string &preNodeName,
                   const std::unordered_map<std::string, IISet *> &setRDict,
                   unsigned II);

  // LLVM Casting support
  NodeKind getKind() const override { return NodeKind::ForkNodeKind; }

  // "classof" needed for dyn_cast
  static bool classof(const AdjNode *node) {
    return node->getKind() == NodeKind::ForkNodeKind;
  }
};

//===----------------------------------------------------------------------===//
//
// Model for CBr Node
//
//===----------------------------------------------------------------------===//
class CBrNode : public AdjNode {
public:
  // Constructor with updated signature.
  CBrNode(mlir::Operation *op, const std::vector<std::string> &predecessors,
          const std::vector<std::string> &successors,
          const std::map<std::string, unsigned> &sucDataWidthMap,
          unsigned latency, const unsigned &bbIndex);

  // Handshake calculation functions:
  // calValidSwitching: Determines valid switching for a given successor based
  // on the condition value and input valid counts.
  void calValidSwitching(const std::string &sucNodeName, unsigned condValue,
                         int numValid0, int numValid1);

  // calValidSet: If valid signal for sucNodeName is 0 (or II==1), then the
  // entire cycle range is active; otherwise, if condValue matches the port
  // number of sucNodeName then assign an empty set, else assign a singleton set
  // {nodeStartTime}.
  void calValidSet(const std::string &sucNodeName, unsigned nodeStartTime,
                   unsigned condValue, unsigned II);

  // calReadySwitching: For the given predecessor, if both numValid and numReady
  // are 0 then ready is 0; else ready is 2.
  void calReadySwitching(const std::string &preNodeName, int numValid,
                         int numReady);

  // calReadySet: If a ready set is provided (non-empty , set bitvector), assign it.
  // Otherwise, if readySignal for the predecessor is 0, assign the full set
  // {0,...,II-1}.
  void calReadySet(const std::string &preNodeName, const IISet &setValid,
                   const IISet &setReady, unsigned II);

  // setReadySet: For convenience, choose a successor with nonzero valid
  // switching (if any) and copy its valid set (from setV) to the ready set for
  // the given predecessor; otherwise, assign full set.
  void setReadySet(const std::string &preNodeName, unsigned II);

  // updateDataout: Update the data output channels for all successors:
  // For each successor, if data already exists, compute an XOR difference with
  // the new input and update toggle counts. Also update the per-channel dataout
  // for channel (1 - condValue).
  void updateDataout(int inputData, unsigned condValue);

  // Print details: call base class printDetail(), then print extra CBrNode
  // info.
  void printDetail() override;

  // LLVM Casting support
  NodeKind getKind() const override { return NodeKind::CBrNodeKind; }

  // "classof" needed for dyn_cast
  static bool classof(const AdjNode *node) {
    return node->getKind() == NodeKind::CBrNodeKind;
  }

  // Extra members.
  std::string condPreNodeName; // Condition predecessor name.
  std::string dataPreNodeName; // Data predecessor name.
  std::string
      trueSucNodeName; // Succeeding node when the condition evaluated to true
  std::string
      falseSucNodeName; // Succeeding node when the condition evaluated to false
  std::map<std::string, unsigned> outChannelNameToIndexMap;
  int lastValidDataValue; // Last valid data value (initialized to -1).
  std::map<int, std::vector<int>>
      per_channel_dataout; // Map: channel -> vector of output data.
};

//===----------------------------------------------------------------------===//
//
// Model for Shli Node
//
//===----------------------------------------------------------------------===//
class ShliNode : public JoinNode {
public:
  ShliNode(mlir::Operation *op, const std::vector<std::string> &predecessors,
           const std::vector<std::string> &successors,
           const std::map<std::string, unsigned> &sucDataWidthMap,
           unsigned latency, const unsigned &bbIndex)
      : JoinNode(op, predecessors, successors, sucDataWidthMap, latency,
                 bbIndex) {}

  // LLVM Casting support
  NodeKind getKind() const override { return NodeKind::ShliNodeKind; }

  // "classof" needed for dyn_cast
  static bool classof(const AdjNode *node) {
    return node->getKind() == NodeKind::ShliNodeKind;
  }
};
//===----------------------------------------------------------------------===//
//
// Model for Shrsi Node
//
//===----------------------------------------------------------------------===//
class ShrsiNode : public JoinNode {
public:
  ShrsiNode(mlir::Operation *op, const std::vector<std::string> &predecessors,
            const std::vector<std::string> &successors,
            const std::map<std::string, unsigned> &sucDataWidthMap,
            unsigned latency, const unsigned &bbIndex)
      : JoinNode(op, predecessors, successors, sucDataWidthMap, latency,
                 bbIndex) {}

  // LLVM Casting support
  NodeKind getKind() const override { return NodeKind::ShrsiNodeKind; }

  // "classof" needed for dyn_cast
  static bool classof(const AdjNode *node) {
    return node->getKind() == NodeKind::ShrsiNodeKind;
  }
};

//===----------------------------------------------------------------------===//
//
// Model for Shrui Node
//
//===----------------------------------------------------------------------===//
class ShruiNode : public JoinNode {
public:
  ShruiNode(mlir::Operation *op, const std::vector<std::string> &predecessors,
            const std::vector<std::string> &successors,
            const std::map<std::string, unsigned> &sucDataWidthMap,
            unsigned latency, const unsigned &bbIndex)
      : JoinNode(op, predecessors, successors, sucDataWidthMap, latency,
                 bbIndex) {}

  // LLVM Casting support
  NodeKind getKind() const override { return NodeKind::ShruiNodeKind; }

  // "classof" needed for dyn_cast
  static bool classof(const AdjNode *node) {
    return node->getKind() == NodeKind::ShruiNodeKind;
  }
};

//===----------------------------------------------------------------------===//
//
// Model for Mux Node
//
//===----------------------------------------------------------------------===//
// MuxNode represents a mux operator in the CFDFC.
// - The condition signal selects the input port whose cond_value equals
// (port_id - 1).
// - It is assumed that in steady state, the condition input is always 0.
class MuxNode : public AdjNode {
public:
  // Constructor.
  //   - op: pointer to the MLIR operation.
  //   - predecessors: a mapping from predecessor names to their port numbers.
  //   - successors: a vector of successor names.
  //   - sucDataWidthMap: mapping from successor names to (data) port numbers.
  //   - latency: the node latency.
  MuxNode(mlir::Operation *op, const std::vector<std::string> &predecessors,
          const std::vector<std::string> &successors,
          const std::map<std::string, unsigned> &sucDataWidthMap,
          unsigned latency, const unsigned &bbIndex);

  // Handshake functions:
  // calValidSwitching: If II == 1, the valid signal is 0; otherwise, it is 2.
  void calValidSwitching(const std::string &sucNodeName, unsigned II);

  // calValidSet: If the valid signal for sucNodeName is 0 then assign full set
  // {0,...,II-1}; otherwise, assign {nodeStartTime}.
  void calValidSet(const std::string &sucNodeName, unsigned nodeStartTime,
                   unsigned II);

  // calReadySwitching:
  //   Parameters:
  //     - preNodeName: name of a predecessor.
  //     - condValue: the condition value at this node.
  //     - num_v: an unsigned representing the valid count.
  //     - setV0: reference to a bitvector (e.g., active cycles) for one input.
  //     - setVSelect: reference to a bitvector for the selected input.
  //     - setR: reference to a bitvector for ready.
  //     - II: initiation interval.
  void calReadySwitching(const std::string &preNodeName, unsigned condValue,
                         int num_v, const IISet &setV0, const IISet &setVSelect,
                         const IISet &setR, unsigned II);

  // calReadySet:
  //   Parameters:
  //     - preNodeName: name of a predecessor.
  //     - setCond: reference to a bitvector for the condition.
  //     - setV: reference to a  valid bitvector.
  //     - setR: reference to a ready bitvector.
  //     - II: initiation interval.
  // If readySignal for preNodeName is 0, assign full set {0,...,II-1};
  // else assign the intersection: setCond ∩ setV ∩ setR.
  void calReadySet(const std::string &preNodeName, const IISet &setCond,
                   const IISet &setV, const IISet &setR, unsigned II);

  // Print detail: call the base class printDetail() and then print extra info.
  void printDetail() override;

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

//===----------------------------------------------------------------------===//
//
// Model for Trunci Node
//
//===----------------------------------------------------------------------===//
class TrunciNode : public PassNode {
public:
  TrunciNode(mlir::Operation *op, const std::vector<std::string> &predecessors,
             const std::vector<std::string> &successors,
             const std::map<std::string, unsigned> &sucDataWidthMap,
             unsigned latency, const unsigned &bbIndex)
      : PassNode(op, predecessors, successors, sucDataWidthMap, latency,
                 bbIndex) {}

  // LLVM Casting support
  NodeKind getKind() const override { return NodeKind::TrunciNodeKind; }

  // "classof" needed for dyn_cast
  static bool classof(const AdjNode *node) {
    return node->getKind() == NodeKind::TrunciNodeKind;
  }
};

//===----------------------------------------------------------------------===//
//
// Model for Extui Node
//
//===----------------------------------------------------------------------===//
class ExtuiNode : public PassNode {
public:
  ExtuiNode(mlir::Operation *op, const std::vector<std::string> &predecessors,
            const std::vector<std::string> &successors,
            const std::map<std::string, unsigned> &sucDataWidthMap,
            unsigned latency, const unsigned &bbIndex)
      : PassNode(op, predecessors, successors, sucDataWidthMap, latency,
                 bbIndex) {}

  // LLVM Casting support
  NodeKind getKind() const override { return NodeKind::ExtuiNodeKind; }

  // "classof" needed for dyn_cast
  static bool classof(const AdjNode *node) {
    return node->getKind() == NodeKind::ExtuiNodeKind;
  }
};

//===----------------------------------------------------------------------===//
//
// Model for Constant Node
//
//===----------------------------------------------------------------------===//
class ConstantNode : public PassNode {
public:
  ConstantNode(mlir::Operation *op,
               const std::vector<std::string> &predecessors,
               const std::vector<std::string> &successors,
               const std::map<std::string, unsigned> &sucDataWidthMap,
               unsigned latency, const unsigned &bbIndex)
      : PassNode(op, predecessors, successors, sucDataWidthMap, latency,
                 bbIndex) {}

  // LLVM Casting support
  NodeKind getKind() const override { return NodeKind::ConstantNodeKind; }

  // "classof" needed for dyn_cast
  static bool classof(const AdjNode *node) {
    return node->getKind() == NodeKind::ConstantNodeKind;
  }
};

//===----------------------------------------------------------------------===//
//
// Model for Ori Node
//
//===----------------------------------------------------------------------===//
class OriNode : public JoinNode {
public:
  OriNode(mlir::Operation *op, const std::vector<std::string> &predecessors,
          const std::vector<std::string> &successors,
          const std::map<std::string, unsigned> &sucDataWidthMap,
          unsigned latency, const unsigned &bbIndex)
      : JoinNode(op, predecessors, successors, sucDataWidthMap, latency,
                 bbIndex) {}

  // LLVM Casting support
  NodeKind getKind() const override { return NodeKind::OriNodeKind; }

  // "classof" needed for dyn_cast
  static bool classof(const AdjNode *node) {
    return node->getKind() == NodeKind::OriNodeKind;
  }
};
//===----------------------------------------------------------------------===//
//
// Model for Andi Node
//
//===----------------------------------------------------------------------===//
class AndiNode : public JoinNode {
public:
  AndiNode(mlir::Operation *op, const std::vector<std::string> &predecessors,
           const std::vector<std::string> &successors,
           const std::map<std::string, unsigned> &sucDataWidthMap,
           unsigned latency, const unsigned &bbIndex)
      : JoinNode(op, predecessors, successors, sucDataWidthMap, latency,
                 bbIndex) {}

  // LLVM Casting support
  NodeKind getKind() const override { return NodeKind::AndiNodeKind; }

  // "classof" needed for dyn_cast
  static bool classof(const AdjNode *node) {
    return node->getKind() == NodeKind::AndiNodeKind;
  }
};

//===----------------------------------------------------------------------===//
//
// Model for Source Node
//
//===----------------------------------------------------------------------===//
class SourceNode : public AdjNode {
public:
  SourceNode(mlir::Operation *op, const std::vector<std::string> &predecessors,
             const std::vector<std::string> &successors,
             const std::map<std::string, unsigned> &sucDataWidthMap,
             unsigned latency, const unsigned &bbIndex)
      : AdjNode(op, predecessors, successors, sucDataWidthMap, latency,
                bbIndex) {}

  void calValidSwitching(const std::string &sucNodeName) {
    validSignal[sucNodeName] = 0;
  }

  void calValidSet(const std::string &sucNodeName, unsigned II) {
    // Build full set {0, 1, ..., II-1}
    IISet fullSet(II, true);
    setV[sucNodeName] = std::move(fullSet);
  }

  void calReadySwitching(const std::string &preNodeName) {
    readySignal[preNodeName] = 0;
  }

  void calReadySet(const std::string &preNodeName, unsigned II) {
    IISet fullSet(II,true);
    setR[preNodeName] = fullSet;
  }

  // LLVM Casting support
  NodeKind getKind() const override { return NodeKind::SourceNodeKind; }

  // "classof" needed for dyn_cast
  static bool classof(const AdjNode *node) {
    return node->getKind() == NodeKind::SourceNodeKind;
  }
};

//===----------------------------------------------------------------------===//
//
// Model for End Node
//
//===----------------------------------------------------------------------===//
class EndNode : public AdjNode {
public:
  EndNode(mlir::Operation *op, const std::vector<std::string> &predecessors,
          const std::vector<std::string> &successors,
          const std::map<std::string, unsigned> &sucDataWidthMap,
          unsigned latency, const unsigned &bbIndex)
      : AdjNode(op, predecessors, successors, sucDataWidthMap, latency,
                bbIndex) {}

  // LLVM Casting support
  NodeKind getKind() const override { return NodeKind::EndNodeKind; }

  // "classof" needed for dyn_cast
  static bool classof(const AdjNode *node) {
    return node->getKind() == NodeKind::EndNodeKind;
  }
};

//----------------------------------------------------------------------------
// SinkNode: Derived directly from AdjNode
//----------------------------------------------------------------------------
class SinkNode : public AdjNode {
public:
  SinkNode(mlir::Operation *op, const std::vector<std::string> &predecessors,
           const std::vector<std::string> &successors,
           const std::map<std::string, unsigned> &sucDataWidthMap,
           unsigned latency, const unsigned &bbIndex)
      : AdjNode(op, predecessors, successors, sucDataWidthMap, latency,
                bbIndex) {}

  // For SinkNode, valid switching and valid set are not defined by hardware;
  // They are intended to be provided via profiling.
  void calValidSwitching() {}
  void calValidSet() {}

  void calReadySwitching(const std::string &preNodeName) {
    // Sink node is always ready.
    readySignal[preNodeName] = 0;
  }

  void calReadySet(const std::string &preNodeName, unsigned II) {
    IISet fullSet(II,true/*default*/);
    setR[preNodeName] = fullSet;
  }

  void calDataout() {}

  // LLVM Casting support
  NodeKind getKind() const override { return NodeKind::SinkNodeKind; }

  // "classof" needed for dyn_cast
  static bool classof(const AdjNode *node) {
    return node->getKind() == NodeKind::SinkNodeKind;
  }
};

//===----------------------------------------------------------------------===//
//
// Model for Start Node
//
//===----------------------------------------------------------------------===//
class StartNode : public AdjNode {
public:
  /// Constructor.
  StartNode(mlir::Operation *op, const std::vector<std::string> &predecessors,
            const std::vector<std::string> &successors,
            const std::map<std::string, unsigned> &sucDataWidthDict,
            unsigned latency, const unsigned &bbIndex);

  // Handshake functions:
  /// Set the valid switching value for a given successor to 2.
  void calValidSwitching(const std::string &sucNodeName);

  /// These methods are not implemented (pass).
  void calValidSet();
  void calReadySwitching();
  void calReadySet();

  // LLVM Casting support
  NodeKind getKind() const override { return NodeKind::StartNodeKind; }

  // "classof" needed for dyn_cast
  static bool classof(const AdjNode *node) {
    return node->getKind() == NodeKind::StartNodeKind;
  }
};

//===----------------------------------------------------------------------===//
//
// Model for Memory controllers, no concrete implementation for now
//
//===----------------------------------------------------------------------===//
class MemConNode : public AdjNode {
public:
  MemConNode(mlir::Operation *op, const std::vector<std::string> &predecessors,
             const std::vector<std::string> &successors,
             const std::map<std::string, unsigned> &sucDataWidthMap,
             unsigned latency, const unsigned &bbIndex)
      : AdjNode(op, predecessors, successors, sucDataWidthMap, latency,
                bbIndex) {}
};

class LSQNode : public AdjNode {
public:
  LSQNode(mlir::Operation *op, const std::vector<std::string> &predecessors,
          const std::vector<std::string> &successors,
          const std::map<std::string, unsigned> &sucDataWidthMap,
          unsigned latency, const unsigned &bbIndex)
      : AdjNode(op, predecessors, successors, sucDataWidthMap, latency,
                bbIndex) {}
};

#endif // EXPERIMENTAL_TRANSFORMS_SWITCHING_EXECUTION_MODEL_H
