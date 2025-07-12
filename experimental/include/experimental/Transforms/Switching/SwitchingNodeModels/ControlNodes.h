//===- ControlNodes.h - Switching Estimation -----*- C++ -*-===//
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

#ifndef EXPERIMENTAL_TRANSFORMS_SWITCHING_CONTROL_NODES_H
#define EXPERIMENTAL_TRANSFORMS_SWITCHING_CONTROL_NODES_H



#include "experimental/Transforms/Switching/NodeInfo.h"
#include "experimental/Transforms/Switching/SwitchingNodeModels/NodeHelpers.h"

#include "experimental/Transforms/Switching/SwitchingSupport.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/DenseMap.h"

#include <map>
#include <memory>
// #include <set>
#include <string>
#include <vector>





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
    void calReadySwitching(const std::string &preNodeName, int numReady){
      unsigned val=(numReady == 0)?0:2;
      setReady(this,preNodeName,val);
  
  }
    // calReadySet:
    //   setR is provided as a bitvector (if not empty with vec.any(), it is used).
    void calReadySet(const std::string &preNodeName,
                     const IISet &setR, unsigned II);
    DEFINE_NODE_KIND(MergeNode,  NodeKind::MergeNodeKind)  ;
    
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
  void calValidSwitching(const std::string &sucNodeName, unsigned II){
    unsigned val=(II == 1)?0:2;
    setValid(this,sucNodeName,val);
  }
  inline void calValidSet(const std::string &sucNodeName, unsigned nodeStartTime,
                   unsigned II){
                    if (validSignal[sucNodeName] == 0)
                    setVSet(this, sucNodeName, fullSet(II));
                else
                    setVSet(this, sucNodeName, singleton(nodeStartTime, II));
                   }

  void calReadySwitching(const std::string &preNodeName){
    setReady(this, preNodeName, 0);
  }
  inline void calReadySet(const std::string &preNodeName, unsigned II){
    setRSet(this,preNodeName,fullSet(II));
  }
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
  void printNodeDetails() override;

  DEFINE_NODE_KIND(CMergeNode,  NodeKind::CMergeNodeKind)  ;


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
  using AdjNode::AdjNode;  // Constructor: simply forwards to AdjNode


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
                    llvm::DenseMap<StringRef, IISet *> &setRDict,
                    const llvm::DenseMap<StringRef,  int> &numReadyDict,
                    unsigned sucNodeStart, unsigned nodeSteadyStart,
                    unsigned II);

  // calValidSet:
  //   If the valid signal for sucNodeName is 0, assign the full set
  //   {0,...,II-1}. Otherwise, if numValid > 0, assign {nodeStartTime}; else,
  //   assign the ready set from setRDict.
  void calValidSet(const std::string &sucNodeName, unsigned nodeStartTime,
                   int numValid,
                   llvm::DenseMap<StringRef, IISet *> &setRDict,

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
                   const llvm::DenseMap<StringRef, IISet *> &setRDict,
                   unsigned II);

  DEFINE_NODE_KIND(ForkNode,  NodeKind::ForkNodeKind)  ;


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
  void printNodeDetails() override;

  DEFINE_NODE_KIND(CBrNode,  NodeKind::CBrNodeKind)  ;

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
  void printNodeDetails() override;

  DEFINE_NODE_KIND(MuxNode,  NodeKind::MuxNodeKind)  ;

  // Extra member: The name of the predecessor on the condition channel.
  std::string conPreNodeName;
  // Map from node name to the corresponding port index
  std::map<std::string, unsigned> preNameToPortIdxMap;
};
//===----------------------------------------------------------------------===//
//
// Model for Source Node
//
//===----------------------------------------------------------------------===//
class SelectNode : public MuxNode {
  public:
    // Constructor.
    //   - op: pointer to the MLIR operation.
    //   - predecessors: a mapping from predecessor names to their port numbers.
    //   - successors: a vector of successor names.
    //   - sucDataWidthMap: mapping from successor names to (data) port numbers.
    //   - latency: the node latency.
    using MuxNode::MuxNode;
  
    // Handshake functions:
    // calValidSwitching: If II == 1, the valid signal is 0; otherwise, it is 2.
    // void calValidSwitching(const std::string &sucNodeName, unsigned II);
  
    // // calValidSet: If the valid signal for sucNodeName is 0 then assign full set
    // // {0,...,II-1}; otherwise, assign {nodeStartTime}.
    // void calValidSet(const std::string &sucNodeName, unsigned nodeStartTime,
    //                  unsigned II);
  
    // calReadySwitching:
    //   Parameters:
    //     - preNodeName: name of a predecessor.
    //     - condValue: the condition value at this node.
    //     - num_v: an unsigned representing the valid count.
    //     - setV0: reference to a bitvector (e.g., active cycles) for one input.
    // //     - setVSelect: reference to a bitvector for the selected input.
    // //     - setR: reference to a bitvector for ready.
    // //     - II: initiation interval.
    // void calReadySwitching(const std::string &preNodeName, unsigned condValue,
    //                        int num_v, const IISet &setV0, const IISet &setVSelect,
    //                        const IISet &setR, unsigned II);
  
    // // calReadySet:
    // //   Parameters:
    // //     - preNodeName: name of a predecessor.
    // //     - setCond: reference to a bitvector for the condition.
    // //     - setV: reference to a  valid bitvector.
    // //     - setR: reference to a ready bitvector.
    // //     - II: initiation interval.
    // // If readySignal for preNodeName is 0, assign full set {0,...,II-1};
    // // else assign the intersection: setCond ∩ setV ∩ setR.
    // void calReadySet(const std::string &preNodeName, const IISet &setCond,
    //                  const IISet &setV, const IISet &setR, unsigned II);
  
    // // Print detail: call the base class printDetail() and then print extra info.
    // void printDetail() override;
  
    DEFINE_NODE_KIND(SelectNode,  NodeKind::SelectNodeKind)  ;
  
    // Extra member: The name of the predecessor on the condition channel.
    // std::string conPreNodeName;
    // Map from node name to the corresponding port index
    std::map<std::string, unsigned> preNameToPortIdxMap;
  };
  
//===----------------------------------------------------------------------===//
//
// Model for Source Node
//
//===----------------------------------------------------------------------===//
class SourceNode : public AdjNode {
public:
    using AdjNode::AdjNode;  // Constructor: simply forwards to AdjNode


  void calValidSwitching(const std::string &sucNodeName) {
    validSignal[sucNodeName] = 0;
  }

  void calValidSet(const std::string &sucNodeName, unsigned II) {
    //  d full set {0, 1, ..., II-1}
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


  DEFINE_NODE_KIND(SourceNode,  NodeKind::SourceNodeKind)  ;

};

//===----------------------------------------------------------------------===//
//
// Model for End Node
//
//===----------------------------------------------------------------------===//
class EndNode : public AdjNode {
public:
  using AdjNode::AdjNode;  // Constructor: simply forwards to AdjNode

  DEFINE_NODE_KIND(EndNode,  NodeKind::EndNodeKind)  ;

};

//----------------------------------------------------------------------------
// SinkNode: Derived directly from AdjNode
//----------------------------------------------------------------------------
class SinkNode : public AdjNode {
public:
  using AdjNode::AdjNode;  // Constructor: simply forwards to AdjNode


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

  DEFINE_NODE_KIND(SinkNode,  NodeKind::SinkNodeKind)  ;

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

  DEFINE_NODE_KIND(StartNode,  NodeKind::StartNodeKind)  ;

};

//===----------------------------------------------------------------------===//
//
// Model for Memory controllers, no concrete implementation for now
//
//===----------------------------------------------------------------------===//
class MemConNode : public AdjNode {
public:
    using AdjNode::AdjNode;  // Constructor: simply forwards to AdjNode

};

class LSQNode : public AdjNode {
public:
    using AdjNode::AdjNode;  // Constructor: simply forwards to AdjNode

};

#endif