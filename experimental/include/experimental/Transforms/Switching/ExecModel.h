//===- ExecModel.h - Switching Estimation -----*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the execution model for all node types used in dynamatic during switching estimation
//
//===----------------------------------------------------------------------===//

#ifndef EXPERIMENTAL_TRANSFORMS_SWITCHING_EXECUTION_MODEL_H
#define EXPERIMENTAL_TRANSFORMS_SWITCHING_EXECUTION_MODEL_H

#include "experimental/Transforms/Switching/SwitchingSupport.h"
#include "mlir/IR/Operation.h"

#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

//===----------------------------------------------------------------------===//
//
// Model for Buffer Node
//
//===----------------------------------------------------------------------===//
class BufferNode : public AdjNode {
public:
  // Constructor
  BufferNode(mlir::Operation *op,
             const std::vector<std::string> &predecessors,
             const std::vector<std::string> &successors,
             const std::map<std::string, unsigned> &sucDataWidthMap,
             const unsigned &latency);

  // Handshake Counting functions
  void calValidSwitching(const std::string &sucNodeName, unsigned II);
  void calReadySwitching(const std::string &preNodeName, unsigned II);

  void calValidSet(const std::string &sucNodeName,
                   std::set<unsigned> &inSetV);
  void calReadySet(const std::string &preNodeName,
                   std::set<unsigned> &inSetR);

  // Override printDetail function
  void printDetail() override;

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
class JoinNode: public AdjNode {
public:
  // Constructor: simply forwards to AdjNode
  JoinNode(mlir::Operation *op,
           const std::vector<std::string> &predecessors,
           const std::vector<std::string> &successors,
           const std::map<std::string, unsigned> &sucDataWidthMap,
           unsigned latency)
      : AdjNode(op, predecessors, successors, sucDataWidthMap, latency) {}
  
  // - calValidSwitching: if either numValid0 or numValid1 > 0, set valid signal to 2;
  //   if both are 0, then set valid signal to 0; if II==1, then also set to 0; otherwise, set to 2.
  virtual void calValidSwitching(const std::string &sucNodeName,
                         unsigned &numValid0,
                         unsigned &numValid1,
                         unsigned &II);
  
  // calValidSet: if the valid signal for sucNodeName is 0 (or II==1), then store the full set {0,...,II-1};
  // otherwise, store only {nodeStartTime}.
  virtual void calValidSet(const std::string &sucNodeName,
                   unsigned &nodeStartTime,
                   unsigned &II);
  
  // calReadySwitching: if both numValid and numReady are 0, set ready signal to 0;
  // if either > 0, set it to 2.
  virtual void calReadySwitching(const std::string &preNodeName,
                         unsigned &numValid,
                         unsigned &numReady);

  // calReadySet: if both set_valid and set_ready are provided (non-null),
  // then take their intersection; otherwise, if readySignal[preNodeName] is 0 then store the full set {0,...,II-1},
  // if >0 then store {nodeStartTime}.
  // (We pass the sets as pointers so that a null pointer represents “None”.)
  virtual void calReadySet(const std::string &preNodeName,
                   const std::set<unsigned> *setValid,
                   const std::set<unsigned> *setReady,
                   unsigned &nodeStartTime,
                   unsigned &II);

  // setReadySet: as a convenience, take the first key from setVMap and copy its set into setRMap for preNodeName.
  void setReadySet(const std::string &preNodeName);
};

//===----------------------------------------------------------------------===//
//
// Model for Pass Node
//
//===----------------------------------------------------------------------===//
class PassNode : public AdjNode {
public:
  // Constructor: forwards to AdjNode
  PassNode(mlir::Operation *op,
           const std::vector<std::string> &predecessors,
           const std::vector<std::string> &successors,
           const std::map<std::string, unsigned> &sucDataWidthMap,
           unsigned latency)
      : AdjNode(op, predecessors, successors, sucDataWidthMap, latency) {}

  // Handshake counting functions:
  // calValidSwitching: if numValid is 0, set valid signal to 0; else to 2.
  virtual void calValidSwitching(const std::string &sucNodeName,
                         unsigned &numValid);

  // calValidSet: if validSignal[sucNodeName] is 0, store full set {0,...,II-1}; else store {nodeStartTime}.
  virtual void calValidSet(const std::string &sucNodeName,
                   unsigned &nodeStartTime,
                   unsigned &II);

  // calReadySwitching: if numReady is 0, set ready signal to 0; else to 2.
  virtual void calReadySwitching(const std::string &preNodeName,
                         unsigned &numReady);

  // calReadySet: if a set (setR) is provided, simply store it; else if readySignal[preNodeName] is 0, store full set; 
  // if >0, store {nodeStartTime}; otherwise, if setVMap is not empty and its full set is not equal to {0,...,II-1}, use that.
  virtual void calReadySet(const std::string &preNodeName,
                   const std::set<unsigned> *setReady,
                   unsigned &nodeStartTime,
                   unsigned &II);

  // setReadySwitching: directly set readySignal.
  void setReadySwitching(const std::string &preNodeName,
                         unsigned &numReadySwitches);

  // setReadySet: store the provided set into setRMap.
  void setReadySet(const std::string &preNodeName,
                   const std::set<unsigned> &setReady);
};

//===----------------------------------------------------------------------===//
//
// Model for Cmpi Node: derived from JoinNode
//
//===----------------------------------------------------------------------===//
class CmpiNode : public JoinNode {
public:
  CmpiNode(mlir::Operation *op,
           const std::vector<std::string> &predecessors,
           const std::vector<std::string> &successors,
           const std::map<std::string, unsigned> &sucDataWidthMap,
           unsigned latency)
      : JoinNode(op, predecessors, successors, sucDataWidthMap, latency) {}

  void calValidSwitching(const std::string &sucNodeName,
                         unsigned &numValid0,
                         unsigned &numValid1,
                         unsigned &II) override {
    JoinNode::calValidSwitching(sucNodeName, numValid0, numValid1, II);
  }
  
  void calValidSet(const std::string &sucNodeName,
                   unsigned &nodeStartTime,
                   unsigned &II) override {
    JoinNode::calValidSet(sucNodeName, nodeStartTime, II);
  }
  
  void calReadySwitching(const std::string &preNodeName,
                         unsigned &numValid,
                         unsigned &numReady) override {
    JoinNode::calReadySwitching(preNodeName, numValid, numReady);
  }
  
  void calReadySet(const std::string &preNodeName,
                   const std::set<unsigned> *setValid,
                   const std::set<unsigned> *setReady,
                   unsigned &nodeStartTime,
                   unsigned &II) override {
    JoinNode::calReadySet(preNodeName, setValid, setReady, nodeStartTime, II);
  }
};

//===----------------------------------------------------------------------===//
//
// Model for Addi Node: derived from JoinNode
//
//===----------------------------------------------------------------------===//
class AddiNode : public JoinNode {
public:
  AddiNode(mlir::Operation *op,
           const std::vector<std::string> &predecessors,
           const std::vector<std::string> &successors,
           const std::map<std::string, unsigned> &sucDataWidthMap,
           unsigned latency)
      : JoinNode(op, predecessors, successors, sucDataWidthMap, latency) {}

  void calValidSwitching(const std::string &sucNodeName,
                         unsigned &numValid0,
                         unsigned &numValid1,
                         unsigned &II) override {
    JoinNode::calValidSwitching(sucNodeName, numValid0, numValid1, II);
  }
  
  void calValidSet(const std::string &sucNodeName,
                   unsigned &nodeStartTime,
                   unsigned &II) override {
    JoinNode::calValidSet(sucNodeName, nodeStartTime, II);
  }
  
  void calReadySwitching(const std::string &preNodeName,
                         unsigned &numValid,
                         unsigned &numReady) override {
    JoinNode::calReadySwitching(preNodeName, numValid, numReady);
  }
  
  void calReadySet(const std::string &preNodeName,
                   const std::set<unsigned> *setValid,
                   const std::set<unsigned> *setReady,
                   unsigned &nodeStartTime,
                   unsigned &II) override {
    JoinNode::calReadySet(preNodeName, setValid, setReady, nodeStartTime, II);
  }
};

//===----------------------------------------------------------------------===//
//
// Model for Subi Node
//
//===----------------------------------------------------------------------===//
class SubiNode : public JoinNode {
public:
  SubiNode(mlir::Operation *op,
           const std::vector<std::string> &predecessors,
           const std::vector<std::string> &successors,
           const std::map<std::string, unsigned> &sucDataWidthMap,
           unsigned latency)
      : JoinNode(op, predecessors, successors, sucDataWidthMap, latency) {}

  void calValidSwitching(const std::string &sucNodeName,
                         unsigned &numValid0,
                         unsigned &numValid1,
                         unsigned &II) override {
    JoinNode::calValidSwitching(sucNodeName, numValid0, numValid1, II);
  }
  
  void calValidSet(const std::string &sucNodeName,
                   unsigned &nodeStartTime,
                   unsigned &II) override {
    JoinNode::calValidSet(sucNodeName, nodeStartTime, II);
  }
  
  void calReadySwitching(const std::string &preNodeName,
                         unsigned &numValid,
                         unsigned &numReady) override {
    JoinNode::calReadySwitching(preNodeName, numValid, numReady);
  }
  
  void calReadySet(const std::string &preNodeName,
                   const std::set<unsigned> *setValid,
                   const std::set<unsigned> *setReady,
                   unsigned &nodeStartTime,
                   unsigned &II) override {
    JoinNode::calReadySet(preNodeName, setValid, setReady, nodeStartTime, II);
  }
};

//===----------------------------------------------------------------------===//
//
// Model for Muli Node
//
//===----------------------------------------------------------------------===//
class MuliNode : public JoinNode {
public:
  MuliNode(mlir::Operation *op,
           const std::vector<std::string> &predecessors,
           const std::vector<std::string> &successors,
           const std::map<std::string, unsigned> &sucDataWidthMap,
           unsigned latency)
      : JoinNode(op, predecessors, successors, sucDataWidthMap, latency) {}

  void calValidSwitching(const std::string &sucNodeName,
                         unsigned &numValid0,
                         unsigned &numValid1,
                         unsigned &II) override {
    JoinNode::calValidSwitching(sucNodeName, numValid0, numValid1, II);
  }
  
  void calValidSet(const std::string &sucNodeName,
                   unsigned &nodeStartTime,
                   unsigned &II) override {
    JoinNode::calValidSet(sucNodeName, nodeStartTime, II);
  }
  
  void calReadySwitching(const std::string &preNodeName,
                         unsigned &numValid,
                         unsigned &numReady) override {
    JoinNode::calReadySwitching(preNodeName, numValid, numReady);
  }
  
  void calReadySet(const std::string &preNodeName,
                   const std::set<unsigned> *setValid,
                   const std::set<unsigned> *setReady,
                   unsigned &nodeStartTime,
                   unsigned &II) override {
    JoinNode::calReadySet(preNodeName, setValid, setReady, nodeStartTime, II);
  }
};

//===----------------------------------------------------------------------===//
//
// Model for Extsi Node
//
//===----------------------------------------------------------------------===//
class ExtsiNode : public PassNode {
public:
  ExtsiNode(mlir::Operation *op,
            const std::vector<std::string> &predecessors,
            const std::vector<std::string> &successors,
            const std::map<std::string, unsigned> &sucDataWidthMap,
            unsigned latency)
      : PassNode(op, predecessors, successors, sucDataWidthMap, latency) {}

  void calValidSwitching(const std::string &sucNodeName,
                         unsigned &numValid) override {
    PassNode::calValidSwitching(sucNodeName, numValid);
  }
  
  void calValidSet(const std::string &sucNodeName,
                   unsigned &nodeStartTime,
                   unsigned &II) override {
    PassNode::calValidSet(sucNodeName, nodeStartTime, II);
  }
  
  void calReadySwitching(const std::string &preNodeName,
                         unsigned &numReady) override {
    PassNode::calReadySwitching(preNodeName, numReady);
  }
  
  void calReadySet(const std::string &preNodeName,
                   const std::set<unsigned> *setR,
                   unsigned &nodeStartTime,
                   unsigned &II) override {
    PassNode::calReadySet(preNodeName, setR, nodeStartTime, II);
  }
};

//===----------------------------------------------------------------------===//
//
// Model for Load Node
//
//===----------------------------------------------------------------------===//
class DLoadNode : public AdjNode {
public:
  DLoadNode(mlir::Operation *op,
            const std::vector<std::string> &predecessors,
            const std::vector<std::string> &successors,
            const std::map<std::string, unsigned> &sucDataWidthMap,
            const unsigned &latency);

  // Handshake functions:
  // For valid switching: if num_valid == 0, valid signal is 0; else if > 0, valid signal is 2.
  void calValidSwitching(const std::string &sucNodeName, unsigned &numValid);

  // For valid set: if valid signal is 0, use the full set {0,...,II-1};
  // else, if an input set is provided, shift it by the node latency.
  void calValidSet(const std::string &sucNodeName, const std::set<unsigned> *inSetV, unsigned &II);

  // For ready switching: if num_ready == 0, ready signal is 0; else ready signal is 2.
  void calReadySwitching(const std::string &preNodeName, unsigned &numReady);

  // For ready set: for a load node, we assume the ready set is taken directly if provided;
  // otherwise, if readySignal is 0 then use full set, or else use {nodeStartTime}.
  void calReadySet(const std::string &preNodeName, const std::set<unsigned> *inSetR, 
                    unsigned &nodeStartTime, unsigned &II);

  // Update the data output channels.
  //   - For data output: if a previous value exists, compute diff (using XOR) and update toggle counts.
  //   - For address output: if the new address is -10, reuse the last valid value.
  void updateDataout(int &inputData, int addressValue);

  void printDetail() override;

  // Extra members for DLoad nodes
  std::string dataOutNodeName;    // Name for the data out channel
  std::string addressOutNodeName; // this should be the corresponding name of the mem_controller
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
  DStoreNode(mlir::Operation *op,
             const std::vector<std::string> &predecessors,
             const std::vector<std::string> &successors,
             const std::map<std::string, unsigned> &sucDataWidthMap,
             unsigned latency);

  // Handshake functions for d_store.
  // For valid switching on the memory controller channel (key "mc").
  void calValidSwitching(unsigned numValid1, unsigned numValid2);
  void calValidSet(const std::set<unsigned> *setV0,
                           const std::set<unsigned> *setV1,
                           unsigned II);

  void calReadySwitching(const std::string &preNodeName,
                                 unsigned numValid1,
                                 unsigned numValid2);
  void calReadySet(const std::string &preNodeName,
                           const std::set<unsigned> *setV0,
                           const std::set<unsigned> *setV1,
                           unsigned II);

  // Update data output based on the source input node.
  void updateDataout(int inputData, const std::string &srcInputNode);

  void printDetail() override;

  // Handshake checking (e.g. comparing size of readySignal with number of predecessors)
  bool handshakeSwitchingChecking();

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
  MergeNode(mlir::Operation *op,
            const std::vector<std::string> &predecessors,
            const std::vector<std::string> &successors,
            const std::map<std::string, unsigned> &sucDataWidthMap,
            unsigned latency)
      : AdjNode(op, predecessors, successors, sucDataWidthMap, latency) {}

  // calValidSwitching:
  //   setVList: vector of active valid sets for each input channel.
  //   numVList: vector of valid signal switching counts for each input channel.
  //   II: initiation interval.
  void calValidSwitching(const std::string &sucNodeName,
                                 const std::vector<std::set<unsigned>> &setVList,
                                 const std::vector<unsigned> &numVList,
                                 unsigned II);

  // calValidSet:
  //   setVList: vector of active valid sets for each input channel.
  //   II: initiation interval.
  void calValidSet(const std::string &sucNodeName,
                           const std::vector<std::set<unsigned>> &setVList,
                           unsigned II);

  // calReadySwitching:
  //   For ready switching we have a single number.
  void calReadySwitching(const std::string &preNodeName,
                                 unsigned numReady);

  // calReadySet:
  //   setR is provided as a pointer (if not null, it is used).
  void calReadySet(const std::string &preNodeName,
                           const std::set<unsigned> *setR,
                           unsigned II);
};


//===----------------------------------------------------------------------===//
//
// Model for CMerge Node
//
//===----------------------------------------------------------------------===//


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



//===----------------------------------------------------------------------===//
//
// Model for CBr Node
//
//===----------------------------------------------------------------------===//


//===----------------------------------------------------------------------===//
//
// Model for Shli Node
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
//
// Model for Shrsi Node
//
//===----------------------------------------------------------------------===//


//===----------------------------------------------------------------------===//
//
// Model for Shrui Node
//
//===----------------------------------------------------------------------===//


//===----------------------------------------------------------------------===//
//
// Model for Mux Node
//
//===----------------------------------------------------------------------===//


//===----------------------------------------------------------------------===//
//
// Model for Trunci Node
//
//===----------------------------------------------------------------------===//


//===----------------------------------------------------------------------===//
//
// Model for Extui Node
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
//
// Model for Constant Node
//
//===----------------------------------------------------------------------===//


//===----------------------------------------------------------------------===//
//
// Model for Ori Node
//
//===----------------------------------------------------------------------===//


//===----------------------------------------------------------------------===//
//
// Model for Andi Node
//
//===----------------------------------------------------------------------===//


//===----------------------------------------------------------------------===//
//
// Model for Source Node
//
//===----------------------------------------------------------------------===//


//===----------------------------------------------------------------------===//
//
// Model for Start Node
//
//===----------------------------------------------------------------------===//


#endif // EXPERIMENTAL_TRANSFORMS_SWITCHING_EXECUTION_MODEL_H
