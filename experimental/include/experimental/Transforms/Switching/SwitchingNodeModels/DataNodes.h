//===- DataNodes.h - Switching Estimation -----*- C++ -*-===//
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


#ifndef EXPERIMENTAL_TRANSFORMS_SWITCHING_DATA_NODES_H
#define EXPERIMENTAL_TRANSFORMS_SWITCHING_DATA_NODES_H

#include "experimental/Transforms/Switching/NodeInfo.h"
#include "experimental/Transforms/Switching/SwitchingNodeModels/NodeHelpers.h"

#include "experimental/Transforms/Switching/SwitchingSupport.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/DenseMap.h"

#include <map>
#include <memory>

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
    BufferNode(mlir::Operation *op, const std::vector<std::string> &predecessors,
               const std::vector<std::string> &successors,
               const std::map<std::string, unsigned> &sucDataWidthMap,
               const unsigned &latency, const unsigned &bbIndex);
  
    // Handshake Counting functions
    inline void calValidSwitching(const std::string &sucNodeName, unsigned II){
      if (setV.find(sucNodeName) != setV.end()) {
        unsigned val=setV[sucNodeName].all()/* every bit is valid*/?0:2;
        setValid(this,sucNodeName,val);
      }
    }
    
    inline void calReadySwitching(const std::string &preNodeName,
      unsigned II) {
        if (setR.find(preNodeName) != setR.end()) {
          unsigned val= (setR[preNodeName].all()) ? 0:2;
          setReady(this,preNodeName,val);
        }
      }
  
    void calValidSet(const std::string &sucNodeName, IISet &inSetV);//  TODO1 change to const
    void calReadySet(const std::string &preNodeName,// TODO1 change to const
                     const IISet &inSetR); // TODO reference  TODO1 change to const
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
  };
  
//===----------------------------------------------------------------------===//
//
// Model for Join Node
//
//===----------------------------------------------------------------------===//
class JoinNode : public AdjNode {
  public:   
    using AdjNode::AdjNode;  // Constructor: simply forwards to AdjNode

    // - calValidSwitching: if either numValid0 or numValid1 > 0, set valid signal
    // to 2;
    //   if both are 0, then set valid signal to 0; if II==1, then also set to 0;
    //   otherwise, set to 2.
    inline virtual void calValidSwitching(const std::string &sucNodeName, int &numValid0,
                                   int &numValid1, unsigned &II){
                                    unsigned val{};
                                    if (numValid0 > 0 || numValid1 > 0) {
                                        val = 2;
                                    } else if (II == 1) {
                                        val = 0;
                                    } else {
                                        val = 0;
                                    }
                                    setValid(this, sucNodeName, val);
                                   }
  
                              
    // calValidSet: if the valid signal for sucNodeName is 0 (or II==1), then
    // store the full set {0,...,II-1}; otherwise, store only {nodeStartTime}.
    inline virtual void calValidSet(const std::string &sucNodeName,
                             unsigned &nodeStartTime, unsigned &II){
     // Check if validSignal contains sucNodeName.
     if (validSignal.find(sucNodeName) != validSignal.end()) {
       if (validSignal[sucNodeName] == 0 || II == 1) {
        setVSet(this, sucNodeName, fullSet(II));
       } else {
        setVSet(this,sucNodeName, singleton(nodeStartTime,II));
       }
     }
   }
   
    // calReadySwitching: if both numValid and numReady are 0, set ready signal to
    // 0; if either > 0, set it to 2.
    inline virtual void calReadySwitching(const std::string &preNodeName, int &numValid,
                                   int &numReady){
                                    unsigned val = (numValid==0 && numReady==0) ? 0:2;
                                    setReady(this, preNodeName, val);
                                   }
  
    // calReadySet: if both set_valid and set_ready are provided (non-null),
    // then take their intersection; otherwise, if readySignal[preNodeName] is 0
    // then store the full set {0,...,II-1}, if >0 then store {nodeStartTime}. (We
    // pass the sets as bitvectors so that a negative  vector.any() represents “None”.)
    inline virtual void calReadySet(const std::string &preNodeName,
                             const IISet &setValid, const IISet &setReady,
  
                             unsigned &nodeStartTime, unsigned &II){
       if (setValid.any() && setReady.any()) {// intersect, since both sets have bits
         setRSet(this, preNodeName, intersection_(setValid,setReady));
       } else if (readySignal.find(preNodeName) != readySignal.end()) {
         if (readySignal[preNodeName] == 0) {
           setR[preNodeName] = fullSet(II);
         } else if (readySignal[preNodeName] > 0) {
          setRSet(this,preNodeName,singleton(nodeStartTime,II));
         }
       }
                             }
  
    // setReadySet: as a convenience, take the first key from setVMap and copy its
    // set into setRMap for preNodeName.
    inline void setReadySet(const std::string &preNodeName){
      if (!setV.empty()) {    // If setVMap is not empty, take the first available set and copy it to
        // setRMap.
        auto it = setV.begin();
        setR[preNodeName] = it->second;
      }
    }
    DEFINE_NODE_KIND(JoinNode,  NodeKind::JoinNodeKind)  ;

  };
  
  //===----------------------------------------------------------------------===//
  //
  // Model for Pass Node
  //
  //===----------------------------------------------------------------------===//
  class PassNode : public AdjNode {
  public:
    // Constructor: forwards to AdjNode
      using AdjNode::AdjNode;  // Constructor: simply forwards to AdjNode

  
    // Handshake counting functions:
    // calValidSwitching: if numValid is 0, set valid signal to 0; else to 2.
    inline virtual void calValidSwitching(const std::string &sucNodeName, int &numValid){
      unsigned val = (numValid==0)?0:2;
      setValid(this,sucNodeName,val);
    }
  
    // calValidSet: if validSignal[sucNodeName] is 0, store full set {0,...,II-1};
    // else store {nodeStartTime}.
    virtual void calValidSet(const std::string &sucNodeName,
                             unsigned &nodeStartTime, unsigned &II);
  
    // calReadySwitching: if numReady is 0, set ready signal to 0; else to 2.
    inline virtual void calReadySwitching(const std::string &preNodeName, int &numReady){
      unsigned val = ( numReady==0) ? 0:2;
      setReady(this, preNodeName, val);
    }
  
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
    DEFINE_NODE_KIND(PassNode,  NodeKind::PassNodeKind)  ;

  };
  
  //===----------------------------------------------------------------------===//
  //
  // Model for Cmpi Node: derived from JoinNode
  //
  //===----------------------------------------------------------------------===//
  
  class CmpiNode : public JoinNode {
  public:
      using JoinNode::JoinNode;  // inherit all of JoinNode’s constructotors
    DEFINE_NODE_KIND(CmpiNode,  NodeKind::CmpiNodeKind)  ;
  };
  
  //===----------------------------------------------------------------------===//
  //
  // Model for Addi Node: derived from JoinNode
  //
  //===----------------------------------------------------------------------===//
  class AddiNode : public JoinNode {
  public:
    using JoinNode::JoinNode;  // inherit all of JoinNode’s constructotors
    DEFINE_NODE_KIND(AddiNode,  NodeKind::AddiNodeKind)  ;

  };
  
  //===----------------------------------------------------------------------===//
  //
  // Model for Subi Node
  //
  //===----------------------------------------------------------------------===//
  class SubiNode : public JoinNode {
  public:
      using JoinNode::JoinNode;  // inherit all of JoinNode’s constructotors  
    DEFINE_NODE_KIND(SubiNode,  NodeKind::SubiNodeKind)  ;
  };
  
  //===----------------------------------------------------------------------===//
  //
  // Model for Muli Node
  //
  //===----------------------------------------------------------------------===//
  class MuliNode : public JoinNode {
  public:
      using JoinNode::JoinNode;  // inherit all of JoinNode’s constructotors
    DEFINE_NODE_KIND(MuliNode,  NodeKind::MuliNodeKind)  ;
  };
  
  //===----------------------------------------------------------------------===//
  //
  // Model for Extsi Node
  //
  //===----------------------------------------------------------------------===//
  class ExtsiNode : public PassNode {
  public:
      using PassNode::PassNode;  // inherit all of PassNode constructotors
    void calValidSwitching(const std::string &sucNodeName,
                           int &numValid) override {
                            unsigned val = (numValid==0)?0:2;
                            setValid(this,sucNodeName,val);
    }
    void calReadySwitching(const std::string &preNodeName,
                           int &numReady) override {
                            unsigned val = ( numReady==0) ? 0:2;
                            setReady(this, preNodeName, val);
    }

    DEFINE_NODE_KIND(ExtsiNode,  NodeKind::ExtsiNodeKind)  ;
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
    inline void calValidSwitching(const std::string &sucNodeName, int &numValid){
      unsigned val= (numValid==0)?0:2;
      setValid(this,sucNodeName,val);
    }
    // For valid set: if valid signal is 0, use the full set {0,...,II-1};
    // else, if an input set is provided, shift it by the node latency.
    inline void calValidSet(const std::string &sucNodeName, const IISet &inSetV,
                     unsigned &II){
                      if (validSignal.find(sucNodeName) != validSignal.end()) {
                          if (validSignal.find(sucNodeName) != validSignal.end()) {
                          if (validSignal[sucNodeName] == 0) {
                          setV[sucNodeName] = IISet(II,true);
                          } else if (inSetV.any() ) {
                          IISet finalSet(II, false);
                          for (unsigned bitidx : inSetV.set_bits())
                          finalSet.set((bitidx + nodeLatency) % II); // shifted
                          setV[sucNodeName] = std::move(finalSet);
                          }
                          }
                          }
                        }
  
                     
  
    // For ready switching: if num_ready == 0, ready signal is 0; else ready
    // signal is 2.
    inline void calReadySwitching(const std::string &preNodeName, int &numReady){
      unsigned val = (numReady==0)?0:2;
      setReady(this,preNodeName,val);
    }
  
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
  
    void printNodeDetails() override;
  
    // LLVM Casting support
    DEFINE_NODE_KIND(DLoadNode,  NodeKind::DLoadNodeKind)  ;

  
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
    inline void calValidSwitching(int numValid1, int numValid2){
      // For the memory controller channel, use key "mc".
      unsigned val = (numValid1 > 0 || numValid2 > 0) ? 4:0;
      setValid(this,"mem",val);
    }
    void calValidSet(const IISet &setV0, const IISet &setV1, unsigned II);
  
    inline void calReadySwitching(const std::string &preNodeName, int numValid1,
                           int numValid2){
  // TODO: May need to change the following modeling as the implementation
  unsigned val = (numValid1 > 0 || numValid2 > 0) ? 2 : 0;
  setReady(this, preNodeName, val);
  }
                           
  
    void calReadySet(const std::string &preNodeName, const IISet &setV0,
                     const IISet &setV1, unsigned II);
  
    // Update data output based on the source input node.
    void updateDataout(int inputData, const std::string &srcInputNode);
  
    void printNodeDetails() override;
  
    // Handshake checking (e.g. comparing size of readySignal with number of
    // predecessors)
    bool handshakeSwitchingChecking();
  
    DEFINE_NODE_KIND(DStoreNode,  NodeKind::DStoreNodeKind)  ;

    // Extra members for DStoreNode.
    std::string dataInNode;
    std::string addressInNode;
    std::string dataInSrcNode;
    std::string addressInSrcNode;
  };
  //===----------------------------------------------------------------------===//
  //
  // Model for Addf Node
  //
  //===----------------------------------------------------------------------===//
  class AddfNode : public AddiNode{
    public:

      using AddiNode::AddiNode;  // inherit all of AddiNode constructotors

      DEFINE_NODE_KIND(AddfNode,  NodeKind::AddfNodeKind)  ;

    };
  
  //===----------------------------------------------------------------------===//
  //
  // Model for Mulf Node
  //
  //===----------------------------------------------------------------------===//
  class MulfNode : public MuliNode {
    public:
      using MuliNode::MuliNode;  // inherit all of MuliNode constructotors
      DEFINE_NODE_KIND(MulfNode,  NodeKind::MulfNodeKind)  ;

    };
  //===----------------------------------------------------------------------===//
  //
  // Model for Cmpf Node: derived from CmpiNode
  //
  //===----------------------------------------------------------------------===//
  class CmpfNode : public CmpiNode {
    public:
        using CmpiNode::CmpiNode;
      DEFINE_NODE_KIND(CmpfNode,  NodeKind::CmpfNodeKind)  ;

    };
  
  //===----------------------------------------------------------------------===//
  //
  // Model for Shli Node
  //
  //===----------------------------------------------------------------------===//
  class ShliNode : public JoinNode {
    public:
      using JoinNode::JoinNode;  // inherit all of JoinNode’s constructotors
      DEFINE_NODE_KIND(ShliNode,  NodeKind::ShliNodeKind)  ;

    };
    //===----------------------------------------------------------------------===//
    //
    // Model for Shrsi Node
    //
    //===----------------------------------------------------------------------===//
    class ShrsiNode : public JoinNode {
    public:
        using JoinNode::JoinNode;  // inherit all of JoinNode’s constructotors
        DEFINE_NODE_KIND(ShrsiNode,  NodeKind::ShrsiNodeKind)  ;

    };
    
    //===----------------------------------------------------------------------===//
    //
    // Model for Shrui Node
    //
    //===----------------------------------------------------------------------===//
    class ShruiNode : public JoinNode {
    public:
        using JoinNode::JoinNode;  // inherit all of JoinNode’s constructotors

        DEFINE_NODE_KIND(ShruiNode,  NodeKind::ShruiNodeKind)  ;

    };
  
  //===----------------------------------------------------------------------===//
  //
  // Model for Trunci Node
  //
  //===----------------------------------------------------------------------===//
  class TrunciNode : public PassNode {
    public:
      using PassNode::PassNode;  // inherit all of PassNode constructotors
      DEFINE_NODE_KIND(TrunciNode,  NodeKind::TrunciNodeKind)  ;
  };
    
    //===----------------------------------------------------------------------===//
    //
    // Model for Extui Node
    //
    //===----------------------------------------------------------------------===//
    class ExtuiNode : public PassNode {
    public:
      using PassNode::PassNode;  // inherit all of PassNode constructotors

      DEFINE_NODE_KIND(ExtuiNode,  NodeKind::ExtuiNodeKind)  ;

    };
    
    //===----------------------------------------------------------------------===//
    //
    // Model for Constant Node
    //
    //===----------------------------------------------------------------------===//
    class ConstantNode : public PassNode {
    public:
      using PassNode::PassNode;  // inherit all of PassNode constructotors

      DEFINE_NODE_KIND(ConstantNode,  NodeKind::ConstantNodeKind)  ;

    };
    
    //===----------------------------------------------------------------------===//
    //
    // Model for Ori Node
    //
    //===----------------------------------------------------------------------===//
    class OriNode : public JoinNode {
    public:
        using JoinNode::JoinNode;  // inherit all of JoinNode’s constructotors

        DEFINE_NODE_KIND(OriNode,  NodeKind::OriNodeKind)  ;

    };
    //===----------------------------------------------------------------------===//
    //
    // Model for Andi Node
    //
    //===----------------------------------------------------------------------===//
    class AndiNode : public JoinNode {
    public:
        using JoinNode::JoinNode;  // inherit all of JoinNode’s constructotors
        DEFINE_NODE_KIND(AndiNode,  NodeKind::AndiNodeKind)  ;

    };
    

#endif


