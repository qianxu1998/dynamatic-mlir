//===- NodeInfo.h - Switching Estimation -----*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the Node for the Adjacency Graph
//
//===----------------------------------------------------------------------===//

#pragma once
#ifndef NODEINFO_HPP
#define NODEINFO_HPP
#include "dynamatic/Support/LLVM.h"
#include "llvm/ADT/SmallBitVector.h"
#include <map>
#include <string>
#include <vector>
#include <set>
using IISet = llvm::SmallBitVector;

using namespace mlir;
using namespace dynamatic;
// using namespace dynamatic::handshake;

class AdjNode {
  public:
    // Define the virtual destructor
    virtual ~AdjNode() = default;
  
    // Constructer
    AdjNode(mlir::Operation* selOp, 
            const std::vector<std::string>& predecessors, const std::vector<std::string>& successors, 
            const std::map<std::string, unsigned>& sucDataWidthMap, const unsigned& latency, const unsigned& bbIndex);
  
    // This funciton checks whether the handshake channel swiching information updating is finished or not
    bool handshakeUpdateFinished();
  
    // This function checks the handshake switching number of the node itself is updated or not
    bool handshakeSwitchingChecking();
  
    // This function prints all information of the node
    virtual void printNodeDetails();
  
    // This function prints details of handshake switching
    void printHandshakeSwitching();
  
    // This function will calculate the total valid and ready swithcing for the node
    void totalHandshakeSwitchingUpdate();
  
    // This function calculate the number of switches in the dataout channel
    void updateDataoutChannel(int inputData);
  
    // Add extra handshake channel switches
    void updateHandshakeChannelSwitching(unsigned validChannelSwitching, unsigned readyChannelSwitching);
  
    // This function calculates the total number of switches of all data out channels
    //  Note: We treat "X" as invalid data and will count it only once
    void totalDataSwitchingCounting(bool mapped);
  
    // This function will return the vector of 1's position in the number's binary format
    std::vector<unsigned> getPositionList(int number);
  
    //
    void printDataChannelSwitching();
    void printPerDataChannelToggleNumber();
    void printPerHandshakeChannelToggleNumber();
  
    // Support LLVM node casting
    // Following enum class is needed for isa<> and dyn_cast<>
    enum class NodeKind {AdjNodeKind, BufferNodeKind, JoinNodeKind, PassNodeKind, CmpiNodeKind, CmpfNodeKind, AddiNodeKind, AddfNodeKind,
                          SubiNodeKind, MuliNodeKind, MulfNodeKind, ExtsiNodeKind, DLoadNodeKind, DStoreNodeKind, MergeNodeKind, SelectNodeKind,
                          CMergeNodeKind, ForkNodeKind, CBrNodeKind, ShliNodeKind, ShrsiNodeKind, ShruiNodeKind, MuxNodeKind, TrunciNodeKind,
                          ExtuiNodeKind, ConstantNodeKind, OriNodeKind, AndiNodeKind, SourceNodeKind, EndNodeKind, SinkNodeKind, StartNodeKind};
  
    // By default, an AdjNode has kind = AdjNodeKind
    virtual NodeKind getKind() const { return NodeKind::AdjNodeKind; }
  
    // "classof" used by llvm casting
    static bool classof(const AdjNode *node) {
      // We claim "yes" if node->getKind() == AdjNodeKind
      // Usually it's trivial for the base class
      return node->getKind() == NodeKind::AdjNodeKind;
    }
  
    // 
    //  Internal Storing Variables
    //
    unsigned nodeLatency = 0;       // Used to store the latency of the chosen node
    mlir::Operation* op;            // Pointer to the operation in the mlir file
    unsigned bbindex;               // BB index for the node
    // Memory controller is exclueded from the pres and sucs
    std::vector<std::string> pres;  // Vector storing the predecessors of the node in the segemnt
    std::vector<std::string> sucs;  // Vector storing the successors of the node in the segement
    std::vector<AdjNode*>    sucsPtrs;  // ← p

    // Handshake Signal Switching
    std::map<std::string, unsigned> validSignal;    // Map used to store number of switching of the node's valid signals (per channel, e.x., {"node_name" : 2})
    std::map<std::string, unsigned> readySignal;    // Map used to store number of switching of the node's ready signals (per channel, e.x., {"node_name" : 2})
    // TODO: Need to improve the way the setR/V is stored
    std::map<std::string, IISet> setV; // Set used to store the active range of the corresponding valid signal in different channels
    std::map<std::string, IISet> setR; // Set used to store active range of the corresponding ready signal in different channels
  
    // Data channel Switching
    std::map<std::string, unsigned> sucsDataWidthMap; // Map from succeeding node name to the channel width
    std::map<std::string, std::map<unsigned, unsigned>> perChannelToggle; // toggle count dict per bitwidth
    std::map<std::string, std::vector<int>> dataOut;  // Map from succeeding node name to the corresponding outptu channel dict
    
    // Overall signal switching status
    std::map<std::string, unsigned> dataSwitches;  // Map storing the number of data switches for different data channels
    std::map<std::string, unsigned> handshakeSwitches;          // Map storing the number of switches in different handshake channels
  
    unsigned totalValidSwitching = 0;
    unsigned totalReadySwitching = 0;
    unsigned totalDataSwitching = 0;
  
    // Update Flag
    bool handshakeUpdateFlag = false;
  };



  #endif