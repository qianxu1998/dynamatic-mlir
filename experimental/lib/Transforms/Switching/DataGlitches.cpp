//===- DataGlitches.cpp - Switching Estimation -----*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares all functions used for data channel switching calculation
//
//===----------------------------------------------------------------------===//#include "experimental/Transforms/Switching/DataGlitches.h"

#include "experimental/Transforms/Switching/GraphModel.h"
#include "experimental/Transforms/Switching/utils.h"
#include "experimental/Transforms/Switching/DataChannelCal.h"
#include "experimental/Transforms/Switching/SwitchingNodeModels/SwitchingNodeModels.h"
#include "experimental/Transforms/Switching/SwitchingSupport.h"
#include "experimental/Transforms/Switching/DFSKernel.h"
#include "experimental/Transforms/Switching/DataGlitches.h"

#include <string>
#include "llvm/Support/Debug.h"
/// Return the first non-buffer predecessor along the **unique** path
/// between `node` and the Mux control port.
/// If you hit more than one predecessor (fan-in) we bail out –
/// only transparent chains are supported.
static std::string peelBufferChain(const SwitchingInfo &SI,
                                   std::string node) {
  // Use find() instead of operator[] to handle const StringMap
  while (contains(node, "buffer")) {
    auto nodeIt = SI.data.dfgBaseNodeValue.find(node);
    if (nodeIt == SI.data.dfgBaseNodeValue.end()) {
      llvm::errs() << "[ERROR] Node " << node << " not found in dfgBaseNodeValue\n";
      return node;
    }
    
    const auto &preds = nodeIt->second->segSucNodeMap; // Note: using segSucNodeMap as it exists
    if (preds.size() != 1) {
      llvm::errs() << "[ERROR] Buffer " << node
                   << " has multiple or no predecessors; can't peel.\n";
      return node;        // give up – caller will still error-out cleanly
    }
    
    const auto& firstSeg = preds.begin()->second;
    if (firstSeg.original.size() != 1) {
      llvm::errs() << "[ERROR] Buffer " << node
                   << " has multiple original predecessors; can't peel.\n";
      return node;
    }
    
    node = firstSeg.original.front(); // hop one step upstream
  }
  return node;
}


void dataGlitchNodeSearch(SwitchingInfo &switchInfo, SCFProfilingResult &profileResults) {
  for (const auto & [label, bblist] : switchInfo.staticinfo.segToBBs) {
    // Skip all "S", "E", and "T" sections
    if (contains(label,"S") || contains(label,"E")||contains(label,"T")) {
      continue;
    }

    // Get the selected AdjGraph
    auto selAdjGraph = switchInfo.staticinfo.segToGraph[label];
    std::string mgBaseNode = selAdjGraph->baseNode;
    auto mgII = switchInfo.staticinfo.cfdfcIIs[std::stoul(std::string(label))];
    auto selDataBaseNodes = switchInfo.data.segToDataBaseVec[label].data;
    auto orderedNodes =selAdjGraph->orderedNodeName;
    //! Testing
    llvm::dbgs() << "[DEBUG] \t\tMG II: " << mgII << "\n";

    std::map<std::string, std::vector<NodeGlitchInfo>> tmpGlitchDict;
    std::vector<std::string> tmpGlitchNodes;

    // Iterate over all mapped nodes (orderedMappedList).
    for (const auto& selNode: orderedNodes) {
      if (containsValue(selDataBaseNodes,selNode) ) {
        // Get the type of the node
        auto nodeType = getNodeType(selNode);

        if (contains(GLITCH_NODE,nodeType)) {
          // ALU will only have two inputs
          auto preNodeLists = selAdjGraph->nodes[selNode]->pres;

          // Unpack the pre_node_list
          std::string preNode1 = preNodeLists[0];
          std::string preNode2 = preNodeLists[1];

          // Get the source of the two inputs
          std::string srcNode1 = segNodeDataSrcSearch(switchInfo, preNode1, selAdjGraph.get());
          std::string srcNode2 = segNodeDataSrcSearch(switchInfo, preNode2, selAdjGraph.get());


          // Check whether this inode can have glitches
          LongestPathResult result1 = selLongestPath(switchInfo, preNode1, std::string(label));
          LongestPathResult result2 = selLongestPath(switchInfo, preNode2, std::string(label));

          //! Testing
          // llvm::dbgs() << "[DEBUG] \tNode : " << selNode << "\n";
          // llvm::dbgs() << "[DEBUG] \t\tsrcNode1 : " << srcNode1 << "\n";
          // llvm::dbgs() << "[DEBUG] \t\tsrcNode2 : " << srcNode2 << "\n";

          // TODO: Validate the following rounding process
          int finStartPoint1Steady = static_cast<int>(std::fmod(static_cast<float_t>(result1.maxLatency), mgII)) + 
                                      selAdjGraph->startBaseNodeShiftMap[result1.selStartNode] + selAdjGraph->startBaseNodeShiftMap[mgBaseNode];
          int finStartPoint2Steady = static_cast<int>(std::fmod(static_cast<float_t>(result2.maxLatency), mgII)) + 
                                      selAdjGraph->startBaseNodeShiftMap[result2.selStartNode] + selAdjGraph->startBaseNodeShiftMap[mgBaseNode];

          // TODO: Need to validate the following assumptions
          // If the node is directly connected with the srcs(without any buffers in between), the steady state value can not be used
          // As the latency can not be hidden
          bool preSrc1Buffered = false;
          bool preSrc2Buffered = false;

          for (const auto &selOriNode: switchInfo.data.dfgBaseNodeValue[srcNode1]->segSucNodeMap[label].original) {
            if (preNode1 == selOriNode) preSrc1Buffered = true;
          }

          for (const auto &selOriNode: switchInfo.data.dfgBaseNodeValue[srcNode2]->segSucNodeMap[label].original) {
            if (preNode2 == selOriNode) preSrc2Buffered = true;
          }

          // We assign the faster one with value 0 and slower one with value 1
          if (!preSrc1Buffered && !preSrc2Buffered) {
            finStartPoint1Steady = static_cast<int>(result1.maxLatency) + 
                                      selAdjGraph->startBaseNodeShiftMap[result1.selStartNode] + selAdjGraph->startBaseNodeShiftMap[mgBaseNode];
            finStartPoint2Steady = static_cast<int>(result2.maxLatency) + 
                                      selAdjGraph->startBaseNodeShiftMap[result2.selStartNode] + selAdjGraph->startBaseNodeShiftMap[mgBaseNode];

            if (finStartPoint1Steady != finStartPoint2Steady) {
              if (finStartPoint1Steady > finStartPoint2Steady) {
                finStartPoint1Steady = 1;
                finStartPoint2Steady = 0;
              } else {
                finStartPoint1Steady = 0;
                finStartPoint2Steady = 1;
              }
            }
          }

          // Define status storing structure
          NodeGlitchInfo tmpNode1, tmpNode2;
          tmpNode1.srcNode = srcNode1;
          tmpNode1.steadyTime = finStartPoint1Steady;
          tmpNode1.buffered = preSrc1Buffered;

          tmpNode2.srcNode = srcNode2;
          tmpNode2.steadyTime = finStartPoint2Steady;
          tmpNode2.buffered = preSrc2Buffered;

          // TODO: Validate the following criterias
          if (finStartPoint1Steady != finStartPoint2Steady) {
            //  Check condition for Case 1
            // C1: None of the src node is a constant
            if ((!contains(srcNode1,"constant"))  && (!contains(srcNode2,"constant"))) {
              tmpGlitchDict[selNode].push_back(tmpNode1);
              tmpGlitchDict[selNode].push_back(tmpNode2);
              tmpGlitchNodes.push_back(selNode);
            }
          } else {
            // Check Case 4
            if (containsValue(tmpGlitchNodes, srcNode1) &&(!containsValue(tmpGlitchNodes ,srcNode2) )){
              if (!preSrc1Buffered) {
                tmpGlitchDict[selNode].push_back(tmpNode1);
                tmpGlitchDict[selNode].push_back(tmpNode2);
                tmpGlitchNodes.push_back(selNode);
              }
            } else if (containsValue(tmpGlitchNodes, srcNode2) &&(!containsValue(tmpGlitchNodes ,srcNode1) )) {
              if (!preSrc2Buffered) {
                tmpGlitchDict[selNode].push_back(tmpNode1);
                tmpGlitchDict[selNode].push_back(tmpNode2);
                tmpGlitchNodes.push_back(selNode);
              }
            }
          }
        } 
      }
    }
    
    // Store the glitching info
    switchInfo.data.glitches[label] = tmpGlitchDict;
  }
}

int calGlitchValue(int op1, int op2, std::string selNode) {
  // TODO: Add supports for other types of nodes in self.GLITCH_NODE
  if ( contains(selNode,"add")) {return op1 + op2;} 
  else if ( contains(selNode,"mul")) {return op1 * op2; } 
  else {
    llvm::errs() << "[ERROR] Unexpected node type during glitching calculation: " << selNode << "\n";
    return 0;
  }
}


void dataBaseNodeGlitchUpdate(SwitchingInfo &switchInfo, SCFProfilingResult &profileResults, bool debug) {
  if (debug) {
    llvm::dbgs() << "[DEBUG]\n[DEBUG] \t\t[NODE GLITCHING VALUE CALCULATION]\n";
  }

  //
  auto segExecTrace = profileResults.executedSegTrace;
  std::map<std::string, unsigned> lastSegIterMap;

  // Iterate through the execution trace
  for (unsigned i = 0; i < segExecTrace.size(); i++) {
    std::string executedSeg = segExecTrace[i];
    lastSegIterMap[executedSeg] = i;
    bool glitchUpdateFlag = false;

    if (debug) {
      llvm::dbgs() << "[DEBUG] \t******** Iter: " << i << ", Seg: " << executedSeg << "\n";
    }

    // Check wehther we need to update the glitch value for this seg
    if  (contains(executedSeg,"S") ||contains(executedSeg,"E") ||contains(executedSeg,"T")) {
      glitchUpdateFlag = false;
    } else {
      glitchUpdateFlag = true;
    }

    // Step 1: Calculate glitching for inner MG nodes -- All ALUs, other nodes shall be excluded
    //! For this type of glitching nodes (only ALUs), we only deal with the following glitching cases:
    //!     - CASE 1: No glitching for the two input operands
    //!     - CASE 2: One of the operand is glitching, but the original arriving time of the two inputs are the same.
    //! We ignore all other glitching cases and directly use the value from software profiling(original value)
    // Iterate through all data base nodes
    std::vector<std::string> indexUpdateList;
    for (const auto& selNode: switchInfo.data.segToOrderedDataBaseNodes[executedSeg]) {
      if (contains(selNode,"constant") || contains(selNode,"source")  ||contains(selNode,"load")  ) continue;
          
      indexUpdateList.push_back(selNode);

      // Define vector to store the tmp glitch value
      std::vector<int> tmpValue;

      // Check whether we need the glitch value calculation
      if (glitchUpdateFlag) {
        // If we have glitch info for this node
        if (contains(switchInfo.data.glitches[executedSeg],selNode)) {
          // Get the source node
          std::string preSrc1 = switchInfo.data.glitches[executedSeg][selNode][0].srcNode;
          std::string preSrc2 = switchInfo.data.glitches[executedSeg][selNode][1].srcNode;
          // Get the starting time
          int preSrc1Start = switchInfo.data.glitches[executedSeg][selNode][0].steadyTime;
          int preSrc2Start = switchInfo.data.glitches[executedSeg][selNode][1].steadyTime;
          std::map<std::string, int> srcStartTimeDict = {{preSrc1, preSrc1Start}, {preSrc2, preSrc2Start}};

          // Get the buffering information
          bool preSrc1Buffered = switchInfo.data.glitches[executedSeg][selNode][0].buffered;
          bool preSrc2Buffered = switchInfo.data.glitches[executedSeg][selNode][1].buffered;
          std::map<std::string, bool> srcBufferedDict = {{preSrc1, preSrc1Buffered}, {preSrc2, preSrc2Buffered}};

          // Defining variables for glitch calculation
          int op1 = 0, op2 = 0;
          unsigned op1PreIndex = 0, op2PreIndex = 0;

          // Status Definition
          std::string fasterNode = "", slowerNode = "";

          //! Testing
          if (debug) {
            llvm::dbgs() << "[DEBUG] \t\tGlitch Node: " << selNode << "\n"
                      << "[DEBUG] \t\t\tPre_src_1: " << preSrc1 << "\n"
                      << "[DEBUG] \t\t\tPre_src_2: " << preSrc2 << "\n";
          }

          // If this is the last iteration, we ignore the glitching value, just copy the original value
          if (i == segExecTrace.size() - 1) {
            //! Testing
            if (debug) {
              llvm::dbgs() << "[DEBUG] \t\t(LAST ITER DURING GLITCH CALCULATION)\n";
            }

            switchInfo.data.dfgBaseNodeValue[selNode]->oriGlitchDataOut[i] = 
                {switchInfo.data.dfgBaseNodeValue[selNode]->originalDataOut[i].value};
            switchInfo.data.dfgBaseNodeValue[selNode]->lastUpdateIndex = i;
            continue;
          }

          // Check the starting time
          if (preSrc1Start != preSrc2Start) {
            if (preSrc1Start > preSrc2Start) {
              fasterNode = preSrc2;
              slowerNode = preSrc1;
            } else {
              fasterNode = preSrc1;
              slowerNode = preSrc2;
            }

            // Value 1: F[x - 1] op S[x - 1] if needed
            if (srcStartTimeDict[fasterNode] > 0) {
              // Initialization
              op1PreIndex = switchInfo.data.dfgBaseNodeValue[fasterNode]->lastUpdateIndex;
              op2PreIndex = switchInfo.data.dfgBaseNodeValue[slowerNode]->lastUpdateIndex;

              //! Testing
              if (debug) {
                llvm::dbgs() << "[DEBUG] \t\t\t[Value 1]: \n"
                          << "[DEBUG] \t\t\t\tOp_1_src_node: " << fasterNode << "\n"
                          << "[DEBUG] \t\t\t\tOp_1_pre_index: " << op1PreIndex << "\n"
                          << "[DEBUG] \t\t\t\tOp_2_src_node: " << slowerNode << "\n"
                          << "[DEBUG] \t\t\t\tOp_2_pre_index: " << op2PreIndex << "\n";
              }

              // Check the existence of the selected iter
              if (contains(switchInfo.data.dfgBaseNodeValue[fasterNode]->originalDataOut,op1PreIndex)) {
                op1 = switchInfo.data.dfgBaseNodeValue[fasterNode]->originalDataOut[op1PreIndex].value;
              } else {
                op1 = 0;
              }

              // Get operand 2, if op_2_pre not in original_dataout keys, we use 0 instead
              if (!(contains(switchInfo.data.dfgBaseNodeValue[slowerNode]->originalDataOut,op2PreIndex))){
                //! Testing
                if (debug) {
                  llvm::dbgs() << "[DEBUG] \t\t\t[Value 1]: \n"
                              << "[DEBUG] \t\t\t\t[Warning] S Last Active Iter Not in the corresponding storing structure\n";
                }
                op2 = 0;
              } else {
                op2 = switchInfo.data.dfgBaseNodeValue[slowerNode]->originalDataOut[op2PreIndex].value;
              }

              tmpValue.push_back(calGlitchValue(op1, op2, selNode));

              //! Testing
              if (debug) {
                llvm::dbgs() << "[DEBUG] \t\t\t\tFaster Node: " << fasterNode << "\n"
                          << "[DEBUG] \t\t\t\tOp_1: " << op1 << "\n"
                          << "[DEBUG] \t\t\t\tSlower Node: " << slowerNode << "\n"
                          << "[DEBUG] \t\t\t\tOp_2: " << op2 << "\n";
              }
            }

            // Value 2: F[x] op S[x - 1]
            //! Testing
            if (debug) {
              llvm::dbgs() << "[DEBUG] \t\t\t[Value 2]: \n";
            }

            int op1 = 0, op2 = 0;
            // Be careful that one of the operand may be generated by the selNode itself, then we need to use
            // the value in the previous iteration in the calculation
            op1PreIndex = switchInfo.data.dfgBaseNodeValue[fasterNode]->lastUpdateIndex;
            op2PreIndex = switchInfo.data.dfgBaseNodeValue[slowerNode]->lastUpdateIndex;

            // Get the value of op1
            if (fasterNode == selNode) {
              if (contains(switchInfo.data.dfgBaseNodeValue[fasterNode]->originalDataOut,op1PreIndex)){
                op1 = switchInfo.data.dfgBaseNodeValue[fasterNode]->originalDataOut[op1PreIndex].value;
              }
            } else if (contains(fasterNode,"mux")) {
              // Check whether the source of mux node is the selected node itself
              std::string tmpMuxDataSrcNode = getMuxDataSrc(switchInfo, fasterNode, i);

              if (tmpMuxDataSrcNode == selNode) {
                // TODO: Validate the following checking mechanism
                if (!contains(switchInfo.data.dfgBaseNodeValue[fasterNode]->originalDataOut,op1PreIndex)) {
                  op1 = 0;
                } else op1 = switchInfo.data.dfgBaseNodeValue[fasterNode]->originalDataOut[op1PreIndex].value;
              } else {
                op1 = switchInfo.data.dfgBaseNodeValue[fasterNode]->originalDataOut[i].value;
              }
            } else if (contains(switchInfo.data.glitches[executedSeg],fasterNode) ) {
              // If the faster node is glitching, we check whether the node is buffered or not
              if (srcBufferedDict[fasterNode]) {
                // Buffered
                op1 = switchInfo.data.dfgBaseNodeValue[fasterNode]->originalDataOut[i].value;
              } else {
                // unsigned tmpSize = switchInfo.data.dfgBaseNodeValue[fasterNode]->oriGlitchDataOut[i].size();
                auto  &glitchvec =switchInfo.data.dfgBaseNodeValue[fasterNode]->oriGlitchDataOut[i];
                // ! If the list is smaller than 2, something is wrong
                int op1 = 0;
                if (glitchvec.size() >=2) {
                  op1 = glitchvec[glitchvec.size() - 2];
                  llvm::errs() << "[ERROR] Glitch Vector for " << selNode << " has a size smaller than 2\n";
                } else if (!glitchvec.empty()) {
                  op1 = glitchvec.back(); 
                } else{          // only one glitch value – use ite
                  op1 = switchInfo.data.dfgBaseNodeValue[fasterNode]->originalDataOut[i].value; // no glitches recorded
                }
              }
            } else {
              op1 = switchInfo.data.dfgBaseNodeValue[fasterNode]->originalDataOut[i].value;
            }

            // Get the value of op2
            if (!contains(switchInfo.data.dfgBaseNodeValue[slowerNode]->originalDataOut,op2PreIndex)) {
              op2 = switchInfo.data.dfgBaseNodeValue[slowerNode]->originalDataOut[op2PreIndex].value;
            }

            tmpValue.push_back(calGlitchValue(op1, op2, selNode));

            //! Testing
            if (debug) {
              llvm::dbgs() << "[DEBUG] \t\t\t\tFaster Node: " << fasterNode << "\n"
                        << "[DEBUG] \t\t\t\tF Last Active Iter: " << op1PreIndex << "\n"
                        << "[DEBUG] \t\t\t\tOp_1: " << op1 << "\n"
                        << "[DEBUG] \t\t\t\tSlower Node: " << slowerNode << "\n"
                        << "[DEBUG] \t\t\t\tS Last Active Iter: " << op2PreIndex << "\n"
                        << "[DEBUG] \t\t\t\tOp_2: " << op2 << "\n";
            }

            // Value 3: F[x] op S[x]
            // Check wheter the faster node is a mux node
            if ( contains(fasterNode,"mux")) {
              std::string tmpMuxDataSrcNode = getMuxDataSrc(switchInfo, fasterNode, i);
            
              if (tmpMuxDataSrcNode == selNode) {
                op1PreIndex = switchInfo.data.dfgBaseNodeValue[fasterNode]->lastUpdateIndex;

                // TODO: Validate the following rounding process
                if (!contains(switchInfo.data.dfgBaseNodeValue[fasterNode]->originalDataOut,op1PreIndex) ){
                  op1 = 0;
                } else {
                  op1 = switchInfo.data.dfgBaseNodeValue[fasterNode]->originalDataOut[op1PreIndex].value;
                }
              } else {
                op1 = switchInfo.data.dfgBaseNodeValue[fasterNode]->originalDataOut[i].value;
              }
            } else {
              op1 = switchInfo.data.dfgBaseNodeValue[fasterNode]->originalDataOut[i].value;
            }

            // Check whether the slower node is a mux node
            // TODO: Check the condition below
            op2 = switchInfo.data.dfgBaseNodeValue[slowerNode]->originalDataOut[i].value;

            tmpValue.push_back(calGlitchValue(op1, op2, selNode));

            if (debug) {
              llvm::dbgs() << "\t[Value 3]: \n"
                        << "\t\tOp_1: " << op1 << "\n"
                        << "\t\tOp_2: " << op2 << "\n"
                        << "\t\t[FINAL] ";
              for (auto v : tmpValue) llvm::dbgs() << v << " ";
              llvm::dbgs() << "\n";
            }
          }
        } else {
          tmpValue.push_back(switchInfo.data.dfgBaseNodeValue[selNode]->originalDataOut[i].value);
        }
      } else {
        // No neeed for glitch value calculation, we directly copy the ori data
        tmpValue.push_back(switchInfo.data.dfgBaseNodeValue[selNode]->originalDataOut[i].value);
      }

      // Update the storing structure
      switchInfo.data.dfgBaseNodeValue[selNode]->oriGlitchDataOut[i] = tmpValue;
    }

    // Step 1.5: Update the list of last update index for all data base nodes
    for (const auto& selNode: indexUpdateList) {
      switchInfo.data.dfgBaseNodeValue[selNode]->lastUpdateIndex = i;
    }

    // Step 2: Update value for all MUXs
    // TODO: Pre store the information like mux etc.
    for (const auto& selMuxNode: switchInfo.data.controlNodes[executedSeg].muxNodeList) {
      std::string selCondInputNode = switchInfo.staticinfo.dataflowGraph->muxToSrcNodeMap[selMuxNode]["control"];

      // We need to check the existence of the control value output
      auto selCMBaseNode = dyn_cast<CMergeData>(switchInfo.data.dfgBaseNodeValue[selCondInputNode].get());
      int condValue = selCMBaseNode->getControlOutput(i);
      std::string selDataSrcNode = switchInfo.staticinfo.dataflowGraph->muxToSrcNodeMap[selMuxNode][std::to_string(condValue)];

      // Temp Value vector definition
      std::vector<int> preValue, curValue, nexValue;

      //! Testing
      if (debug) {
        llvm::dbgs() << "Mux Node: " << selMuxNode << "\n"
                    << "\t[CUR_VALUE]\n"
                    << "\t\tCond Node: " << selCondInputNode << "\n"
                    << "\t\tCond_value: " << condValue << "\n"
                    << "\t\tCur data src: " << selDataSrcNode << "\n";
      }

      // TODO: Validate the following indexing mechanism   
      // Check whether we have glitches from the srcs or not
      if ( contains(selDataSrcNode,"constant") || contains(selDataSrcNode,"source")  || contains(selDataSrcNode,"start")) {
        curValue.push_back(switchInfo.data.dfgBaseNodeValue[selDataSrcNode]->originalDataOut[0].value);
      } else if (selDataSrcNode == selMuxNode) {
        // The src node is the selected node itself
        unsigned preIndex = switchInfo.data.dfgBaseNodeValue[selDataSrcNode]->lastUpdateIndex;
        // Check whether the preIndex exists or not
        if (contains(switchInfo.data.dfgBaseNodeValue[selDataSrcNode]->originalDataOut,preIndex)) {
          curValue.push_back(switchInfo.data.dfgBaseNodeValue[selDataSrcNode]->originalDataOut[preIndex].value);
        } else curValue.push_back(0);
      } else if (glitchUpdateFlag &&
                  (contains(switchInfo.data.glitches[executedSeg],selDataSrcNode) ) )  {
        // Check whether there are buffers in between
        std::string preNode = "";
        auto selMuxNodeStructure = dyn_cast<MuxNode>(switchInfo.staticinfo.dataflowGraph->nodes[selMuxNode].get());

        // Get the actual preNode
        // TODO: Need to check the portidx to name mapping
        for (const auto& [nodeName, portIdx] : selMuxNodeStructure->preNameToPortIdxMap) {
          //* Here we need to do cond + 1, as the port map of muxnode assigns 0 to its control input.
          const unsigned int cond_add1= condValue + 1;
          if (portIdx == (cond_add1)) preNode = nodeName;
        }

        bool bufferedFlag = false;
        if ( containsValue(switchInfo.data.dfgBaseNodeValue[selDataSrcNode]->segSucNodeMap[executedSeg].original,preNode) ) bufferedFlag = true;
        
        //! Testing
        if (debug) {
          llvm::dbgs() << "\t\tCur src node has glitches, buffered: " << bufferedFlag << "\n";
        }

        if (bufferedFlag) {
          // Src glitching but buffered
          curValue.push_back(switchInfo.data.dfgBaseNodeValue[selDataSrcNode]->originalDataOut[i].value);
        } else {
          curValue = switchInfo.data.dfgBaseNodeValue[selDataSrcNode]->oriGlitchDataOut[i];
        }
      } else {
        // Handling special case for cascaded Muxes
        if (contains(selDataSrcNode,"mux") && 
            (switchInfo.data.dfgBaseNodeValue[selDataSrcNode]->originalDataOut.find(i) == switchInfo.data.dfgBaseNodeValue[selDataSrcNode]->originalDataOut.end())) {
          curValue.push_back(switchInfo.data.dfgBaseNodeValue[selDataSrcNode]->originalDataOut[i - 1].value);
        } else {
          curValue.push_back(switchInfo.data.dfgBaseNodeValue[selDataSrcNode]->originalDataOut[i].value);
        }
      }
      
      //! Testing
      if (debug) {
        llvm::dbgs() << "\t\tCur_value: ";
        for (auto v : curValue) llvm::dbgs() << v << " ";
        llvm::dbgs() << "\n\t[TRANSATION GLITCHES]\n";
      }

      // Calculate control flow glitches
      if (!switchInfo.data.dfgBaseNodeValue[selMuxNode]->skipControlCal) {
        std::string nodePreValidSeg = switchInfo.data.dfgBaseNodeValue[selMuxNode]->lastValidSeg;
        bool doubleTransFlag = false;

        //! Testing
        if (debug) {
          llvm::dbgs() << "\t\tNode Pre MG: " << nodePreValidSeg << "\n"
                    << "\t\tNode Cur MG: " << executedSeg << "\n";
        }

        // Check whether we have transition between different MGs
        // We have two types of MG transitions
        //    TYPE 1: MG 0 -> MG 1 -> MG 1
        //    TYPE 2: MG 0 -> MG 1 -> MG 0
        if (nodePreValidSeg != "" && (nodePreValidSeg != executedSeg)) {
          // Transition detected, check the next iter mg label
          if (i != segExecTrace.size() - 1) {
            // Check the existence of the mux node
            if (std::find(switchInfo.data.controlNodes[segExecTrace[i + 1]].muxNodeList.begin(), 
                        switchInfo.data.controlNodes[segExecTrace[i + 1]].muxNodeList.end(), selMuxNode) != switchInfo.data.controlNodes[segExecTrace[i + 1]].muxNodeList.end()) {
              std::string nextExecSegLabel = segExecTrace[i + 1];

              if (nextExecSegLabel != executedSeg) {
                // Type 2 detected
                doubleTransFlag = true;
              }
            }
          }

          // Calculate preList
          if (!contains(executedSeg,"E")) {
            // Cond value will be the same for the last segment
            int preCondValue = 1 - condValue;
            std::string preDataSrc = switchInfo.staticinfo.dataflowGraph->muxToSrcNodeMap[selMuxNode][std::to_string(preCondValue)];

            // Check whether the value exist or not
            if (contains(preDataSrc,"constant")) {
              preValue.push_back(switchInfo.data.dfgBaseNodeValue[preDataSrc]->originalDataOut[0].value);
            } else if (switchInfo.data.dfgBaseNodeValue[preDataSrc]->originalDataOut.find(i) == switchInfo.data.dfgBaseNodeValue[preDataSrc]->originalDataOut.end()) {
              preValue.push_back(switchInfo.data.dfgBaseNodeValue[preDataSrc]->originalDataOut[i].value);
            }
          } 

          // If double transition
          if (doubleTransFlag) {
            switchInfo.data.dfgBaseNodeValue[selMuxNode]->skipControlCal = true;
            nexValue = preValue;
          }
          // TODO: Check the following CondValue condition, it maybe 0 now
        } else if (nodePreValidSeg != "" && condValue == 1) {
          //! Con_Merge's control output automatically go back to 0 need to check why!!!!!!!
          if (!contains(executedSeg,"E")) {
            // Cond value will be the same for the last segment
            int preCondValue = 1 - condValue;
            std::string preDataSrc = switchInfo.staticinfo.dataflowGraph->muxToSrcNodeMap[selMuxNode][std::to_string(preCondValue)];
            [[maybe_unused]] unsigned newIterIdx = i;

            // If we have the same sources for both cond_value
            if (preDataSrc == selDataSrcNode) {
              // Change the iterindex
              if (contains(switchInfo.segToBackedgePairMap,executedSeg)) {
                auto mgIndexList = switchInfo.staticinfo.backEdgeToCFDFC[switchInfo.segToBackedgePairMap[executedSeg]];
                if (mgIndexList.size() > 1) {
                  // Get the first one
                  for (const auto & selMG: mgIndexList) {
                    if (std::to_string(selMG) != executedSeg) {
                      newIterIdx = lastSegIterMap[std::to_string(selMG)];
                      break;
                    }
                  }
                }
              } else {
                newIterIdx = i;
              }
            } else {
              newIterIdx = i;
            }

            // Check whether the value exist or not
            if (contains(preDataSrc,"constant")){
              preValue.push_back(switchInfo.data.dfgBaseNodeValue[preDataSrc]->originalDataOut[0].value);
            } else if (switchInfo.data.dfgBaseNodeValue[preDataSrc]->originalDataOut.find(i) != switchInfo.data.dfgBaseNodeValue[preDataSrc]->originalDataOut.end()) {
              preValue.push_back(switchInfo.data.dfgBaseNodeValue[preDataSrc]->originalDataOut[i].value);
            }
          }
        }

        //! Testing
        if (debug) {
          llvm::dbgs() << "\t\tDouble transition: " << doubleTransFlag << "\n"
                        << "\t\tPre_value: ";
              for (auto v : preValue) llvm::dbgs() << v << " ";
              llvm::dbgs() << "\n\t\tNext value: ";
              for (auto v : nexValue) llvm::dbgs() << v << " ";
              llvm::dbgs() << "\n";
        }
      } else {
        switchInfo.data.dfgBaseNodeValue[selMuxNode]->skipControlCal = false;
      }

      // Get the final mux output data list
      std::vector<int> finalMuxOutputList;

      for (const auto& selValue: preValue) finalMuxOutputList.push_back(selValue);
      for (const auto& selValue: curValue) finalMuxOutputList.push_back(selValue);
      for (const auto& selValue: nexValue) finalMuxOutputList.push_back(selValue);
    }

    // Step 3: Relay memory load node's data
    for (const auto& selNode: switchInfo.data.segToOrderedDataBaseNodes[executedSeg]) {
      if (contains(selNode,"load")) {
        switchInfo.data.dfgBaseNodeValue[selNode]->lastUpdateIndex = i;
      }
    }
  }
}





std::string segNodeDataSrcSearch(SwitchingInfo &switchInfo,
                                 std::string startNode,
                                 AdjGraph *selGraph) {
  // 0) trivial case
  if (isDataBase(switchInfo, startNode))
    return startNode;

  std::string foundDataSrc;
  std::unordered_set<std::string> pathSet;

  // 1) backwards‐successors in reverse order (to match your old .back())
  auto succOf = [&](const std::string &node) -> std::vector<std::string> {
    if (!foundDataSrc.empty())
      return {};
    std::vector<std::string> preds;
    if (contains(node, "cond_br")) {
      if (auto *c = dyn_cast<CBrNode>(switchInfo.staticinfo.dataflowGraph->nodes[node].get())) {
        preds.emplace_back(
          !c->dataPreNodeName.empty() ? c->dataPreNodeName
                                      : c->condPreNodeName
        );
      }
    } else if (selGraph->nodes.count(node)) {
      for (auto it = selGraph->nodes[node]->pres.rbegin();
               it != selGraph->nodes[node]->pres.rend(); ++it)
        preds.push_back(*it);
    }
    // drop any already‐in‐path
    preds.erase(std::remove_if(preds.begin(), preds.end(),
                               [&](auto &m){ return isInPath(pathSet, m); }),
                preds.end());
    return preds;
  };

  // 2) onEnter: mark path & catch first base‐node
  auto onEnter = [&](const std::string &n) {
    pathSet.insert(n);
    if (foundDataSrc.empty() && isDataBase(switchInfo, n))
      foundDataSrc = n;
  };

  // 3) onExit: un‐mark path
  auto onExit = [&](const std::string &n) {
    pathSet.erase(n);
  };

  // 4) run your existing DFS kernel
  dfsWithEvents({startNode}, succOf, onEnter, onExit);
  // TODO add Early exit : FoundBaseNode  with try catch
  // 5) report failure
  if (foundDataSrc.empty()) {
    llvm::dbgs() << "[ERROR] Could not find base node for " << startNode << "\n";
    return "";
  }

  return foundDataSrc;
}



LongestPathResult selLongestPath(SwitchingInfo &switchInfo, std::string dstNode,
                                 std::string mgLabel) {
  //
  unsigned maxLatency = 0;
  std::string selStartNode = "";
  std::string lastSecondBuff = "";
  LongestPathResult returnValue;
  auto selGraph = switchInfo.staticinfo.segToGraph[mgLabel];
  [[maybe_unused]] auto selMGII = switchInfo.staticinfo.cfdfcIIs[std::stoul(mgLabel)];

  // Get the list of buffers
  std::vector<std::string> selBuffList;
  for (const auto &selNode : selGraph->orderedNodeName) {
    if (contains(selNode, "buffer")) {
      selBuffList.push_back(selNode);
    }
  }

  // For all start node
  for (const auto &startNode : selGraph->segStartNodes) {
    auto [latency, bestsrcnode] =
        selGraph->getMaxLatency(startNode, dstNode, false, true);
    // llvm::dbgs() << "step5 longest path data channel "<< startNode<<" "<<
    // dstNode <<"\n";

    std::vector<std::pair<unsigned, std::string>> tmpLastSecondBufferList;
    if (latency > maxLatency) {
      maxLatency = latency;
      selStartNode = bestsrcnode;
    }
  }

  // 2. walk the paths just for the last second buffer
  std::vector<std::pair<unsigned, std::string>> tmpLastSecondBufferList;
  for (const auto &startNode : selGraph->segStartNodes) {
    // Find all paths from startNode to dstNode
    auto paths = selGraph->findPaths(startNode, dstNode, false, true);// TODO  this is also hashable with getmaxlatency if the last two nodes can be saevd
    for (const auto &selPath : paths) {
      if (selPath.latency == maxLatency) {
        // Find the list of buffer nodes along the path
        std::vector<std::string> tmpBufferList;
        for (const auto &tmpSelNode : selPath.nodeList) {
          if (contains(tmpSelNode,"buffer")) {
            tmpBufferList.push_back(tmpSelNode);
          }
        }
        // if there are more than 2 buffers
        if (tmpBufferList.size() > 1) {
          tmpLastSecondBufferList.push_back(
              {maxLatency, tmpBufferList[tmpBufferList.size() - 2]});
        }
      }
    }
  }

  // Select the wanted last second buffer for set_r calc
  for (const auto &selPair : tmpLastSecondBufferList) {
    if (selPair.first == maxLatency) {
      // pick the buffer with the lowest occupancy??
      auto selBuffer = dyn_cast<BufferNode>(selGraph->nodes[selPair.second].get());
      float_t oriOcc = 0;
      if (!lastSecondBuff.empty()) {
        auto oriBuffer =dyn_cast<BufferNode>(selGraph->nodes[lastSecondBuff].get());
        oriOcc = oriBuffer->occupancy;
      }
      if (lastSecondBuff.empty() || (selBuffer->occupancy < oriOcc)) {
        lastSecondBuff = selPair.second;
      }
    }
  }

  // Check the validity of the start node
  if (selStartNode.empty()) {
    selStartNode = selGraph->baseNode;
  }

  // Construct the return value
  returnValue.lastSecondBuffer = lastSecondBuff;
  returnValue.maxLatency = maxLatency;
  returnValue.selStartNode = selStartNode;

  return returnValue;
}


std::string getMuxDataSrc(SwitchingInfo &switchInfo, std::string selMuxNode,
                          unsigned selIter) {
  const auto& selMuxSrcMap = switchInfo.staticinfo.dataflowGraph->muxToSrcNodeMap;
  
  // Check if the mux node exists in the map
  auto muxIt = selMuxSrcMap.find(selMuxNode);
  if (muxIt == selMuxSrcMap.end()) {
    llvm::errs() << "[ERROR] Mux node " << selMuxNode << " not found in muxToSrcNodeMap\n";
    return ""; // Return empty string on error
  }
  
  // Check if "control" key exists
  auto controlIt = muxIt->second.find("control");
  if (controlIt == muxIt->second.end()) {
    llvm::errs() << "[ERROR] Control input not found for mux node " << selMuxNode << "\n";
    return ""; // Return empty string on error
  }
  
  // The cond input src will always be a control merge node
  std::string muxCondInput = controlIt->second;
  muxCondInput = peelBufferChain(switchInfo, muxCondInput);

  // TODO: Prevent the following situation from happening
  if (contains(muxCondInput,"buffer")) {
    llvm::errs() << "[ERROR] A buffer is directly preceding " << selMuxNode
                 << "'s control input port\n";
  }
  
  // Use find() instead of operator[] for safer access
  auto nodeIt = switchInfo.data.dfgBaseNodeValue.find(muxCondInput);
  if (nodeIt == switchInfo.data.dfgBaseNodeValue.end()) {
    llvm::errs() << "[ERROR] Node " << muxCondInput << " not found in dfgBaseNodeValue\n";
    return ""; // Return empty string on error
  }
  
  // Get the corresponding Control Merge Data base node
  auto* selCMBaseNode = dyn_cast<CMergeData>(nodeIt->second.get());
  if (!selCMBaseNode) {
    llvm::errs() << "[ERROR] No selCMBaseNode, got " << muxCondInput << "'s control input port\n";
    return ""; // Return empty string on error
  }
  
  int ctrlInValue = selCMBaseNode->getControlOutput(selIter);
  
  // Check if the control value key exists
  auto dataIt = muxIt->second.find(std::to_string(ctrlInValue));
  if (dataIt == muxIt->second.end()) {
    llvm::errs() << "[ERROR] Data input for control value " << ctrlInValue 
                 << " not found for mux node " << selMuxNode << "\n";
    return ""; // Return empty string on error
  }

  return dataIt->second;
}


std::string getMuxDataSrcold(SwitchingInfo &switchInfo, std::string selMuxNode,
  unsigned selIter) {
auto selMuxSrcMap = switchInfo.staticinfo.dataflowGraph->muxToSrcNodeMap;
// The cond input src will always be a control merge node
std::string muxCondInput = selMuxSrcMap[selMuxNode]["control"];
auto *val = switchInfo.data.dfgBaseNodeValue[muxCondInput].get();

// TODO: Prevent the following situation from happening
if (muxCondInput.find("buffer")) {
llvm::errs() << "[ERROR] A buffer is directly preceding " << selMuxNode
<< "'s control input port\n";
}
// Get the corresponding Control Merge Data base node
auto selCMBaseNode =dyn_cast<CMergeData>(switchInfo.data.dfgBaseNodeValue[muxCondInput].get());
int controlInputValue = selCMBaseNode->getControlOutput(selIter);

return selMuxSrcMap[selMuxNode][std::to_string(controlInputValue)];
}

std::string getMuxDataSrco(SwitchingInfo &SI,
                          const std::string &selMuxNode,
                          unsigned iter) {
  const auto &srcMap = SI.staticinfo.dataflowGraph->muxToSrcNodeMap;
  std::string muxCondInput = srcMap.at(selMuxNode).at("control");

  // Peel transparent buffers, if any
  muxCondInput = peelBufferChain(SI, muxCondInput);

  auto *cmNode =
      llvm::dyn_cast_or_null<CMergeData>(SI.data.dfgBaseNodeValue[muxCondInput].get());
  if (!cmNode) {
    llvm::errs() << "[ERROR] Expected a CMergeData after peeling, got "
                 << muxCondInput << "\n";
    return "";
  }

  int ctrlValue = cmNode->getControlOutput(iter);
  return srcMap.at(selMuxNode).at(std::to_string(ctrlValue));
}



