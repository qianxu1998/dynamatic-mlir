#include "experimental/Transforms/Switching/DataGlitches.h"

#include "experimental/Transforms/Switching/GraphModel.h"
#include "experimental/Transforms/Switching/DataChannelCal.h"
#include <string>
#include "llvm/Support/Debug.h"


void dataGlitchNodeSearch(SwitchingInfo &switchInfo, SCFProfilingResult &profileResults) {
  for (const auto & [label, bblist] : switchInfo.segToBBListMap) {
    // Skip all "S", "E", and "T" sections
    if (label.find("S") != std::string::npos || label.find("E") != std::string::npos || label.find("T") != std::string::npos) {
      continue;
    }

    // Get the selected AdjGraph
    auto selAdjGraph = switchInfo.segToAdjGraphMap[label];
    std::string mgBaseNode = selAdjGraph->baseNode;
    auto mgII = switchInfo.cfdfcIIs[std::stoul(label)];

    //! Testing
    llvm::dbgs() << "[DEBUG] \t\tMG II: " << mgII << "\n";

    std::map<std::string, std::vector<NodeGlitchInfo>> tmpGlitchDict;
    std::vector<std::string> tmpGlitchNodes;

    // Iterate over all mapped nodes (orderedMappedList).
    for (const auto& selNode: switchInfo.segToAdjGraphMap[label]->orderedNodeName) {

      auto selDataBaseNodes = switchInfo.segToDataBaseVecMap[label].data;
      if (std::find(selDataBaseNodes.begin(), selDataBaseNodes.end(), selNode) != selDataBaseNodes.end()) {
        // Get the type of the node
        auto nodeType = getNodeType(selNode);

        if (GLITCH_NODE.find(nodeType) != GLITCH_NODE.end()) {
          // ALU will only have two inputs
          auto preNodeLists = selAdjGraph->nodes[selNode]->pres;

          // Unpack the pre_node_list
          std::string preNode1 = preNodeLists[0];
          std::string preNode2 = preNodeLists[1];

          // Get the source of the two inputs
          std::string srcNode1 = segNodeDataSrcSearch(switchInfo, preNode1, selAdjGraph.get());
          std::string srcNode2 = segNodeDataSrcSearch(switchInfo, preNode2, selAdjGraph.get());

          // Check whether this inode can have glitches
          LongestPathResult result1 = selLongestPath(switchInfo, preNode1, label);
          LongestPathResult result2 = selLongestPath(switchInfo, preNode2, label);

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

          for (const auto &selOriNode: switchInfo.dfgBaseNodeValueMap[srcNode1]->segSucNodeMap[label].original) {
            if (preNode1 == selOriNode) preSrc1Buffered = true;
          }

          for (const auto &selOriNode: switchInfo.dfgBaseNodeValueMap[srcNode2]->segSucNodeMap[label].original) {
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
            if (srcNode1.find("constant") == std::string::npos && srcNode2.find("constant") == std::string::npos) {
              tmpGlitchDict[selNode].push_back(tmpNode1);
              tmpGlitchDict[selNode].push_back(tmpNode2);
              tmpGlitchNodes.push_back(selNode);
            }
          } else {
            // Check Case 4
            if (std::find(tmpGlitchNodes.begin(), tmpGlitchNodes.end(), srcNode1) != tmpGlitchNodes.end() &&
                  (std::find(tmpGlitchNodes.begin(), tmpGlitchNodes.end(), srcNode2) == tmpGlitchNodes.end())){
              if (!preSrc1Buffered) {
                tmpGlitchDict[selNode].push_back(tmpNode1);
                tmpGlitchDict[selNode].push_back(tmpNode2);
                tmpGlitchNodes.push_back(selNode);
              }
            } else if (std::find(tmpGlitchNodes.begin(), tmpGlitchNodes.end(), srcNode2) != tmpGlitchNodes.end() &&
                  (std::find(tmpGlitchNodes.begin(), tmpGlitchNodes.end(), srcNode1) == tmpGlitchNodes.end())) {
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
    switchInfo.mgGlitchNodeDict[label] = tmpGlitchDict;
  }
}

int 
calGlitchValue(int op1, int op2, std::string selNode) {
  // TODO: Add supports for other types of nodes in self.GLITCH_NODE
  if (selNode.find("add") != std::string::npos) {
    return op1 + op2;
  } else if (selNode.find("mul") != std::string::npos) {
    return op1 * op2;
  } else {
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
    if (executedSeg.find("S") != std::string::npos || 
          executedSeg.find("E") != std::string::npos ||
          executedSeg.find("T") != std::string::npos) {
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
    for (const auto& selNode: switchInfo.segToOrderedDataBaseNodes[executedSeg]) {
      if (selNode.find("constant") != std::string::npos ||
          selNode.find("source") != std::string::npos ||
          selNode.find("load") != std::string::npos) continue;

      indexUpdateList.push_back(selNode);

      // Define vector to store the tmp glitch value
      std::vector<int> tmpValue;

      // Check whether we need the glitch value calculation
      if (glitchUpdateFlag) {
        // If we have glitch info for this node
        if (switchInfo.mgGlitchNodeDict[executedSeg].find(selNode) != switchInfo.mgGlitchNodeDict[executedSeg].end()) {
          // Get the source node
          std::string preSrc1 = switchInfo.mgGlitchNodeDict[executedSeg][selNode][0].srcNode;
          std::string preSrc2 = switchInfo.mgGlitchNodeDict[executedSeg][selNode][1].srcNode;
          // Get the starting time
          int preSrc1Start = switchInfo.mgGlitchNodeDict[executedSeg][selNode][0].steadyTime;
          int preSrc2Start = switchInfo.mgGlitchNodeDict[executedSeg][selNode][1].steadyTime;
          std::map<std::string, int> srcStartTimeDict = {{preSrc1, preSrc1Start}, {preSrc2, preSrc2Start}};

          // Get the buffering information
          bool preSrc1Buffered = switchInfo.mgGlitchNodeDict[executedSeg][selNode][0].buffered;
          bool preSrc2Buffered = switchInfo.mgGlitchNodeDict[executedSeg][selNode][1].buffered;
          std::map<std::string, bool> srcBufferedDict = {{preSrc1, preSrc1Buffered}, {preSrc2, preSrc2Buffered}};

          // Defining variables for glitch calculation
          int op1 = 0, op2 = 0;
          unsigned op1PreIndex = 0, op2PreIndex = 0;

          // Status Definition
          std::string fasterNode = "";
          std::string slowerNode = "";


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

            switchInfo.dfgBaseNodeValueMap[selNode]->oriGlitchDataOut[i] = 
                {switchInfo.dfgBaseNodeValueMap[selNode]->originalDataOut[i].value};
            switchInfo.dfgBaseNodeValueMap[selNode]->lastUpdateIndex = i;
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
              op1PreIndex = switchInfo.dfgBaseNodeValueMap[fasterNode]->lastUpdateIndex;
              op2PreIndex = switchInfo.dfgBaseNodeValueMap[slowerNode]->lastUpdateIndex;

              //! Testing
              if (debug) {
                llvm::dbgs() << "[DEBUG] \t\t\t[Value 1]: \n"
                          << "[DEBUG] \t\t\t\tOp_1_src_node: " << fasterNode << "\n"
                          << "[DEBUG] \t\t\t\tOp_1_pre_index: " << op1PreIndex << "\n"
                          << "[DEBUG] \t\t\t\tOp_2_src_node: " << slowerNode << "\n"
                          << "[DEBUG] \t\t\t\tOp_2_pre_index: " << op2PreIndex << "\n";
              }

              // Check the existence of the selected iter
              if (switchInfo.dfgBaseNodeValueMap[fasterNode]->originalDataOut.find(op1PreIndex) != switchInfo.dfgBaseNodeValueMap[fasterNode]->originalDataOut.end()) {
                op1 = switchInfo.dfgBaseNodeValueMap[fasterNode]->originalDataOut[op1PreIndex].value;
              } else {
                op1 = 0;
              }

              // Get operand 2, if op_2_pre not in original_dataout keys, we use 0 instead
              if (switchInfo.dfgBaseNodeValueMap[slowerNode]->originalDataOut.find(op2PreIndex) == switchInfo.dfgBaseNodeValueMap[slowerNode]->originalDataOut.end()) {
                //! Testing
                if (debug) {
                  llvm::dbgs() << "[DEBUG] \t\t\t[Value 1]: \n"
                              << "[DEBUG] \t\t\t\t[Warning] S Last Active Iter Not in the corresponding storing structure\n";
                }
                op2 = 0;
              } else {
                op2 = switchInfo.dfgBaseNodeValueMap[slowerNode]->originalDataOut[op2PreIndex].value;
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
            op1PreIndex = switchInfo.dfgBaseNodeValueMap[fasterNode]->lastUpdateIndex;
            op2PreIndex = switchInfo.dfgBaseNodeValueMap[slowerNode]->lastUpdateIndex;

            // Get the value of op1
            if (fasterNode == selNode) {
              if (switchInfo.dfgBaseNodeValueMap[fasterNode]->originalDataOut.find(op1PreIndex) != switchInfo.dfgBaseNodeValueMap[fasterNode]->originalDataOut.end()) {
                op1 = switchInfo.dfgBaseNodeValueMap[fasterNode]->originalDataOut[op1PreIndex].value;
              }
            } else if (fasterNode.find("mux")) {
              // Check whether the source of mux node is the selected node itself
              std::string tmpMuxDataSrcNode = getMuxDataSrc(switchInfo, fasterNode, i);

              if (tmpMuxDataSrcNode == selNode) {
                // TODO: Validate the following checking mechanism
                if (switchInfo.dfgBaseNodeValueMap[fasterNode]->originalDataOut.find(op1PreIndex) == switchInfo.dfgBaseNodeValueMap[fasterNode]->originalDataOut.end()) {
                  op1 = 0;
                } else op1 = switchInfo.dfgBaseNodeValueMap[fasterNode]->originalDataOut[op1PreIndex].value;
              } else {
                op1 = switchInfo.dfgBaseNodeValueMap[fasterNode]->originalDataOut[i].value;
              }
            } else if (switchInfo.mgGlitchNodeDict[executedSeg].find(fasterNode) != switchInfo.mgGlitchNodeDict[executedSeg].end()) {
              // If the faster node is glitching, we check whether the node is buffered or not
              if (srcBufferedDict[fasterNode]) {
                // Buffered
                op1 = switchInfo.dfgBaseNodeValueMap[fasterNode]->originalDataOut[i].value;
              } else {
                // unsigned tmpSize = switchInfo.dfgBaseNodeValueMap[fasterNode]->oriGlitchDataOut[i].size();
                auto  &glitchvec =switchInfo.dfgBaseNodeValueMap[fasterNode]->oriGlitchDataOut[i];
                // ! If the list is smaller than 2, something is wrong
                int op1 = 0;
                if (glitchvec.size() >=2) {
                  op1 = glitchvec[glitchvec.size() - 2];
                  llvm::errs() << "[ERROR] Glitch Vector for " << selNode << " has a size smaller than 2\n";
                }
                else if (!glitchvec.empty()) {
                  op1 = glitchvec.back(); 
                }          // only one glitch value – use it

                else {
                  op1 = switchInfo.dfgBaseNodeValueMap[fasterNode]
                            ->originalDataOut[i].value; // no glitches recorded
                }
              }
            } else {
              op1 = switchInfo.dfgBaseNodeValueMap[fasterNode]->originalDataOut[i].value;
            }

            // Get the value of op2
            if (switchInfo.dfgBaseNodeValueMap[slowerNode]->originalDataOut.find(op2PreIndex) == switchInfo.dfgBaseNodeValueMap[slowerNode]->originalDataOut.end()) {
              op2 = switchInfo.dfgBaseNodeValueMap[slowerNode]->originalDataOut[op2PreIndex].value;
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
            if (fasterNode.find("mux") != std::string::npos) {
              std::string tmpMuxDataSrcNode = getMuxDataSrc(switchInfo, fasterNode, i);

              if (tmpMuxDataSrcNode == selNode) {
                op1PreIndex = switchInfo.dfgBaseNodeValueMap[fasterNode]->lastUpdateIndex;

                // TODO: Validate the following rounding process
                if (switchInfo.dfgBaseNodeValueMap[fasterNode]->originalDataOut.find(op1PreIndex) == switchInfo.dfgBaseNodeValueMap[fasterNode]->originalDataOut.end()) {
                  op1 = 0;
                } else {
                  op1 = switchInfo.dfgBaseNodeValueMap[fasterNode]->originalDataOut[op1PreIndex].value;
                }
              } else {
                op1 = switchInfo.dfgBaseNodeValueMap[fasterNode]->originalDataOut[i].value;
              }
            } else {
              op1 = switchInfo.dfgBaseNodeValueMap[fasterNode]->originalDataOut[i].value;
            }

            // Check whether the slower node is a mux node
            // TODO: Check the condition below
            op2 = switchInfo.dfgBaseNodeValueMap[slowerNode]->originalDataOut[i].value;

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
          tmpValue.push_back(switchInfo.dfgBaseNodeValueMap[selNode]->originalDataOut[i].value);
        }
      } else {
        // No neeed for glitch value calculation, we directly copy the ori data
        tmpValue.push_back(switchInfo.dfgBaseNodeValueMap[selNode]->originalDataOut[i].value);
      }

      // Update the storing structure
      switchInfo.dfgBaseNodeValueMap[selNode]->oriGlitchDataOut[i] = tmpValue;
    }

    // Step 1.5: Update the list of last update index for all data base nodes
    for (const auto& selNode: indexUpdateList) {
      switchInfo.dfgBaseNodeValueMap[selNode]->lastUpdateIndex = i;
    }

    // Step 2: Update value for all MUXs
    // TODO: Pre store the information like mux etc.
    for (const auto& selMuxNode: switchInfo.segToControlNodeList[executedSeg].muxNodeList) {
      std::string selCondInputNode = switchInfo.dataflowGraph->muxToSrcNodeMap[selMuxNode]["control"];

      // We need to check the existence of the control value output
      auto selCMBaseNode = dyn_cast<CMergeData>(switchInfo.dfgBaseNodeValueMap[selCondInputNode].get());
      int condValue = selCMBaseNode->getControlOutput(i);
      std::string selDataSrcNode = switchInfo.dataflowGraph->muxToSrcNodeMap[selMuxNode][std::to_string(condValue)];

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
      if (selDataSrcNode.find("constant") != std::string::npos ||
          selDataSrcNode.find("source") != std::string::npos ||
          selDataSrcNode.find("start") != std::string::npos) {
        curValue.push_back(switchInfo.dfgBaseNodeValueMap[selDataSrcNode]->originalDataOut[0].value);
      } else if (selDataSrcNode == selMuxNode) {
        // The src node is the selected node itself
        unsigned preIndex = switchInfo.dfgBaseNodeValueMap[selDataSrcNode]->lastUpdateIndex;
        // Check whether the preIndex exists or not
        if (switchInfo.dfgBaseNodeValueMap[selDataSrcNode]->originalDataOut.find(preIndex) != switchInfo.dfgBaseNodeValueMap[selDataSrcNode]->originalDataOut.end()) {
          curValue.push_back(switchInfo.dfgBaseNodeValueMap[selDataSrcNode]->originalDataOut[preIndex].value);
        } else curValue.push_back(0);
      } else if (glitchUpdateFlag &&
                  (switchInfo.mgGlitchNodeDict[executedSeg].find(selDataSrcNode) != switchInfo.mgGlitchNodeDict[executedSeg].end())) {
        // Check whether there are buffers in between
        std::string preNode = "";
        auto selMuxNodeStructure = dyn_cast<MuxNode>(switchInfo.dataflowGraph->nodes[selMuxNode].get());

        // Get the actual preNode
        // TODO: Need to check the portidx to name mapping
        for (const auto& [nodeName, portIdx] : selMuxNodeStructure->preNameToPortIdxMap) {
          //* Here we need to do cond + 1, as the port map of muxnode assigns 0 to its control input.
          const unsigned int cond_add1= condValue + 1;
          if (portIdx == (cond_add1)) preNode = nodeName;
        }

        bool bufferedFlag = false;
        if (std::find(switchInfo.dfgBaseNodeValueMap[selDataSrcNode]->segSucNodeMap[executedSeg].original.begin(),
              switchInfo.dfgBaseNodeValueMap[selDataSrcNode]->segSucNodeMap[executedSeg].original.end(), preNode) != 
              switchInfo.dfgBaseNodeValueMap[selDataSrcNode]->segSucNodeMap[executedSeg].original.end()) bufferedFlag = true;
        
        //! Testing
        if (debug) {
          llvm::dbgs() << "\t\tCur src node has glitches, buffered: " << bufferedFlag << "\n";
        }

        if (bufferedFlag) {
          // Src glitching but buffered
          curValue.push_back(switchInfo.dfgBaseNodeValueMap[selDataSrcNode]->originalDataOut[i].value);
        } else {
          curValue = switchInfo.dfgBaseNodeValueMap[selDataSrcNode]->oriGlitchDataOut[i];
        }
      } else {
        // Handling special case for cascaded Muxes
        if (selDataSrcNode.find("mux") != std::string::npos && 
            (switchInfo.dfgBaseNodeValueMap[selDataSrcNode]->originalDataOut.find(i) == switchInfo.dfgBaseNodeValueMap[selDataSrcNode]->originalDataOut.end())) {
          curValue.push_back(switchInfo.dfgBaseNodeValueMap[selDataSrcNode]->originalDataOut[i - 1].value);
        } else {
          curValue.push_back(switchInfo.dfgBaseNodeValueMap[selDataSrcNode]->originalDataOut[i].value);
        }
      }

      //! Testing
      if (debug) {
        llvm::dbgs() << "\t\tCur_value: ";
        for (auto v : curValue) llvm::dbgs() << v << " ";
        llvm::dbgs() << "\n\t[TRANSATION GLITCHES]\n";
      }

      // Calculate control flow glitches
      if (!switchInfo.dfgBaseNodeValueMap[selMuxNode]->skipControlCal) {
        std::string nodePreValidSeg = switchInfo.dfgBaseNodeValueMap[selMuxNode]->lastValidSeg;
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
            if (std::find(switchInfo.segToControlNodeList[segExecTrace[i + 1]].muxNodeList.begin(), 
                        switchInfo.segToControlNodeList[segExecTrace[i + 1]].muxNodeList.end(), selMuxNode) != switchInfo.segToControlNodeList[segExecTrace[i + 1]].muxNodeList.end()) {
              std::string nextExecSegLabel = segExecTrace[i + 1];

              if (nextExecSegLabel != executedSeg) {
                // Type 2 detected
                doubleTransFlag = true;
              }
            }
          }

          // Calculate preList
          if (executedSeg.find("E") == std::string::npos) {
            // Cond value will be the same for the last segment
            int preCondValue = 1 - condValue;
            std::string preDataSrc = switchInfo.dataflowGraph->muxToSrcNodeMap[selMuxNode][std::to_string(preCondValue)];

            // Check whether the value exist or not
            if (preDataSrc.find("constant") != std::string::npos) {
              preValue.push_back(switchInfo.dfgBaseNodeValueMap[preDataSrc]->originalDataOut[0].value);
            } else if (switchInfo.dfgBaseNodeValueMap[preDataSrc]->originalDataOut.find(i) == switchInfo.dfgBaseNodeValueMap[preDataSrc]->originalDataOut.end()) {
              preValue.push_back(switchInfo.dfgBaseNodeValueMap[preDataSrc]->originalDataOut[i].value);
            }
          } 

          // If double transition
          if (doubleTransFlag) {
            switchInfo.dfgBaseNodeValueMap[selMuxNode]->skipControlCal = true;
            nexValue = preValue;
          }
          // TODO: Check the following CondValue condition, it maybe 0 now
        } else if (nodePreValidSeg != "" && condValue == 1) {
          //! Con_Merge's control output automatically go back to 0 need to check why!!!!!!!
          if (executedSeg.find("E") == std::string::npos) {
            // Cond value will be the same for the last segment
            int preCondValue = 1 - condValue;
            std::string preDataSrc = switchInfo.dataflowGraph->muxToSrcNodeMap[selMuxNode][std::to_string(preCondValue)];
            [[maybe_unused]] unsigned newIterIdx = i;

            // If we have the same sources for both cond_value
            if (preDataSrc == selDataSrcNode) {
              // Change the iterindex
              if (switchInfo.segToBackedgePairMap.find(executedSeg) != switchInfo.segToBackedgePairMap.end()) {
                auto mgIndexList = switchInfo.backEdgeToCFDFCMap[switchInfo.segToBackedgePairMap[executedSeg]];
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
            if (preDataSrc.find("constant")){
              preValue.push_back(switchInfo.dfgBaseNodeValueMap[preDataSrc]->originalDataOut[0].value);
            } else if (switchInfo.dfgBaseNodeValueMap[preDataSrc]->originalDataOut.find(i) != switchInfo.dfgBaseNodeValueMap[preDataSrc]->originalDataOut.end()) {
              preValue.push_back(switchInfo.dfgBaseNodeValueMap[preDataSrc]->originalDataOut[i].value);
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
        switchInfo.dfgBaseNodeValueMap[selMuxNode]->skipControlCal = false;
      }

      // Get the final mux output data list
      std::vector<int> finalMuxOutputList;

      for (const auto& selValue: preValue) finalMuxOutputList.push_back(selValue);
      for (const auto& selValue: curValue) finalMuxOutputList.push_back(selValue);
      for (const auto& selValue: nexValue) finalMuxOutputList.push_back(selValue);
    }

    // Step 3: Relay memory load node's data
    for (const auto& selNode: switchInfo.segToOrderedDataBaseNodes[executedSeg]) {
      if (selNode.find("load") != std::string::npos) {
        switchInfo.dfgBaseNodeValueMap[selNode]->lastUpdateIndex = i;
      }
    }
  }
}
