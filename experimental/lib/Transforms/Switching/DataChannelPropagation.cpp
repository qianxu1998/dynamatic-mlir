#include "experimental/Transforms/Switching/DataChannelPropagation.h"
#include "experimental/Transforms/Switching/DFSKernel.h"
#include "experimental/Transforms/Switching/SwitchingNodeModels/SwitchingNodeModels.h"




void dfgDataChannelPropagate(SwitchingInfo &switchInfo,
                             SCFProfilingResult &profileResults, bool debug) {
  //
  auto segExecTrace = profileResults.executedSegTrace;

  // Iterate through the execution trace and propagate base nodes values in the
  // selected segment
  for (unsigned i = 0; i < segExecTrace.size(); i++) {
    std::string executedSeg = segExecTrace[i];

    // List storing nodes that need special treatment
    std::vector<std::string> pendingMemList;

    //! Testing
    if (debug)
      llvm::dbgs() << "[DEBUG] ==================================\n";

    // Propagate data values for all non_memory related nodes
    for (const auto &selNode :
         switchInfo.data.segToDataBaseVec[executedSeg].all) {
      //! Testing
      if (debug)
        llvm::dbgs() << "[DEBUG] Node: " << selNode << "\n";

      // Case 1: Mapped Nodes --> All ALUs 
      if (containsValue(switchInfo.data.segToOrderedALUNodes[executedSeg], selNode)) {
        //! Testing
        if (debug)
          llvm::dbgs() << "[DEBUG] ALU Node Detected \n";
        
        // Defensive check for ALU node
        
        if ( !contains(switchInfo.data.dfgBaseNodeValue,selNode) ){
          llvm::dbgs() << "[DEBUG] Warning: ALU node " << selNode << " not found in dfgBaseNodeValue at iteration " << i << "\n";
          continue;
        }
        
        // The value shall be obtained from ori_glitch_dataout
        // Step 1: Update the selected node itself
        for (const auto &selValue :
             switchInfo.data.dfgBaseNodeValue[selNode]->oriGlitchDataOut[i]) {
          switchInfo.staticinfo.dataflowGraph->nodes[selNode]->updateDataoutChannel(
              selValue);

          // Step 2: Update the glitching suceeding list
          //! Testing
          if (debug)
            llvm::dbgs() << "[DEBUG] \t[Glitch Output]\n";
          
          // Defensive check for segSucNodeMap access
          if (!contains(switchInfo.data.dfgBaseNodeValue[selNode]->segSucNodeMap,executedSeg) ){
            // llvm::dbgs() << "[DEBUG] Warning: segment " << executedSeg << " not found in segSucNodeMap for ALU node " << selNode << "\n";
            continue;
          }
          
          for (const auto &selSucNode : switchInfo.data.dfgBaseNodeValue[selNode]->segSucNodeMap[executedSeg].glitch) {
            //! Testing
            if (debug)
              llvm::dbgs() << "[DEBUG] \t\tSucNode: " << selSucNode << "\n";

            // Get the node bitwidth
            unsigned tmpNodeWidth = switchInfo.data.dfgBaseNodeValue[selNode]->segSucNodeMap[executedSeg].dataWidthMap[selSucNode];

            //! Testing
            if (debug)
              llvm::dbgs() << "[DEBUG] \t\t\tDataWidth: " << tmpNodeWidth
                           << "\n";

            if (contains(selSucNode, "store")) {
              // Get the actual node storing structure
              auto selStoreNode = dyn_cast<DStoreNode>(
                  switchInfo.staticinfo.dataflowGraph->nodes[selSucNode].get());
              selStoreNode->updateDataout(reduceBits(selValue, tmpNodeWidth),
                                          selNode);
            } else if (contains(selSucNode, "cond_br")) {
              // Get the node
              auto selCondStoreNode = dyn_cast<CBrNode>(switchInfo.staticinfo.dataflowGraph->nodes[selSucNode].get());
              // Get the control value
              // TODO: Validate the assumption that cond_value is always updated
              // before the data_in of the node
              int tmpCondValue =
                  getCondBrNodeCondValue(switchInfo, i, selSucNode);

              selCondStoreNode->updateDataout(reduceBits(selValue, tmpNodeWidth), tmpCondValue);
            } else {
              switchInfo.staticinfo.dataflowGraph->nodes[selSucNode]->updateDataoutChannel(reduceBits(selValue, tmpNodeWidth));
            }
          }

          //! Testing
          if (debug)
            llvm::dbgs() << "[DEBUG] \t[DONE]\n";
        }

        // Step 3: Update ori succeeding node list
        //! Testing
        if (debug)
          llvm::dbgs() << "[DEBUG] \t[Ori Output]\n";
        for (const auto &selSucNode : switchInfo.data.dfgBaseNodeValue[selNode]->segSucNodeMap[executedSeg].original) {
          // Get Node Bitwidth
          unsigned tmpNodeWidth = switchInfo.data.dfgBaseNodeValue[selNode]->segSucNodeMap[executedSeg].dataWidthMap[selSucNode];

          //! Testing
          if (debug)
            llvm::dbgs() << "[DEBUG] \t\tSuc Node: " << selSucNode << "\n";

          //
          if (contains(selSucNode, "store")) {
            //
            auto selStoreNode = dyn_cast<DStoreNode>(
                switchInfo.staticinfo.dataflowGraph->nodes[selSucNode].get());
            selStoreNode->updateDataout(
                reduceBits(switchInfo.data.dfgBaseNodeValue[selNode]->originalDataOut[i].value,tmpNodeWidth),
                selNode);
          } else if (contains(selSucNode, "cond_br")) {
            // Get the node
            // TODO: Validate the assumption that cond_value is always updated
            // before the data_in of the node
            auto selCondStoreNode = dyn_cast<CBrNode>(
                switchInfo.staticinfo.dataflowGraph->nodes[selSucNode].get());
            int tmpCondValue =
                getCondBrNodeCondValue(switchInfo, i, selSucNode);

            selCondStoreNode->updateDataout(
                reduceBits(switchInfo.data.dfgBaseNodeValue[selNode]
                               ->originalDataOut[i]
                               .value,
                           tmpNodeWidth),
                tmpCondValue);
          } else {
            switchInfo.staticinfo.dataflowGraph->nodes[selSucNode]->updateDataoutChannel(
                reduceBits(switchInfo.data.dfgBaseNodeValue[selNode]
                               ->originalDataOut[i]
                               .value,
                           tmpNodeWidth));
          }
        }
        //! Testing
        if (debug)
          llvm::dbgs() << "[DEBUG] \t[DONE]\n";
      } else {
        // Case 2: This is a control merge node

        if (contains(selNode, "control_merge")) {
          //! Testing
          if (debug)
            llvm::dbgs() << "[DEBUG] Control Merge Detected \n";

          // Defensive check for control merge node
          if (!contains(switchInfo.data.dfgBaseNodeValue,selNode)) {
            llvm::dbgs() << "[DEBUG] Warning: Control merge node " << selNode << " not found in dfgBaseNodeValue at iteration " << i << "\n";
            continue;
          }

          // Get node
          auto selCMNode = dyn_cast<CMergeData>(
              switchInfo.data.dfgBaseNodeValue[selNode].get());
          int initValue = 0;

          //! For all transition related nodes, we need to check the iter 0 as
          //! well
          if (i == 1) {
            initValue = selCMNode->getControlOutput(0);
            switchInfo.staticinfo.dataflowGraph->nodes[selNode]->updateDataoutChannel(
                initValue);
          } else {
            initValue = 0;
          }

          // Step 1: Update the node itself
          int selValue = selCMNode->getControlOutput(i);
          switchInfo.staticinfo.dataflowGraph->nodes[selNode]->updateDataoutChannel(
              selValue);

          if (selValue == 1) {
            switchInfo.staticinfo.dataflowGraph->nodes[selNode]->updateDataoutChannel(0);
          }

          // Step 2: Update all nodes in the control succeeding node list
          //! Testing
          if (debug)
            llvm::dbgs() << "[DEBUG] \t[Control Output]\n";
          
          // Defensive check for mgSucNodeDict access
          if (!contains(selCMNode->mgSucNodeDict,executedSeg)) {
            // llvm::dbgs() << "[DEBUG] Warning: segment " << executedSeg << " not found in mgSucNodeDict for control merge node " << selNode << "\n";
            continue;
          }
          
          for (const auto &selSucNode :
               selCMNode->mgSucNodeDict[executedSeg].control) {
            //! Testing
            if (debug)
              llvm::dbgs() << "[DEBUG] \t\tSuc Node: " << selSucNode << "\n";

            if (i == 1) {
              switchInfo.staticinfo.dataflowGraph->nodes[selSucNode]->updateDataoutChannel(
                  initValue);
            }
            switchInfo.staticinfo.dataflowGraph->nodes[selSucNode]->updateDataoutChannel(
                selValue);

            // Step 2.5: Update all glitch node, if cm_value = 1, then change it
            // back to 0, unless II == 1
            if (selValue == 1) {
              if (std::find(
                      selCMNode->mgSucNodeDict[executedSeg].glitch.begin(),
                      selCMNode->mgSucNodeDict[executedSeg].glitch.end(),
                      selSucNode) !=
                  selCMNode->mgSucNodeDict[executedSeg].glitch.end()) {
                switchInfo.staticinfo.dataflowGraph->nodes[selSucNode]
                    ->updateDataoutChannel(0);
              }
            }
          }

          //! Testing
          if (debug)
            llvm::dbgs() << "[DEBUG] \t[DONE]\n";

          // Step 3: Update all data channel values
          //! Testing
          if (debug)
            llvm::dbgs() << "[DEBUG] \t[Data Output]\n";
          for (const auto &selSucNode :
               selCMNode->mgSucNodeDict[executedSeg].data) {
            //! Testing
            if (debug)
              llvm::dbgs() << "[DEBUG] \t\tSuc Node: " << selSucNode << "\n";

            if (contains(selSucNode, "cond_br")) {
              // Get the node
              auto selCondStoreNode = dyn_cast<CBrNode>(
                  switchInfo.staticinfo.dataflowGraph->nodes[selSucNode].get());

              int tmpCondValue =
                  getCondBrNodeCondValue(switchInfo, i, selSucNode);
              selCondStoreNode->updateDataout(-1, tmpCondValue);
            } else {
              switchInfo.staticinfo.dataflowGraph->nodes[selSucNode]->updateDataoutChannel(
                  -1);
            }
          }

          //! Testing
          if (debug)
            llvm::dbgs() << "[DEBUG] \t[DONE]\n";
        } else if (contains(selNode, "mux")) {
          // Case 3: MUX Node
          // In the last iteration or MG transitions, we take the
          // ori_glitch_data out value
          //! For all transition related nodes, we need to check the iter 0 as
          //! well Testing
          if (debug)
            llvm::dbgs() << "[DEBUG] MUX Detected \n";
          
          // Defensive check for MUX node
          if (!contains(switchInfo.data.dfgBaseNodeValue,selNode)) {
            llvm::dbgs() << "[DEBUG] Warning: MUX node " << selNode << " not found in dfgBaseNodeValue at iteration " << i << "\n";
            continue;
          }
          
          for (const auto &selValue :
               switchInfo.data.dfgBaseNodeValue[selNode]->oriGlitchDataOut[i]) {
            switchInfo.staticinfo.dataflowGraph->nodes[selNode]->updateDataoutChannel(
                selValue);

            // Step 2: Update the glitching succeeding list
            //! Testing
            if (debug)
              llvm::dbgs() << "[DEBUG] \t[Glitch Output]\n";
            
            // Defensive check for segSucNodeMap access
            if (!contains(switchInfo.data.dfgBaseNodeValue[selNode]->segSucNodeMap,executedSeg)) {
              // llvm::dbgs() << "[DEBUG] Warning: segment " << executedSeg << " not found in segSucNodeMap for node " << selNode << "\n";
              continue;
            }
            
            for (const auto &selSucNode :
                 switchInfo.data.dfgBaseNodeValue[selNode]
                     ->segSucNodeMap[executedSeg]
                     .glitch) {
              //! Testing
              if (debug)
                llvm::dbgs() << "[DEBUG] \t\tSuc Node: " << selSucNode << "\n";

              // Get the node output bitwidth
              unsigned tmpNodeWidth = switchInfo.data.dfgBaseNodeValue[selNode]->segSucNodeMap[executedSeg].dataWidthMap[selSucNode];

              if (contains(selSucNode, "cond_br")) {
                // Get the node
                auto selCondStoreNode = dyn_cast<CBrNode>(switchInfo.staticinfo.dataflowGraph->nodes[selSucNode].get());

                int tmpCondValue =
                    getCondBrNodeCondValue(switchInfo, i, selSucNode);
                selCondStoreNode->updateDataout(
                    reduceBits(selValue, tmpNodeWidth), tmpCondValue);
              } else {
                switchInfo.staticinfo.dataflowGraph->nodes[selSucNode]
                    ->updateDataoutChannel(reduceBits(selValue, tmpNodeWidth));
              }
            }
            //! Testing
            if (debug)
              llvm::dbgs() << "[DEBUG] \t[DONE]\n";
          }

          // Step 3: Update the ori succeeding node list
          //! Testing
          if (debug)
            llvm::dbgs() << "[DEBUG] \t[Ori Output]\n";
          for (const auto &selSucNode : switchInfo.data.dfgBaseNodeValue[selNode]
                                            ->segSucNodeMap[executedSeg]
                                            .original) {
            //! Testing
            if (debug)
              llvm::dbgs() << "[DEBUG] \t\tSuc Node: " << selSucNode << "\n";
            // Get bitwidth
            unsigned tmpNodeWidth = switchInfo.data.dfgBaseNodeValue[selNode]
                                        ->segSucNodeMap[executedSeg]
                                        .dataWidthMap[selSucNode];

            //
            if (contains(switchInfo.data.dfgBaseNodeValue[selNode]->originalDataOut,i)) {
              if (contains(selSucNode, "store")) {
                auto selStoreNode = dyn_cast<DStoreNode>(switchInfo.staticinfo.dataflowGraph->nodes[selSucNode].get());
                selStoreNode->updateDataout(reduceBits(switchInfo.data.dfgBaseNodeValue[selNode] ->originalDataOut[i] .value,
                               tmpNodeWidth),
                    selNode);
              } else if (contains(selSucNode, "cond_br")) {
                auto selCondBrNode = dyn_cast<CBrNode>(
                    switchInfo.staticinfo.dataflowGraph->nodes[selSucNode].get());
                int tmpCondValue =
                    getCondBrNodeCondValue(switchInfo, i, selSucNode);

                selCondBrNode->updateDataout(
                    reduceBits(switchInfo.data.dfgBaseNodeValue[selNode]
                                   ->originalDataOut[i]
                                   .value,
                               tmpNodeWidth),
                    tmpCondValue);
              } else {
                switchInfo.staticinfo.dataflowGraph->nodes[selSucNode]
                    ->updateDataoutChannel(
                        reduceBits(switchInfo.data.dfgBaseNodeValue[selNode]
                                       ->originalDataOut[i]
                                       .value,
                                   tmpNodeWidth));
              }
            }
          }
          //! Testing
          if (debug)
            llvm::dbgs() << "[DEBUG] \t[DONE]\n";
        } else if (contains(selNode, "load")) {
          //! Testing
          if (debug)
            llvm::dbgs() << "[DEBUG] LOAD Detected \n";
          // Case 4: load node
          pendingMemList.push_back(selNode);
          continue;
        } else {
          //! Testing
          if (debug)
            llvm::dbgs() << "[DEBUG] Other Nodes Detected\n";
          // All other nodes
          int selValue = 0;
          if (contains(selNode, "source") || contains(selNode, "start")) {
            if (i == switchInfo.data.firstExecutedIter[executedSeg]) {
              selValue = switchInfo.data.dfgBaseNodeValue[selNode]
                             ->originalDataOut[0]
                             .value;
            } else
              continue;
          } else if (contains(selNode, "constant")) {
            selValue = switchInfo.data.dfgBaseNodeValue[selNode]
                           ->originalDataOut[0]
                           .value;
          } else {
            // Defensive check to prevent crash
            if ( ! contains(switchInfo.data.dfgBaseNodeValue,selNode)) {
              llvm::dbgs() << "[DEBUG] Warning: selNode " << selNode << " not found in dfgBaseNodeValue at iteration " << i << "\n";
              selValue = 0; // Default value
            } else if (!contains(switchInfo.data.dfgBaseNodeValue[selNode]->originalDataOut ,i )) {
              llvm::dbgs() << "[DEBUG] Warning: No data found for node " << selNode << " at iteration " << i << "\n";
              selValue = 0; // Default value
            } else {
              selValue = switchInfo.data.dfgBaseNodeValue[selNode]->originalDataOut[i].value;
            }
          }

          // Step 1: update the node itself
          switchInfo.staticinfo.dataflowGraph->nodes[selNode]->updateDataoutChannel(selValue);

          // Step 2: Update all glitching nodes in the succeeding list
          // Defensive check for segSucNodeMap access
          if (contains(switchInfo.data.dfgBaseNodeValue[selNode]->segSucNodeMap,executedSeg)) {
            // llvm::dbgs() << "[DEBUG] Warning: segment " << executedSeg << " not found in segSucNodeMap for other node " << selNode << "\n";
            continue;
          }
          
          for (const auto &selSucNode : switchInfo.data.dfgBaseNodeValue[selNode]
                                            ->segSucNodeMap[executedSeg]
                                            .glitch) {
            if (contains(selSucNode, "cond_br")) {
              auto selCondStoreNode = dyn_cast<CBrNode>(
                  switchInfo.staticinfo.dataflowGraph->nodes[selSucNode].get());
              int tmpCondValue =
                  getCondBrNodeCondValue(switchInfo, i, selSucNode);
              selCondStoreNode->updateDataout(selValue, tmpCondValue);
            } else {
              switchInfo.staticinfo.dataflowGraph->nodes[selSucNode]->updateDataoutChannel(
                  selValue);
            }
          }

          // Step 3: Update all non glitching nodes in the succeeding list
          for (const auto &selSucNode : switchInfo.data.dfgBaseNodeValue[selNode]
                                            ->segSucNodeMap[executedSeg]
                                            .original) {
            switchInfo.staticinfo.dataflowGraph->nodes[selSucNode]->updateDataoutChannel(
                selValue);
          }

          //! Testing
          if (debug)
            llvm::dbgs() << "[DEBUG] [DONE]\n";
        }
      }
    }

    // Case 6: Nodes related to memory accesses
    for (const auto &selNode : pendingMemList) {
      // Defensive check for memory node
      if (!contains(switchInfo.data.dfgBaseNodeValue,selNode)) {
        llvm::dbgs() << "[DEBUG] Warning: Memory node " << selNode << " not found in dfgBaseNodeValue at iteration " << i << "\n";
        continue;
      }
      
      // We need to update the address out and data out separatly
      // Step 1: Get the address source
      auto selMemNode =
          dyn_cast<DLoadNode>(switchInfo.staticinfo.dataflowGraph->nodes[selNode].get());

      std::string tmpAddrPreNode = selMemNode->addressInNodeName;
      std::string tmpAddrPreSrcNode = "";
      if (executedSeg == "E") {
        tmpAddrPreSrcNode =segENodeSrcSearch(switchInfo, tmpAddrPreNode, profileResults);


      } else {
        tmpAddrPreSrcNode = memAddrSrcSearch(switchInfo, profileResults,tmpAddrPreNode, executedSeg, i);

      }

      // We assume there will not be any glitches for the address value
      int addrValue = 0;
      if (contains(tmpAddrPreSrcNode,"constant") ) {
        addrValue = switchInfo.data.dfgBaseNodeValue[tmpAddrPreSrcNode]
                        ->originalDataOut[0]
                        .value;
      } else {
        // TODO: Validate the following assumption
        if (contains(switchInfo.data.dfgBaseNodeValue[tmpAddrPreSrcNode]->originalDataOut,i)) {
          addrValue = switchInfo.data.dfgBaseNodeValue[tmpAddrPreSrcNode]->originalDataOut[i].value;
        } else {
          addrValue = -10;
        }
      }

      int selValue =
          switchInfo.data.dfgBaseNodeValue[selNode]->originalDataOut[i].value;

      // Step 1: Update the node itself
      selMemNode->updateDataout(selValue, addrValue);

      // Step 2: Update teh nodes in glitching list
      // Defensive check for segSucNodeMap access
      if (!contains(switchInfo.data.dfgBaseNodeValue[selNode]->segSucNodeMap,executedSeg)) {
        llvm::dbgs() << "[DEBUG] Warning: segment " << executedSeg << " not found in segSucNodeMap for memory node " << selNode << "\n";
        continue;
      }
      
      for (const auto &selSucNode : switchInfo.data.dfgBaseNodeValue[selNode]->segSucNodeMap[executedSeg].glitch) {
        //! Exclude the memory controller
        if (  (!contains(selSucNode,"mem_controller"))   && (  (!contains(selSucNode,"end"))   ))  {
          // Get node bitwidth
          unsigned tmpNodeWidth = switchInfo.data.dfgBaseNodeValue[selNode]->segSucNodeMap[executedSeg].dataWidthMap[selSucNode];

          if (contains(selSucNode, "cond_br")) {
            // Get the node
            auto selCondStoreNode = dyn_cast<CBrNode>(switchInfo.staticinfo.dataflowGraph->nodes[selSucNode].get());

            int tmpCondValue =getCondBrNodeCondValue(switchInfo, i, selSucNode);
            selCondStoreNode->updateDataout(reduceBits(selValue, tmpNodeWidth),
                                            tmpCondValue);
          } else {
            switchInfo.staticinfo.dataflowGraph->nodes[selSucNode]->updateDataoutChannel(
                reduceBits(selValue, tmpNodeWidth));
          }
        }
      }

      // Step 3: Update nodes in non glitching list
      for (const auto &selSucNode : switchInfo.data.dfgBaseNodeValue[selNode]->segSucNodeMap[executedSeg].original) {
        // Get bitwidth
        unsigned tmpNodeWidth = switchInfo.data.dfgBaseNodeValue[selNode]->segSucNodeMap[executedSeg].dataWidthMap[selSucNode];

        //
        if (contains(selSucNode, "cond_br")) {
          // Get the node
          auto selCondStoreNode = dyn_cast<CBrNode>(
              switchInfo.staticinfo.dataflowGraph->nodes[selSucNode].get());

          int tmpCondValue = getCondBrNodeCondValue(switchInfo, i, selSucNode);
          selCondStoreNode->updateDataout(reduceBits(selValue, tmpNodeWidth),
                                          tmpCondValue);
        } else {
          switchInfo.staticinfo.dataflowGraph->nodes[selSucNode]->updateDataoutChannel(
              reduceBits(selValue, tmpNodeWidth));
        }
      }
    }
  }

  // Final step, change the results of buffers directly connected to cond_br
  // node for (auto& [selCondNode, bufferList] :
  // switchInfo.staticinfo.dataflowGraph->condBrToBufferMap) {
  //   // Get the node
  //   auto selCondStoringNode =
  //   dyn_cast<CBrNode>(switchInfo.staticinfo.dataflowGraph->nodes[selCondNode].get());
  //   for (auto& [selBuffer, portIdx]: bufferList) {
  //     // Change the resutls of the buffer node
  //     for (auto& [selIdx, valueVec]:
  //     switchInfo.staticinfo.dataflowGraph->nodes[selBuffer]->dataOut) {
  //       valueVec = selCondStoringNode->per_channel_dataout[portIdx];
  //     }
  //   }
  // }
}

// Masks 'value' to 'targetBitWidth' bits.
// E.g. reduceBits(0xABCD, 8) => 0xCD
int reduceBits(int value, unsigned targetBitWidth) {
  // If the target bit width is >= 32, we leave 'value' unchanged
  // because an int on most platforms is 32 bits, so no further masking is
  // needed.
  if (targetBitWidth >= 32) {
    return value;
  }

  // Build the mask. E.g. if targetBitWidth=8 => mask=0xFF
  // 1U << targetBitWidth is well-defined for 0 <= targetBitWidth <= 31
  // (shifting by 32 is undefined in 32-bit, hence the check above).
  unsigned mask = (1U << targetBitWidth) - 1U;

  // Apply the mask; cast to int to keep the function's return type int.
  return static_cast<int>(static_cast<unsigned>(value) & mask);
}



int getCondBrNodeCondValue(SwitchingInfo &switchInfo, unsigned iterIndex, std::string nodeName) {
  // Defensive check for condBrToConSrcMap
  if (!contains(switchInfo.staticinfo.dataflowGraph->condBrToConSrcMap,nodeName)) {
    llvm::dbgs() << "[DEBUG] Warning: cond_br node " << nodeName << " not found in condBrToConSrcMap\n";
    return 1; // Default value
  }

  std::string condSrcNode = switchInfo.staticinfo.dataflowGraph->condBrToConSrcMap[nodeName];
  int condValue = 1;

  // Defensive check for dfgBaseNodeValue
  if (!contains(switchInfo.data.dfgBaseNodeValue,condSrcNode) ) {
    llvm::dbgs() << "[DEBUG] Warning: condSrcNode " << condSrcNode << " not found in dfgBaseNodeValue\n";
    return 1; // Default value
  }

  if (contains(switchInfo.data.dfgBaseNodeValue[condSrcNode]->originalDataOut,iterIndex)  ) {
  condValue = switchInfo.data.dfgBaseNodeValue[condSrcNode]->originalDataOut[iterIndex].value;
  } else {
  // Note: This line seems incorrect - it's assigning to condSrcNode instead of condValue
  // condSrcNode = 1;  // This was wrong
  condValue = 1;  // Fixed: assign to condValue instead
  }

  return condValue;
  }





  static std::optional<std::string>
  findSrcInSegment(const SwitchingInfo &si,
                   const std::string& nodeName,
                   const std::string& segLabel) {// TODO use string veiew
    static constexpr std::array<llvm::StringRef,4> cats{
      "control","data","glitch","original"
    };
  
    // 1) Check if segment exists
    if (!contains(si.data.segToDataBaseVec, segLabel))
      return std::nullopt;
  
    // iterate all mapped roots
    for (auto const &mapped : si.data.segToDataBaseVec.at(segLabel).all) {
      // 2) direct name‐match
      if (contains(nodeName, mapped))
        return mapped;
  
      // 3) Check if DFG node exists
      if (!contains(si.data.dfgBaseNodeValue, mapped))
        continue;
      auto const &nodeVal = si.data.dfgBaseNodeValue.at(mapped);
  
      // 4) Check if segSuc entry exists
      if (!contains(nodeVal->segSucNodeMap, segLabel))
        continue;
      auto const &mSuc = nodeVal->segSucNodeMap.at(segLabel);
  
      // 5) Search in successor lists
      for (auto cat : cats) {
        auto const &lst = (cat=="control" ? mSuc.control 
                       : cat=="data"    ? mSuc.data
                       : cat=="glitch"  ? mSuc.glitch
                                       : mSuc.original);
        if (std::any_of(lst.begin(), lst.end(),
              [&](auto const &s){ return s.find(nodeName) != std::string::npos; }))  // Fixed: std::string::npos
          return mapped;
      }
    }
    return std::nullopt;
  }
  
  
  // Now both functions collapse to:
  std::string segENodeSrcSearch(SwitchingInfo &si,
                                const std::string& nodeName,
                                SCFProfilingResult &profile) {
    if (auto src = findSrcInSegment(si, nodeName, "E"))
      return *src;
    // fallback to the penultimate trace entry
  
    if( profile.executedSegTrace.size()>=2){
    auto prev = profile.executedSegTrace[
                  profile.executedSegTrace.size()-2];
    if (auto src = findSrcInSegment(si, nodeName, prev))
      return *src;
    }
    return "";
  }
  
  std::string memAddrSrcSearch(SwitchingInfo &si,
                               SCFProfilingResult &profile,
                               const std::string&  nodeName,
                               const std::string&  selSeg,
                               unsigned iterIdx) {
    if (auto src = findSrcInSegment(si, nodeName, selSeg))
      return *src;
    // fallback to the previous segment in the trace
    if (iterIdx == 0 || iterIdx > profile.executedSegTrace.size()-1)
      return "";
    auto prev = profile.executedSegTrace[iterIdx-1];
    if (auto src = findSrcInSegment(si, nodeName, prev))
      return *src;
    return "";
  }
  