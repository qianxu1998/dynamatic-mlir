#include "experimental/Transforms/Switching/HandShakeBufferAnalysis.h"
#include "experimental/Transforms/Switching/DFSKernel.h"
#include "experimental/Transforms/Switching/DataGlitches.h"

void updateMGBufferSwitching(SwitchingInfo &switchInfo, std::string selMG,
                       bool debug) {
  // Get the corresponding graph
  auto selAdjGraph = switchInfo.staticinfo.segToGraph[selMG];

  // Get needed information
  double_t selMgThroughput = switchInfo.staticinfo.cfdfcThroughput[std::stoi(selMG)];
  // TODO: Validate the following rounding
  float rawII = 1.0f / selMgThroughput;
  unsigned selMGII = static_cast<unsigned>(std::round(rawII));
  std::string baseNode = selAdjGraph->baseNode;

  //! Testing
  if (debug) {
    llvm::dbgs() << "[MG " << selMG << "]\n";
    llvm::dbgs() << "[MG INFO] Throughput: " << selMgThroughput << "\n";
    llvm::dbgs() << "[MG INFO] II: " << selMGII << "\n";
    llvm::dbgs() << "[MG INFO] Base Node: " << baseNode << "\n";
  }

  // Create the "universe" set: set of all cycle indices from 0..selCfdfcII-1
  IISet selMGUniverseSet(selMGII, true /* all 1's*/);
  // Analyze timing info for each of the buffers
  for (const auto &selBuffName : selAdjGraph->orderedNodeName) {
    if (contains(selBuffName, "buffer")) {
      if (debug) {
        llvm::dbgs() << "\t[" << selBuffName << "]\n";
      }

      // Get the buffer node
      auto selBuffNode =
          dyn_cast<BufferNode>(selAdjGraph->nodes[selBuffName].get());

      // Get the longest path
      LongestPathResult tmpLongPath =selLongestPath(switchInfo, selBuffName, selMG);

      if (debug) {
        llvm::dbgs() << "\t\tStarting node: " << tmpLongPath.selStartNode
                     << "\n";
        llvm::dbgs() << "\t\tPath Latency: " << tmpLongPath.maxLatency << "\n";
        llvm::dbgs() << "\t\tPath Last second buffer: "
                     << tmpLongPath.lastSecondBuffer << "\n";
        llvm::dbgs()
            << "\t\tStart Node Shift: "
            << selAdjGraph->startBaseNodeShiftMap[tmpLongPath.selStartNode]
            << "\n\n";
      }

      // Step 1: Calculate D. Get the steady state buffer starting point
      // fin_start_point = (path_latency % sel_cfdfc_II)
      //                 + cfdfc_tim_start_nodes[cfdfcIndex][pathStartNode]
      //                 + cfdfc_tim_start_nodes[cfdfcIndex][baseNode]
      unsigned pathLatencyMod = tmpLongPath.maxLatency % selMGII;
      unsigned finStartPoint =
          pathLatencyMod +
          static_cast<unsigned>(
              selAdjGraph->startBaseNodeShiftMap[tmpLongPath.selStartNode]) +
          static_cast<unsigned>(selAdjGraph->startBaseNodeShiftMap[baseNode]);

      if (debug) {
        llvm::dbgs() << "\t\tpathLatencyMod: " << pathLatencyMod << "\n";
        llvm::dbgs() << "\t\tD : " << finStartPoint
                     << " regarding the start of " << baseNode << "\n";
      }

      selBuffNode->START = finStartPoint;

      // Step 2: Calculate SET_V
      float_t selBuffOcc = selBuffNode->occupancy;
      unsigned selBufSlots = selBuffNode->numSlots;
      bool selBuffTransparent = selBuffNode->transparent;

      // Store occupancy of direct preceding buffer, if exist
      float_t tmpPreBuffOcc = 0;
      float_t tmpNumCycles = 0;

      // Handle Transparent buffers
      if (selBuffTransparent) {
        if (tmpLongPath.lastSecondBuffer != "") {
          auto selLastSecondBuff = dyn_cast<BufferNode>(
              selAdjGraph->nodes[tmpLongPath.lastSecondBuffer].get());
          tmpPreBuffOcc = selLastSecondBuff->occupancy;
        }

        tmpNumCycles =
            ((tmpPreBuffOcc + selBuffOcc) / selMgThroughput) -
            selAdjGraph->startBaseNodeShiftMap[tmpLongPath.selStartNode];
      } else {
        tmpNumCycles =
            (selBuffOcc / selMgThroughput) -
            selAdjGraph->startBaseNodeShiftMap[tmpLongPath.selStartNode];
      }

      IISet tmpBufferValidSet(selMGII, false /*all 0's*/);
      if(tmpNumCycles<0.0f) tmpNumCycles=0.0f;
      for (unsigned i = 0; i < getUnsigned(tmpNumCycles); ++i) {
        unsigned relativeActiveCycle =
            static_cast<unsigned>(finStartPoint + i) % selMGII;
        tmpBufferValidSet.set(relativeActiveCycle);
      }

      //! Testing
      if (debug) {
        llvm::dbgs() << "\t\tSET_V: { ";
        for (auto x : tmpBufferValidSet.set_bits())
          llvm::dbgs() << x << " ";
        llvm::dbgs() << "}\n";
      }
      selBuffNode->calValidSet(selBuffNode->sucs[0], tmpBufferValidSet);

      // Step 3: Calculate SET_R
      IISet tmpBufferReadySet(selMGII, false /*all bits are 0*/);
      // Handle cascaded buffers (opaque buffer -> transparent buffer)
      // TODO: Validate the following operations through simulation
      if (selBuffTransparent) {
        // Check the direct predecessor
        std::string predName = selBuffNode->pres[0];
        if (contains(predName, "buffer")) {
          auto selPredBufferNode =
              dyn_cast<BufferNode>(selAdjGraph->nodes[predName].get());
          // The direct predecessor is an opaque buffer
          if (!selPredBufferNode->transparent) {
            selBuffNode->occupancy -= selPredBufferNode->occupancy;
            selBuffOcc = selBuffNode->occupancy;
          }
        }
      }

      if (selBuffTransparent && (selBufSlots == 1) && (selBuffOcc == 0.0)) {
        // TODO: Check why it's different when the transparent just have one
        // slot Special case for transparent buffer
        tmpBufferReadySet = IISet(selMGII, true /* all bits 1 */);

      } else if (!selBuffTransparent &&
                 (selBufSlots - selBuffOcc >= (1 - selMgThroughput))) {
        // The corresponding buffer is always ready
        tmpBufferReadySet = IISet(selMGII, true /* all bits 1 */);
      } else {
        // The corresponding buffer is not always ready
        float_t tmpTransparentOffset = 0;
        if (selBuffTransparent) {
          // Transparent buffer
          // Check the existence fo the last second buffer
          if (tmpLongPath.lastSecondBuffer != "") {
            auto selLastSecondBuff = dyn_cast<BufferNode>(
                selAdjGraph->nodes[tmpLongPath.lastSecondBuffer].get());
            tmpTransparentOffset =
                selLastSecondBuff->occupancy / selMgThroughput;
          } else {
            // TODO: Validate the following assumption
            llvm::dbgs() << "[WARNING] Transparent Buffer " << selBuffName
                         << " has no preceding opaque buffer!\n";
            tmpTransparentOffset = 0;
          }
        }

        //TODO Changed here from unisgned to int
        int tmpMissingCycles = 0;
        if (selBuffTransparent) {
          if (selBufSlots == 1) {
            tmpMissingCycles = static_cast<int>(
                ((1 - selMgThroughput) -
                 (selBufSlots - selBuffOcc - tmpPreBuffOcc)) /
                selMgThroughput);
          } else {
            tmpMissingCycles = static_cast<int>(
                ((1 - selMgThroughput) - (selBufSlots - selBuffOcc)) /
                selMgThroughput);
          }
        } else {
          // Opaque buffer
          tmpMissingCycles = static_cast<int>(
              ((1 - selMgThroughput) - (selBufSlots - selBuffOcc)) /
              selMgThroughput);
        }

        //
        IISet offsetSet(selMGII, false /*all bits are 0*/);
        //TODO Changed here from unisgned to int and guarded
        llvm::dbgs()<<" Missed Cycles temp "<<tmpMissingCycles<<" \n";
        if(tmpMissingCycles<0) tmpMissingCycles=0;
        for (unsigned i = 0; i < tmpMissingCycles; i++) {
          // TODO: Validate the following rounding
          unsigned relativeActiveCycle =
              (finStartPoint + i +
               static_cast<unsigned>(tmpTransparentOffset)) %
              selMGII;
          offsetSet.set(relativeActiveCycle);
        }
        
        // Now the buffer is NOT ready in offsetSet, so we want
        // tmpBufferReadySet = Universe - offsetSet
        // i.e. difference: in bitwise operation it translates to   Universe &
        // ~offsetSet We'll start with entire universe, then remove each element
        // from offsetSet

        tmpBufferReadySet = selMGUniverseSet;
        tmpBufferReadySet &= (~offsetSet);
      }

      //! Testing
      if (debug) {
        llvm::dbgs() << "\t\tSET_R: { ";
        for (auto x : tmpBufferReadySet.set_bits())
          llvm::dbgs() << x << " ";
        llvm::dbgs() << "}\n";
      }
      selBuffNode->calReadySet(selBuffNode->pres[0], tmpBufferReadySet);

      // Store the other data
      if (debug) {
        llvm::dbgs() << "\t\tOccupancy: " << selBuffOcc << "\n";
        llvm::dbgs() << "\t\tNum Slots: " << selBufSlots << "\n";
        llvm::dbgs() << "\t\tTransparent: " << selBuffTransparent << "\n";
      }

      // Calculate switching
      selBuffNode->calValidSwitching(selBuffNode->sucs[0], selMGII);
      selBuffNode->calReadySwitching(selBuffNode->pres[0], selMGII);

      if (debug) {
        selBuffNode->printNodeDetails();
      }

      // Update the buffer status
      selBuffNode->totalHandshakeSwitchingUpdate();
    }
  }
}
