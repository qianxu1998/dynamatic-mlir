//===- HandShakeChannelCal.cpp - Estimate Swithicng Activities ------*- C++
//-*-===//
//
// This file declares all functions used for data channel switching calculation
//
//===----------------------------------------------------------------------===//

#include "experimental/Transforms/Switching/HandShakeChannelCal.h"
#include "experimental/Transforms/Switching/DFSKernel.h"
#include "experimental/Transforms/Switching/DataGlitches.h"

#include "dynamatic/Dialect/Handshake/HandshakeAttributes.h"
#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Support/Attribute.h"
#include "dynamatic/Support/Backedge.h"
#include "dynamatic/Support/CFG.h"
#include "dynamatic/Support/DynamaticPass.h"
#include "dynamatic/Support/LLVM.h"
#include "dynamatic/Support/Logging.h"
#include "dynamatic/Support/TimingModels.h"


#include "experimental/Transforms/Switching/utils.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/Support/Path.h"
// #include "llvm/Support/Debug.h"
//  #include <unordered_map>
#include "llvm/ADT/DenseMap.h"
using IISet = llvm::SmallBitVector;


void mgHandshakeSwitchingCounting(SwitchingInfo &switchInfo, HandshakeInfo& hs, std::string selMG,
                                  bool debug) {
  // Get the corresponding graph
  auto selAdjGraph = switchInfo.staticinfo.segToGraph[selMG];
  double_t selMgThroughput = switchInfo.staticinfo.cfdfcThroughput[std::stoi(selMG)];
  // TODO: Validate the following rounding
  float rawII = 1.0f / selMgThroughput;
  unsigned selMGII = static_cast<unsigned>(std::round(rawII));

  // Create a list of pending update nodes
  std::vector<std::string> pendingList;
  std::vector<std::string> bufferList;
  for (const auto &selNode : selAdjGraph->orderedNodeName) {
    if (selNode.find("buffer") == std::string::npos)
      pendingList.push_back(selNode);
    else
      bufferList.push_back(selNode);
  }

  // Get the influenced load nodes list
  auto selInfluencedLoadUnits = getLoadsInfluencedByBuffers(switchInfo, selMG, bufferList);
      hs.mgInfluencedLoadUnits.push_back(selInfluencedLoadUnits);

  // Status
  unsigned numIter = 0;
  unsigned deadlockCounter = 0;
  unsigned listLength = pendingList.size();

  while (pendingList.size() > 0) {
    // List of finished nodes in this iteration
    std::vector<std::string> tmpFinished;

    // Status
    numIter++;
    if (debug) {
      llvm::dbgs() << "Iter: " << numIter << "\n";
      llvm::dbgs() << "Pending: \n\t[";
      for (const auto &selNode : pendingList)
        llvm::dbgs() << selNode << ", ";
      llvm::dbgs() << "]\n";
    }

    // Calculate switching
    for (auto &selNode : pendingList) {
      //
      nodeHandshakeUpdate(switchInfo, hs,selNode, selMG, selMGII, debug = false);

      // If the node finished
      if (selAdjGraph->nodes[selNode]->handshakeUpdateFinished()) {
        tmpFinished.push_back(selNode);

        // Update total switching for finished node
        selAdjGraph->nodes[selNode]->totalHandshakeSwitchingUpdate();

        if (debug) {
          llvm::dbgs() << "Finished node: " << selNode << "\n";
          selAdjGraph->nodes[selNode]->printHandshakeSwitching();
        }
      }
    }

    // remove the finished from pending
    for (auto &fn : tmpFinished) {
      auto it = std::find(pendingList.begin(), pendingList.end(), fn);
      if (it != pendingList.end()) {
        pendingList.erase(it);
      }
    }

    // Check the nodes in the pending list
    if (pendingList.size() == listLength) {
      if (debug) {
        llvm::dbgs() << "[WARNING] No Node Gets Updated\n";
      }
      if (deadlockCounter > selMGII+1) {
        if (debug) {
          llvm::dbgs() << "[Ending] # Iter: " << numIter << "\n";
        }

        for (auto &pendingNode : pendingList) {
          if (!selAdjGraph->nodes[pendingNode]->handshakeSwitchingChecking()) {
            llvm::dbgs() << "[Unsolved] " << pendingNode << "\n";
          } else {
            selAdjGraph->nodes[pendingNode]->totalHandshakeSwitchingUpdate();
          }
        }

        // TODO: Check the following breaking rule
        break;
      }

      // Update all Join type node
      std::vector<std::string> tmpFinishedNode =
          breakHandshakeUpdateDeadlock(switchInfo, pendingList, selMG, selMGII);

      for (auto &fn2 : tmpFinishedNode) {
        auto it2 = std::find(pendingList.begin(), pendingList.end(), fn2);
        if (it2 != pendingList.end()) {
          pendingList.erase(it2);
        }
      }

      deadlockCounter++;
    }

    // Update the list length
    listLength = pendingList.size();
  }
  // After the iterative update, make sure all nodes have their total handshake
  // switching tallied. Some nodes might not have reached the "finished" state
  // (e.g., constants/sources with missing sets) but they still have valid/ready
  // counts registered in validSignal/readySignal. We therefore run a final
  // pass to accumulate per-node totals.
  for (const auto &entry : selAdjGraph->nodes) {
    entry.second->totalHandshakeSwitchingUpdate();
  }

  //! Testing
  if(debug){
  llvm::dbgs() << "[DEBUG] [HANDSHAKE CHANNEL SWITCHING SUMMARY]\n";
  for (const auto &selNode : selAdjGraph->nodes) {
    llvm::dbgs() << "[DEBUG] \t=================================\n";
    llvm::dbgs() << "[DEBUG] \tNode Name: " << selNode.first() << "\n";
    selNode.second->printNodeDetails();
  }
}
}

void nodeHandshakeUpdate(SwitchingInfo &switchInfo, HandshakeInfo& hs, std::string &selNode, std::string &selMG, unsigned selMGII, bool debug) {
  // Get the corresponding storing structure
  auto selAdjGraph = switchInfo.staticinfo.segToGraph[selMG];
  auto selNodeStoringDict = selAdjGraph->nodes;

  // Get all needed information if available
  std::vector<int> tmpPValidList;
  llvm::DenseMap<StringRef, int> tmpPValidDict;
  std::vector<IISet *> tmpPValidSetList;
  llvm::DenseMap<StringRef, IISet *> tmpPValidSetDict;

  std::vector<int> tmpNReadyList;
  llvm::DenseMap<StringRef, int> tmpNReadyDict;

  std::vector<IISet *> tmpNReadySetList;
  llvm::DenseMap<StringRef, IISet *> tmpNReadySetDict;

  // Get all needed pValid signal info
  for (const auto &selPre : selNodeStoringDict[selNode]->pres) {
    // Switching numbers
    if (selNodeStoringDict[selPre]->validSignal.find(selNode) !=
        selNodeStoringDict[selPre]->validSignal.end()) {
      // The pValid value exists!
      tmpPValidList.push_back(selNodeStoringDict[selPre]->validSignal[selNode]);
      tmpPValidDict[selPre] = selNodeStoringDict[selPre]->validSignal[selNode];
    } else {
      // The value is not ready yet
      tmpPValidList.push_back(-1);
      tmpPValidDict[selPre] = -1;
    }

    // Active Range
    if (selNodeStoringDict[selPre]->setV.find(selNode) !=
        selNodeStoringDict[selPre]->setV.end()) {
      // The active range info is available
      tmpPValidSetList.push_back(&(selNodeStoringDict[selPre]->setV[selNode]));
      tmpPValidSetDict[selPre] = &(selNodeStoringDict[selPre]->setV[selNode]);
    } else {
      // The set information is not available yet
      tmpPValidSetList.push_back(nullptr);
      tmpPValidSetDict[selPre] = nullptr;
    }
  }

  // Get all needed nReady signal info
  for (const auto &selSuc : selNodeStoringDict[selNode]->sucs) {
    // Switching numbers
    if (selNodeStoringDict[selSuc]->readySignal.find(selNode) !=
        selNodeStoringDict[selSuc]->readySignal.end()) {
      // The nReady signal exists
      tmpNReadyList.push_back(selNodeStoringDict[selSuc]->readySignal[selNode]);
      tmpNReadyDict[selSuc] = selNodeStoringDict[selSuc]->readySignal[selNode];
    } else {
      // The value is not ready yet
      tmpNReadyList.push_back(-1);
      tmpNReadyDict[selSuc] = -1;
    }

    // Active Range
    if (selNodeStoringDict[selSuc]->setR.find(selNode) !=
        selNodeStoringDict[selSuc]->setR.end()) {
      // The active range info is available
      tmpNReadySetList.push_back(&(selNodeStoringDict[selSuc]->setR[selNode]));
      tmpNReadySetDict[selSuc] = &(selNodeStoringDict[selSuc]->setR[selNode]);
    } else {
      // The set information is not available yet
      tmpNReadySetList.push_back(nullptr);
      tmpNReadySetDict[selSuc] = nullptr;
    }
  }

  // Get node steady state start time
  unsigned tmpNodeSSStart =
      static_cast<unsigned>(mgGetNodeStartingPoint(switchInfo, selNode, selMG));

  // Get node steady state start time
  //! Testing
  if (debug) {
    llvm::dbgs() << "============================= \n";
    llvm::dbgs() << "CURRENT NODE: " << selNode << "\n";
    llvm::dbgs() << "\tpValid List: ";
    for (int v : tmpPValidList)
      llvm::dbgs() << v << " ";
    llvm::dbgs() << "\n";
    llvm::dbgs() << "\tpValid Dict: ";
    for (const auto &p : tmpPValidDict)
      llvm::dbgs() << "{" << p.first << ": " << p.second << "} ";
    llvm::dbgs() << "\n";
    llvm::dbgs() << "\tpValid Set List: ";
    for (auto ptr : tmpPValidSetList) {
      if (ptr) {
        llvm::dbgs() << "{";
        for (int x : ptr->set_bits())
          llvm::dbgs() << x << ",";
        llvm::dbgs() << "} ";
      } else {
        llvm::dbgs() << "None ";
      }
    }
    llvm::dbgs() << "\n";
    llvm::dbgs() << "\tpValid Set Dict: ";
    for (const auto &p : tmpPValidSetDict) {
      llvm::dbgs() << "{" << p.first << ": ";
      if (p.second) {
        llvm::dbgs() << "{";
        for (int x : p.second->set_bits())
          llvm::dbgs() << x << ",";
        llvm::dbgs() << "}";
      } else {
        llvm::dbgs() << "None";
      }
      llvm::dbgs() << "} ";
    }
    llvm::dbgs() << "\n";
    llvm::dbgs() << "\tnReady List: ";
    for (int v : tmpNReadyList)
      llvm::dbgs() << v << " ";
    llvm::dbgs() << "\n";
    llvm::dbgs() << "\tnReady Dict: ";
    for (const auto &p : tmpNReadyDict)
      llvm::dbgs() << "{" << p.first << ": " << p.second << "} ";
    llvm::dbgs() << "\n";
    llvm::dbgs() << "\tnReady Set List: ";
    for (auto ptr : tmpNReadySetList) {
      if (ptr) {
        llvm::dbgs() << "{";
        for (int x : ptr->set_bits())
          llvm::dbgs() << x << ",";
        llvm::dbgs() << "} ";
      } else {
        llvm::dbgs() << "None ";
      }
    }
    llvm::dbgs() << "\n";
    llvm::dbgs() << "\tnReady Set Dict: ";
    for (const auto &p : tmpNReadySetDict) {
      llvm::dbgs() << "{" << p.first << ": ";
      if (p.second) {
        llvm::dbgs() << "{";
        for (int x : p.second->set_bits())
          llvm::dbgs() << x << ",";
        llvm::dbgs() << "}";
      } else {
        llvm::dbgs() << "None";
      }
      llvm::dbgs() << "} ";
    }
    llvm::dbgs() << "\n";
    llvm::dbgs() << "\tSteady State Start Cycle: " << tmpNodeSSStart << "\n";
  }


  // Early exit for  nodes withou successors): setting predecessors as always-ready (as in python)
  auto nodePtr = selNodeStoringDict[selNode];
  if (nodePtr->sucs.empty()) {
    if (debug)
      llvm::dbgs() << "[DEBUG] Leaf node `" << selNode << "`: no successors, marking all ready early.\n";
    for (const auto &pre : nodePtr->pres) {
      setReady(nodePtr.get(), pre, 0);
      setRSet(nodePtr.get(), pre, IISet(selMGII, true));

    }
    return;
  }

  // Calculate handshake switching for different types of node
  auto node = selNodeStoringDict[selNode];

  // helper to fetch an IISet from a dict or return all-false since we switched
  // from set  to bitvector with references
  auto getSet = [&](const llvm::DenseMap<StringRef, IISet *> &dict,
                    const std::string &name) {
    auto it = dict.find(name);
    return (it != dict.end() && it->second) ? *it->second
                                            : IISet(selMGII, false);
  };

  // Use llvm::TypeSwitch to branch by node type.
  llvm::TypeSwitch<AdjNode *, void>(node.get())
      .Case<CmpiNode>([&](CmpiNode *cmpi) {
        // CMPI NODE
        //! Testing
        if (debug) llvm::dbgs() << "[CMPI NODE] \n";
        std::string outName = cmpi->sucs[0];
        // Valid Signal
        for (const auto &selSuc : cmpi->sucs) {
          // Calculate the number of switching
          cmpi->calValidSwitching(selSuc, tmpPValidList[0], tmpPValidList[1],
                                  selMGII);
          // Calculate active range
          cmpi->calValidSet(selSuc, tmpNodeSSStart, selMGII);
        }

        // Ready signal
        for (const auto &selPre : cmpi->pres) {
          // Get the node name of the other input port
          std::string other;
          for (const auto &tmpNode : cmpi->pres)
            if (tmpNode != selPre)
              other = tmpNode;

          IISet validSet = getSet(tmpPValidSetDict, other);
          IISet readySet = getSet(tmpNReadySetDict, outName);

          cmpi->calReadySwitching(selPre, tmpPValidDict[other],
                                  tmpNReadyList[0]);
          cmpi->calReadySet(selPre, validSet, readySet, tmpNodeSSStart,
                            selMGII);
        }
      })
      .Case<AddiNode>([&](AddiNode *addi) {
        // ADDI NODE
        //! Testing
        if (debug)
          llvm::dbgs() << "[ADDI NODE] \n";
        std::string outName = addi->sucs[0];
        // Valid Signal
        for (const auto &selSuc : addi->sucs) {
          // Calculate the number of switching
          addi->calValidSwitching(selSuc, tmpPValidList[0], tmpPValidList[1],
                                  selMGII);
          // Calculate active range
          addi->calValidSet(selSuc, tmpNodeSSStart, selMGII);
        }

        // Ready Signals
        for (const auto &selPre : addi->pres) {
          // Get the node name of the other input port
          std::string other;
          for (const auto &tmpNode : addi->pres)
            if (tmpNode != selPre)
              other = tmpNode;

          IISet validSet = getSet(tmpPValidSetDict, other);
          IISet readySet = getSet(tmpNReadySetDict, outName);

          addi->calReadySwitching(selPre, tmpPValidDict[other],
                                  tmpNReadyList[0]);
          addi->calReadySet(selPre, validSet, readySet, tmpNodeSSStart,
                            selMGII);
        }
      })
      .Case<OriNode>([&](OriNode *ori) {
        // ORI NODE
        //! Testing
        if (debug)
          llvm::dbgs() << "[ORI NODE] \n";
        std::string outName = ori->sucs[0];
        // Valid Signal
        for (const auto &selSuc : ori->sucs) {
          // Calculate the number of switching
          ori->calValidSwitching(selSuc, tmpPValidList[0], tmpPValidList[1],
                                 selMGII);
          // Calculate active range
          ori->calValidSet(selSuc, tmpNodeSSStart, selMGII);
        }

        // Ready Signals
        for (const auto &selPre : ori->pres) {
          // Get the node name of the other input port
          std::string other;
          for (const auto &tmpNode : ori->pres)
            if (tmpNode != selPre)
              other = tmpNode;

          IISet validSet = getSet(tmpPValidSetDict, other);
          IISet readySet = getSet(tmpNReadySetDict, outName);

          ori->calReadySwitching(selPre, tmpPValidDict[other],
                                 tmpNReadyList[0]);
          ori->calReadySet(selPre, validSet, readySet, tmpNodeSSStart, selMGII);
        }
      })
      .Case<AndiNode>([&](AndiNode *andi) {
        // ANDI NODE
        //! Testing
        if (debug)
          llvm::dbgs() << "[ANDI NODE] \n";
        std::string outName = andi->sucs[0];
        // Valid Signal
        for (const auto &selSuc : andi->sucs) {
          // Calculate the number of switching
          andi->calValidSwitching(selSuc, tmpPValidList[0], tmpPValidList[1],
                                  selMGII);
          // Calculate active range
          andi->calValidSet(selSuc, tmpNodeSSStart, selMGII);
        }

        // Ready Signals
        for (const auto &selPre : andi->pres) {
          // Get the node name of the other input port
          std::string other;
          for (const auto &tmpNode : andi->pres)
            if (tmpNode != selPre)
              other = tmpNode;

          IISet validSet = getSet(tmpPValidSetDict, other);
          IISet readySet = getSet(tmpNReadySetDict, outName);

          andi->calReadySwitching(selPre, tmpPValidDict[other],
                                  tmpNReadyList[0]);
          andi->calReadySet(selPre, validSet, readySet, tmpNodeSSStart,
                            selMGII);
        }
      })
      .Case<SubiNode>([&](SubiNode *subi) {
        // SUBI NODE
        //! Testing
        if (debug)
          llvm::dbgs() << "[SUBI NODE] \n";
        // Determine output successor name for ready-set lookup
        std::string outName = subi->sucs[0];
        // Valid Signal
        for (const auto &selSuc : subi->sucs) {
          // Calculate the number of switching
          subi->calValidSwitching(selSuc, tmpPValidList[0], tmpPValidList[1],
                                  selMGII);
          // Calculate active range
          subi->calValidSet(selSuc, tmpNodeSSStart, selMGII);
        }

        // Ready Signals
        for (const auto &selPre : subi->pres) {
          // Determine the other input port
          std::string other;
          for (const auto &tmpNode : subi->pres)
            if (tmpNode != selPre)
              other = tmpNode;

          IISet validSet = getSet(tmpPValidSetDict, other);
          IISet readySet = getSet(tmpNReadySetDict, outName);

          subi->calReadySwitching(selPre, tmpPValidDict[other],
                                  tmpNReadyList[0]);
          subi->calReadySet(selPre, validSet, readySet, tmpNodeSSStart,
                            selMGII);
        }
      })
      .Case<MuliNode>([&](MuliNode *muli) {
        // MULI NODE
        //! Testing
        if (debug)
          llvm::dbgs() << "[MULI NODE] \n";
        // Determine output successor name for ready-set lookup
        std::string outName = muli->sucs[0];
        // Valid Signal
        for (const auto &selSuc : muli->sucs) {
          // Calculate the number of switching
          muli->calValidSwitching(selSuc, tmpPValidList[0], tmpPValidList[1],
                                  selMGII);
          // Calculate active range
          muli->calValidSet(selSuc, tmpNodeSSStart, selMGII);
        }

        // Ready Signals
        for (const auto &selPre : muli->pres) {
          // Get the node name of the other input port
          std::string other;
          for (const auto &tmpNode : muli->pres)
            if (tmpNode != selPre)
              other = tmpNode;

          IISet validSet = getSet(tmpPValidSetDict, other);
          IISet readySet = getSet(tmpNReadySetDict, outName);

          muli->calReadySwitching(selPre, tmpPValidDict[other],
                                  tmpNReadyList[0]);
          muli->calReadySet(selPre, validSet, readySet, tmpNodeSSStart,
                            selMGII);
        }
      })
      .Case<ExtsiNode>([&](ExtsiNode *ext) {
        // EXTSI NODE
        //! Testing
        if (debug)
          llvm::dbgs() << "[EXTSI NODE] \n";
        std::string outName = ext->sucs[0];
        // Valid Signal
        for (const auto &selSuc : ext->sucs) {
          // Number of switching
          ext->calValidSwitching(selSuc, tmpPValidList[0]);
          // Active Range
          ext->calValidSet(selSuc, tmpNodeSSStart, selMGII);
        }

        // Ready Signal
        for (const auto &selPre : ext->pres) {
          if (tmpNReadyList.empty()) {
            unsigned numReadySwitches = 0;
            ext->setReadySwitching(selPre, numReadySwitches);
            IISet tmpSet(selMGII, true);
            ext->setReadySet(selPre, tmpSet);
          } else {
            IISet readySet = getSet(tmpNReadySetDict, outName);
            ext->calReadySwitching(selPre, tmpNReadyList[0]);
            ext->calReadySet(selPre, readySet, tmpNodeSSStart, selMGII);
          }
        }
      })
      .Case<ExtuiNode>([&](ExtuiNode *ext) {
        // EXTUI NODE
        //! Testing
        if (debug)
          llvm::dbgs() << "[EXTUI NODE] \n";
        std::string outName = ext->sucs[0];
        // Valid Signal
        for (const auto &selSuc : ext->sucs) {
          // Number of switching
          ext->calValidSwitching(selSuc, tmpPValidList[0]);
          ext->calValidSet(selSuc, tmpNodeSSStart, selMGII);
        }

        // Ready Signal
        for (const auto &selPre : ext->pres) {
          IISet readySet = getSet(tmpNReadySetDict, outName);
          ext->calReadySwitching(selPre, tmpNReadyList[0]);
          ext->calReadySet(selPre, readySet, tmpNodeSSStart, selMGII);
        }
      })
      .Case<DLoadNode>([&](DLoadNode *load) {
        // DLoadNode NODE: This node will only have 1 input and 1 output, all
        // connections to mem_con ignored
        //! Testing
        if (debug)
          llvm::dbgs() << "[DLoad NODE] \n";
        std::string outName = load->sucs[0];
        // Valid Signal
        for (const auto &selSuc : load->sucs) {
          // Number of switching
          load->calValidSwitching(selSuc, tmpPValidList[0]);
          // Active Range
          load->calValidSet(selSuc, *tmpPValidSetList[0], selMGII);
        }

        // Ready Signal
        for (const auto &selPre : load->pres) {
          IISet readySet = getSet(tmpNReadySetDict, outName);
          // Number of switching
          if (std::find(
            hs.mgInfluencedLoadUnits[std::stoul(selMG)].begin(),
            hs.mgInfluencedLoadUnits[std::stoul(selMG)].end(),
                  selNode) !=
                  hs.mgInfluencedLoadUnits[std::stoul(selMG)].end()) {
            int tmpSwitching = 2;
            load->calReadySwitching(selPre, tmpSwitching);
          } else {
            int tmpSwitching = 0;
            load->calReadySwitching(selPre, tmpSwitching);
          }

          // Active Range
          load->calReadySet(selPre, readySet, tmpNodeSSStart, selMGII);
        }
      })
      .Case<DStoreNode>([&](DStoreNode *store) {
        // DStoreNode NODE
        //! Testing
        if (debug)
          llvm::dbgs() << "[DStore NODE] \n";
        // Valid Signal: This node will not have any successors in the extracted
        // CFDFC, but we still model the switching of MC
        store->calValidSwitching(tmpPValidList[0], tmpPValidList[1]);
        // Active Range
        store->calValidSet(*tmpPValidSetList[0], *tmpPValidSetList[1], selMGII);

        // Ready Signal
        for (const auto &selPre : store->pres) {
          store->calReadySwitching(selPre, tmpPValidList[0], tmpPValidList[1]);
          store->calReadySet(selPre, *tmpPValidSetList[0], *tmpPValidSetList[1],
                             selMGII);
        }
      })
      .Case<ForkNode>([&](ForkNode *fork) {
        // ForkNode NODE
        //! Testing
        if (debug)
          llvm::dbgs() << "[Fork NODE] \n";
        // Valid Signal
        for (auto &selSuc : fork->sucs) {
          // Get Suc Node Start
          int sucNodeStart = mgGetNodeStartingPoint(switchInfo, selSuc, selMG);
          // Number of switching
          fork->calValidSwitching(
              selSuc, tmpPValidList[0], tmpNReadySetDict, tmpNReadyDict,
              static_cast<unsigned>(sucNodeStart), tmpNodeSSStart, selMGII);
          // Active Range
          fork->calValidSet(selSuc, tmpNodeSSStart, tmpPValidList[0],
                            tmpNReadySetDict, selMGII);

          //! Testing
          llvm::dbgs() << "Updating Value for Suc: " << selSuc << "\n";
        }

        // Ready Signal
        for (const auto &selPre : fork->pres) {
          // Safe default ready set if none exists yet
          IISet defaultReady(selMGII, false);
          IISet *readyPtr = tmpNReadySetDict[selPre] ? tmpNReadySetDict[selPre]
                                                     : &defaultReady;
          // Number of switching
          fork->calReadySwitching(selPre, tmpNReadyList);
          // build a one-entry map for this predecessor's ready set
          llvm::DenseMap<StringRef, IISet *> singleR{{selPre, readyPtr}};

          fork->calReadySet(selPre, singleR, selMGII);
        }
      })
      // TODO: Add support for lazy fork Node
      .Case<MuxNode>([&](MuxNode *mux) {
        // MuxNode NODE
        //! Testing
        if (debug)
          llvm::dbgs() << "[Mux NODE] \n";
        std::string outName = mux->sucs[0];
        // Get the cond input port name
        std::string condPortName = "";
        std::string dataInputPortName = "";
        auto nameToPortIdxMap = mux->preNameToPortIdxMap;
        for (auto &selPreName : mux->pres) {
          if (nameToPortIdxMap[selPreName] == 0) {
            condPortName = selPreName;
          } else {
            dataInputPortName = selPreName;
          }
        }

        // Get the cond value
        // ! Check the following default value
        // TODO: May need to change this value
        int condValue = 1;

        // Get node starting time in steady state
        int nodeStartTime = mgGetNodeStartingPoint(switchInfo, selNode, selMG);

        // Valid Signal
        for (auto &selSuc : mux->sucs) {
          // Number of switching
          mux->calValidSwitching(selSuc, selMGII);
          // Active Range
          mux->calValidSet(selSuc, static_cast<unsigned>(nodeStartTime),
                           selMGII);
        }

        // Ready Signal
        for (auto &selPre : mux->pres) {
          IISet condSet = getSet(tmpPValidSetDict, condPortName);
          IISet dataSet = getSet(tmpPValidSetDict, dataInputPortName);
          IISet readySet = getSet(tmpNReadySetDict, outName);
          // Number of Switching
          mux->calReadySwitching(selPre, static_cast<unsigned>(condValue),
                                 tmpPValidDict[selPre], condSet, dataSet,
                                 readySet, selMGII);
          // Active Range
          mux->calReadySet(selPre, condSet, dataSet, readySet, selMGII);
        }
      })
      .Case<TrunciNode>([&](TrunciNode *trunci) {
        // TrunciNode NODE
        //! Testing
        if (debug)
          llvm::dbgs() << "[Trunci NODE] \n";
        std::string outName = trunci->sucs[0];
        // Valid Signal
        for (const auto &selSuc : trunci->sucs) {
          trunci->calValidSwitching(selSuc, tmpPValidList[0]);
          trunci->calValidSet(selSuc, tmpNodeSSStart, selMGII);
        }

        // Ready Signal
        for (auto &selPre : trunci->pres) {
          IISet readySet = getSet(tmpNReadySetDict, outName);
          trunci->calReadySwitching(selPre, tmpNReadyList[0]);
          trunci->calReadySet(selPre, readySet, tmpNodeSSStart, selMGII);
        }
      })
      .Case<CMergeNode>([&](CMergeNode *cmerge) {
        // CMergeNode NODE
        //! Testing
        if (debug)
          llvm::dbgs() << "[CMerge NODE] \n";
        // Get node starting time in steady state
        int nodeStartTime = mgGetNodeStartingPoint(switchInfo, selNode, selMG);

        // Valid Signal
        for (auto &selSuc : cmerge->sucs) {
          // Numebr of Switching
          cmerge->calValidSwitching(selSuc, selMGII);
          // Active Range
          cmerge->calValidSet(selSuc, static_cast<unsigned>(nodeStartTime),
                              selMGII);
        }

        // Ready Signal
        for (const auto &selPre : cmerge->pres) {
          // Number of Switching
          cmerge->calReadySwitching(selPre);
          // Active Range
          cmerge->calReadySet(selPre, selMGII);
        }
      })
      .Case<CBrNode>([&](CBrNode *cbr) {
        // CBrNode NODE
        //! Testing
        if (debug)
          llvm::dbgs() << "[CBr NODE] \n";
        // Get the conditional value
        std::string condPortName = cbr->condPreNodeName;

        // Check the existence of the cond_port_name,
        // sometime the cond_port_name is the same as the data_port_name
        if (condPortName == "") {
          condPortName = cbr->dataPreNodeName;
        }

        //! Testing
        // TODO: Calibrate the cond value with the simulation
        if (debug) {
          llvm::dbgs() << "Node: " << selNode << "\n";
          llvm::dbgs() << "\tCond_input Port: " << condPortName << "\n";
        }

        // Check whether the index exist or not
        int condValue = 0;
        if (switchInfo.staticinfo.dataflowGraph->nodes[condPortName]
                ->dataOut[selNode]
                .size() > switchInfo.data.firstExecutedIter[selMG]) {
          condValue =
              switchInfo.staticinfo.dataflowGraph->nodes[condPortName]
                  ->dataOut[selNode][switchInfo.data.firstExecutedIter[selMG]];
        } else {
          condValue = switchInfo.staticinfo.dataflowGraph->nodes[condPortName]
                          ->dataOut[selNode][0];
        }

        //! Testing
        if (debug) {
          llvm::dbgs() << "\tCond_value: " << condValue << "\n";
        }

        std::string selOutputPort = "";
        auto selNameToPortIdxMap = cbr->outChannelNameToIndexMap;

        for (auto &selSuc : cbr->sucs) {
          if (selNameToPortIdxMap[selSuc] != static_cast<unsigned>(condValue)) {
            selOutputPort = selSuc;
          }
        }

        //! Testing
        if (debug) {
          llvm::dbgs() << "\tSelected Suc Node: " << selOutputPort << "\n";
        }
        // TODO Changed out of bounds but not good imo
        // Sometime we just have one output for cond_br
        if (selOutputPort == "") {
          if( cbr->sucs.empty()){
            llvm::dbgs() << "[ERROR] \tSelected CBr Node " << condPortName << " has no successors! Skipping\n";
            return;
          }
          selOutputPort = cbr->sucs[0];
          condValue = 1 - condValue;
        }

        // Check the number of input ports
        unsigned numInputs = cbr->pres.size();
        std::string outName = selOutputPort;
        // Valid Signal
        for (auto &selSuc : cbr->sucs) {
          if (numInputs > 1) {
            // Number of switching
            cbr->calValidSwitching(selSuc, static_cast<unsigned>(condValue),
                                   tmpPValidList[0], tmpPValidList[1]);
            // Active Range
            cbr->calValidSet(selSuc, tmpNodeSSStart,
                             static_cast<unsigned>(condValue), selMGII);
          } else {
            // Number of switching
            cbr->calValidSwitching(selSuc, static_cast<unsigned>(condValue),
                                   tmpPValidList[0], tmpPValidList[0]);
            // Active Range
            cbr->calValidSet(selSuc, tmpNodeSSStart,
                             static_cast<unsigned>(condValue), selMGII);
          }
        }

        // Ready Signal
        for (auto &selPre : cbr->pres) {
          // Determine the "other" input port for multi-input
          std::string other;
          for (const auto &tmpNode : cbr->pres)
            if (tmpNode != selPre)
              other = tmpNode;

          // Safe fallback for valid and ready sets
          IISet validSet = getSet(tmpPValidSetDict, other);
          IISet readySet = getSet(tmpNReadySetDict, outName);
          if (numInputs > 1) {
            // Use the "other" port's pValid count and the selected suc's nReady
            // count
            cbr->calReadySwitching(selPre, tmpPValidDict[other],
                                   tmpNReadyDict[selOutputPort]);
            cbr->calReadySet(selPre, validSet, readySet, selMGII);
          } else {
            // Single-input case: use selPre for both pValid and nReady
            IISet singleValid = getSet(tmpPValidSetDict, selPre);
            IISet singleReady = getSet(tmpNReadySetDict, selOutputPort);
            cbr->calReadySwitching(selPre, tmpPValidDict[selPre],
                                   tmpNReadyDict[selOutputPort]);
            cbr->calReadySet(selPre, singleValid, singleReady, selMGII);
          }
        }
      })
      .Case<SourceNode>([&](SourceNode *source) {
        // Source node: Normally no pre node
        //! Testing
        if (debug)
          llvm::dbgs() << "[Source NODE] \n";
        // Valid Signals
        for (auto &selSuc : source->sucs) {
          // Number of Switching
          source->calValidSwitching(selSuc);
          // Active Range
          source->calValidSet(selSuc, selMGII);
        }
      })
      .Case<ConstantNode>([&](ConstantNode *constant) {
        // ConstantNode NODE
        //! Testing
        if (debug)
          llvm::dbgs() << "[Constant NODE] \n";
        std::string outName = constant->sucs[0];
        // Valid Signal
        for (const auto &selSuc : constant->sucs) {
          constant->calValidSwitching(selSuc, tmpPValidList[0]);
          constant->calValidSet(selSuc, tmpNodeSSStart, selMGII);
        }

        // Ready Signal
        for (auto &selPre : constant->pres) {
          IISet readySet = getSet(tmpNReadySetDict, outName);
          constant->calReadySwitching(selPre, tmpNReadyList[0]);
          constant->calReadySet(selPre, readySet, tmpNodeSSStart, selMGII);
        }
      })
      .Case<ShliNode>([&](ShliNode *shli) {
        // ShliNode NODE
        //! Testing
        if (debug)
          llvm::dbgs() << "[Shli NODE] \n";
        std::string outName = shli->sucs[0];
        // Valid Signal
        for (const auto &selSuc : shli->sucs) {
          // Calculate the number of switching
          shli->calValidSwitching(selSuc, tmpPValidList[0], tmpPValidList[1],
                                  selMGII);
          // Calculate active range
          shli->calValidSet(selSuc, tmpNodeSSStart, selMGII);
        }

        // Ready Signals
        for (const auto &selPre : shli->pres) {
          // Get the node name of the other input port
          std::string other;
          for (const auto &tmpNode : shli->pres)
            if (tmpNode != selPre)
              other = tmpNode;

          IISet validSet = getSet(tmpPValidSetDict, other);
          IISet readySet = getSet(tmpNReadySetDict, outName);

          shli->calReadySwitching(selPre, tmpPValidDict[other],
                                  tmpNReadyList[0]);
          shli->calReadySet(selPre, validSet, readySet, tmpNodeSSStart,
                            selMGII);
        }
      })
      .Case<ShrsiNode>([&](ShrsiNode *shrsi) {
        // ShrsiNode NODE
        //! Testing
        if (debug)
          llvm::dbgs() << "[Shrsi NODE] \n";
        std::string outName = shrsi->sucs[0];
        // Valid Signal
        for (const auto &selSuc : shrsi->sucs) {
          // Calculate the number of switching
          shrsi->calValidSwitching(selSuc, tmpPValidList[0], tmpPValidList[1],
                                   selMGII);
          // Calculate active range
          shrsi->calValidSet(selSuc, tmpNodeSSStart, selMGII);
        }

        // Ready Signals
        for (const auto &selPre : shrsi->pres) {
          // Get the node name of the other input port
          std::string other;
          for (const auto &tmpNode : shrsi->pres)
            if (tmpNode != selPre)
              other = tmpNode;

          IISet validSet = getSet(tmpPValidSetDict, other);
          IISet readySet = getSet(tmpNReadySetDict, outName);

          shrsi->calReadySwitching(selPre, tmpPValidDict[other],
                                   tmpNReadyList[0]);
          shrsi->calReadySet(selPre, validSet, readySet, tmpNodeSSStart,
                             selMGII);
        }
      })
      .Case<ShruiNode>([&](ShruiNode *shrui) {
        // ShruiNode NODE
        //! Testing
        if (debug)
          llvm::dbgs() << "[Shrui NODE] \n";
        std::string outName = shrui->sucs[0];
        // Valid Signal
        for (const auto &selSuc : shrui->sucs) {
          // Calculate the number of switching
          shrui->calValidSwitching(selSuc, tmpPValidList[0], tmpPValidList[1],
                                   selMGII);
          // Calculate active range
          shrui->calValidSet(selSuc, tmpNodeSSStart, selMGII);
        }

        // Ready Signals
        for (const auto &selPre : shrui->pres) {
          // Get the node name of the other input port
          std::string other;
          for (const auto &tmpNode : shrui->pres)
            if (tmpNode != selPre)
              other = tmpNode;

          IISet validSet = getSet(tmpPValidSetDict, other);
          IISet readySet = getSet(tmpNReadySetDict, outName);

          shrui->calReadySwitching(selPre, tmpPValidDict[other],
                                   tmpNReadyList[0]);
          shrui->calReadySet(selPre, validSet, readySet, tmpNodeSSStart,
                             selMGII);
        }
      })
      .Case<SinkNode>([&](SinkNode *sink) {
        // SinkNode NODE
        //! Testing
        if (debug)
          llvm::dbgs() << "[Sink NODE] \n";
        // Ready Signal
        for (auto &selPre : sink->pres) {
          sink->calReadySwitching(selPre);
          sink->calReadySet(selPre, selMGII);
        }
      })
      .Default([&](AdjNode *n) {
        // Unknown node type.
        llvm::errs() << "[ERROR] Unknown node type in nodeHandshakeUpdate: "
                     << selNode << "\n";
      });

  // Print node status
  if (debug) {
    // selAdjGraph->nodes[selNode]->printHandshakeSwitching();
    selAdjGraph->nodes[selNode]->printNodeDetails();
  }
}

//===---------------------------------------------------------------------------------===//
//
// Functions for DFS in the graph, should be merged with the other functions if
// possible
//
//===---------------------------------------------------------------------------------===//
std::vector<std::string>
breakHandshakeUpdateDeadlock(SwitchingInfo &switchInfo,
                             const std::vector<std::string> &pendingNodeList,
                             std::string selMG, unsigned selMGII) {
  auto selAdjGraph = switchInfo.staticinfo.segToGraph[selMG];
  // Get all join type node in the pending list
  std::vector<std::string> selNodeList;
  for (auto &selNode : pendingNodeList) {
    // Get the node type
    auto curNodeType = getNodeType(selNode);
    if (JOIN_NODE.find(curNodeType) != JOIN_NODE.end()) {
      selNodeList.push_back(selNode);
    }
  }

  // Final list
  std::vector<std::string> finishedNodeList;

  // Update all selected node
  for (auto &selNode : selNodeList) {
    // Pending Channel Name
    std::vector<std::string> nameUpdateList;

    // Step 1: Get unresolved pValid pre_node name
    for (auto &selPre : selAdjGraph->nodes[selNode]->pres) {
      // Check the existence of pValid
      if (selAdjGraph->nodes[selPre]->validSignal.find(selNode) ==
          selAdjGraph->nodes[selPre]->validSignal.end()) {
        nameUpdateList.push_back(selPre);
      } else if (selAdjGraph->nodes[selPre]->validSignal[selNode] == 2) {
        nameUpdateList.push_back(selPre);
      }
    }

    // Update all corresponding ready signal
    for (auto &selPreNode : nameUpdateList) {
      // Get the Join type node
      auto selJoinTypeNode =
          dyn_cast<JoinNode>(selAdjGraph->nodes[selNode].get());
      if (selJoinTypeNode) {
        if (selMGII != 1) {
          int tmp1 = 2;
          int tmp2 = 0;
          selJoinTypeNode->calReadySwitching(selPreNode, tmp1, tmp2);
          selJoinTypeNode->setReadySet(selPreNode);
        } else {
          int tmp1 = 0;
          int tmp2 = 0;
          selJoinTypeNode->calReadySwitching(selPreNode, tmp1, tmp2);
          selJoinTypeNode->setReadySet(selPreNode);
        }
      } else {
        llvm::errs() << "[ERROR] " << selNode << " can't be cast to JoinNode\n";
      }
    }

    // Check the node is finished or not
    if (selAdjGraph->nodes[selNode]->handshakeUpdateFinished()) {
      finishedNodeList.push_back(selNode);

      // Check the node is finished or not
      selAdjGraph->nodes[selNode]->totalHandshakeSwitchingUpdate();
    }
  }

  return finishedNodeList;
}

std::vector<std::string>
getLoadsInfluencedByBuffers(SwitchingInfo &switchInfo, std::string selMG,
                        std::vector<std::string> bufferList) {
  auto selAdjGraph = switchInfo.staticinfo.segToGraph[selMG];

  //
  std::vector<std::string> transBufferList;
  for (const auto &selBuffer : bufferList) {
    auto selBufferNode =
        dyn_cast<BufferNode>(selAdjGraph->nodes[selBuffer].get());
    if (selBufferNode->transparent)
      transBufferList.push_back(selBuffer);
  }

  // Get the list of load nodes
  std::vector<std::string> loadNodeList;
  for (const auto &selNode : selAdjGraph->orderedNodeName) {
    if (contains(selNode, "load"))
      loadNodeList.push_back(selNode);
  }

  // Define the list of influenced nodes
  std::vector<std::string> influencedList;

  // Get the influenced load ndoes
  for (const auto &selLoadNode : loadNodeList) {
    // DFS STACK: None recrusive approach
    std::vector<std::string> mainStack;
    std::vector<std::vector<std::string>> adjStack;

    //
    mainStack.push_back(selLoadNode);
    adjStack.push_back(selAdjGraph->nodes[selLoadNode]->pres);

    while (!mainStack.empty()) {
      std::vector<std::string> curAdjList = adjStack.back();
      adjStack.pop_back();

      if (!curAdjList.empty()) {
        std::string curNode = curAdjList.back();
        curAdjList.pop_back();
        adjStack.push_back(curAdjList);

        // Get the type of the node
        auto curNodeType = getNodeType(curNode);

        if (std::find(transBufferList.begin(), transBufferList.end(),
                      curNode) != transBufferList.end()) {
          influencedList.push_back(selLoadNode);
          break;
        } else if (std::find(bufferList.begin(), bufferList.end(), curNode) !=
                   bufferList.end()) {
          continue;
        } else if (contains(curNode, "fork")) {
          continue;
        } else if (JOIN_NODE.find(curNodeType) != JOIN_NODE.end()) {
          continue;
        } else {
          mainStack.push_back(curNode);

          // Continue the search process
          if (contains(curNodeType, "cond_br")) {
            // This is a condbr node
            auto selCondNode =
                dyn_cast<CBrNode>(selAdjGraph->nodes[curNode].get());
            if (selCondNode->dataPreNodeName != "") {
              if (std::find(mainStack.begin(), mainStack.end(),
                            selCondNode->dataPreNodeName) == mainStack.end()) {
                adjStack.push_back({selCondNode->dataPreNodeName});
              }
            } else {
              if (std::find(mainStack.begin(), mainStack.end(),
                            selCondNode->condPreNodeName) == mainStack.end()) {
                adjStack.push_back({selCondNode->condPreNodeName});
              }
            }
          } else {
            // Normal node
            std::vector<std::string> newAdjList =
                selAdjGraph->nodes[curNode]->pres;
            std::vector<std::string> tmpAdjList;
            for (const auto &selNode : newAdjList) {
              if (std::find(mainStack.begin(), mainStack.end(), selNode) ==
                  mainStack.end())
                tmpAdjList.push_back(selNode);
            }

            adjStack.push_back(tmpAdjList);
          }
        }
      } else {
        mainStack.pop_back();
      }
    }
  }

  return influencedList;
}

int mgGetNodeStartingPoint(SwitchingInfo &switchInfo, std::string &selNode,
                           std::string &selMG) {
  // MG Info
  // Get the corresponding graph
  auto selAdjGraph = switchInfo.staticinfo.segToGraph[selMG];
  double_t selMgThroughput = switchInfo.staticinfo.cfdfcThroughput[std::stoi(selMG)];
  float rawII = 1.0f / selMgThroughput;
  unsigned selMGII = static_cast<unsigned>(std::round(rawII));
  std::string baseNode = selAdjGraph->baseNode;

  // Get the longest path from all starting node
  LongestPathResult selPath = selLongestPath(switchInfo, selNode, selMG);

  int finStartPoint = (selPath.maxLatency % selMGII) +
                      selAdjGraph->startBaseNodeShiftMap[selPath.selStartNode] +
                      selAdjGraph->startBaseNodeShiftMap[baseNode];

  return finStartPoint;
}
