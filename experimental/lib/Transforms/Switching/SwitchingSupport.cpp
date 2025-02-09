//===- SwitchingSupport.cpp - Estimate Swithicng Activities ------*- C++ -*-===//
//
// Implements the supporting datastructures for the switching estimation
// pass
//
//===----------------------------------------------------------------------===//

#include "experimental/Transforms/Switching/SwitchingSupport.h"
#include "experimental/Transforms/Switching/ExecModel.h"
#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Dialect/Handshake/HandshakeDialect.h"
#include "dynamatic/Dialect/Handshake/HandshakeInterfaces.h"
#include "dynamatic/Dialect/Handshake/HandshakeAttributes.h"
#include "dynamatic/Dialect/Handshake/MemoryInterfaces.h"
#include "dynamatic/Support/CFG.h"
#include "dynamatic/Transforms/BufferPlacement/CFDFC.h"
#include "dynamatic/Support/DynamaticPass.h"
#include "dynamatic/Support/LLVM.h"
#include "dynamatic/Support/Logging.h"
#include "dynamatic/Support/Backedge.h"
#include "dynamatic/Support/TimingModels.h"
#include "dynamatic/Support/Attribute.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/IR/Value.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Debug.h"

using namespace mlir;
using namespace dynamatic;

/// Function declaration
static unsigned extractNodeLatency(mlir::Operation *op, TimingDatabase timingDB);

void SwitchingInfo::insertBE(unsigned srcBB, unsigned dstBB, StringRef mgLabel) {
  std::pair<unsigned, unsigned> BBPair = {srcBB, dstBB};

  // Check the existence of the backedge pair
  if (backEdgeToCFDFCMap.find(BBPair) != backEdgeToCFDFCMap.end()) {
    backEdgeToCFDFCMap[BBPair].push_back(static_cast<unsigned>(std::stoul(mgLabel.str())));
  } else {
    std::vector<unsigned> tmpVector{static_cast<unsigned>(std::stoul(mgLabel.str()))};
    backEdgeToCFDFCMap[std::make_pair(srcBB, dstBB)] = tmpVector;
  }
}

//===----------------------------------------------------------------------===//
//
// Definitions of AdjNode
//
//===----------------------------------------------------------------------===//
AdjNode::AdjNode(mlir::Operation* selOp, 
          const std::vector<std::string>& predecessors, const std::vector<std::string>& successors, 
          const std::map<std::string, unsigned>& sucDataWidthMap, const unsigned& latency) {
  // Initialize all variables
  this->op = selOp;
  this->pres = predecessors;
  this->sucs = successors;
  this->sucsDataWidthMap = sucDataWidthMap;
  this->nodeLatency = latency;

  // Initialize toggle count map
  for (auto& [key, value] : sucsDataWidthMap) {
    std::map<unsigned, unsigned> tmpMap;

    for (int i = 0; i < value; i++) {
      tmpMap[i] = 0;
    }

    perChannelToggle[key] = tmpMap;
  }
}

bool AdjNode::handshakeUpdateFinished() {
  if (validSignal.size() == sucs.size()) {
    if (readySignal.size() == pres.size()) {
      if (setR.size() == pres.size()) {
        if (setV.size() == sucs.size()) {
          return true;
        }
      }
    }
  }

  return false;
}

bool AdjNode::handshakeSwitchingChecking() {
  if (validSignal.size() == sucs.size()) {
    if (readySignal.size() == pres.size()) {
      return true;
    }
  }

  return false;
}

void AdjNode::totalHandshakeSwitchingUpdate() {
  for (const auto& [s, selValue]: validSignal) {
    totalValidSwitching += selValue;
  }

  for (const auto& [p, selValue]: readySignal) {
    totalReadySwitching += selValue;
  }

}

void AdjNode::updateDataoutChannel(int inputData) {
  // For each successor, we do xor for the data init
  for (const auto& suc: sucs) {
    int diff = 0;

    // Check whether the key is in dataOut or not
    if (dataOut.find(suc) != dataOut.end()) {
      auto &hist = dataOut[suc];
      diff = hist.back() ^ inputData;
      hist.push_back(inputData);
    } else {
      diff = inputData;
      std::vector<int> tmpVec = {inputData};
      dataOut[suc] = tmpVec;
    }

    // Update the per_channel count
    auto positions = getPositionList(diff);

    for (const auto& selPos : positions) {
      if (perChannelToggle[suc].find(selPos) != perChannelToggle[suc].end()) {
        perChannelToggle[suc][selPos] += 1;
      }
    }

  }
}

void AdjNode::updateHandshakeChannelSwitching(unsigned validChannelSwitching, unsigned readyChannelSwitching) {
  totalValidSwitching += validChannelSwitching;
  totalReadySwitching += readyChannelSwitching;

  // Update the handshake status as well
  handshakeUpdateFlag = true;
}

void AdjNode::totalDataSwitchingCounting(bool mapped) {
  for (const auto& [suc, valueVec] : dataOut) {
    // Define tmp storing structure
    unsigned numSwitches = 0;

    // If this suc is one of the units at scf level
    int lastInput = mapped && valueVec[0] == -1 ? 1 : valueVec[0];

    // The corresponding output channel maynot have dataout
    // Check the validity of the data channel value
    for (unsigned i = 0; i < valueVec.size(); i++) {
      int curVal = (valueVec[i] == -1) ? 1 : valueVec[i];
      int diff = lastInput ^ curVal;

      // Count bits
      unsigned bitCount = 0;
      while(diff) {
        bitCount += (diff & 1);
        diff >>= 1;
      }
      numSwitches += bitCount;
      lastInput = curVal;
    }

    // Store the total number of channel switches
    dataSwitches[suc] = numSwitches;
    totalDataSwitching += numSwitches;
  }
}

std::vector<unsigned> AdjNode::getPositionList(int number) {
  std::vector<unsigned> positions;
  unsigned idx = 0;
  int tmp = number;
  while (tmp) {
    if (tmp & 1) positions.push_back(idx);
    tmp >>= 1;
    idx++;
  }

  return positions;
}

// Define all the printing functions to faciliate debug
void AdjNode::printDetail() {
  llvm::dbgs() << "[DEBUG] \t=============================================================\n";
  llvm::dbgs() << "[DEBUG] \t[Node Info Start]\n";

  //
  llvm::dbgs() << "[DEBUG] \t\tLatency: " << nodeLatency <<";\n";
  llvm::dbgs() << "[DEBUG] \t\tPredecessors: [";
  for (const auto& p: pres) {
    llvm::dbgs() << p << " ";
  }
  llvm::dbgs() << "]\n[DEBUG] \t\tSuccessors: [";
  for (const auto& s: sucs) {
    llvm::dbgs() << s << " ";
  }
  llvm::dbgs() << "]\n";

  llvm::dbgs() << "[DEBUG] \t\tSuccessor Channel DataWidth: \n";
  for (const auto& [s, width] : sucsDataWidthMap) {
    llvm::dbgs() << "[DEBUG] \t\t\tNode: " << s << ", Data_width: " << width <<"\n";
  }

  llvm::dbgs() << "[DEBUG] \t\tSuccessor Channel Data Value\n";
  for (const auto& [s, valueVec] : dataOut) {
    llvm::dbgs() << "[DEBUG] \t\t\tNode: " << s << ", Output Value Vector: \n";
    printVector(valueVec);
    llvm::dbgs() << "\n";
  }

  llvm::dbgs() << "[DEBUG] \t\tValid Channel Switching: \n";
  for (const auto& [s, numSwitches] : validSignal) {
    llvm::dbgs() << "[DEBUG] \t\t\tNode: " << s << ", Data_width: " << numSwitches <<"\n";
  }

  llvm::dbgs() << "[DEBUG] \t\tReady Channel Switching: \n";
  for (const auto& [s, numSwitches] : readySignal) {
    llvm::dbgs() << "[DEBUG] \t\t\tNode: " << s << ", Data_width: " << numSwitches <<"\n";
  }

  llvm::dbgs() << "[DEBUG] \t\tValid Active Range: \n";
  for (const auto& [s, valueVec] : setV) {
    llvm::dbgs() << "[DEBUG] \t\t\tNode: " << s << " ; Active Range: [";
    for (const auto& selValue : valueVec) {
      llvm::dbgs() << selValue << " ";
    }
    llvm::dbgs() << "]\n";
  }

  llvm::dbgs() << "[DEBUG] \t\tReady Active Range: \n";
  for (const auto& [s, valueVec] : setR) {
    llvm::dbgs() << "[DEBUG] \t\t\tNode: " << s << " ; Active Range: [";
    for (const auto& selValue : valueVec) {
      llvm::dbgs() << selValue << " ";
    }
    llvm::dbgs() << "]\n";
  }

}

void AdjNode::printHandshakeSwitching() {
  llvm::dbgs() << "[DEBUG] \t\tTotal Valid Switching Number: " << totalValidSwitching << "\n";
  llvm::dbgs() << "[DEBUG] \t\tTotal Ready Switching Number: " << totalReadySwitching << "\n";

  llvm::dbgs() << "[DEBUG] \t\tValid Channel Switching: \n";
  for (const auto& [s, numSwitches] : validSignal) {
    llvm::dbgs() << "[DEBUG] \t\t\tNode: " << s << ", Data_width: " << numSwitches <<"\n";
  }

  llvm::dbgs() << "[DEBUG] \t\tReady Channel Switching: \n";
  for (const auto& [s, numSwitches] : readySignal) {
    llvm::dbgs() << "[DEBUG] [DEBUG] \t\t\t\tNode: " << s << ", Data_width: " << numSwitches <<"\n";
  }

  llvm::dbgs() << "[DEBUG] \t\tValid Active Range: \n";
  for (const auto& [s, valueVec] : setV) {
    llvm::dbgs() << "[DEBUG] \t\t\tNode: " << s << " ; Active Range: [";
    for (const auto& selValue : valueVec) {
      llvm::dbgs() << selValue << " ";
    }
    llvm::dbgs() << "]\n";
  }

  llvm::dbgs() << "[DEBUG] \t\tReady Active Range: \n";
  for (const auto& [s, valueVec] : setR) {
    llvm::dbgs() << "[DEBUG] \t\t\tNode: " << s << " ; Active Range: [";
    for (const auto& selValue : valueVec) {
      llvm::dbgs() << selValue << " ";
    }
    llvm::dbgs() << "]\n";
  }
}

void AdjNode::printDataChannelSwitching() {
  llvm::dbgs() << "[DEBUG] \t\tTotal Number of Data Channel Switches: " << totalDataSwitching << "\n";

  // Print per channel data switches
  for (const auto& [suc, value] : dataSwitches) {
    llvm::dbgs() << "[DEBUG] \t\t\tChannel: " << suc << ": " << value <<"\n";
  }
}

void AdjNode::printPerDataChannelToggleNumber() {
  llvm::dbgs() << "[DEBUG] \t\tData Channel Per BIT Switches: \n";

  for (const auto& [suc, valueVec] : perChannelToggle) {
    llvm::dbgs() << "[DEBUG] \t\t Node: " << suc << "\n";

    for (const auto& [selBit, value]: valueVec) {
      llvm::dbgs() << "[DEBUG] \t\t\tBit " << selBit << ": " << value << "\n";
    }
  }

}

void AdjNode::printPerHandshakeChannelToggleNumber() {
  llvm::dbgs() << "[DEBUG] \t\t[VALID CHANNEL]\n";
  for (const auto& [suc, validSwitch]: validSignal) {
    llvm::dbgs() << "[DEBUG] \t\t\tNode: " << suc << ", Valid Switching: " << validSwitch << "\n";
  }

  llvm::dbgs() << "[DEBUG] \t\t[READY CHANNEL]\n";
  for (const auto& [pre, readySwitching]: readySignal) {
    llvm::dbgs() << "[DEBUG] \t\t\tNode: " << pre << ", Ready Switching: " << readySwitching << "\n";
  }
}

//===----------------------------------------------------------------------===//
//
// Definitions of AdjGraph
//
//===----------------------------------------------------------------------===//
// Initialize the whole adjacency graph for the selected segment
AdjGraph::AdjGraph(const buffer::CFDFC& cfdfc, const TimingDatabase& timingDB, 
                    const unsigned &II, const unsigned &mgIndex) {
  cfdfcIndex = mgIndex;
  cfdfcII = II;
  
  std::map<std::string, std::vector<std::string>> nodeToPresMap;
  std::map<std::string, std::vector<std::string>> nodeToSucsMap;

  // Step 1: Construct the pres and sucs map for all units in the CFDFC
  for (const auto& selChannel: cfdfc.channels) {
    mlir::Operation* srcOp = selChannel.getDefiningOp();
    std::string srcName = srcOp->getAttrOfType<StringAttr>("handshake.name").str();

    for (const auto& dstOp: selChannel.getUsers()) {
      std::string dstName = dstOp->getAttrOfType<StringAttr>("handshake.name").str();

      // Get the BB of the dstOp
      unsigned dstBB;
      if (std::optional<unsigned> optBB = getLogicBB(dstOp); !optBB.has_value())
        continue;
      else
        dstBB = *optBB;

      // Check whether this node is in the cfdfc
      if (!cfdfc.cycle.contains(dstBB)) continue;

      // Check whether this is a backedge
      if (cfdfc.isCFDFCBackedge(selChannel)) {
        backedges.push_back(std::make_pair(srcName, dstName));

        // Add the name of dstNode to start node vector
        segStartNodes.push_back(dstName);
      }
      
      // Insert to sucs
      insertToSurroundingList(nodeToSucsMap, srcName, dstName);

      // Insert the pres
      insertToSurroundingList(nodeToPresMap, dstName, srcName); 
    }
  }

  // Step 2:Traverse all nodes in the MG
  for (auto& selNode: cfdfc.units) {
    std::string unitType = selNode->getName().getStringRef().str();
    std::string unitName = selNode->getAttrOfType<StringAttr>("handshake.name").str();

    // Preserve the relative order of nodes
    orderedNodeName.push_back(unitName);

    // Get the unit latency
    unsigned nodeLatency = extractNodeLatency(selNode, timingDB);

    //! Testing
    // llvm::dbgs() << "[DEBUG] \t=================================\n";
    // llvm::dbgs() << "[DEBUG] \tNode Name: " << unitName << "\n";
    // llvm::dbgs() << "[DEBUG] \tNode Latency From DataBase: " << nodeLatency << "\n";

    // Step 2.1: Construct the node storing structure
    auto newNode = createNodeFromOperation(
      selNode, nodeToPresMap[unitName], nodeToSucsMap[unitName], nodeLatency);

    if (!newNode) continue;

    //! Testing
    // newNode->printDetail();

    // Store the new node
    nodes[unitName] = newNode;
  }

}

AdjGraph::AdjGraph(const TimingDatabase& timingDB, const unsigned &II, 
            handshake::FuncOp funcOp,
            std::vector<std::pair<std::string, std::string>> &allBackedges) : backedges(allBackedges) {
  cfdfcII = II;
  // Iterate over all the ops in the funcop
  for (Operation &op: funcOp.getOps()) {
    std::string unitName = op.getAttrOfType<StringAttr>("handshake.name").str();
    // For now, we exclude lsq and mem_controller
    if (unitName.find("lsq") != std::string::npos || unitName.find("mem_controller") != std::string::npos) continue;

    // Preserve the relative order of nodes
    orderedNodeName.push_back(unitName);

    //! Testing
    //! Testing
    llvm::dbgs() << "[DEBUG] \t=================================\n";
    llvm::dbgs() << "[DEBUG] \tNode Name: " << unitName << "\n";

    std::vector<std::string> pres;
    std::vector<std::string> sucs;
    // Construct the sucs
    for (OpResult res: op.getResults()) {
      assert(std::distance(res.getUsers().begin(), res.getUsers().end()) == 1 &&
             "value must have unique user");

      Operation *user = *res.getUsers().begin();
      std::string dstName = user->getAttrOfType<StringAttr>("handshake.name").str();

      // Excluding lsq and mem_controller
      if (dstName.find("lsq") == std::string::npos && dstName.find("mem_controller") == std::string::npos)
        sucs.push_back(dstName);
    }

    // Construct pres
    for (auto operand: op.getOperands()) {
      // TODO: Need to check do we need to include the block argument in the graph or not
      if (operand.getDefiningOp()) {
        std::string preName = operand.getDefiningOp()->getAttrOfType<StringAttr>("handshake.name").str();
        if (preName.find("lsq") == std::string::npos && preName.find("mem_controller") == std::string::npos)
          pres.push_back(preName);
      }
    }

    unsigned nodeLatency = extractNodeLatency(&op, timingDB);

    auto newNode = createNodeFromOperation(
      &op, pres, sucs, nodeLatency); 
    
    if (!newNode) continue;

    //! Testing
    newNode->printDetail();

    // Store the new node
    nodes[unitName] = newNode;
  }
}

void AdjGraph::insertToSurroundingList(std::map<std::string, std::vector<std::string>>& selMap, 
                                std::string& key, std::string& value) {
  if (selMap.find(key) != selMap.end()) {
    selMap[key].push_back(value);
  } else {
    std::vector<std::string> tmpVec = {value};
    selMap[key] = tmpVec;
  }

}

std::shared_ptr<AdjNode> AdjGraph::createNodeFromOperation(mlir::Operation *op,
                                                    std::vector<std::string> &pres, std::vector<std::string> &sucs,
                                                    unsigned &nodeLatency) {
    std::map<std::string, unsigned> nodeSucsDataWidthMap;
    // Get the successor channels' dataWidth
    for (unsigned resIndex = 0, e = op->getNumResults(); resIndex < e; ++resIndex) {
      mlir::Value selRes = op->getResult(resIndex);

      // The result is of type !handshake.channel<...>
      if (auto chanTy = selRes.getType().dyn_cast<handshake::ChannelType>()) {
        // Extract the underlying data width.
        unsigned dataWidth = chanTy.getDataBitWidth();

        // Now, iterate over all users of this result.
        for (mlir::Operation *user : selRes.getUsers()) {
          // Try to get the successor's name attribute.
          if (auto nameAttr = user->getAttrOfType<mlir::StringAttr>("handshake.name")) {
            nodeSucsDataWidthMap[nameAttr.getValue().str()] = dataWidth;

            //! Testing
            // llvm::dbgs() << "[DEBUG] \t\t" << nameAttr.getValue() << "\n";
          }
        }
      }
    }
    
    // Return a unique pointer
    return llvm::TypeSwitch<Operation *, std::shared_ptr<AdjNode>>(op)
      // handshake::AddIOp operator
      .Case<handshake::AddIOp>([&](auto selNode) {
        auto node = std::make_shared<AddiNode>(op, pres, sucs, nodeSucsDataWidthMap, nodeLatency);
        return node;
      })
      // handshake::SubIOp operator
      .Case<handshake::SubIOp>([&](auto selNode) {
        auto node = std::make_shared<SubiNode>(op, pres, sucs, nodeSucsDataWidthMap, nodeLatency);
        return node;
      })
      // handshake::MulIOp operator
      .Case<handshake::MulIOp>([&](auto selNode) {
        auto node = std::make_shared<MuliNode>(op, pres, sucs, nodeSucsDataWidthMap, nodeLatency);
        return node;
      })
      // handshake::CmpIOp operator
      .Case<handshake::CmpIOp>([&](handshake::CmpIOp selNode) {
        auto node = std::make_shared<CmpiNode>(op, pres, sucs, nodeSucsDataWidthMap, nodeLatency);
        return node;
      })
      // handshake::BufferOp operator
      .Case<handshake::BufferOp>([&](handshake::BufferOp selNode) {
        // For the buffer nodes, we need to get the number of slots and transparency
        auto params = selNode->getAttrOfType<DictionaryAttr>(RTL_PARAMETERS_ATTR_NAME);
        if (!params) {
          llvm::errs() << "[ERROR] " << RTL_PARAMETERS_ATTR_NAME << " is missing for buffer op in handshake_export.mlir\n";
          exit(-1);
        }

        // Retrieve the number of slots
        auto optSlots = params.getNamed(BufferOp::NUM_SLOTS_ATTR_NAME);
        unsigned numSlotsValue = 0;
        if (optSlots) {
          if (auto numSlots = dyn_cast<IntegerAttr>(optSlots->getValue())) {
            if (numSlots.getType().isUnsignedInteger())
              numSlotsValue = numSlots.getUInt();
          }
        }

        // Get the occupancy value
        float_t occValue = -1;
        DictionaryAttr occAttr = getDialectAttr<handshake::BufferOccupancyAttr>(op).getBufferOccMap();
        auto occValueAttr = occAttr.get(std::to_string(cfdfcIndex)).dyn_cast<mlir::FloatAttr>();
        if (occValueAttr) {
          // Successfully retrieved the occ attribute
          occValue = occValueAttr.getValueAsDouble();
        } else {
          llvm::errs() << "[ERROR] Failed to retrieve the buffer occupancy attribute\n";
          exit(-1);
        }

        auto node = std::make_shared<BufferNode>(op, pres, sucs, nodeSucsDataWidthMap, nodeLatency);

        // Update the desired inforamtion for the buffer node
        node->occupancy = occValue;
        node->numSlots = numSlotsValue;
        node->transparent = nodeLatency ? false : true;

        return node;
      })
      // handshake::MuxOp operator
      .Case<handshake::MuxOp>([&](handshake::MuxOp selNode) {
        auto node = std::make_shared<MuxNode>(op, pres, sucs, nodeSucsDataWidthMap, nodeLatency);
        return node;
      })
      // handshake::OrIOp operator
      .Case<handshake::OrIOp>([&](handshake::OrIOp selNode) {
        auto node = std::make_shared<OriNode>(op, pres, sucs, nodeSucsDataWidthMap, nodeLatency);
        return node;
      })
      // handshake::AndIOp operator
      .Case<handshake::AndIOp>([&](handshake::AndIOp selNode) {
        auto node = std::make_shared<AndiNode>(op, pres, sucs, nodeSucsDataWidthMap, nodeLatency);
        return node;
      })
      // handshake::ForkOp operator
      .Case<handshake::ForkOp>([&](handshake::ForkOp selNode) {
        auto node = std::make_shared<ForkNode>(op, pres, sucs, nodeSucsDataWidthMap, nodeLatency);
        return node;
      })
      // handshake::LazyForkOp operator
      .Case<handshake::LazyForkOp>([&](handshake::LazyForkOp selNode) {
        // TODO: Add model for lazy fork
        llvm::dbgs() << "[DEBUG] \t\t Missing Implementation for LAZY FORK NODE\n";
        return std::shared_ptr<AdjNode>(nullptr);
      })
      // handshake::TruncIOp operator
      .Case<handshake::TruncIOp>([&](auto selNode) {
        auto node = std::make_shared<TrunciNode>(op, pres, sucs, nodeSucsDataWidthMap, nodeLatency);
        return node;
      })
      // handshake::ExtSIOp operator
      .Case<handshake::ExtSIOp>([&](auto selNode) {
        auto node = std::make_shared<ExtsiNode>(op, pres, sucs, nodeSucsDataWidthMap, nodeLatency);
        return node;
      })
      // handshake::ExtUIOp operator
      .Case<handshake::ExtUIOp>([&](auto selNode) {
        auto node = std::make_shared<ExtuiNode>(op, pres, sucs, nodeSucsDataWidthMap, nodeLatency);
        return node;
      })
      // handshake::ControlMergeOp operator
      .Case<handshake::ControlMergeOp>([&](handshake::ControlMergeOp selNode) {
        auto node = std::make_shared<CMergeNode>(op, pres, sucs, nodeSucsDataWidthMap, nodeLatency);
        return node;
      })
      // handshake::ConditionalBranchOp operator
      .Case<handshake::ConditionalBranchOp>([&](handshake::ConditionalBranchOp selNode) {
        auto node = std::make_shared<CBrNode>(op, pres, sucs, nodeSucsDataWidthMap, nodeLatency);
        return node;
      })
      // handshake::SourceOp operator
      .Case<handshake::SourceOp>([&](auto selNode) {
        auto node = std::make_shared<SourceNode>(op, pres, sucs, nodeSucsDataWidthMap, nodeLatency);
        return node;
      })
      // handshake::ConstantOp operator
      .Case<handshake::ConstantOp>([&](auto selNode) {
        auto node = std::make_shared<ConstantNode>(op, pres, sucs, nodeSucsDataWidthMap, nodeLatency);
        return node;
      })
      // handshake::LoadOp operator
      .Case<handshake::LoadOp>([&](handshake::LoadOp selNode) {
        // Check whether this is a LSQ Load or not
        auto memOp = findMemInterface(selNode.getAddressResult());
        if (isa_and_present<handshake::LSQOp>(memOp)) {
          // TODO: Need to change the latency obtaining method for lsq load op
          auto node = std::make_shared<DLoadNode>(op, pres, sucs, nodeSucsDataWidthMap, nodeLatency);
          return node;
        } else {
          auto node = std::make_shared<DLoadNode>(op, pres, sucs, nodeSucsDataWidthMap, nodeLatency);
          return node;
        }
      })
      // handshake::StoreOp operator
      .Case<handshake::StoreOp>([&](handshake::StoreOp selNode) {
        auto node = std::make_shared<DStoreNode>(op, pres, sucs, nodeSucsDataWidthMap, nodeLatency);
        return node;
      })
      // handshake::ShLIOp operator
      .Case<handshake::ShLIOp>([&](auto selNode) {
        auto node = std::make_shared<ShliNode>(op, pres, sucs, nodeSucsDataWidthMap, nodeLatency);
        return node;
      })
      // handshake::ShRSIOp operator
      .Case<handshake::ShRSIOp>([&](auto selNode) {
        auto node = std::make_shared<ShrsiNode>(op, pres, sucs, nodeSucsDataWidthMap, nodeLatency);
        return node;
      })
      // handshake::ShRUIOp operator
      .Case<handshake::ShRUIOp>([&](auto selNode) {
        auto node = std::make_shared<ShruiNode>(op, pres, sucs, nodeSucsDataWidthMap, nodeLatency);
        return node;
      })
      // handshake::SinkOp operator
      .Case<handshake::SinkOp>([&](auto selNode) {
        auto node = std::make_shared<SinkNode>(op, pres, sucs, nodeSucsDataWidthMap, nodeLatency);
        return node;
      })
      // handshake::EndOp operator
      .Case<handshake::EndOp>([&](auto selNode) {
        auto node = std::make_shared<EndNode>(op, pres, sucs, nodeSucsDataWidthMap, nodeLatency);
        return node;
      })
      .Default([](auto selNode) {
          selNode->emitOpError() << "Unknown operation!";

          return std::shared_ptr<AdjNode>(nullptr);
      });
}

unsigned AdjGraph::calPathLatency(const Path &selPath, bool useGlobalOrder) {
  unsigned latencySum = 0;

  for (const auto& selNode: selPath.nodeList) {
    latencySum += nodes[selNode]->nodeLatency;

    if (useGlobalOrder) {
      if ((std::find(segStartNodes.begin(), segStartNodes.end(), selNode) == segStartNodes.end()) &&
            latencySum < graphGlobalOrder[selNode].second) {
        latencySum = graphGlobalOrder[selNode].second;
      }
    }
  }

  // Check backedges
  if (selPath.contain_backedge) {
    latencySum -= (selPath.backedges.size() * cfdfcII);
  }

  return latencySum;
}

void AdjGraph::obtainNodeGlobalOrder() {
  // Iterate over all nodes in the AdjGraph
  for (const auto& name: orderedNodeName) {
    if (std::find(segStartNodes.begin() , segStartNodes.end(), name) != segStartNodes.end()) {
      continue;
    } else {
      unsigned maxLatency = 0;
      std::string finalStartNode = "";

      for (const auto& selStartNode: segStartNodes) {
        auto foundPaths = findPaths(selStartNode, name, true, false);

        if (foundPaths.size() > 0) {
          for (const auto& selPath: foundPaths) {
            auto tmpPathLat = selPath.latency;
            if (tmpPathLat >= maxLatency) {
              maxLatency = tmpPathLat;
              finalStartNode = selStartNode;
            }
          }
        }
      }
      // Store the global order
      graphGlobalOrder[name] = std::make_pair(finalStartNode, maxLatency);

      //! Testing
      // llvm::dbgs() << "[DEBUG] \tNode: " << name << "; Global Order: (" << finalStartNode << ", " << maxLatency << ");\n";
    }
  }
}

//! Testing
// 1) Print mainStack
static void printMainStack(const std::vector<std::string> &mainStack) {
  llvm::dbgs() << "mainStack: [ ";
  for (const auto &node : mainStack) {
    llvm::dbgs() << node << " ";
  }
  llvm::dbgs() << "]\n";
}

// 2) Print adjStack
static void printAdjStack(const std::vector<std::vector<std::string>> &adjStack) {
  llvm::dbgs() << "adjStack:\n";
  // Each element of adjStack is a vector of strings
  for (size_t i = 0; i < adjStack.size(); ++i) {
    llvm::dbgs() << "  Level " << i << ": [ ";
    for (const auto &node : adjStack[i]) {
     llvm::dbgs() << node << " ";
    }
    llvm::dbgs() << "]\n";
  }
}

//
std::vector<Path> AdjGraph::findPaths(const std::string &srcNode, const std::string &dstNode,
                              bool noStartingNode, bool useGlobalOrder) {
  // Build excluding list
  // TODO: replace noStartingNode with an actual excluding list
  std::set<std::string> excludingSet;
  if (noStartingNode) {
    for (const auto &sn : segStartNodes) {
      if (sn != dstNode) excludingSet.insert(sn);
    }
  }

  // Define storing structure
  std::vector<std::string> mainStack;
  std::vector<std::vector<std::string>> adjStack;
  std::vector<Path> foundPaths;

  // Initialization
  mainStack.push_back(srcNode);
  adjStack.push_back(nodes[srcNode]->sucs);

  //
  while (!mainStack.empty()) {
    std::vector<std::string> curAdjList = adjStack.back();
    adjStack.pop_back();

    if (!curAdjList.empty()) {
      std::string curNode = curAdjList.back();
      curAdjList.pop_back();

      mainStack.push_back(curNode);
      adjStack.push_back(curAdjList);

      // Insert new adj_list
      std::vector<std::string> tmpAdjList;
      std::vector<std::string> newAdjList = nodes[curNode]->sucs;

      // If node not in the mainStack and the excluding list
      for (const auto &n : newAdjList) {
        bool inStack = (std::find(mainStack.begin(), mainStack.end(), n) != mainStack.end());
        bool isExcluded = (excludingSet.find(n) != excludingSet.end());
        if (!inStack && !isExcluded) {
          tmpAdjList.push_back(n);
        }
      }

      adjStack.push_back(tmpAdjList);      
    } else {
      mainStack.pop_back();
    }

    // Found a path
    if (!mainStack.empty() && mainStack.back() == dstNode) {
      std::vector<std::string> pathList = mainStack;

      // Build edge list
      std::vector<std::pair<std::string, std::string>> edgeList;
      edgeList.reserve(pathList.size() > 1 ? pathList.size() - 1 : 0);
      for (size_t i = 0; i < pathList.size() - 1; ++i) {
        edgeList.push_back({pathList[i], pathList[i + 1]});
      }

      Path selPath(pathList);

      // Retrieve all the backedges in the path
      for (auto &edge: edgeList) {
        if (std::find(backedges.begin(), backedges.end(), edge) != backedges.end()) selPath.add_backedge(edge);
      }

      // store the path
      foundPaths.push_back(selPath);
      unsigned tmpPathLatency = calPathLatency(selPath, useGlobalOrder);
      foundPaths.back().set_latency(tmpPathLatency);

      //
      mainStack.pop_back();
      if (!adjStack.empty()) adjStack.pop_back();
    }
  }

  return foundPaths;
}

std::string AdjGraph::graphBacktrack(std::string srcNode, std::unordered_set<std::string> &baseNodeSet) {
  // //! Testing
  // llvm::dbgs() << "[DEBUG] [BASE NODE SET]\n[DEBUG]\t\t";
  // for (const auto& selNode: baseNodeSet) {
  //   llvm::dbgs() << selNode << ", ";
  // }
  // llvm::dbgs() << "\n";

  // If srcNode is in baseNodes => return it
  if (baseNodeSet.find(srcNode) != baseNodeSet.end()) {
    return srcNode;
  }

  // Define storing structure
  std::vector<std::string> mainStack;
  std::vector<std::vector<std::string>> adjStack;

  // Initialization
  mainStack.push_back(srcNode);
  if (srcNode.find("cond_br") != std::string::npos) {
    if (auto *cbrNode = dyn_cast<CBrNode>(nodes[srcNode].get())) {
      std::string tmpDataPreNode = cbrNode->dataPreNodeName;
      adjStack.push_back({ tmpDataPreNode });
    }
  } else {
    adjStack.push_back(nodes[srcNode]->pres);
  }

  // //! Testing
  // llvm::dbgs() << "=================Initial==================\n";
  // printMainStack(mainStack);
  // printAdjStack(adjStack);

  //
  while (!mainStack.empty()) {
    std::vector<std::string> curAdjList = adjStack.back();
    adjStack.pop_back();

    if (!curAdjList.empty()) {
      //
      std::string curNode = curAdjList.back();
      curAdjList.pop_back();
      adjStack.push_back(curAdjList);

      if (baseNodeSet.find(curNode) != baseNodeSet.end()) {
        return curNode;
      } else {
        mainStack.push_back(curNode);

        if (curNode.find("cond_br") != std::string::npos) {
          if (auto *cbrNode = dyn_cast<CBrNode>(nodes[curNode].get())){
            std::string tmpDataPreNode = cbrNode->dataPreNodeName;
            if (std::find(mainStack.begin(), mainStack.end(), tmpDataPreNode) == mainStack.end()) {
              adjStack.push_back({ tmpDataPreNode });
            } 
          }
        } else {
          std::vector<std::string> tmpAdjList;
          std::vector<std::string> newAdjList = nodes[curNode]->pres;

          for (const auto&n : newAdjList) {
            bool inStack = (std::find(mainStack.begin(), mainStack.end(), n) != mainStack.end());
            if (!inStack) tmpAdjList.push_back(n);
          }

          adjStack.push_back(tmpAdjList);
        }
      }

      // //! Testing
      // llvm::dbgs() << "=================Internal==================\n";
      // printMainStack(mainStack);
      // printAdjStack(adjStack);
    } else {
      mainStack.pop_back();
    }
  }

  llvm::dbgs() << "[ERROR] Could not find base node for " << srcNode << "\n";
  return "";
}

void AdjGraph::analyzeStartNodeShifting() {
  // Find the largest value in the global order map
  unsigned tmpMaxValue = 0;
  for (const auto& [nodeName, delayPair]: graphGlobalOrder) {
    if (delayPair.second > tmpMaxValue) {
      tmpMaxValue = delayPair.second;
      baseNode = delayPair.first;
    }
  }

  // Update the cycle time of the MG
  for (const auto& selBackedge: backedges) {
    unsigned tmpLonPath = 0;
    auto tmpPaths = findPaths(selBackedge.second, selBackedge.first, false, true);

    for (auto selPath: tmpPaths) {
      if (selPath.latency > tmpLonPath) {
        tmpLonPath = selPath.latency;
      }
    }
    cycleTimeMap[selBackedge.second] = tmpLonPath;
  }

  // Analyze the shifting between different start ndoes and the base node
  for (const auto& selStart : segStartNodes) {
    auto nodeVal = cycleTimeMap[selStart];
    auto baseVal = cycleTimeMap[baseNode];

    if (selStart == baseNode) {
      startBaseNodeShiftMap[selStart] = baseVal % cfdfcII;
      continue;
    }

    if ((nodeVal % cfdfcII) == (baseVal % cfdfcII)) {
      startBaseNodeShiftMap[selStart] = 0;
    } else {
      if (nodeVal > baseVal) {
        int tmpDiff = nodeVal - baseVal;
        startBaseNodeShiftMap[selStart] = tmpDiff % cfdfcII;
      } else {
        int tmpDiff = baseVal - nodeVal;
        startBaseNodeShiftMap[selStart] = -1 * (tmpDiff % cfdfcII);
      }
    }
  }
}

void AdjGraph::buildMuxSrcMap() {
  // Traverse all nodes in the dataflow graph
  for (const auto& selNode: orderedNodeName) {
    // If this is a mux node
    if (selNode.find("mux") != std::string::npos) {
      //! Testing
      llvm::dbgs() << "[DEBUG] \t\tMux Node Name: " << selNode << "\n";

      // Ger the mux node storing structure
      auto *selMuxNode = dyn_cast<MuxNode>(nodes[selNode].get());
      // 
      std::map<std::string, std::string> tmpMuxPortMap;
      std::string conPreNodeName = selMuxNode->conPreNodeName;
      std::string dataPre0NodeName = "";
      std::string dataPre1NodeName = "";

      for (const auto& [nodeName, portIdx]: selMuxNode->preNameToPortIdxMap) {
        if (portIdx == 1) {
          dataPre0NodeName = nodeName;
        } else if (portIdx == 2) {
          dataPre1NodeName = nodeName;
        }
      }

      // Get the source node for all three ports
      std::string conSrcNodeName = graphBacktrack(conPreNodeName, allDataBaseNode);
      std::string dataPre0SrcName = graphBacktrack(dataPre0NodeName, allDataBaseNode);
      std::string dataPre1SrcName = graphBacktrack(dataPre1NodeName, allDataBaseNode);

      // Update the control_merge to mux map
      if (cmToMuxMap.find(conSrcNodeName) != cmToMuxMap.end()) {
        cmToMuxMap[conSrcNodeName].push_back(selNode);
      } else {
        cmToMuxMap[conSrcNodeName] = { selNode };
      }

      // Build mux src map
      tmpMuxPortMap["control"] = conSrcNodeName;
      tmpMuxPortMap["0"] = dataPre0SrcName;
      tmpMuxPortMap["1"] = dataPre1SrcName;

      // Update the global map
      muxToSrcNodeMap[selNode] = tmpMuxPortMap;

      // Update the src to mux map
      if (srcNodeToMuxMap.find(dataPre0SrcName) != srcNodeToMuxMap.end()) {
        srcNodeToMuxMap[dataPre0SrcName].push_back(std::make_pair(selNode, 0));
      } else {
        srcNodeToMuxMap[dataPre0SrcName] = {std::make_pair(selNode, 0)};
      }

      if (srcNodeToMuxMap.find(dataPre1SrcName) != srcNodeToMuxMap.end()) {
        srcNodeToMuxMap[dataPre1SrcName].push_back(std::make_pair(selNode, 1));
      } else {
        srcNodeToMuxMap[dataPre1SrcName] = {std::make_pair(selNode, 1)};
      }
    }
  }
}

void AdjGraph::buildCondandStoreSrcMap() {
  // Traverse all nodes in the dataflow graph
  for (const auto& selNode: orderedNodeName) {
    // If this is a cond_br node
    if (selNode.find("cond_br") != std::string::npos) {
      // Ger the cond_br node storing structure
      auto *selCBrNode = dyn_cast<CBrNode>(nodes[selNode].get());

      std::string controlSrcNode = graphBacktrack(selCBrNode->condPreNodeName, allDataBaseNode);
      condBrToConSrcMap[selNode] = controlSrcNode;
    } else if (selNode.find("store") != std::string::npos) {
      auto *selStoreNode = dyn_cast<DStoreNode>(nodes[selNode].get());

      std::string addrPreNode = selStoreNode->addressInNode;
      std::string dataPreNode = selStoreNode->dataInNode;

      selStoreNode->addressInSrcNode = graphBacktrack(addrPreNode, allDataBaseNode);
      selStoreNode->dataInSrcNode = graphBacktrack(dataPreNode, allDataBaseNode);

      //! Testing
      llvm::dbgs() << "[DEBUG] \tStore Node: " << selNode << "\n";
      llvm::dbgs() << "[DEBUG] \t\tData Src Node: " << selStoreNode->dataInSrcNode << "\n";
      llvm::dbgs() << "[DEBUG] \t\tAddr Src Node: " << selStoreNode->addressInSrcNode << "\n";

    }
  }
}

//===----------------------------------------------------------------------===//
//
// Helper Functions
//
//===----------------------------------------------------------------------===//
std::string getHandshakeNodeName(mlir::Value &selRes) {
  for (mlir::Operation *user : selRes.getUsers()) {
    // Try to get the successor's name attribute.
    if (auto nameAttr = user->getAttrOfType<mlir::StringAttr>("handshake.name")) {
      return nameAttr.getValue().str();
    }
  }
};

void printBEToCFDFCMap(const std::map<std::pair<unsigned, unsigned>, std::vector<unsigned>>& selMap) {
  for (const auto& selPair : selMap) {
    const std::pair<unsigned, unsigned>& key = selPair.first;
    const std::vector<unsigned> mgList = selPair.second;

    llvm::dbgs() << "[DEBUG] \tBackEdge Pair: (" << key.first << ", " << key.second <<") : [";

    for (const auto& selMG: mgList) {
      llvm::dbgs() << selMG << ", ";
    }

    llvm::dbgs() << "]\n";
  }
}

void printSegToBBListMap(const std::map<std::string, mlir::SetVector<unsigned>>& selMap) {
  for (const auto& selPair: selMap) {
    const std::string segLabel = selPair.first;
    const mlir::SetVector<unsigned> BBList = selPair.second;

    llvm::dbgs() << "[DEBUG] \tSeg Label: " << segLabel <<" : [";

    for (const auto& selBB: BBList) {
      llvm::dbgs() << selBB << ", ";
    }

    llvm::dbgs() << "]\n";
  }
}

std::string removeDigits(const std::string& inStr) {
  std::regex digitsRegex("\\d");

  std::string outStr = std::regex_replace(inStr, digitsRegex, "");

  return outStr;
}

std::vector<std::string> split(const std::string& s, const std::string& delimiter) {
  std::vector<std::string> tokens;
  std::regex re(delimiter);
  std::sregex_token_iterator it(s.begin(), s.end(), re, -1);
  std::sregex_token_iterator end;

  for (; it != end; ++it) {
    tokens.push_back(it->str());
  }

  return tokens;
}

std::string strip(const std::string &inputStr, const std::string &toRemove) {
  // Trim leading and trailing whitespace
  size_t start = inputStr.find_first_not_of(" \t\n\r\f\v");
  if (start == std::string::npos) {
      return "";  // Return an empty string if there are only whitespaces
  }
  size_t end = inputStr.find_last_not_of(" \t\n\r\f\v");
  std::string stripped = inputStr.substr(start, end - start + 1);

  if (toRemove == "") {
    return stripped;
  }

  // Remove all occurrences of the specified substring 'toRemove'
  size_t pos = stripped.find(toRemove);
  while (pos != std::string::npos) {
      stripped.erase(pos, toRemove.length());
      pos = stripped.find(toRemove, pos);
  }

  return stripped;
}

/// Extracts the latency for each operation
/// This is done in 3 ways:
/// 1. If the operation is in the timingDB, the latency is extracted from the
/// timingDB
/// 2. If the operation is a buffer operation, the latency is extracted from the
/// timing attribute
/// 3. If the operation is neither, then its latency is set to 0
static unsigned extractNodeLatency(mlir::Operation *op, TimingDatabase timingDB) {
  double latency = 0;

  if (!failed(timingDB.getLatency(op, SignalType::DATA, latency))) {
    return latency;
  }

  if (isa<handshake::BufferOp>(op)) {
    auto params = op->getAttrOfType<DictionaryAttr>(RTL_PARAMETERS_ATTR_NAME);
    if (!params)
      return 0;

    auto optTiming = params.getNamed(handshake::BufferOp::TIMING_ATTR_NAME);
    if (!optTiming)
      return 0;

    if (auto timing = dyn_cast<handshake::TimingAttr>(optTiming->getValue())) {
      handshake::TimingInfo info = timing.getInfo();
      return info.getLatency(SignalType::DATA).value_or(0);
    }
  }

  return 0;
}
