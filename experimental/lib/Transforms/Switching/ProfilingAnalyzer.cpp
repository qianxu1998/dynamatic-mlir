//===- SwitchingSupport.cpp - Estimate Swithicng Activities ------*- C++ -*-===//
//
// Implements the Analyzer Class for the functional profiling results
//
//===----------------------------------------------------------------------===//

#include "experimental/Transforms/Switching/ProfilingAnalyzer.h"
#include <cassert>
using namespace mlir;
using namespace dynamatic;
using namespace dynamatic::handshake;

// Constructor for the SCF parsing class
SCFProfilingResult::SCFProfilingResult(StringRef dataTrace, SwitchingInfo& switchInfo) {
  // Step 0: Get the directory path
  std::filesystem::path pathObj(dataTrace.str());
  std::string resultDir = pathObj.parent_path().string();
  std::string scfFilePath = resultDir + "/cf_dyn_transformed.mlir";
  llvm::dbgs() << "[DEBUG] \tResult Dir : " << resultDir << "\n";

  // Step 0: Parse the SCF level mlir file to get the op name vector
  SCFFile scfFile(scfFilePath);
  buildScfToHSMap(switchInfo, scfFile);

  // Step 2: Parse the actual data log file
  parseUnifiedLogFile(dataTrace, switchInfo);
  llvm::dbgs() << "[DEBUG] \t\tDONE\n";

  // Step 3: Construct the map for seg execution count
  constructSegExeCount();

  //! Testing, 01/09/2024
  // for (const auto& [key, value]: execPhaseToSegExecNumMap) {
  //   llvm::dbgs() << "[DEBUG] Seg Name: " << key << "\n";

  //   llvm::dbgs() << "[DEBUG] \tValue: " << value.first << ", Iter Index: " << value.second << "\n";
    
  // }
}





void SCFProfilingResult::parseUnifiedLogFile(StringRef tracePath, SwitchingInfo& switchInfo) {
  llvm::dbgs() << "[DEBUG] \t[PARSING UNIFIED TRACE LOG FILE]\n";

  // STEP 0: Initialize structures
  std::vector<unsigned> tmpMGTrace;
  std::vector<unsigned> tmpBBTrace;
  int numTransSections = 0;
  executedBBTrace.clear();
  executedBBTrace.push_back(0);    // Always start with BB 0
  tmpBBTrace.push_back(0);
  std::vector<std::vector<unsigned>> transactionBBLists;
  unsigned curIter = 0;

  // Read entire trace into memory
  std::vector<std::string> lines;
  {
    std::ifstream file(tracePath.str());
    if (!file.is_open()) {
      llvm::errs() << "[ERROR] Can't open file " << tracePath << "\n";
      return;
    }
    std::string rawLine;
    while (std::getline(file, rawLine))
      lines.push_back(rawLine);
    file.close();
  }

  // PASS 1: Build BB trace from [Edge] / [BEdge]
  for (auto &rawLine : lines) {
    std::string line = strip(rawLine, "");
    auto parts = splitf(line, ' ');
    if (parts[0] == "[Edge]") {
      std::string tuple = strip(parts.back(), "(");
      tuple = strip(tuple, ")");
      auto edgeTuple = splitf(tuple, ',');
      unsigned dst = std::stoul(edgeTuple[1]);
      executedBBTrace.push_back(dst);
      tmpBBTrace.push_back(dst);
    } else if (parts[0] == "[BEdge]") {
      std::string tuple = strip(parts.back(), "(");
      tuple = strip(tuple, ")");
      auto backEdgeTuple = splitf(tuple, ',');
      unsigned srcBB = std::stoul(backEdgeTuple[0]);
      unsigned dstBB = std::stoul(backEdgeTuple[1]);
      executedBBTrace.push_back(dstBB);
      // determine MG
      auto CFDFCVec = switchInfo.staticinfo.backEdgeToCFDFC[{srcBB, dstBB}];
      unsigned travMG = 0;
      if (CFDFCVec.size() > 1) {
        bool matched = false;
        unsigned bestSize = 0;
        for (auto selMG : CFDFCVec) {
          auto &bbList = switchInfo.staticinfo.segToBBs[std::to_string(selMG)];
          std::set<unsigned> mgSet(bbList.begin(), bbList.end());
          std::set<unsigned> traceSet(tmpBBTrace.begin(), tmpBBTrace.end());
          if (std::includes(traceSet.begin(), traceSet.end(), mgSet.begin(), mgSet.end())) {
            matched = true;
            if (mgSet.size() > bestSize) {
              bestSize = mgSet.size();
              travMG = selMG;
            }
          }
        }
        if (!matched)
          llvm::errs() << "[ERROR] Can't match backedge (" << srcBB << "," << dstBB << ")\n";
      } else {
        travMG = CFDFCVec[0];
      }
      tmpMGTrace.push_back(travMG);
      tmpBBTrace = {dstBB};
    }
  }

  // STEP 2: Partition the BB trace into segments and build iterEndIndex
  unsigned segmentIter = 0, ptr = 0;
  for (auto selMG : tmpMGTrace) {
    auto &mgBBs = switchInfo.staticinfo.segToBBs[std::to_string(selMG)];
    if (segmentIter == 0) {
      std::vector<unsigned> startBBs;
      while (ptr < executedBBTrace.size() &&
             std::find(mgBBs.begin(), mgBBs.end(), executedBBTrace[ptr]) == mgBBs.end()) {
        startBBs.push_back(executedBBTrace[ptr++]);
      }
      switchInfo.staticinfo.segToBBs["S"] = startBBs;
      executedSegTrace.push_back("S");
      iterEndIndex.push_back(ptr - 1);
      segmentIter++;
      ptr += mgBBs.size();
      executedSegTrace.push_back(std::to_string(selMG));
      iterEndIndex.push_back(ptr - 1);
      segmentIter++;
    } else {
      if (std::find(mgBBs.begin(), mgBBs.end(), executedBBTrace[ptr]) == mgBBs.end()) {
        std::vector<unsigned> transBBs;
        while (ptr < executedBBTrace.size() &&
               std::find(mgBBs.begin(), mgBBs.end(), executedBBTrace[ptr]) == mgBBs.end()) {
          transBBs.push_back(executedBBTrace[ptr++]);
        }
        bool found = false;
        for (size_t i = 0; i < transactionBBLists.size(); ++i) {
          if (transactionBBLists[i] == transBBs) {
            executedSegTrace.push_back("T" + std::to_string(i));
            found = true;
            break;
          }
        }
        iterEndIndex.push_back(ptr - 1);
        curIter++;
        if (!found) {
          transactionBBLists.push_back(transBBs);
          executedSegTrace.push_back("T" + std::to_string(numTransSections));
          switchInfo.staticinfo.segToBBs["T" + std::to_string(numTransSections)] = transBBs;
          switchInfo.staticinfo.transToSucMGMap["T" + std::to_string(numTransSections)] = std::to_string(selMG);
          numTransSections++;
        }
      }
      ptr += mgBBs.size();
      executedSegTrace.push_back(std::to_string(selMG));
      iterEndIndex.push_back(ptr - 1);
      segmentIter++;
    }
  }
  // Ending segment
  std::vector<unsigned> endBBs(executedBBTrace.begin() + ptr, executedBBTrace.end());
  executedSegTrace.push_back("E");
  switchInfo.staticinfo.segToBBs["E"] = endBBs;

  // Build bbToIterMap
  unsigned mapCounter = 0, iterCounter = 0;
  for (auto &seg : executedSegTrace) {
    for (size_t i = 0; i < switchInfo.staticinfo.segToBBs[seg].size(); ++i) {
      bbToIterMap[mapCounter++] = iterCounter;
    }
    iterCounter++;
  }

  // PASS 2: Data parsing with curIter bumps on segment boundaries
  curIter = 0;
  for (auto &rawLine : lines) {
    std::string line = strip(rawLine, "");
    auto parts = splitf(line, ' ');
    if (parts[0] == "[DATA]") {
      std::string valStr = strip(parts[1], "(");
      valStr = strip(valStr, ")");
      auto tup = splitf(valStr, ',');
      std::string name = strip(tup[0], "\"");
      std::string hsName = scfToHandshakeNameMap[name];
      int val = tup.size() > 1 ? std::stoi(tup.back()) : 0;
      insertValuePair(val, curIter, hsName);
    } else if (parts[0] == "[ARG]") {
      std::string valStr = strip(parts[1], "(");
      valStr = strip(valStr, ")");
      auto tup = splitf(valStr, ',');
      std::string arg = strip(tup[0], "\"");
      int val = std::stoi(tup.back());
      insertValuePair(val, curIter, arg);
      argNamesVec.push_back(arg);
    } else if (parts[0] == "[Edge]" || parts[0] == "[BEdge]") {
      unsigned idx = std::stoul(parts[1]);
      if (std::find(iterEndIndex.begin(), iterEndIndex.end(), idx) != iterEndIndex.end()) {
        ++curIter;
      }
    }
  }
}









void SCFProfilingResult::buildScfToHSMap(SwitchingInfo& switchInfo, SCFFile& scfFile) {
  for (size_t i = 0; i < switchInfo.staticinfo.funcOpNames.size(); i++) {
    std::string scfName = scfFile.opNameList[i];
    std::string hsName = switchInfo.staticinfo.funcOpNames[i];

    scfToHandshakeNameMap[scfName] = hsName;

    // llvm::dbgs() << "[DEBUG] \t SCF OP Name : " << scfName << "; Handshake Name: " << hsName << "\n";
  }
}

//===----------------------------------------------------------------------===//
//
// Helper Class
//
//===----------------------------------------------------------------------===//
SCFFile::SCFFile(StringRef scfFile) {
  // Define the matching pattern
  std::regex nonNameExpr("[^a-zA-Z_0-9]+");
  std::ifstream file(scfFile.str());

  if (!file.is_open()) {
    llvm::errs() << "[ERROR] Can't Open file " << scfFile << "\n";
    return;
  }

  // Get the operation name
  std::string line;
  while (std::getline(file, line)) {
    line = strip(line, "");
    auto lineSplit = split(line, " ");

    if ((lineSplit[0].find("%") != std::string::npos) || (lineSplit[0].find("memref") != std::string::npos)) {
      std::regex handshakeExpr("#handshake\\.name.*");
      std::vector<std::string>::iterator iterBegin = lineSplit.begin();
      std::vector<std::string>::iterator iterEnd = lineSplit.end();

      // Iterate through the found line
      for (std::vector<std::string>::iterator vecIter = iterBegin; vecIter != iterEnd; vecIter++) {
        if (*vecIter == "{handshake.name" || *vecIter == "handshake.name") {
          // Handshake name matched
          // TODO: Change the name analysis process to make sure the name of the ALU is not changed
          vecIter += 2;

          // If not constant
          // Remove unwanted characters
          std::string tmpName = std::regex_replace(*vecIter, nonNameExpr, "");
          
          if (tmpName.find("constant") == std::string::npos) {
            // Also remove all index_cast node
            if (tmpName.find("index_cast") == std::string::npos) {
              opNameList.push_back(tmpName);
            }
          }
        }
      }
    }
  }
  // Close the file
  file.close();
}

void SCFProfilingResult::constructSegExeCount() {
  unsigned segCounter = 0;
  unsigned numExecPhase = 0;
  unsigned globalCounter = 0;
  std::string prevSeg = "S";

  for (const auto& selSeg: executedSegTrace) {
    if (selSeg != prevSeg) {
      execPhaseToSegExecNumMap[numExecPhase] = std::make_pair(prevSeg, segCounter);
      prevSeg = selSeg;
      numExecPhase++;
      segCounter = 1;
    } else {
      segCounter++;
    }

    if (segToStartIterIndexMap.find(selSeg) == segToStartIterIndexMap.end()) {
      segToStartIterIndexMap[selSeg] = globalCounter;
    }

    // Update the globalCounter
    globalCounter++;
  }

  // Add the ending section
  execPhaseToSegExecNumMap[numExecPhase] = std::make_pair("E", 1);
}

void SCFProfilingResult::insertValuePair(int opValue, unsigned iterIndex, std::string opName) {
  // Check wheter the key exist in the map or not
  if (opNameToValueListMap.find(opName) != opNameToValueListMap.end()) {
    opNameToValueListMap[opName].push_back(std::make_pair(opValue, iterIndex));
  } else {
    std::vector<std::pair<int, unsigned>> newVector = {std::make_pair(opValue, iterIndex)};
    opNameToValueListMap[opName] = newVector;
  }
}