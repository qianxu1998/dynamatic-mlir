//===- ProfilingAnalyzer.cpp - Estimate Switching Activities ------*- C++ -*-===//
//
// Implements the Analyzer Class for the functional profiling results
//
//===----------------------------------------------------------------------===//

#include "experimental/Analysis/SwitchingEstimation/ProfilingAnalyzer.h"

#include <cassert>

using namespace llvm;
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

  // Step 1: Parse the actual data log file
  parseUnifiedLogFile(dataTrace, switchInfo);
  llvm::dbgs() << "[DEBUG] \t\tDONE\n";

  // Step 2: Construct the map for seg execution count
  constructSegExeCount();
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

void SCFProfilingResult::parseUnifiedLogFile(StringRef tracePath, SwitchingInfo& switchInfo) {
  LLVM_DEBUG(llvm::dbgs() << "[DEBUG] [Step 1] [PARSING UNIFIED TRACE LOG FILE]\n");

  // STEP 0: Initialize structures
  std::vector<unsigned> tmpMGTrace;
  std::vector<unsigned> tmpBBTrace; // List of traversed BBs
  int numTransSections = 0;
  executedBBTrace.clear();
  executedBBTrace.push_back(0);    // Always start with BB 0
  tmpBBTrace.push_back(0);

  // Record all transaction section
  std::vector<std::vector<unsigned>> transactionBBLists;
  
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
    std::string line = customStrip(rawLine, "");
    auto parts = customSplit(line, " ");

    if (parts[0] == "[Edge]") {
      std::string tuple = customStrip(parts.back(), "(");
      tuple = customStrip(tuple, ")");

      auto edgeTuple = customSplit(tuple, ",");
      unsigned dst = std::stoul(edgeTuple[1]);

      executedBBTrace.push_back(dst);
      tmpBBTrace.push_back(dst);
    } else if (parts[0] == "[BEdge]") {
      std::string tuple = customStrip(parts.back(), "(");
      tuple = customStrip(tuple, ")");

      auto backEdgeTuple = customSplit(tuple, ",");
      unsigned srcBB = std::stoul(backEdgeTuple[0]);
      unsigned dstBB = std::stoul(backEdgeTuple[1]);
      executedBBTrace.push_back(dstBB);

      // determine MG
      auto CFDFCVec = switchInfo.staticinfo.backEdgeToCFDFC[std::make_pair(srcBB, dstBB)];
      unsigned travMG = 0;
      if (CFDFCVec.size() > 1) {
        // Multiple MGs, need to match the BB trace
        bool matched = false;
        unsigned bestSize = 0;

        for (auto selMG : CFDFCVec) {
          // Get the BB list for this MG
          // We choose the one with the most number of bbs matched with the selBBList
          auto &bbList = switchInfo.staticinfo.segToBBs[std::to_string(selMG)];
          // Create sets from traversed BB and the potential MG
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

  // STEP 2: Partition the executedBB Trace into different sections
  // Identify all segments in the execution trace
  // Including both MGs and Transitions sections, which will not be repeatedly executed
  unsigned curIter = 0, tracePointer = 0;

  for (auto selMG : tmpMGTrace) {
    auto &mgBBs = switchInfo.staticinfo.segToBBs[std::to_string(selMG)];

    if (curIter == 0) {
      std::vector<unsigned> startBBs;

      while (tracePointer < executedBBTrace.size() &&
             std::find(mgBBs.begin(), mgBBs.end(), executedBBTrace[tracePointer]) == mgBBs.end()) {
        startBBs.push_back(executedBBTrace[tracePointer++]);
      }

      // Update the CFDFC BB storing dict
      switchInfo.staticinfo.segToBBs["S"] = startBBs;

      // Store the end of the initialization
      executedSegTrace.push_back("S");

      // Update the interation end status
      iterEndIndex.push_back(tracePointer - 1);
      curIter++;

      // First Marked Graph entered, update the trace pointer
      tracePointer += mgBBs.size();
      executedSegTrace.push_back(std::to_string(selMG));
      iterEndIndex.push_back(tracePointer - 1);
      curIter++;
    } else {
      // Check whether a transition section is encountered
      if (std::find(mgBBs.begin(), mgBBs.end(), executedBBTrace[tracePointer]) == mgBBs.end()) {
        std::vector<unsigned> transBBs;

        while (tracePointer < executedBBTrace.size() &&
               std::find(mgBBs.begin(), mgBBs.end(), executedBBTrace[tracePointer]) == mgBBs.end()) {
          transBBs.push_back(executedBBTrace[tracePointer++]);
        }

        bool found = false;
        for (size_t i = 0; i < transactionBBLists.size(); ++i) {
          if (transactionBBLists[i] == transBBs) {
            executedSegTrace.push_back("T" + std::to_string(i));
            found = true;
            break;
          }
        }

        iterEndIndex.push_back(tracePointer - 1);
        curIter++;

        if (!found) {
          std::string segName = "T" + std::to_string(numTransSections);

          transactionBBLists.push_back(transBBs);
          executedSegTrace.push_back(segName);
          switchInfo.staticinfo.segToBBs[segName] = transBBs;
          switchInfo.staticinfo.transToSucMGMap[segName] = std::to_string(selMG);
          numTransSections++;
        }
      }

      tracePointer += mgBBs.size();
      executedSegTrace.push_back(std::to_string(selMG));
      iterEndIndex.push_back(tracePointer - 1);
      curIter++;
    }
  }

  // Ending segment, "E"
  std::vector<unsigned> endBBs(executedBBTrace.begin() + tracePointer, executedBBTrace.end());
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
    std::string line = customStrip(rawLine, "");
    auto parts = customSplit(line, " ");

    if (parts[0] == "[DATA]") {
      std::string valStr = customStrip(parts[1], "(");
      valStr = customStrip(valStr, ")");

      auto tup = customSplit(valStr, ",");
      std::string hsName = customStrip(tup[0], "\"");
      
      if (parts.size() > 1) {
        int opValue = std::stoi(tup.back());
        insertValuePair(opValue, curIter, hsName);
      } else {
        continue;
      }
    } else if (parts[0] == "[ARG]") {
      // Get the (op_name, value) tuple
      std::string valStr = customStrip(parts[1], "(");
      valStr = customStrip(valStr, ")");

      auto tup = customSplit(valStr, ",");
      std::string arg = customStrip(tup[0], "\"");

      int val = std::stoi(tup.back());
      insertValuePair(val, curIter, arg);

      // Record the name of the argument
      argNamesVec.push_back(arg);
    } else if (parts[0] == "[Edge]" || parts[0] == "[BEdge]") {
      unsigned idx = std::stoul(parts[1]);
      if (std::find(iterEndIndex.begin(), iterEndIndex.end(), idx) != iterEndIndex.end()) {
        ++curIter;
      }
    }
  }
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

//===----------------------------------------------------------------------===//
//
//  Helper Functions
//
//===----------------------------------------------------------------------===//

std::vector<std::string> customSplit(const std::string &s,
                               const std::string &delimiter) {
  std::vector<std::string> tokens;
  std::regex re(delimiter);
  std::sregex_token_iterator it(s.begin(), s.end(), re, -1);
  std::sregex_token_iterator end;

  for (; it != end; ++it) {
    tokens.push_back(it->str());
  }

  return tokens;
}

std::string customStrip(const std::string &inputStr, const std::string &toRemove) {
  // Trim leading and trailing whitespace
  size_t start = inputStr.find_first_not_of(" \t\n\r\f\v");
  if (start == std::string::npos) {
    return ""; // Return an empty string if there are only whitespaces
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

