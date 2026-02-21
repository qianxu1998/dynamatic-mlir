//===- ProfilingAnalyzer.cpp - Estimate Switching Activities ------*- C++
//-*-===//
//
// Implements the Analyzer Class for the functional profiling results
//
//===----------------------------------------------------------------------===//

#include "experimental/Analysis/SwitchingEstimation/ProfilingAnalyzer.h"

#include <cassert>
#include <cctype>
#include <cerrno>
#include <cstdlib>
#include <limits>

using namespace llvm;
using namespace mlir;
using namespace dynamatic;
using namespace dynamatic::handshake;

// Constructor for the SCF parsing class
SCFProfilingResult::SCFProfilingResult(StringRef dataTrace,
                                       SwitchingInfo &switchInfo) {
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

void SCFProfilingResult::insertValuePair(int opValue, unsigned iterIndex,
                                         std::string opName) {
  // Check wheter the key exist in the map or not
  if (opNameToValueListMap.find(opName) != opNameToValueListMap.end()) {
    opNameToValueListMap[opName].push_back(std::make_pair(opValue, iterIndex));
  } else {
    std::vector<std::pair<int, unsigned>> newVector = {
        std::make_pair(opValue, iterIndex)};
    opNameToValueListMap[opName] = newVector;
  }
}

void SCFProfilingResult::parseUnifiedLogFile(StringRef tracePath,
                                             SwitchingInfo &switchInfo) {
  
      llvm::dbgs() << "[DEBUG] [Step 1] [PARSING UNIFIED TRACE LOG FILE]\n";

  // STEP 0: Initialize structures
  iterEndIndex.clear();
  executedSegTrace.clear();
  bbToIterMap.clear();
  opNameToValueListMap.clear();
  argNamesVec.clear();

  std::vector<unsigned> tmpMGTrace;
  std::vector<unsigned> tmpBBTrace; // List of traversed BBs
  int numTransSections = 0;
  executedBBTrace.clear();
  executedBBTrace.push_back(0); // Always start with BB 0
  tmpBBTrace.push_back(0);

  // Record all transaction section
  std::vector<std::vector<unsigned>> transactionBBLists;
  std::set<std::pair<unsigned, unsigned>> warnedUnmatchedBackedges;

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

  auto parseUnsigned = [](const std::string &token, unsigned &out) -> bool {
    if (token.empty())
      return false;
    char *end = nullptr;
    errno = 0;
    unsigned long value = std::strtoul(token.c_str(), &end, 10);
    if (errno != 0 || end == token.c_str() || *end != '\0' ||
        value > std::numeric_limits<unsigned>::max())
      return false;
    out = static_cast<unsigned>(value);
    return true;
  };

  auto parseSigned = [](const std::string &token, int &out) -> bool {
    if (token.empty())
      return false;
    char *end = nullptr;
    errno = 0;
    long value = std::strtol(token.c_str(), &end, 10);
    if (errno != 0 || end == token.c_str() || *end != '\0' ||
        value < std::numeric_limits<int>::min() ||
        value > std::numeric_limits<int>::max())
      return false;
    out = static_cast<int>(value);
    return true;
  };

  auto chooseFallbackMGForDst = [&](unsigned dstBB) -> unsigned {
    for (const auto &[segLabel, bbList] : switchInfo.staticinfo.segToBBs) {
      if (segLabel.empty() || !std::isdigit(segLabel[0]))
        continue;
      if (std::find(bbList.begin(), bbList.end(), dstBB) != bbList.end()) {
        unsigned mgIdx = 0;
        if (parseUnsigned(segLabel.str(), mgIdx))
          return mgIdx;
      }
    }
    return 0;
  };

  auto belongsToAnyMG = [&](unsigned bb) -> bool {
    for (const auto &[segLabel, bbList] : switchInfo.staticinfo.segToBBs) {
      if (segLabel.empty() || !std::isdigit(segLabel[0]))
        continue;
      if (std::find(bbList.begin(), bbList.end(), bb) != bbList.end())
        return true;
    }
    return false;
  };

  auto filterToTransitionBBs = [&](const std::vector<unsigned> &bbs) {
    std::vector<unsigned> filtered;
    filtered.reserve(bbs.size());
    for (unsigned bb : bbs) {
      if (!belongsToAnyMG(bb))
        filtered.push_back(bb);
    }
    return filtered;
  };

  // PASS 1: Build BB trace from [Edge] / [BEdge]
  for (auto &rawLine : lines) {
    std::string line = customStrip(rawLine, "");
    auto parts = customSplit(line, " ");
    if (parts.empty() || parts[0].empty())
      continue;

    if (parts[0] == "[Edge]") {
      if (parts.size() < 2)
        continue;
      std::string tuple = customStrip(parts.back(), "(");
      tuple = customStrip(tuple, ")");

      auto edgeTuple = customSplit(tuple, ",");
      if (edgeTuple.size() < 2)
        continue;

      unsigned dst = 0;
      if (!parseUnsigned(edgeTuple[1], dst))
        continue;

      executedBBTrace.push_back(dst);
      tmpBBTrace.push_back(dst);
    } else if (parts[0] == "[BEdge]") {
      if (parts.size() < 2)
        continue;
      std::string tuple = customStrip(parts.back(), "(");
      tuple = customStrip(tuple, ")");

      auto backEdgeTuple = customSplit(tuple, ",");
      if (backEdgeTuple.size() < 2)
        continue;

      unsigned srcBB = 0, dstBB = 0;
      if (!parseUnsigned(backEdgeTuple[0], srcBB) ||
          !parseUnsigned(backEdgeTuple[1], dstBB))
        continue;
      executedBBTrace.push_back(dstBB);

      // determine MG
      std::vector<unsigned> CFDFCVec;
      if (auto it = switchInfo.staticinfo.backEdgeToCFDFC.find(
              std::make_pair(srcBB, dstBB));
          it != switchInfo.staticinfo.backEdgeToCFDFC.end())
        CFDFCVec = it->second;

      unsigned travMG = chooseFallbackMGForDst(dstBB);
      bool matched = false;

      if (CFDFCVec.size() == 1) {
        travMG = CFDFCVec[0];
        matched = true;
      } else if (CFDFCVec.size() > 1) {
        // Multiple MGs, need to match the BB trace
        unsigned bestOverlap = 0;

        for (auto selMG : CFDFCVec) {
          // Get the BB list for this MG
          auto segIt =
              switchInfo.staticinfo.segToBBs.find(std::to_string(selMG));
          if (segIt == switchInfo.staticinfo.segToBBs.end())
            continue;
          auto &bbList = segIt->second;

          unsigned overlap = 0;
          for (auto bb : tmpBBTrace) {
            if (std::find(bbList.begin(), bbList.end(), bb) != bbList.end())
              overlap++;
          }

          if (overlap > bestOverlap) {
            bestOverlap = overlap;
            travMG = selMG;
            matched = true;
          }
        }
        if (!matched)
          llvm::errs() << "[ERROR] Can't match backedge (" << srcBB << ","
                       << dstBB << ")\n";
      }

      if (!matched && warnedUnmatchedBackedges.insert({srcBB, dstBB}).second) {
        llvm::errs() << "[WARNING] Can't match backedge (" << srcBB << ","
                     << dstBB << "), fallback MG = " << travMG << "\n";
      }
      tmpMGTrace.push_back(travMG);
      tmpBBTrace = {dstBB};
    }
  }

  // STEP 2: Partition the executedBB Trace into different sections
  // Identify all segments in the execution trace
  // Including both MGs and Transitions sections, which will not be repeatedly
  // executed
  unsigned curIter = 0, tracePointer = 0;

  for (auto selMG : tmpMGTrace) {
    auto segIt = switchInfo.staticinfo.segToBBs.find(std::to_string(selMG));
    if (segIt == switchInfo.staticinfo.segToBBs.end() || segIt->second.empty())
      continue;
    auto &mgBBs = segIt->second;

    if (tracePointer >= executedBBTrace.size())
      break;

    if (curIter == 0) {
      std::vector<unsigned> startBBs;

      while (tracePointer < executedBBTrace.size() &&
             std::find(mgBBs.begin(), mgBBs.end(),
                       executedBBTrace[tracePointer]) == mgBBs.end()) {
        startBBs.push_back(executedBBTrace[tracePointer++]);
      }

      // Keep S as true pre-MG transition BBs only.
      switchInfo.staticinfo.segToBBs["S"] = filterToTransitionBBs(startBBs);

      // Store the end of the initialization
      executedSegTrace.push_back("S");

      // Update the interation end status
      iterEndIndex.push_back(tracePointer == 0 ? 0 : tracePointer - 1);
      curIter++;

      // First Marked Graph entered, update the trace pointer
      tracePointer =
          std::min(tracePointer + std::max<unsigned>(1, mgBBs.size()),
                   static_cast<unsigned>(executedBBTrace.size()));
      executedSegTrace.push_back(std::to_string(selMG));
      iterEndIndex.push_back(tracePointer == 0 ? 0 : tracePointer - 1);
      curIter++;
    } else {
      // Check whether a transition section is encountered
      if (tracePointer >= executedBBTrace.size())
        break;
      // Check whether a transition section is encountered
      if (std::find(mgBBs.begin(), mgBBs.end(),
                    executedBBTrace[tracePointer]) == mgBBs.end()) {
        std::vector<unsigned> transBBs;

        while (tracePointer < executedBBTrace.size() &&
               std::find(mgBBs.begin(), mgBBs.end(),
                         executedBBTrace[tracePointer]) == mgBBs.end()) {
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

        iterEndIndex.push_back(tracePointer == 0 ? 0 : tracePointer - 1);
        curIter++;

        if (!found) {
          std::string segName = "T" + std::to_string(numTransSections);

          transactionBBLists.push_back(transBBs);
          executedSegTrace.push_back(segName);
          switchInfo.staticinfo.segToBBs[segName] =
              filterToTransitionBBs(transBBs);
          switchInfo.staticinfo.transToSucMGMap[segName] =
              std::to_string(selMG);
          numTransSections++;
        }
      }

      tracePointer =
          std::min(tracePointer + std::max<unsigned>(1, mgBBs.size()),
                   static_cast<unsigned>(executedBBTrace.size()));
      executedSegTrace.push_back(std::to_string(selMG));
      iterEndIndex.push_back(tracePointer == 0 ? 0 : tracePointer - 1);
      curIter++;
    }
  }

  // Ending segment, "E"
  tracePointer =
      std::min(tracePointer, static_cast<unsigned>(executedBBTrace.size()));
  std::vector<unsigned> endBBs(executedBBTrace.begin() + tracePointer,
                               executedBBTrace.end());
  executedSegTrace.push_back("E");
  // Keep E as true post-MG transition BBs only.
  switchInfo.staticinfo.segToBBs["E"] = filterToTransitionBBs(endBBs);

  {
    llvm::dbgs() << "[DEBUG] [Profiling] segToBBs summary:\n";
    for (const auto &[segLabelRef, bbList] : switchInfo.staticinfo.segToBBs) {
      std::set<unsigned> uniq(bbList.begin(), bbList.end());
      llvm::dbgs() << "[DEBUG]   seg=" << segLabelRef
                   << " bb_count=" << bbList.size() << " bb_unique={";
      bool first = true;
      for (unsigned bb : uniq) {
        if (!first)
          llvm::dbgs() << ",";
        llvm::dbgs() << bb;
        first = false;
      }
      llvm::dbgs() << "}\n";
    }
  };

  // Build edge-index -> execution-iteration map from the concrete
  // segment-boundary indices observed in the trace.
  //
  // `iterEndIndex` stores the last edge index of each executed segment.
  // Data propagation queries this map with edge indices derived from the
  // executed BB trace.
  unsigned edgeIter = 0;
  const unsigned numMapEntries =
      executedBBTrace.empty()
          ? 0U
          : static_cast<unsigned>(executedBBTrace.size() - 1);
  for (unsigned edgeIdx = 0; edgeIdx <= numMapEntries; ++edgeIdx) {
    while ((edgeIter + 1) < iterEndIndex.size() &&
           edgeIdx > iterEndIndex[edgeIter]) {
      ++edgeIter;
    }
    bbToIterMap[edgeIdx] = edgeIter;
  }

  // PASS 2: Data parsing with curIter bumps on segment boundaries
  curIter = 0;
  for (auto &rawLine : lines) {
    std::string line = customStrip(rawLine, "");
    auto parts = customSplit(line, " ");
    if (parts.empty() || parts[0].empty())
      continue;

    if (parts[0] == "[DATA]") {
      if (parts.size() < 2)
        continue;
      std::string valStr = customStrip(parts[1], "(");
      valStr = customStrip(valStr, ")");

      auto tup = customSplit(valStr, ",");
      if (tup.empty())
        continue;
      std::string hsName = customStrip(tup[0], "\"");

      if (parts.size() > 1 && !tup.empty()) {
        int opValue = 0;
        if (!parseSigned(tup.back(), opValue))
          continue;
        insertValuePair(opValue, curIter, hsName);
      } else {
        continue;
      }
    } else if (parts[0] == "[ARG]") {
      if (parts.size() < 2)
        continue;
      // Get the (op_name, value) tuple
      std::string valStr = customStrip(parts[1], "(");
      valStr = customStrip(valStr, ")");

      auto tup = customSplit(valStr, ",");
      if (tup.empty())
        continue;
      std::string arg = customStrip(tup[0], "\"");

      int val = 0;
      if (!parseSigned(tup.back(), val))
        continue;
      insertValuePair(val, curIter, arg);

      // Record the name of the argument
      argNamesVec.push_back(arg);
    } else if (parts[0] == "[Edge]" || parts[0] == "[BEdge]") {
      if (parts.size() < 2)
        continue;
      unsigned idx = 0;
      if (!parseUnsigned(parts[1], idx))
        continue;
      if (std::find(iterEndIndex.begin(), iterEndIndex.end(), idx) !=
          iterEndIndex.end()) {
        ++curIter;
      }
    }
  }
}

void SCFProfilingResult::constructSegExeCount() {
  unsigned segCounter = 0;
  unsigned numExecPhase = 0;
  unsigned globalCounter = 0;
  std::string prevSeg = "";

  execPhaseToSegExecNumMap.clear();
  segToStartIterIndexMap.clear();

  if (executedSegTrace.empty()) {
    execPhaseToSegExecNumMap[numExecPhase] = std::make_pair("E", 1);
    return;
  }

  prevSeg = executedSegTrace.front();

  for (const auto &selSeg : executedSegTrace) {
    if (selSeg != prevSeg) {
      execPhaseToSegExecNumMap[numExecPhase] =
          std::make_pair(prevSeg, segCounter);
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

  // Store the final real execution phase. `executedSegTrace` already contains
  // the terminal "E" segment, so appending another synthetic ending phase here
  // would double-count teardown transitions.
  execPhaseToSegExecNumMap[numExecPhase] = std::make_pair(prevSeg, segCounter);

  {
    llvm::dbgs() << "[DEBUG] [Profiling] executedSegTrace size: "
                 << executedSegTrace.size() << "\n";
    llvm::dbgs() << "[DEBUG] [Profiling] exec phases:\n";
    for (const auto &[phase, segExec] : execPhaseToSegExecNumMap) {
      llvm::dbgs() << "[DEBUG]   phase=" << phase << " seg=" << segExec.first
                   << " count=" << segExec.second << "\n";
    }
    llvm::dbgs() << "[DEBUG] [Profiling] segToStartIterIndexMap:\n";
    for (const auto &[seg, idx] : segToStartIterIndexMap)
      llvm::dbgs() << "[DEBUG]   seg=" << seg << " firstIter=" << idx << "\n";
  };
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

std::string customStrip(const std::string &inputStr,
                        const std::string &toRemove) {
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
