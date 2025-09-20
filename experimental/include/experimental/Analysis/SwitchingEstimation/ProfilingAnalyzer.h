//===- ProfilingAnalyzer.h - Switching Estimation -----*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the Analyzer Class to analyze the functional 
// profiling result.
//
//===----------------------------------------------------------------------===//

#ifndef EXPERIMENTAL_ANALYSIS_SWITCHINGESTIMATION_PROFILINGANALYZER_H
#define EXPERIMENTAL_ANALYSIS_SWITCHINGESTIMATION_PROFILINGANALYZER_H

#include "experimental/Analysis/SwitchingEstimation/SwitchingSupport.h"
#include "dynamatic/Support/LLVM.h"
#include "dynamatic/Support/DynamaticPass.h"
#include "dynamatic/Support/LLVM.h"
#include "dynamatic/Support/Logging.h"
#include "dynamatic/Support/Backedge.h"
#include "dynamatic/Support/Attribute.h"

#include <algorithm>
#include <cmath>
#include <vector>
#include <iostream>
#include <fstream>
#include <string>
#include <map>
#include <set>
#include <sstream>
#include <filesystem>

using namespace llvm;
using namespace mlir;
using namespace dynamatic;
using namespace dynamatic::handshake;

//===----------------------------------------------------------------------===//
//
//  Profiler Analyzer Class
//
//===----------------------------------------------------------------------===//

// Class used to analyze the scf-level software profiling
// 
class SCFProfilingResult {
public:

  SCFProfilingResult(StringRef dataTrace,
                     SwitchingInfo& switchInfo);

  // Parse the data trace file
  void parseUnifiedLogFile(StringRef tracePath, SwitchingInfo& switchInfo);

  // This function insert new (value, iter index) pair to the value list
  void insertValuePair(int opValue, unsigned iterIndex, std::string opName);

  // This function constructs the overall segment execution counts
  void constructSegExeCount();

  //
  //  Global Storing Structure for analyzing the profiling results
  //
  std::vector<unsigned> executedBBTrace;                      // Vector used to store the execution trace of BB labels
  std::vector<unsigned> iterEndIndex;                         // Vector storing the ending edge index on the boundary of different segments
  std::vector<std::string> executedSegTrace;                  // Vector used to store the execution trace consists of segment labels
  std::map<unsigned, unsigned> bbToIterMap;                   // Map from the index of a BB in the executedBBTrace to the corresponding Iteration index
  std::map<unsigned, std::pair<std::string, unsigned>> execPhaseToSegExecNumMap;     // Map from execution stage to (segLabel, numExec) pair
  std::map<std::string, std::vector<std::pair<int, unsigned>>> opNameToValueListMap; // Map from Hndshake level opName to the list of value in the data profiling process, format: (value, iterIdx)

  // Map from SCF level op name to Handshake level op name
  std::map<std::string, std::string> scfToHandshakeNameMap;

  // Map from the segment label to the first iteration that the segment starts execution
  std::map<std::string, unsigned> segToStartIterIndexMap;

  // Vector of arguments names
  std::vector<std::string> argNamesVec;
};

//===----------------------------------------------------------------------===//
//
//  Helper Functions
//
//===----------------------------------------------------------------------===//

// This function split a given string into a vector based on the delimiter
std::vector<std::string> customSplit(const std::string &s, const std::string& delimiter);

// This function removes the starting and ending empty space
std::string customStrip(const std::string &inputStr, const std::string &toRemove);

#endif // EXPERIMENTAL_ANALYSIS_SWITCHINGESTIMATION_PROFILINGANALYZER_H
