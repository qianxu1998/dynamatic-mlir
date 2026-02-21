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

#include "dynamatic/Support/Attribute.h"
#include "dynamatic/Support/Backedge.h"
#include "dynamatic/Support/DynamaticPass.h"
#include "dynamatic/Support/LLVM.h"
#include "dynamatic/Support/Logging.h"
#include "experimental/Analysis/SwitchingEstimation/SwitchingSupport.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>

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
  SCFProfilingResult(StringRef dataTrace, SwitchingInfo &switchInfo);

  // Parse the data trace file
  void parseUnifiedLogFile(StringRef tracePath, SwitchingInfo &switchInfo);

  // This function insert new (value, iter index) pair to the value list
  void insertValuePair(int opValue, unsigned iterIndex, std::string opName);

  // This function constructs the overall segment execution counts
  void constructSegExeCount();

  // Global storage extracted from one unified profiling trace.
  std::vector<unsigned> executedBasicBlockTrace;
  std::vector<unsigned> segmentEndEdgeIndices;
  std::vector<std::string> executedSegmentTrace;
  // Edge index in `executedBasicBlockTrace` -> segment iteration index.
  std::map<unsigned, unsigned> edgeIndexToIterationMap;
  // Execution phase index -> (segment label, number of consecutive executions).
  // Example: 5 -> ("2", 3) means phase #5 executed segment "2" three times.
  std::map<unsigned, std::pair<std::string, unsigned>>
      executionPhaseToSegmentExecCount;
  // Handshake node name -> [(value, iteration index), ...].
  std::map<std::string, std::vector<std::pair<int, unsigned>>> nodeToValueTrace;

  // Map from SCF level op name to Handshake level op name
  std::map<std::string, std::string> scfToHandshakeNameMap;

  // Map from the segment label to the first iteration that the segment starts
  // execution
  std::map<std::string, unsigned> segmentToFirstIteration;

  // List of traced argument names in the order they appear in the log.
  std::vector<std::string> argumentNames;
};

//===----------------------------------------------------------------------===//
//
//  Helper Functions
//
//===----------------------------------------------------------------------===//

// This function split a given string into a vector based on the delimiter
std::vector<std::string> customSplit(const std::string &s,
                                     const std::string &delimiter);

// This function removes the starting and ending empty space
std::string customStrip(const std::string &inputStr,
                        const std::string &toRemove);

#endif // EXPERIMENTAL_ANALYSIS_SWITCHINGESTIMATION_PROFILINGANALYZER_H
