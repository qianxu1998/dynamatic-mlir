//===- SwitchingEstimation.h - Switching Estimation -----*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the Analyzer Class to analyze the functional 
// profiling result (Both Data Trace and the BB list).
//
//===----------------------------------------------------------------------===//

#ifndef EXPERIMENTAL_TRANSFORMS_PROFILING_ANALYZER_H
#define EXPERIMENTAL_TRANSFORMS_PROFILING_ANALYZER_H

#include "experimental/Transforms/Switching/SwitchingSupport.h"
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

using namespace mlir;
using namespace dynamatic;
using namespace dynamatic::handshake;

//===----------------------------------------------------------------------===//
//
//  Helper Class
//
//===----------------------------------------------------------------------===//

class SCFFile {
public:
  SCFFile(StringRef scfFile);

  // Vector storing all SCF level op names in the mlir file
  std::vector<std::string> opNameList;

};

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



  // Parse the  data profiling trace log file

  void parseUnifiedLogFile(StringRef dataTrace, SwitchingInfo& switchInfo);
  // Construct the map from scf level profiling to handshake level IR
  void buildScfToHSMap(SwitchingInfo& switchInfo, SCFFile& scfFile);

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


// This function split a given string into a vector based on the delimiter
// std::vector<std::string> split(const std::string &s, const std::string& delimiter);

// This function removes the starting and ending empty space
// std::string strip(const std::string &inputStr, const std::string &toRemove);

inline std::vector<std::string> split(const std::string &s,
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

inline std::vector<std::string> splitf(const std::string &s,
  char delimiter) {
std::vector<std::string> tokens;
// std::regex re(delimiter);
std::istringstream tokenstream(s);
std::string token;
// std::sregex_token_iterator it(s.begin(), s.end(), re, -1);
// std::sregex_token_iterator end;

while (std::getline(tokenstream,token,delimiter)) {
tokens.push_back(token);
}

return tokens;
}

inline void splitfast(const std::string &s, char delim,
  std::vector<std::string_view> &out) {
out.clear();
size_t start = 0;
while (true) {
size_t pos = s.find(delim, start);
out.emplace_back(s.data() + start, pos == std::string::npos ? 
                   s.size() - start : pos - start);
if (pos == std::string::npos) break;
start = pos + 1;
}
}

inline std::string strip(const std::string &inputStr, const std::string &toRemove) {
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
inline std::string trim(const std::string &s) {
  auto start = s.find_first_not_of(" \t\n\r\f\v");
  auto end = s.find_last_not_of(" \t\n\r\f\v");
  return (start == std::string::npos) ? "" : s.substr(start, end - start + 1);
}






static inline void split_sv(const std::string &in,
                            char delim,
                            std::vector<std::string_view> &out) {
  out.clear();
  auto str = std::string_view(in);
  size_t start = 0;
  while (start <= str.size()) {
    size_t pos = str.find(delim, start);
    if (pos == std::string_view::npos) {
      out.emplace_back(str.data() + start, str.size() - start);
      break;
    }
    out.emplace_back(str.data() + start, pos - start);
    start = pos + 1;
  }
}




#endif // EXPERIMENTAL_TRANSFORMS_PROFILING_ANALYZER_H


