#ifndef DYNAMATIC_TRANSFORMS_BUFFERPLACEMENT_CFDFC_CACHE_H
#define DYNAMATIC_TRANSFORMS_BUFFERPLACEMENT_CFDFC_CACHE_H

#include "dynamatic/Analysis/CFDFCAnalysis.h"
#include "dynamatic/Analysis/NameAnalysis.h"
#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Support/LLVM.h"
#include "mlir/IR/BuiltinOps.h"

#include <optional>
#include <string>

namespace dynamatic {
namespace buffer {

struct CFDFCCacheProvenance {
  std::optional<std::string> timingModels;
  std::optional<double> targetPeriod;
  std::optional<std::string> bufferAlgorithm;
};

LogicalResult writeCFDFCCache(ModuleOp modOp, const CFDFCAnalysis &analysis,
                              NameAnalysis &nameAnalysis, StringRef filepath,
                              const CFDFCCacheProvenance &provenance);

LogicalResult readCFDFCCache(ModuleOp modOp, NameAnalysis &nameAnalysis,
                             StringRef filepath,
                             std::map<handshake::FuncOp, ListOfCFDFCs> &out,
                             const CFDFCCacheProvenance &expected);

} // namespace buffer
} // namespace dynamatic

#endif // DYNAMATIC_TRANSFORMS_BUFFERPLACEMENT_CFDFC_CACHE_H
