//===- CFDFCCache.cpp - CFDFC cache I/O ------------------------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implements JSON serialization for finalized CFDFC placement data reused by
// switching-estimation.
//
//===----------------------------------------------------------------------===//

#include "dynamatic/Transforms/BufferPlacement/CFDFCCache.h"

#include "dynamatic/Support/JSON/JSON.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"

#include <cmath>

using namespace llvm;
using namespace mlir;
using namespace dynamatic;
using namespace dynamatic::buffer;

namespace llvm_json = llvm::json;

namespace {

constexpr unsigned kSchemaVersion = 1;

struct SerializedChannelDesc {
  std::string producer;
  unsigned resultIndex = 0;
  std::string consumer;
};

struct SerializedUnitOccupancy {
  std::string unit;
  double occupancy = 0.0;
};

struct SerializedCFDFC {
  unsigned index = 0;
  std::vector<unsigned> cycleBBs;
  double throughput = 0.0;
  std::vector<std::string> units;
  std::vector<SerializedChannelDesc> channels;
  std::vector<SerializedChannelDesc> backedges;
  std::vector<SerializedUnitOccupancy> unitOccupancy;
};

struct SerializedFunctionCFDFCs {
  std::string name;
  std::vector<SerializedCFDFC> cfdfcs;
};

static std::string normalizePathForComparison(StringRef path) {
  SmallString<256> normalized(path);
  if (!normalized.empty()) {
    (void)sys::fs::make_absolute(normalized);
    sys::path::remove_dots(normalized, /*remove_dot_dot=*/true);
  }
  return std::string(normalized.str());
}

static std::optional<std::string>
normalizeOptionalPath(const std::optional<std::string> &path) {
  if (!path.has_value() || path->empty())
    return std::nullopt;
  return normalizePathForComparison(*path);
}

static bool parseUnsignedArray(const llvm_json::Value &value,
                               std::vector<unsigned> &out,
                               llvm_json::Path path) {
  const llvm_json::Array *array = value.getAsArray();
  if (!array) {
    path.report(dynamatic::json::ERR_EXPECTED_ARRAY);
    return false;
  }

  out.clear();
  out.reserve(array->size());
  for (auto [idx, element] : enumerate(*array)) {
    unsigned parsed = 0;
    if (!llvm_json::fromJSON(element, parsed, path.index(idx)))
      return false;
    out.push_back(parsed);
  }
  return true;
}

static bool parseStringArray(const llvm_json::Value &value,
                             std::vector<std::string> &out,
                             llvm_json::Path path) {
  const llvm_json::Array *array = value.getAsArray();
  if (!array) {
    path.report(dynamatic::json::ERR_EXPECTED_ARRAY);
    return false;
  }

  out.clear();
  out.reserve(array->size());
  for (auto [idx, element] : enumerate(*array)) {
    std::string parsed;
    if (!llvm_json::fromJSON(element, parsed, path.index(idx)))
      return false;
    out.push_back(std::move(parsed));
  }
  return true;
}

static bool fromJSON(const llvm_json::Value &value,
                     SerializedChannelDesc &channel, llvm_json::Path path) {
  dynamatic::json::ObjectDeserializer mapper(value, path);
  return mapper.map("producer", channel.producer)
      .map("result_index", channel.resultIndex)
      .map("consumer", channel.consumer)
      .exhausted();
}

static bool fromJSON(const llvm_json::Value &value,
                     SerializedUnitOccupancy &occupancy,
                     llvm_json::Path path) {
  dynamatic::json::ObjectDeserializer mapper(value, path);
  return mapper.map("unit", occupancy.unit)
      .map("occupancy", occupancy.occupancy)
      .exhausted();
}

static bool fromJSON(const llvm_json::Value &value, SerializedCFDFC &cfdfc,
                     llvm_json::Path path) {
  const llvm_json::Object *object = value.getAsObject();
  if (!object) {
    path.report(dynamatic::json::ERR_EXPECTED_OBJECT);
    return false;
  }

  dynamatic::json::ObjectDeserializer mapper(*object, path);
  if (!mapper.map("index", cfdfc.index)
           .map("throughput", cfdfc.throughput)
           .map("cycle_bbs",
                std::function<bool(const llvm_json::Value &, llvm_json::Path)>(
                    [&](const llvm_json::Value &arr,
                        llvm_json::Path arrPath) -> bool {
                      return parseUnsignedArray(arr, cfdfc.cycleBBs, arrPath);
                    }))
           .map("units",
                std::function<bool(const llvm_json::Value &, llvm_json::Path)>(
                    [&](const llvm_json::Value &arr,
                        llvm_json::Path arrPath) -> bool {
                      return parseStringArray(arr, cfdfc.units, arrPath);
                    }))
           .map("channels", cfdfc.channels)
           .map("backedges", cfdfc.backedges)
           .map("unit_occupancy", cfdfc.unitOccupancy)
           .exhausted())
    return false;

  return true;
}

static bool fromJSON(const llvm_json::Value &value,
                     SerializedFunctionCFDFCs &func, llvm_json::Path path) {
  dynamatic::json::ObjectDeserializer mapper(value, path);
  return mapper.map("name", func.name).map("cfdfcs", func.cfdfcs).exhausted();
}

static llvm_json::Value toJSON(const SerializedChannelDesc &channel) {
  return llvm_json::Object{{"producer", channel.producer},
                           {"result_index", channel.resultIndex},
                           {"consumer", channel.consumer}};
}

static llvm_json::Value toJSON(const SerializedUnitOccupancy &occupancy) {
  return llvm_json::Object{{"unit", occupancy.unit},
                           {"occupancy", occupancy.occupancy}};
}

static llvm_json::Value toJSON(const SerializedCFDFC &cfdfc) {
  llvm_json::Array cycleBBs;
  for (unsigned bb : cfdfc.cycleBBs)
    cycleBBs.push_back(bb);

  llvm_json::Array units;
  for (const std::string &unit : cfdfc.units)
    units.push_back(unit);

  llvm_json::Array channels;
  for (const SerializedChannelDesc &channel : cfdfc.channels)
    channels.push_back(toJSON(channel));

  llvm_json::Array backedges;
  for (const SerializedChannelDesc &channel : cfdfc.backedges)
    backedges.push_back(toJSON(channel));

  llvm_json::Array occupancies;
  for (const SerializedUnitOccupancy &occupancy : cfdfc.unitOccupancy)
    occupancies.push_back(toJSON(occupancy));

  return llvm_json::Object{{"index", cfdfc.index},
                           {"cycle_bbs", std::move(cycleBBs)},
                           {"throughput", cfdfc.throughput},
                           {"units", std::move(units)},
                           {"channels", std::move(channels)},
                           {"backedges", std::move(backedges)},
                           {"unit_occupancy", std::move(occupancies)}};
}

static llvm_json::Value toJSON(const SerializedFunctionCFDFCs &func) {
  llvm_json::Array cfdfcs;
  for (const SerializedCFDFC &cfdfc : func.cfdfcs)
    cfdfcs.push_back(toJSON(cfdfc));
  return llvm_json::Object{{"name", func.name},
                           {"cfdfcs", std::move(cfdfcs)}};
}

static FailureOr<SerializedChannelDesc> buildSerializedChannel(Value channel,
                                                               NameAnalysis &na) {
  auto result = dyn_cast<OpResult>(channel);
  if (!result)
    return failure();

  Operation *producer = result.getOwner();
  auto userIt = channel.getUsers().begin();
  if (userIt == channel.getUsers().end())
    return failure();

  Operation *consumer = *userIt;
  ++userIt;
  if (userIt != channel.getUsers().end())
    return failure();

  return SerializedChannelDesc{na.getName(producer).str(),
                               result.getResultNumber(),
                               na.getName(consumer).str()};
}

static LogicalResult resolveFunctionByName(ModuleOp modOp, StringRef funcName,
                                           handshake::FuncOp &funcOp) {
  for (handshake::FuncOp candidate : modOp.getOps<handshake::FuncOp>()) {
    if (candidate.getName() == funcName) {
      funcOp = candidate;
      return success();
    }
  }
  modOp->emitError() << "CFDFC cache references unknown function '" << funcName
                     << "'";
  return failure();
}

static LogicalResult resolveChannel(handshake::FuncOp funcOp, NameAnalysis &na,
                                    const SerializedChannelDesc &desc,
                                    Value &channel) {
  Operation *producer = na.getOp(desc.producer);
  if (!producer) {
    funcOp.emitError() << "CFDFC cache references unknown producer op '"
                       << desc.producer << "'";
    return failure();
  }
  if (producer->getParentOfType<handshake::FuncOp>() != funcOp) {
    funcOp.emitError()
        << "CFDFC cache producer op '" << desc.producer
        << "' does not belong to function '" << funcOp.getName() << "'";
    return failure();
  }
  if (desc.resultIndex >= producer->getNumResults()) {
    funcOp.emitError() << "CFDFC cache references invalid result index "
                       << desc.resultIndex << " on producer '" << desc.producer
                       << "'";
    return failure();
  }

  channel = producer->getResult(desc.resultIndex);
  for (Operation *user : channel.getUsers()) {
    if (na.getName(user) == desc.consumer) {
      if (user->getParentOfType<handshake::FuncOp>() != funcOp) {
        funcOp.emitError()
            << "CFDFC cache consumer op '" << desc.consumer
            << "' does not belong to function '" << funcOp.getName() << "'";
        return failure();
      }
      return success();
    }
  }

  funcOp.emitError() << "CFDFC cache channel " << desc.producer << "["
                     << desc.resultIndex << "] does not feed consumer '"
                     << desc.consumer << "'";
  return failure();
}

static LogicalResult validateProvenance(ModuleOp modOp,
                                        const llvm_json::Object &object,
                                        const CFDFCCacheProvenance &expected) {
  auto expectPathMatch = [&](StringLiteral key,
                             const std::optional<std::string> &expectedPath,
                             StringRef what) -> LogicalResult {
    if (!expectedPath.has_value())
      return success();
    auto cachePath = object.getString(key);
    if (!cachePath) {
      modOp->emitError() << "CFDFC cache missing required provenance field '"
                         << key << "' for " << what;
      return failure();
    }
    if (normalizePathForComparison(*cachePath) != *expectedPath) {
      modOp->emitError() << "CFDFC cache " << what << " mismatch: cache='"
                         << *cachePath << "', current='" << *expectedPath
                         << "'";
      return failure();
    }
    return success();
  };

  if (failed(expectPathMatch("timing_models",
                             normalizeOptionalPath(expected.timingModels),
                             "timing-models path")))
    return failure();

  if (expected.targetPeriod.has_value()) {
    auto cacheTargetPeriod = object.getNumber("target_period");
    if (!cacheTargetPeriod) {
      modOp->emitError()
          << "CFDFC cache missing required provenance field 'target_period'";
      return failure();
    }
    if (std::fabs(*cacheTargetPeriod - *expected.targetPeriod) > 1e-9) {
      modOp->emitError() << "CFDFC cache target-period mismatch: cache="
                         << *cacheTargetPeriod
                         << ", current=" << *expected.targetPeriod;
      return failure();
    }
  }

  if (expected.bufferAlgorithm.has_value()) {
    auto cacheAlgorithm = object.getString("buffer_algorithm");
    if (!cacheAlgorithm) {
      modOp->emitError()
          << "CFDFC cache missing required provenance field 'buffer_algorithm'";
      return failure();
    }
    if (*cacheAlgorithm != *expected.bufferAlgorithm) {
      modOp->emitError() << "CFDFC cache buffer-algorithm mismatch: cache='"
                         << *cacheAlgorithm << "', current='"
                         << *expected.bufferAlgorithm << "'";
      return failure();
    }
  }

  return success();
}

} // namespace

LogicalResult dynamatic::buffer::writeCFDFCCache(
    ModuleOp modOp, const CFDFCAnalysis &analysis, NameAnalysis &nameAnalysis,
    StringRef filepath, const CFDFCCacheProvenance &provenance) {
  llvm_json::Array functions;

  for (handshake::FuncOp funcOp : modOp.getOps<handshake::FuncOp>()) {
    SerializedFunctionCFDFCs func;
    func.name = funcOp.getName().str();

    auto it = analysis.mapFuncOpToCFDFCs.find(funcOp);
    if (it != analysis.mapFuncOpToCFDFCs.end()) {
      for (auto [index, cfdfc] : enumerate(it->second)) {
        SerializedCFDFC serialized;
        serialized.index = index;
        serialized.throughput = cfdfc.throughput;
        serialized.cycleBBs.assign(cfdfc.cycle.begin(), cfdfc.cycle.end());

        for (Operation *unit : cfdfc.units)
          serialized.units.push_back(nameAnalysis.getName(unit).str());

        for (Value channel : cfdfc.channels) {
          FailureOr<SerializedChannelDesc> desc =
              buildSerializedChannel(channel, nameAnalysis);
          if (failed(desc)) {
            funcOp.emitError()
                << "failed to serialize CFDFC channel into cache JSON";
            return failure();
          }
          serialized.channels.push_back(*desc);
        }

        for (Value backedge : cfdfc.backedges) {
          FailureOr<SerializedChannelDesc> desc =
              buildSerializedChannel(backedge, nameAnalysis);
          if (failed(desc)) {
            funcOp.emitError()
                << "failed to serialize CFDFC backedge into cache JSON";
            return failure();
          }
          serialized.backedges.push_back(*desc);
        }

        for (auto &[unit, occupancy] : cfdfc.unitOccupancy) {
          serialized.unitOccupancy.push_back(
              SerializedUnitOccupancy{nameAnalysis.getName(unit).str(),
                                      occupancy});
        }

        func.cfdfcs.push_back(std::move(serialized));
      }
    }

    functions.push_back(toJSON(func));
  }

  llvm_json::Object topLevel{{"schema_version", kSchemaVersion},
                             {"functions", std::move(functions)}};
  if (auto normalized = normalizeOptionalPath(provenance.timingModels))
    topLevel["timing_models"] = *normalized;
  if (provenance.targetPeriod.has_value())
    topLevel["target_period"] = *provenance.targetPeriod;
  if (provenance.bufferAlgorithm.has_value())
    topLevel["buffer_algorithm"] = *provenance.bufferAlgorithm;

  std::error_code ec;
  raw_fd_ostream os(filepath, ec);
  if (ec) {
    modOp->emitError() << "failed to open CFDFC cache file '" << filepath
                       << "': " << ec.message();
    return failure();
  }

  os << formatv("{0:2}", llvm_json::Value(std::move(topLevel)));
  os << "\n";
  return success();
}

LogicalResult dynamatic::buffer::readCFDFCCache(
    ModuleOp modOp, NameAnalysis &nameAnalysis, StringRef filepath,
    std::map<handshake::FuncOp, ListOfCFDFCs> &out,
    const CFDFCCacheProvenance &expected) {
  out.clear();

  auto bufferOrErr = MemoryBuffer::getFile(filepath);
  if (!bufferOrErr) {
    modOp->emitError() << "failed to read CFDFC cache file '" << filepath
                       << "'";
    return failure();
  }

  Expected<llvm_json::Value> value =
      llvm_json::parse(bufferOrErr.get()->getBuffer());
  if (!value) {
    modOp->emitError() << "failed to parse CFDFC cache JSON '" << filepath
                       << "'";
    return failure();
  }

  const llvm_json::Object *object = value->getAsObject();
  if (!object) {
    modOp->emitError() << "CFDFC cache root must be a JSON object";
    return failure();
  }

  unsigned schemaVersion = 0;
  llvm_json::Path::Root root(filepath);
  dynamatic::json::ObjectDeserializer mapper(*object, root);
  std::vector<SerializedFunctionCFDFCs> serializedFunctions;
  if (!mapper.map("schema_version", schemaVersion)
           .map("functions", serializedFunctions)
           .mapOptional(
               "timing_models",
               std::function<bool(const llvm_json::Value &, llvm_json::Path)>(
                   [&](const llvm_json::Value &, llvm_json::Path) {
                     return true;
                   }))
           .mapOptional(
               "target_period",
               std::function<bool(const llvm_json::Value &, llvm_json::Path)>(
                   [&](const llvm_json::Value &, llvm_json::Path) {
                     return true;
                   }))
           .mapOptional(
               "buffer_algorithm",
               std::function<bool(const llvm_json::Value &, llvm_json::Path)>(
                   [&](const llvm_json::Value &, llvm_json::Path) {
                     return true;
                   }))
           .exhausted())
    return failure();

  if (schemaVersion != kSchemaVersion) {
    modOp->emitError() << "unsupported CFDFC cache schema version "
                       << schemaVersion << " (expected " << kSchemaVersion
                       << ")";
    return failure();
  }

  if (failed(validateProvenance(modOp, *object, expected)))
    return failure();

  DenseSet<StringRef> seenFunctions;
  for (const SerializedFunctionCFDFCs &serializedFunc : serializedFunctions) {
    handshake::FuncOp funcOp;
    if (failed(resolveFunctionByName(modOp, serializedFunc.name, funcOp)))
      return failure();

    if (!seenFunctions.insert(funcOp.getName()).second) {
      funcOp.emitError() << "CFDFC cache contains duplicate function entry '"
                         << serializedFunc.name << "'";
      return failure();
    }

    ListOfCFDFCs cfdfcs;
    cfdfcs.reserve(serializedFunc.cfdfcs.size());
    for (const SerializedCFDFC &serializedCFDFC : serializedFunc.cfdfcs) {
      unsigned expectedIndex = cfdfcs.size();
      if (serializedCFDFC.index != expectedIndex) {
        funcOp.emitError() << "CFDFC cache index mismatch for function '"
                           << serializedFunc.name << "': expected "
                           << expectedIndex << ", got "
                           << serializedCFDFC.index;
        return failure();
      }

      CFDFC cfdfc;
      for (unsigned bb : serializedCFDFC.cycleBBs)
        cfdfc.cycle.insert(bb);
      cfdfc.throughput = serializedCFDFC.throughput;

      for (const std::string &unitName : serializedCFDFC.units) {
        Operation *unit = nameAnalysis.getOp(unitName);
        if (!unit) {
          funcOp.emitError() << "CFDFC cache references unknown unit op '"
                             << unitName << "'";
          return failure();
        }
        if (unit->getParentOfType<handshake::FuncOp>() != funcOp) {
          funcOp.emitError()
              << "CFDFC cache unit op '" << unitName
              << "' does not belong to function '" << funcOp.getName() << "'";
          return failure();
        }
        cfdfc.units.insert(unit);
      }

      for (const SerializedChannelDesc &desc : serializedCFDFC.channels) {
        Value channel;
        if (failed(resolveChannel(funcOp, nameAnalysis, desc, channel)))
          return failure();
        cfdfc.channels.insert(channel);
      }

      for (const SerializedChannelDesc &desc : serializedCFDFC.backedges) {
        Value backedge;
        if (failed(resolveChannel(funcOp, nameAnalysis, desc, backedge)))
          return failure();
        if (!cfdfc.channels.contains(backedge)) {
          funcOp.emitError()
              << "CFDFC cache backedge is not present in channel set for "
              << desc.producer << "[" << desc.resultIndex << "]";
          return failure();
        }
        cfdfc.backedges.insert(backedge);
      }

      for (const SerializedUnitOccupancy &occupancy :
           serializedCFDFC.unitOccupancy) {
        Operation *unit = nameAnalysis.getOp(occupancy.unit);
        if (!unit) {
          funcOp.emitError()
              << "CFDFC cache references unknown occupancy unit '"
              << occupancy.unit << "'";
          return failure();
        }
        if (!cfdfc.units.contains(unit)) {
          funcOp.emitError()
              << "CFDFC cache occupancy unit '" << occupancy.unit
              << "' is not present in the serialized CFDFC unit list";
          return failure();
        }
        cfdfc.unitOccupancy[unit] = occupancy.occupancy;
      }

      cfdfcs.push_back(std::move(cfdfc));
    }

    out[funcOp] = std::move(cfdfcs);
  }

  for (handshake::FuncOp funcOp : modOp.getOps<handshake::FuncOp>()) {
    if (!seenFunctions.contains(funcOp.getName())) {
      funcOp.emitError() << "CFDFC cache does not contain function '"
                         << funcOp.getName() << "'";
      return failure();
    }
  }

  return success();
}
