//===- CFDFCCacheTest.cpp ---------------------------------------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "dynamatic/Analysis/NameAnalysis.h"
#include "dynamatic/Dialect/Handshake/HandshakeDialect.h"
#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Transforms/BufferPlacement/CFDFC.h"
#include "dynamatic/Transforms/BufferPlacement/CFDFCCache.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"
#include "gtest/gtest.h"

#include <tuple>

using namespace mlir;
using namespace dynamatic;
using namespace dynamatic::buffer;

namespace llvm_json = llvm::json;

namespace {

constexpr llvm::StringLiteral kModuleIR = R"mlir(
module {
  handshake.func @test(%arg0: !handshake.channel<i16>, %start: !handshake.control<>) -> !handshake.channel<i32> {
    sink %start : <>
    %ext0 = extsi %arg0 : <i16> to <i32>
    %buf = buffer %ext0, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 : <i32>
    end %buf : <i32>
  }
}
)mlir";

struct ParsedModule {
  OwningOpRef<ModuleOp> module;
  std::unique_ptr<NameAnalysis> names;
  handshake::FuncOp func;
  Operation *extsi = nullptr;
  Operation *buffer = nullptr;
  Operation *end = nullptr;
  CFDFCCacheProvenance provenance;

  bool valid() const { return module && names && func; }
};

class CFDFCCacheTest : public ::testing::Test {
protected:
  DialectRegistry registry;
  MLIRContext context;

  CFDFCCacheTest() {
    registry.insert<func::FuncDialect, handshake::HandshakeDialect>();
    context.appendDialectRegistry(registry);
    context.loadDialect<func::FuncDialect, handshake::HandshakeDialect>();
  }

  static std::string normalizePath(StringRef inputPath) {
    llvm::SmallString<256> normalized(inputPath);
    if (!normalized.empty()) {
      (void)llvm::sys::fs::make_absolute(normalized);
      llvm::sys::path::remove_dots(normalized, /*remove_dot_dot=*/true);
    }
    return std::string(normalized.str());
  }

  ParsedModule createParsedModule() {
    ParsedModule parsed;
    parsed.module = parseSourceString<ModuleOp>(kModuleIR, &context);
    EXPECT_TRUE(static_cast<bool>(parsed.module));
    if (!parsed.module)
      return parsed;

    parsed.names = std::make_unique<NameAnalysis>(parsed.module->getOperation());
    EXPECT_TRUE(parsed.names->isAnalysisValid());
    if (!parsed.names->isAnalysisValid())
      return parsed;
    parsed.names->nameAllUnnamedOps();

    auto funcs = parsed.module->getOps<handshake::FuncOp>();
    EXPECT_EQ(std::distance(funcs.begin(), funcs.end()), 1);
    if (funcs.begin() == funcs.end())
      return parsed;
    parsed.func = *funcs.begin();

    auto exts = parsed.func.getOps<handshake::ExtSIOp>();
    EXPECT_EQ(std::distance(exts.begin(), exts.end()), 1);
    if (exts.begin() != exts.end()) {
      handshake::ExtSIOp extOp = *exts.begin();
      parsed.extsi = extOp.getOperation();
    }

    auto buffers = parsed.func.getOps<handshake::BufferOp>();
    EXPECT_EQ(std::distance(buffers.begin(), buffers.end()), 1);
    if (buffers.begin() != buffers.end()) {
      handshake::BufferOp bufferOp = *buffers.begin();
      parsed.buffer = bufferOp.getOperation();
    }

    parsed.end = parsed.func.front().getTerminator();
    parsed.provenance.timingModels = "/tmp/test-components.json";
    parsed.provenance.targetPeriod = 8.0;
    parsed.provenance.bufferAlgorithm = "fpga20";
    return parsed;
  }

  static CFDFC buildReferenceCFDFC(const ParsedModule &parsed) {
    CFDFC cfdfc;
    cfdfc.cycle.insert(0);
    cfdfc.throughput = 0.5;
    cfdfc.units.insert(parsed.extsi);
    cfdfc.units.insert(parsed.buffer);
    cfdfc.units.insert(parsed.end);
    cfdfc.channels.insert(parsed.extsi->getResult(0));
    cfdfc.channels.insert(parsed.buffer->getResult(0));
    cfdfc.backedges.insert(parsed.buffer->getResult(0));
    cfdfc.unitOccupancy[parsed.buffer] = 1.0;
    return cfdfc;
  }

  static std::vector<std::string> collectUnitNames(const CFDFC &cfdfc,
                                                   NameAnalysis &names) {
    std::vector<std::string> ordered;
    for (Operation *unit : cfdfc.units)
      ordered.push_back(names.getName(unit).str());
    return ordered;
  }

  static std::vector<std::tuple<std::string, unsigned, std::string>>
  collectChannelTriples(const mlir::SetVector<Value> &channels,
                        NameAnalysis &names) {
    std::vector<std::tuple<std::string, unsigned, std::string>> ordered;
    for (Value channel : channels) {
      auto result = cast<OpResult>(channel);
      Operation *producer = result.getOwner();
      Operation *consumer = *channel.getUsers().begin();
      ordered.emplace_back(names.getName(producer).str(), result.getResultNumber(),
                           names.getName(consumer).str());
    }
    return ordered;
  }

  static std::vector<unsigned> collectCycleBBs(const CFDFC &cfdfc) {
    return std::vector<unsigned>(cfdfc.cycle.begin(), cfdfc.cycle.end());
  }

  llvm_json::Object buildValidCacheObject(ParsedModule &parsed) {
    auto &names = *parsed.names;
    llvm_json::Array functions;
    llvm_json::Array cfdfcs;
    llvm_json::Array cycleBBs;
    cycleBBs.push_back(0);

    llvm_json::Array units;
    units.push_back(names.getName(parsed.extsi).str());
    units.push_back(names.getName(parsed.buffer).str());
    units.push_back(names.getName(parsed.end).str());

    llvm_json::Array channels;
    channels.push_back(llvm_json::Object{
        {"producer", names.getName(parsed.extsi).str()},
        {"result_index", 0},
        {"consumer", names.getName(parsed.buffer).str()}});
    channels.push_back(llvm_json::Object{
        {"producer", names.getName(parsed.buffer).str()},
        {"result_index", 0},
        {"consumer", names.getName(parsed.end).str()}});

    llvm_json::Array backedges;
    backedges.push_back(llvm_json::Object{
        {"producer", names.getName(parsed.buffer).str()},
        {"result_index", 0},
        {"consumer", names.getName(parsed.end).str()}});

    llvm_json::Array unitOccupancy;
    unitOccupancy.push_back(llvm_json::Object{
        {"unit", names.getName(parsed.buffer).str()}, {"occupancy", 1.0}});

    cfdfcs.push_back(llvm_json::Object{
        {"index", 0},
        {"cycle_bbs", std::move(cycleBBs)},
        {"throughput", 0.5},
        {"units", std::move(units)},
        {"channels", std::move(channels)},
        {"backedges", std::move(backedges)},
        {"unit_occupancy", std::move(unitOccupancy)},
    });

    functions.push_back(llvm_json::Object{
        {"name", parsed.func.getName().str()}, {"cfdfcs", std::move(cfdfcs)}});

    return llvm_json::Object{
        {"schema_version", 1},
        {"timing_models", normalizePath(*parsed.provenance.timingModels)},
        {"target_period", *parsed.provenance.targetPeriod},
        {"buffer_algorithm", *parsed.provenance.bufferAlgorithm},
        {"functions", std::move(functions)},
    };
  }

  static std::string writeJSONToTempFile(llvm_json::Value value) {
    int fd = -1;
    llvm::SmallString<128> path;
    if (std::error_code ec =
            llvm::sys::fs::createTemporaryFile("cfdfc-cache-test", "json", fd,
                                               path)) {
      ADD_FAILURE() << "failed to create temporary file: " << ec.message();
      return "";
    }

    llvm::raw_fd_ostream os(fd, /*shouldClose=*/true);
    os << llvm::formatv("{0:2}", value) << "\n";
    return std::string(path.str());
  }
};

TEST_F(CFDFCCacheTest, RoundTripPreservesFinalizedCFDFCData) {
  ParsedModule parsed = createParsedModule();
  ASSERT_TRUE(parsed.valid());

  CFDFCAnalysis analysis(parsed.module->getOperation());
  CFDFC expectedCFDFC = buildReferenceCFDFC(parsed);
  analysis.mapFuncOpToCFDFCs[parsed.func].push_back(expectedCFDFC);

  std::string cachePath =
      writeJSONToTempFile(llvm_json::Object{{"schema_version", 0}});
  ASSERT_FALSE(cachePath.empty());

  ASSERT_TRUE(succeeded(writeCFDFCCache(*parsed.module, analysis, *parsed.names,
                                        cachePath, parsed.provenance)));

  std::map<handshake::FuncOp, ListOfCFDFCs> loaded;
  CFDFCCacheProvenance expected{
      *parsed.provenance.timingModels, *parsed.provenance.targetPeriod,
      std::nullopt};
  ASSERT_TRUE(succeeded(readCFDFCCache(*parsed.module, *parsed.names, cachePath,
                                       loaded, expected)));

  auto it = loaded.find(parsed.func);
  ASSERT_NE(it, loaded.end());
  ASSERT_EQ(it->second.size(), 1u);

  CFDFC &loadedCFDFC = it->second.front();
  EXPECT_EQ(collectCycleBBs(loadedCFDFC), collectCycleBBs(expectedCFDFC));
  EXPECT_DOUBLE_EQ(loadedCFDFC.throughput, expectedCFDFC.throughput);
  EXPECT_EQ(collectUnitNames(loadedCFDFC, *parsed.names),
            collectUnitNames(expectedCFDFC, *parsed.names));
  EXPECT_EQ(collectChannelTriples(loadedCFDFC.channels, *parsed.names),
            collectChannelTriples(expectedCFDFC.channels, *parsed.names));
  EXPECT_EQ(collectChannelTriples(loadedCFDFC.backedges, *parsed.names),
            collectChannelTriples(expectedCFDFC.backedges, *parsed.names));
  ASSERT_EQ(loadedCFDFC.unitOccupancy.size(), 1u);
  EXPECT_DOUBLE_EQ(loadedCFDFC.unitOccupancy.lookup(parsed.buffer), 1.0);

  (void)llvm::sys::fs::remove(cachePath);
}

TEST_F(CFDFCCacheTest, RejectsBadSchemaVersion) {
  ParsedModule parsed = createParsedModule();
  ASSERT_TRUE(parsed.valid());

  llvm_json::Object cache = buildValidCacheObject(parsed);
  cache["schema_version"] = 99;

  std::string cachePath = writeJSONToTempFile(llvm_json::Value(std::move(cache)));
  ASSERT_FALSE(cachePath.empty());

  std::map<handshake::FuncOp, ListOfCFDFCs> loaded;
  EXPECT_TRUE(failed(readCFDFCCache(*parsed.module, *parsed.names, cachePath,
                                    loaded, parsed.provenance)));
  (void)llvm::sys::fs::remove(cachePath);
}

TEST_F(CFDFCCacheTest, RejectsMissingFunction) {
  ParsedModule parsed = createParsedModule();
  ASSERT_TRUE(parsed.valid());

  llvm_json::Object cache = buildValidCacheObject(parsed);
  auto &functions = *cache.getArray("functions");
  auto &funcObj = *functions.front().getAsObject();
  funcObj["name"] = "missing_func";

  std::string cachePath = writeJSONToTempFile(llvm_json::Value(std::move(cache)));
  ASSERT_FALSE(cachePath.empty());

  std::map<handshake::FuncOp, ListOfCFDFCs> loaded;
  EXPECT_TRUE(failed(readCFDFCCache(*parsed.module, *parsed.names, cachePath,
                                    loaded, parsed.provenance)));
  (void)llvm::sys::fs::remove(cachePath);
}

TEST_F(CFDFCCacheTest, RejectsMissingUnitName) {
  ParsedModule parsed = createParsedModule();
  ASSERT_TRUE(parsed.valid());

  llvm_json::Object cache = buildValidCacheObject(parsed);
  auto &functions = *cache.getArray("functions");
  auto &funcObj = *functions.front().getAsObject();
  auto &cfdfcs = *funcObj.getArray("cfdfcs");
  auto &cfdfcObj = *cfdfcs.front().getAsObject();
  auto &units = *cfdfcObj.getArray("units");
  units.front() = "missing_unit";

  std::string cachePath = writeJSONToTempFile(llvm_json::Value(std::move(cache)));
  ASSERT_FALSE(cachePath.empty());

  std::map<handshake::FuncOp, ListOfCFDFCs> loaded;
  EXPECT_TRUE(failed(readCFDFCCache(*parsed.module, *parsed.names, cachePath,
                                    loaded, parsed.provenance)));
  (void)llvm::sys::fs::remove(cachePath);
}

TEST_F(CFDFCCacheTest, RejectsMismatchedCFDFCIndex) {
  ParsedModule parsed = createParsedModule();
  ASSERT_TRUE(parsed.valid());

  llvm_json::Object cache = buildValidCacheObject(parsed);
  auto &functions = *cache.getArray("functions");
  auto &funcObj = *functions.front().getAsObject();
  auto &cfdfcs = *funcObj.getArray("cfdfcs");
  auto &cfdfcObj = *cfdfcs.front().getAsObject();
  cfdfcObj["index"] = 7;

  std::string cachePath = writeJSONToTempFile(llvm_json::Value(std::move(cache)));
  ASSERT_FALSE(cachePath.empty());

  std::map<handshake::FuncOp, ListOfCFDFCs> loaded;
  EXPECT_TRUE(failed(readCFDFCCache(*parsed.module, *parsed.names, cachePath,
                                    loaded, parsed.provenance)));
  (void)llvm::sys::fs::remove(cachePath);
}

TEST_F(CFDFCCacheTest, RejectsInvalidResultIndex) {
  ParsedModule parsed = createParsedModule();
  ASSERT_TRUE(parsed.valid());

  llvm_json::Object cache = buildValidCacheObject(parsed);
  auto &functions = *cache.getArray("functions");
  auto &funcObj = *functions.front().getAsObject();
  auto &cfdfcs = *funcObj.getArray("cfdfcs");
  auto &cfdfcObj = *cfdfcs.front().getAsObject();
  auto &channels = *cfdfcObj.getArray("channels");
  auto &firstChannel = *channels.front().getAsObject();
  firstChannel["result_index"] = 99;

  std::string cachePath = writeJSONToTempFile(llvm_json::Value(std::move(cache)));
  ASSERT_FALSE(cachePath.empty());

  std::map<handshake::FuncOp, ListOfCFDFCs> loaded;
  EXPECT_TRUE(failed(readCFDFCCache(*parsed.module, *parsed.names, cachePath,
                                    loaded, parsed.provenance)));
  (void)llvm::sys::fs::remove(cachePath);
}

TEST_F(CFDFCCacheTest, RejectsWrongConsumerMapping) {
  ParsedModule parsed = createParsedModule();
  ASSERT_TRUE(parsed.valid());

  llvm_json::Object cache = buildValidCacheObject(parsed);
  auto &functions = *cache.getArray("functions");
  auto &funcObj = *functions.front().getAsObject();
  auto &cfdfcs = *funcObj.getArray("cfdfcs");
  auto &cfdfcObj = *cfdfcs.front().getAsObject();
  auto &channels = *cfdfcObj.getArray("channels");
  auto &firstChannel = *channels.front().getAsObject();
  firstChannel["consumer"] = parsed.names->getName(parsed.end).str();

  std::string cachePath = writeJSONToTempFile(llvm_json::Value(std::move(cache)));
  ASSERT_FALSE(cachePath.empty());

  std::map<handshake::FuncOp, ListOfCFDFCs> loaded;
  EXPECT_TRUE(failed(readCFDFCCache(*parsed.module, *parsed.names, cachePath,
                                    loaded, parsed.provenance)));
  (void)llvm::sys::fs::remove(cachePath);
}

TEST_F(CFDFCCacheTest, RejectsProvenanceMismatch) {
  ParsedModule parsed = createParsedModule();
  ASSERT_TRUE(parsed.valid());

  llvm_json::Object cache = buildValidCacheObject(parsed);
  cache["timing_models"] = normalizePath("/tmp/different-components.json");

  std::string cachePath = writeJSONToTempFile(llvm_json::Value(std::move(cache)));
  ASSERT_FALSE(cachePath.empty());

  std::map<handshake::FuncOp, ListOfCFDFCs> loaded;
  EXPECT_TRUE(failed(readCFDFCCache(*parsed.module, *parsed.names, cachePath,
                                    loaded, parsed.provenance)));
  (void)llvm::sys::fs::remove(cachePath);
}

} // namespace
