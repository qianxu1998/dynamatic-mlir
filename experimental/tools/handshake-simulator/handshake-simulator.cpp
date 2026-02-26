#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Support/LLVM.h"
#include "dynamatic/Transforms/HandshakeMaterialize.h"
#include "experimental/Support/HandshakeSimulator.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Attributes.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include <string>
#include <utility>

using namespace llvm;
using namespace mlir;
using namespace dynamatic;

static cl::OptionCategory mainCategory("Tool options");

static cl::opt<std::string> inputFilename(cl::Positional, cl::Required,
                                          cl::desc("<input file>"),
                                          cl::cat(mainCategory));

static cl::list<std::string> inputArgs(cl::Positional, cl::desc("<input args>"),
                                       cl::ZeroOrMore, cl::cat(mainCategory));

static cl::opt<std::string>
    inputVectorsDir("input-vectors-dir",
                    cl::desc("Directory containing Dynamatic input_*.dat files"),
                    cl::init(""), cl::cat(mainCategory));

static cl::opt<std::string>
    inputArgsFile("input-args-file",
                  cl::desc("Plain text input file for handshake arguments"),
                  cl::init(""), cl::cat(mainCategory));

static cl::opt<std::string> inputFormat(
    "input-format", cl::desc("Input format: auto|dat|plain"), cl::init("auto"),
    cl::cat(mainCategory));

static cl::opt<unsigned>
    maxCycles("max-cycles", cl::desc("Maximum cycles per transaction"),
              cl::init(1000000), cl::cat(mainCategory));

static cl::opt<std::string>
    topLevelFunction("top-level-function",
                     cl::desc("Handshake function symbol to simulate"),
                     cl::init(""), cl::cat(mainCategory));

static cl::opt<std::string>
    dumpSwitchingJSON("dump-switching-json",
                      cl::desc("Dump channel switching summary JSON"),
                      cl::init(""), cl::cat(mainCategory));

static cl::opt<std::string>
    dumpWaveJSON("dump-wave-json",
                 cl::desc("Dump full per-cycle channel wave JSON"),
                 cl::init(""), cl::cat(mainCategory));

static cl::opt<std::string>
    dumpResultsJSON("dump-results-json",
                    cl::desc("Dump function results and memories JSON"),
                    cl::init(""), cl::cat(mainCategory));

static cl::opt<std::string> dumpSwitchingEstCSV(
    "dump-switching-est-csv",
    cl::desc("Dump node switching CSV in switching-estimation format"),
    cl::init(""), cl::cat(mainCategory));

static cl::opt<std::string>
    topVerilog("top-verilog", cl::desc("Top-level Verilog netlist for VCD mapping"),
               cl::init(""), cl::cat(mainCategory));

static cl::opt<std::string> vcdFile(
    "vcd", cl::desc("VCD file to compare against simulator traces"),
    cl::init(""), cl::cat(mainCategory));

static cl::opt<std::string>
    verifyVCD("verify-vcd", cl::desc("VCD verification mode: exact"),
              cl::init(""), cl::cat(mainCategory));

static cl::opt<std::string>
    lsqConfigDir("lsq-config-dir",
                 cl::desc("Directory containing handshake_lsq_<name>.json"),
                 cl::init(""), cl::cat(mainCategory));

using namespace dynamatic::experimental;

static HSInputFormat parseInputFormatOption(StringRef format) {
  if (format.equals_insensitive("auto"))
    return HSInputFormat::Auto;
  if (format.equals_insensitive("dat"))
    return HSInputFormat::Dat;
  if (format.equals_insensitive("plain"))
    return HSInputFormat::Plain;
  return HSInputFormat::Auto;
}

static handshake::FuncOp selectTopFunction(ModuleOp modOp, StringRef name) {
  if (!name.empty()) {
    for (handshake::FuncOp funcOp : modOp.getOps<handshake::FuncOp>()) {
      auto symName = funcOp->getAttrOfType<StringAttr>(
          SymbolTable::getSymbolAttrName());
      if (symName && symName.getValue() == name)
        return funcOp;
    }
    return {};
  }

  auto funcs = modOp.getOps<handshake::FuncOp>();
  if (funcs.empty())
    return {};
  return *funcs.begin();
}

static int emitFailure(const Simulator &sim) {
  if (sim.hasFailed()) {
    llvm::errs() << "Simulation failed: " << sim.getFailureMessage() << "\n";
  } else if (sim.getVCDCheck().enabled) {
    const auto &check = sim.getVCDCheck();
    llvm::errs() << "VCD verification failed: " << check.message
                 << ", mismatches=" << check.mismatchCount
                 << ", first_mismatch_cycle=" << check.firstMismatchCycle
                 << "\n";
    unsigned emitted = 0;
    for (const auto &mismatch : check.mismatches) {
      if (emitted >= 16)
        break;
      llvm::errs() << "  mismatch[" << emitted << "]: channel " << mismatch.src
                   << " -> " << mismatch.dst << " (operand "
                   << mismatch.operandIndex << "), signal=" << mismatch.signal
                   << ", cycle=" << mismatch.cycle
                   << ", sim=" << mismatch.simulated
                   << ", vcd=" << mismatch.vcd << "\n";
      ++emitted;
    }
  } else {
    llvm::errs() << "Simulation failed\n";
  }
  return 1;
}

int main(int argc, char **argv) {
  InitLLVM y(argc, argv);
  cl::ParseCommandLineOptions(argc, argv, "simulator");

  HSInputFormat parsedInputFormat = parseInputFormatOption(inputFormat);
  if (!StringRef(inputFormat).equals_insensitive("auto") &&
      !StringRef(inputFormat).equals_insensitive("dat") &&
      !StringRef(inputFormat).equals_insensitive("plain")) {
    llvm::errs() << "Invalid --input-format value '" << inputFormat
                 << "' (expected auto|dat|plain)\n";
    return 1;
  }

  if (!verifyVCD.empty() && !StringRef(verifyVCD).equals_insensitive("exact")) {
    llvm::errs() << "Invalid --verify-vcd value '" << verifyVCD
                 << "' (expected exact)\n";
    return 1;
  }

  auto fileOrErr = MemoryBuffer::getFileOrSTDIN(inputFilename.c_str());
  if (std::error_code error = fileOrErr.getError()) {
    llvm::errs() << argv[0] << ": could not open input file '" << inputFilename
                 << "': " << error.message() << "\n";
    return 1;
  }

  MLIRContext context;
  context.loadDialect<handshake::HandshakeDialect, arith::ArithDialect>();

  // Load the MLIR module
  SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(*fileOrErr), SMLoc());
  mlir::OwningOpRef<mlir::ModuleOp> modOp(
      mlir::parseSourceFile<mlir::ModuleOp>(sourceMgr, &context));
  if (!modOp)
    return 1;

  handshake::FuncOp funcOp = selectTopFunction(*modOp, topLevelFunction);
  if (!funcOp) {
    if (!topLevelFunction.empty()) {
      llvm::errs() << "Could not find handshake function '" << topLevelFunction
                   << "'\n";
    } else {
      llvm::errs() << "No handshake function found in module\n";
    }
    return 1;
  }

  SimulatorOptions options;
  options.maxCycles = maxCycles;
  options.lsqConfigDir = lsqConfigDir;

  Simulator sim(funcOp, options);

  bool hasFileInputs = !inputVectorsDir.empty() || !inputArgsFile.empty();
  if (hasFileInputs) {
    if (parsedInputFormat == HSInputFormat::Dat && inputVectorsDir.empty()) {
      llvm::errs() << "--input-format=dat requires --input-vectors-dir\n";
      return 1;
    }
    if (parsedInputFormat == HSInputFormat::Plain && inputArgsFile.empty()) {
      llvm::errs() << "--input-format=plain requires --input-args-file\n";
      return 1;
    }

    SimulationInputs inputs;
    if (failed(sim.loadInputs(inputVectorsDir, inputArgsFile, parsedInputFormat,
                              inputs))) {
      return emitFailure(sim);
    }

    if (failed(sim.simulate(inputs)))
      return emitFailure(sim);
  } else {
    sim.simulate(inputArgs);
    if (sim.hasFailed())
      return emitFailure(sim);
  }

  bool verifyFailed = false;
  if (StringRef(verifyVCD).equals_insensitive("exact")) {
    if (topVerilog.empty() || vcdFile.empty()) {
      llvm::errs()
          << "--verify-vcd=exact requires both --top-verilog and --vcd\n";
      return 1;
    }
    verifyFailed = failed(sim.verifyVCDExact(topVerilog, vcdFile));
  }

  if (!dumpSwitchingJSON.empty() &&
      failed(sim.dumpSwitchingJSON(dumpSwitchingJSON)))
    return emitFailure(sim);
  if (!dumpWaveJSON.empty() && failed(sim.dumpWaveJSON(dumpWaveJSON)))
    return emitFailure(sim);
  if (!dumpResultsJSON.empty() && failed(sim.dumpResultsJSON(dumpResultsJSON)))
    return emitFailure(sim);
  if (!dumpSwitchingEstCSV.empty() &&
      failed(sim.dumpSwitchingEstimationCSV(dumpSwitchingEstCSV)))
    return emitFailure(sim);

  if (verifyFailed)
    return emitFailure(sim);

  sim.printResults();
  return 0;
}
