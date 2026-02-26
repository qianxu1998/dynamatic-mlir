#include "experimental/Support/HandshakeSimulator.h"
#include "dynamatic/Dialect/Handshake/HandshakeAttributes.h"
#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Dialect/Handshake/HandshakeTypes.h"
#include "dynamatic/Support/JSON/JSON.h"
#include "dynamatic/Support/LLVM.h"
#include "dynamatic/Transforms/HandshakeMaterialize.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/Any.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/IR/Attributes.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <fstream>
#include <functional>
#include <limits>
#include <map>
#include <optional>
#include <regex>
#include <set>
#include <sstream>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

using namespace llvm;
using namespace mlir;
using namespace dynamatic;

using namespace dynamatic::experimental;

Data::Data(const APInt &value) {
  data = value;
  hash = llvm::hash_value(value);
  bitwidth = value.getBitWidth();
}

Data::Data(const APFloat &value) {
  data = value;
  hash = llvm::hash_value(value);
  bitwidth = value.getSizeInBits(value.getSemantics());
}

Data &Data::operator=(const APInt &value) {
  this->data = value;
  hash = llvm::hash_value(value);
  bitwidth = value.getBitWidth();
  return *this;
}

Data &Data::operator=(const APFloat &value) {
  this->data = value;
  hash = llvm::hash_value(value);
  Data::bitwidth = value.getSizeInBits(value.getSemantics());
  return *this;
}

bool Data::hasValue() const { return data.has_value(); }

ValueState::ValueState(Value val) : val(val) {}

template <typename Ty>
TypedValueState<Ty>::TypedValueState(TypedValue<Ty> val) : ValueState(val) {}

ChannelState::ChannelState(TypedValue<handshake::ChannelType> channel)
    : TypedValueState<handshake::ChannelType>(channel) {

  llvm::TypeSwitch<mlir::Type>(channel.getType().getDataType())
      .Case<IntegerType>([&](IntegerType intType) {
        data =
            APInt(std::max(1, (int)intType.getWidth()), 0, intType.isSigned());
      })
      .Case<FloatType>([&](FloatType floatType) {
        data = APFloat(floatType.getFloatSemantics());
      })
      .Default([&](auto) {
        emitError(channel.getLoc())
            << "Unsuported date type " << channel.getType()
            << ", we111 should probably report an error and stop";
      });
}

ControlState::ControlState(TypedValue<handshake::ControlType> control)
    : TypedValueState<handshake::ControlType>(control) {}

template <typename State>
DoubleUpdater<State>::DoubleUpdater(State &oldState, State &newState)
    : oldState(oldState), newState(newState) {}

ChannelUpdater::ChannelUpdater(ChannelState &oldState, ChannelState &newState)
    : DoubleUpdater<ChannelState>(oldState, newState) {}

bool ChannelUpdater::check() {
  return newState.valid == oldState.valid && newState.ready == oldState.ready &&
         newState.data.hash == oldState.data.hash;
}

void ChannelUpdater::setValid() {
  newState.valid = true;
  update();
}

void ChannelUpdater::resetValid() {
  if (newState.ready) {
    newState.valid = false;
  }
  update();
}

void ChannelUpdater::update() {
  oldState.valid = newState.valid;
  oldState.ready = newState.ready;
  oldState.data = newState.data;
}

ControlUpdater::ControlUpdater(ControlState &oldState, ControlState &newState)
    : DoubleUpdater<ControlState>(oldState, newState) {}

bool ControlUpdater::check() {
  return newState.valid == oldState.valid && newState.ready == oldState.ready;
}

void ControlUpdater::setValid() {
  newState.valid = true;
  update();
}

void ControlUpdater::resetValid() {
  if (newState.ready)
    newState.valid = false;
  update();
}

void ControlUpdater::update() {
  oldState.valid = newState.valid;
  oldState.ready = newState.ready;
}

ProducerRW::ProducerDescendants ProducerRW::getType() const { return prod; }

ProducerRW::ProducerRW(bool &valid, const bool &ready, ProducerDescendants p)
    : valid(valid), ready(ready), prod(p) {}

ProducerRW::ProducerRW(ProducerRW &p)
    : valid(p.valid), ready(p.ready), prod(p.prod) {}

ConsumerRW::ConsumerDescendants ConsumerRW::getType() const { return cons; }

ConsumerRW::ConsumerRW(const bool &valid, bool &ready, ConsumerDescendants c)
    : valid(valid), ready(ready), cons(c) {}

ConsumerRW::ConsumerRW(ConsumerRW &c)
    : valid(c.valid), ready(c.ready), cons(c.cons) {}

ControlConsumerRW::ControlConsumerRW(ControlState &reader, ControlState &writer)
    : ConsumerRW(reader.valid, writer.ready, D_ControlConsumerRW) {}

bool ControlConsumerRW::classof(const ConsumerRW *c) {
  return c->getType() == D_ControlConsumerRW;
}

ControlProducerRW::ControlProducerRW(ControlState &reader, ControlState &writer)
    : ProducerRW(writer.valid, reader.ready, D_ControlProducerRW) {}

bool ControlProducerRW::classof(const ProducerRW *c) {
  return c->getType() == D_ControlProducerRW;
}

ChannelConsumerRW::ChannelConsumerRW(ChannelState &reader, ChannelState &writer)
    : ConsumerRW(reader.valid, writer.ready, D_ChannelConsumerRW),
      data(reader.data) {}

bool ChannelConsumerRW::classof(const ConsumerRW *c) {
  return c->getType() == D_ChannelConsumerRW;
}

ChannelProducerRW::ChannelProducerRW(ChannelState &reader, ChannelState &writer)
    : ProducerRW(writer.valid, reader.ready, D_ChannelProducerRW),
      data(writer.data) {}

bool ChannelProducerRW::classof(const ProducerRW *c) {
  return c->getType() == D_ChannelProducerRW;
}

ConsumerData::ConsumerData(ConsumerRW *ins) {
  if (auto *p = dyn_cast_if_present<ChannelConsumerRW>(ins)) {
    data = &p->data;
    dataWidth = data->bitwidth;
  } else
    data = nullptr;
}

bool ConsumerData::hasValue() const { return data != nullptr; }

ProducerData::ProducerData(ProducerRW *outs) {
  if (auto *p = dyn_cast_if_present<ChannelProducerRW>(outs)) {
    data = &p->data;
  } else
    data = nullptr;
}

ProducerData &ProducerData::operator=(const ConsumerData &value) {
  if (data)
    *data = *value.data;
  return *this;
}

bool ProducerData::hasValue() const { return data != nullptr; }

ExecutionModel::ExecutionModel(Operation *op) : op(op) {}

void Antotokens::reset(const bool &pvalid1, const bool &pvalid0, bool &kill1,
                       bool &kill0, const bool &generateAt1,
                       const bool &generateAt0, bool &stopValid) {
  regOut0 = false;
  regOut1 = false;

  regIn0 = !pvalid0 && (generateAt0 || regOut0);
  regIn1 = !pvalid1 && (generateAt1 || regOut1);

  stopValid = regOut0 || regOut1;

  kill0 = generateAt0 || regOut0;
  kill1 = generateAt1 || regOut1;
}

void Antotokens::exec(bool isClkRisingEdge, const bool &pvalid1,
                      const bool &pvalid0, bool &kill1, bool &kill0,
                      const bool &generateAt1, const bool &generateAt0,
                      bool &stopValid) {
  if (isClkRisingEdge) {
    regOut0 = regIn0;
    regOut1 = regIn1;
  }
  regIn0 = !pvalid0 && (generateAt0 || regOut0);
  regIn1 = !pvalid1 && (generateAt1 || regOut1);

  stopValid = regOut0 || regOut1;

  kill0 = generateAt0 || regOut0;
  kill1 = generateAt1 || regOut1;
}

ForkSupport::ForkSupport(unsigned size, unsigned datawidth)
    : size(size), datawidth(datawidth) {
  transmitValue.resize(size, false);
  keepValue.resize(size, false);
  blockStopArray.resize(size, false);
}

void ForkSupport::resetDataless(ConsumerRW *ins,
                                std::vector<ProducerRW *> &outs) {
  for (unsigned i = 0; i < size; ++i) {
    transmitValue[i] = true;
    keepValue[i] = !outs[i]->ready;
    outs[i]->valid = ins->valid;
    blockStopArray[i] = !outs[i]->ready;
  }

  // or_n
  anyBlockStop = false;
  for (bool c : blockStopArray)
    anyBlockStop = anyBlockStop || c;

  ins->ready = !anyBlockStop;
  backpressure = ins->valid && anyBlockStop;
}

void ForkSupport::execDataless(bool isClkRisingEdge, ConsumerRW *ins,
                               std::vector<ProducerRW *> &outs) {
  // Combinational view from current register state.
  for (unsigned i = 0; i < outs.size(); ++i) {
    keepValue[i] = !outs[i]->ready && transmitValue[i];
    outs[i]->valid = transmitValue[i] && ins->valid;
    blockStopArray[i] = keepValue[i];
  }

  // or_n
  anyBlockStop = false;
  for (bool c : blockStopArray)
    anyBlockStop = anyBlockStop || c;

  ins->ready = !anyBlockStop;
  backpressure = ins->valid && anyBlockStop;

  // Sequential update uses the current-cycle backpressure/keepValue.
  if (isClkRisingEdge) {
    for (unsigned i = 0; i < outs.size(); ++i)
      transmitValue[i] = keepValue[i] || !backpressure;
  }
}

void ForkSupport::reset(ConsumerRW *ins, std::vector<ProducerRW *> &outs,
                        const ConsumerData &insData,
                        std::vector<ProducerData> &outsData) {
  resetDataless(ins, outs);
  for (auto &out : outsData)
    out = insData;
}

void ForkSupport::exec(bool isClkRisingEdge, ConsumerRW *ins,
                       std::vector<ProducerRW *> &outs,
                       const ConsumerData &insData,
                       std::vector<ProducerData> &outsData) {
  execDataless(isClkRisingEdge, ins, outs);
  for (auto &out : outsData)
    out = insData;
}

JoinSupport::JoinSupport(unsigned size) : size(size) {}

void JoinSupport::exec(std::vector<ConsumerRW *> &ins, ProducerRW *outs) {
  outs->valid = true;
  for (unsigned i = 0; i < size; ++i)
    outs->valid = outs->valid && ins[i]->valid;

  for (unsigned i = 0; i < size; ++i) {
    ins[i]->ready = outs->ready;
    for (unsigned j = 0; j < size; ++j)
      if (i != j)
        ins[i]->ready = ins[i]->ready && ins[j]->valid;
  }
}

OEHBSupport::OEHBSupport(unsigned datawidth) : datawidth(datawidth) {}

void OEHBSupport::resetDataless(ConsumerRW *ins, ProducerRW *outs) {
  outputValid = false;
  ins->ready = !outputValid || outs->ready;
  outs->valid = false;
}

void OEHBSupport::execDataless(bool isClkRisingEdge, ConsumerRW *ins,
                               ProducerRW *outs) {
  ins->ready = !outputValid || outs->ready;
  outs->valid = outputValid;

  if (isClkRisingEdge)
    outputValid = ins->valid || (outputValid && !outs->ready);
}

void OEHBSupport::reset(ConsumerRW *ins, ProducerRW *outs, const Data *insData,
                        Data *outsData) {
  resetDataless(ins, outs);
  if (insData)
    *outsData = APInt(datawidth, 0);
  regEn = ins->ready && ins->valid;
}

void OEHBSupport::exec(bool isClkRisingEdge, ConsumerRW *ins, ProducerRW *outs,
                       const Data *insData, Data *outsData) {
  // Match oehb.v:
  // inputReady = ~outputValid | outs_ready
  // regEn      = inputReady & ins_valid
  // outputValid <= ins_valid | (~outs_ready & outputValid)
  ins->ready = !outputValid || outs->ready;
  outs->valid = outputValid;
  regEn = ins->ready && ins->valid;

  if (isClkRisingEdge) {
    if (regEn && insData)
      *outsData = *insData;
    outputValid = ins->valid || (outputValid && !outs->ready);
  }
}

TEHBSupport::TEHBSupport(unsigned datawidth) : datawidth(datawidth) {}

void TEHBSupport::resetDataless(ConsumerRW *ins, ProducerRW *outs) {
  fullReg = false;
  outputValid = ins->valid || fullReg;
  ins->ready = !fullReg;
  outs->valid = outputValid;
}

void TEHBSupport::execDataless(bool isClkRisingEdge, ConsumerRW *ins,
                               ProducerRW *outs) {
  outputValid = ins->valid || fullReg;
  ins->ready = !fullReg;
  outs->valid = outputValid;

  if (isClkRisingEdge)
    fullReg = (ins->valid || fullReg) && !outs->ready;
}

void TEHBSupport::reset(ConsumerRW *ins, ProducerRW *outs, const Data *insData,
                        Data *outsData) {
  if (insData)
    resetDataFull(ins, outs, insData, outsData);
  else
    resetDataless(ins, outs);
}

void TEHBSupport::exec(bool isClkRisingEdge, ConsumerRW *ins, ProducerRW *outs,
                       const Data *insData, Data *outsData) {
  if (insData)
    execDataFull(isClkRisingEdge, ins, outs, insData, outsData);
  else
    execDataless(isClkRisingEdge, ins, outs);
}

void TEHBSupport::resetDataFull(ConsumerRW *ins, ProducerRW *outs,
                                const Data *insData, Data *outsData) {
  fullReg = false;
  regNotFull = !fullReg;
  outputValid = ins->valid || fullReg;
  outs->valid = outputValid;
  ins->ready = regNotFull;
  regEnable = regNotFull && ins->valid && !outs->ready;
  dataReg = APInt(datawidth, 0);
  *outsData = regNotFull ? *insData : dataReg;
};

void TEHBSupport::execDataFull(bool isClkRisingEdge, ConsumerRW *ins,
                               ProducerRW *outs, const Data *insData,
                               Data *outsData) {
  // Match tehb.v:
  // regNotFull = ~fullReg
  // outs_valid = ins_valid | fullReg
  // regEnable  = regNotFull & ins_valid & ~outs_ready
  // outs       = regNotFull ? ins : dataReg
  regNotFull = !fullReg;
  outputValid = ins->valid || fullReg;
  outs->valid = outputValid;
  ins->ready = regNotFull;
  regEnable = regNotFull && ins->valid && !outs->ready;
  *outsData = regNotFull ? *insData : dataReg;

  if (isClkRisingEdge) {
    if (regEnable)
      dataReg = *insData;
    fullReg = (ins->valid || fullReg) && !outs->ready;
  }
}

BranchModel::BranchModel(handshake::BranchOp branchOp,
                         mlir::DenseMap<Value, RW *> &subset)
    : OpExecutionModel<handshake::BranchOp>(branchOp),
      // get the exact structure for the particular value
      ins(getState<ConsumerRW>(branchOp.getOperand(), subset)),
      outs(getState<ProducerRW>(branchOp.getResult(), subset)),
      // initialize data (nullptr if dataless)
      insData(ins), outsData(outs) {}

void BranchModel::reset() {
  outs->valid = ins->valid;
  ins->ready = outs->ready;
  outsData = insData;
};

void BranchModel::exec(bool isClkRisingEdge) { reset(); }

void BranchModel::printStates() {
  printValue<ConsumerRW, const Data>("ins", ins, insData.data);
  printValue<ProducerRW, Data>("outs", outs, outsData.data);
}

CondBranchModel::CondBranchModel(handshake::ConditionalBranchOp condBranchOp,
                                 mlir::DenseMap<Value, RW *> &subset)
    : OpExecutionModel<handshake::ConditionalBranchOp>(condBranchOp),
      data(getState<ConsumerRW>(condBranchOp.getDataOperand(), subset)),
      condition(getState<ChannelConsumerRW>(condBranchOp.getConditionOperand(),
                                            subset)),
      trueOut(getState<ProducerRW>(condBranchOp.getTrueResult(), subset)),
      falseOut(getState<ProducerRW>(condBranchOp.getFalseResult(), subset)),
      dataData(data), trueOutData(trueOut), falseOutData(falseOut),
      condBrJoin(2) {}

void CondBranchModel::reset() {
  auto k = dataCast<APInt>(condition->data);

  auto cond = k.getBoolValue();
  // join
  std::vector<ConsumerRW *> insJoin = {data, condition};
  ProducerRW outsJoin(brInpValid,
                      (falseOut->ready && !cond) || (trueOut->ready && cond));
  condBrJoin.exec(insJoin, &outsJoin);

  trueOut->valid = cond && brInpValid;
  falseOut->valid = !cond && brInpValid;
  trueOutData = dataData;
  falseOutData = dataData;
}

void CondBranchModel::exec(bool isClkRisingEdge) { reset(); }

void CondBranchModel::printStates() {
  printValue<ConsumerRW, const Data>("data", data, dataData.data);
  printValue<ChannelConsumerRW, const Data>("condition", condition,
                                            &condition->data);
  printValue<ProducerRW, Data>("trueOut", trueOut, trueOutData.data);
  printValue<ProducerRW, Data>("falseOut", falseOut, falseOutData.data);
}

ConstantModel::ConstantModel(handshake::ConstantOp constOp,
                             mlir::DenseMap<Value, RW *> &subset)
    : OpExecutionModel<handshake::ConstantOp>(constOp),
      value(dyn_cast<mlir::IntegerAttr>(constOp.getValue()).getValue()),
      ctrl(getState<ControlConsumerRW>(constOp.getCtrl(), subset)),
      outs(getState<ChannelProducerRW>(constOp.getResult(), subset)) {}

void ConstantModel::reset() {
  outs->data = value;
  outs->valid = ctrl->valid;
  ctrl->ready = outs->ready;
}

void ConstantModel::exec(bool isClkRisingEdge) { reset(); }

void ConstantModel::printStates() {
  llvm::outs() << "ctrl: " << ctrl->valid << " " << ctrl->ready << "\n";
  printValue<ChannelProducerRW, Data>("outs", outs, &outs->data);
}

ControlMergeModel::ControlMergeModel(handshake::ControlMergeOp cMergeOp,
                                     mlir::DenseMap<Value, RW *> &subset)
    : OpExecutionModel<handshake::ControlMergeOp>(cMergeOp),
      size(op->getNumOperands()),
      indexWidth(cMergeOp.getIndex().getType().getDataBitWidth()),
      cMergeTEHB(indexWidth), cMergeFork(2),
      outs(getState<ProducerRW>(cMergeOp.getResult(), subset)),
      index(getState<ChannelProducerRW>(cMergeOp.getIndex(), subset)),
      outsData(outs), insTEHB(dataAvailable, tehbOutReady),
      outsTEHB(tehbOutValid, readyToFork), insFork(tehbOutValid, readyToFork) {
  for (auto oper : cMergeOp->getOperands())
    ins.push_back(getState<ConsumerRW>(oper, subset));

  outsFork = {outs, index};

  for (unsigned i = 0; i < size; ++i)
    insData.emplace_back(ins[i]);

  cMergeFork.datawidth = outsData.dataWidth;
}

void ControlMergeModel::reset() {
  indexTEHB = APInt(indexWidth, 0);

  // process (ins_valid)
  for (unsigned i = 0; i < size; ++i)
    if (ins[i]->valid) {
      indexTEHB = APInt(indexWidth, i);
      break;
    }

  // mergeNotehbDataless
  dataAvailable = false;
  for (auto &in : ins)
    dataAvailable = dataAvailable || in->valid;

  // tehb
  cMergeTEHB.reset(&insTEHB, &outsTEHB, &indexTEHB, &index->data);

  // merge_dataless priority semantics: only the first valid input sees ready.
  bool granted = false;
  for (auto *in : ins) {
    if (!granted && in->valid) {
      in->ready = tehbOutReady;
      granted = true;
    } else {
      in->ready = false;
    }
  }

  // fork dataless
  cMergeFork.resetDataless(&insFork, outsFork);

  outsData = insData[dataCast<APInt>(index->data).getZExtValue()];
}

void ControlMergeModel::exec(bool isClkRisingEdge) {
  indexTEHB = APInt(indexWidth, 0);

  // process (ins_valid)
  for (unsigned i = 0; i < size; ++i)
    if (ins[i]->valid) {
      indexTEHB = APInt(indexWidth, i);
      break;
    }

  // mergeNotehbDataless
  dataAvailable = false;
  for (auto &in : ins)
    dataAvailable = dataAvailable || in->valid;

  // tehb
  cMergeTEHB.exec(isClkRisingEdge, &insTEHB, &outsTEHB, &indexTEHB,
                  &index->data);

  // merge_dataless priority semantics: only the first valid input sees ready.
  bool granted = false;
  for (auto *in : ins) {
    if (!granted && in->valid) {
      in->ready = tehbOutReady;
      granted = true;
    } else {
      in->ready = false;
    }
  }

  // fork dataless
  cMergeFork.execDataless(isClkRisingEdge, &insFork, outsFork);

  outsData = insData[dataCast<APInt>(index->data).getZExtValue()];
}

void ControlMergeModel::printStates() {
  for (unsigned i = 0; i < size; ++i)
    printValue<ConsumerRW, const Data>("ins", ins[i], insData[i].data);
  printValue<ProducerRW, Data>("outs", outs, outsData.data);
  printValue<ChannelProducerRW, Data>("index", index, &index->data);
}

ForkModel::ForkModel(handshake::ForkOp forkOp,
                     mlir::DenseMap<Value, RW *> &subset)
    : OpExecutionModel<handshake::ForkOp>(forkOp), size(op->getNumResults()),
      forkSupport(size), ins(getState<ConsumerRW>(forkOp.getOperand(), subset)),
      insData(ins) {
  for (unsigned i = 0; i < size; ++i)
    outs.push_back(getState<ProducerRW>(forkOp->getResult(i), subset));

  for (unsigned i = 0; i < size; ++i)
    outsData.emplace_back(outs[i]);

  forkSupport.datawidth = insData.dataWidth;
}

void ForkModel::reset() { forkSupport.reset(ins, outs, insData, outsData); }

void ForkModel::exec(bool isClkRisingEdge) {
  forkSupport.exec(isClkRisingEdge, ins, outs, insData, outsData);
}

void ForkModel::printStates() {
  printValue<ConsumerRW, const Data>("ins", ins, insData.data);
  for (unsigned i = 0; i < size; ++i)
    printValue<ProducerRW, Data>("outs", outs[i], outsData[i].data);
}

JoinModel::JoinModel(handshake::JoinOp joinOp,
                     mlir::DenseMap<Value, RW *> &subset)
    : OpExecutionModel<handshake::JoinOp>(joinOp),
      outs(getState<ProducerRW>(joinOp.getResult(), subset)),
      join(joinOp->getNumOperands()) {
  for (auto oper : joinOp->getOperands())
    ins.push_back(getState<ConsumerRW>(oper, subset));
}

void JoinModel::reset() { join.exec(ins, outs); }

void JoinModel::exec(bool isClkRisingEdge) { reset(); }

void JoinModel::printStates() {
  for (auto *in : ins)
    llvm::outs() << "Ins: " << in->valid << " " << in->ready << "\n";
  llvm::outs() << "Outs: " << outs->valid << " " << outs->ready << "\n";
}

LazyForkModel::LazyForkModel(handshake::LazyForkOp lazyForkOp,
                             mlir::DenseMap<Value, RW *> &subset)
    : OpExecutionModel<handshake::LazyForkOp>(lazyForkOp),
      size(lazyForkOp->getNumResults()),
      ins(getState<ConsumerRW>(lazyForkOp.getOperand(), subset)), insData(ins) {

  for (unsigned i = 0; i < size; ++i)
    outs.push_back(getState<ProducerRW>(lazyForkOp->getResult(i), subset));

  for (unsigned i = 0; i < size; ++i)
    outsData.emplace_back(outs[i]);
}

void LazyForkModel::reset() {
  ins->ready = true;
  for (unsigned i = 0; i < size; ++i) {
    bool tempReady = true;
    for (unsigned j = 0; j < size; ++j)
      if (i != j)
        tempReady = tempReady && outs[j]->ready;

    ins->ready = ins->ready && outs[i]->ready;
    outs[i]->valid = ins->valid && tempReady;
  }
  for (unsigned i = 0; i < size; ++i)
    outsData[i] = insData;
}

void LazyForkModel::exec(bool isClkRisingEdge) { reset(); }

void LazyForkModel::printStates() {
  printValue<ConsumerRW, const Data>("ins", ins, insData.data);
  for (unsigned i = 0; i < size; ++i)
    printValue<ProducerRW, Data>("outs", outs[i], outsData[i].data);
}

MergeModel::MergeModel(handshake::MergeOp mergeOp,
                       mlir::DenseMap<Value, RW *> &subset)
    : OpExecutionModel<handshake::MergeOp>(mergeOp),
      size(mergeOp->getNumOperands()),
      outs(getState<ProducerRW>(mergeOp.getResult(), subset)), outsData(outs),
      insTEHB(tehbValid, tehbReady) {
  for (unsigned i = 0; i < size; ++i)
    ins.push_back(getState<ConsumerRW>(mergeOp->getOperand(i), subset));
  for (unsigned i = 0; i < size; ++i)
    insData.emplace_back(ins[i]);

  mergeTEHB.datawidth = outsData.dataWidth;
}

void MergeModel::reset() {
  if (outsData.hasValue())
    execDataFull();
  else
    execDataless();
}

void MergeModel::exec(bool isClkRisingEdge) {
  (void)isClkRisingEdge;
  if (outsData.hasValue())
    execDataFull();
  else
    execDataless();
}

void MergeModel::printStates() {
  for (unsigned i = 0; i < size; ++i)
    printValue<ConsumerRW, const Data>("ins", ins[i], insData[i].data);
  printValue<ProducerRW, Data>("outs", outs, outsData.data);
}

void MergeModel::execDataless() {
  bool selectedValid = false;
  int selected = -1;
  for (unsigned i = 0; i < size; ++i) {
    if (!selectedValid && ins[i]->valid) {
      selectedValid = true;
      selected = static_cast<int>(i);
    }
  }

  outs->valid = selectedValid;
  for (unsigned i = 0; i < size; ++i)
    ins[i]->ready = (static_cast<int>(i) == selected) ? outs->ready : false;
}

void MergeModel::execDataFull() {
  bool selectedValid = false;
  int selected = -1;
  Data selectedData = outsData.hasValue() ? *outsData.data : Data();
  if (!insData.empty() && insData[0].hasValue())
    selectedData = *insData[0].data;

  for (unsigned i = 0; i < size; ++i) {
    if (!selectedValid && ins[i]->valid) {
      selectedValid = true;
      selected = static_cast<int>(i);
      if (insData[i].hasValue())
        selectedData = *insData[i].data;
    }
  }

  outs->valid = selectedValid;
  for (unsigned i = 0; i < size; ++i)
    ins[i]->ready = (static_cast<int>(i) == selected) ? outs->ready : false;
  if (outsData.data)
    *outsData.data = selectedData;
}

MuxModel::MuxModel(handshake::MuxOp muxOp, mlir::DenseMap<Value, RW *> &subset)
    : OpExecutionModel<handshake::MuxOp>(muxOp),
      size(muxOp.getDataOperands().size()),
      selectWidth(muxOp.getSelectOperand().getType().getDataBitWidth()),
      index(getState<ChannelConsumerRW>(muxOp.getSelectOperand(), subset)),
      outs(getState<ProducerRW>(muxOp.getResult(), subset)), outsData(outs) {
  for (auto oper : muxOp.getDataOperands())
    ins.push_back(getState<ChannelConsumerRW>(oper, subset));
  for (unsigned i = 0; i < size; ++i)
    insData.emplace_back(ins[i]);
}

void MuxModel::reset() {
  indexNum = dataCast<APInt>(index->data).getZExtValue();
  if (outsData.hasValue())
    execDataFull();
  else
    execDataless();
}

void MuxModel::exec(bool isClkRisingEdge) {
  (void)isClkRisingEdge;
  indexNum = dataCast<APInt>(index->data).getZExtValue();
  if (outsData.hasValue())
    execDataFull();
  else
    execDataless();
}

void MuxModel::printStates() {
  for (unsigned i = 0; i < size; ++i)
    printValue<ConsumerRW, const Data>("ins", ins[i], insData[i].data);
  printValue<ProducerRW, Data>("outs", outs, outsData.data);
  printValue<ChannelConsumerRW, const Data>("index", index, &index->data);
}

void MuxModel::execDataless() {
  bool selectedDataValid = false;
  bool indexEqual = false;
  for (unsigned i = 0; i < size; ++i) {
    indexEqual = (i == indexNum);

    if (indexEqual && index->valid && ins[i]->valid)
      selectedDataValid = true;
    ins[i]->ready =
        (indexEqual && index->valid && ins[i]->valid && outs->ready) ||
        !ins[i]->valid;
  }
  index->ready = !index->valid || (selectedDataValid && outs->ready);
  outs->valid = selectedDataValid;
}

void MuxModel::execDataFull() {
  bool selectedDataValid = false;
  bool indexEqual = false;
  Data selectedData = *insData[0].data;
  for (unsigned i = 0; i < size; ++i) {
    indexEqual = (i == indexNum);

    if (indexEqual && index->valid && ins[i]->valid) {
      selectedDataValid = true;
      selectedData = *insData[i].data;
    }
    ins[i]->ready =
        (indexEqual && index->valid && ins[i]->valid && outs->ready) ||
        !ins[i]->valid;
  }
  index->ready = !index->valid || (selectedDataValid && outs->ready);
  outs->valid = selectedDataValid;
  if (outsData.data)
    *outsData.data = selectedData;
}

OEHBModel::OEHBModel(handshake::BufferOp oehbOp,
                     mlir::DenseMap<Value, RW *> &subset)
    : OpExecutionModel<handshake::BufferOp>(oehbOp),
      ins(getState<ConsumerRW>(oehbOp.getOperand(), subset)),
      outs(getState<ProducerRW>(oehbOp.getResult(), subset)), insData(ins),
      outsData(outs), oehbDl(insData.dataWidth) {}

void OEHBModel::reset() {
  oehbDl.reset(ins, outs, insData.data, outsData.data);
}

void OEHBModel::exec(bool isClkRisingEdge) {
  oehbDl.exec(isClkRisingEdge, ins, outs, insData.data, outsData.data);
}

void OEHBModel::printStates() {
  printValue<ConsumerRW, const Data>("ins", ins, insData.data);
  printValue<ProducerRW, Data>("outs", outs, outsData.data);
}

SinkModel::SinkModel(handshake::SinkOp sinkOp,
                     mlir::DenseMap<Value, RW *> &subset)
    : OpExecutionModel<handshake::SinkOp>(sinkOp),
      ins(getState<ConsumerRW>(sinkOp.getOperand(), subset)), insData(ins) {}

void SinkModel::reset() { ins->ready = true; }

void SinkModel::exec(bool isClkRisingEdge) { reset(); }

void SinkModel::printStates() {
  printValue<ConsumerRW, const Data>("ins", ins, insData.data);
}

SourceModel::SourceModel(handshake::SourceOp sourceOp,
                         mlir::DenseMap<Value, RW *> &subset)
    : OpExecutionModel<handshake::SourceOp>(sourceOp),
      outs(getState<ProducerRW>(sourceOp.getResult(), subset)) {}

void SourceModel::reset() { outs->valid = true; }

void SourceModel::exec(bool isClkRisingEdge) { reset(); }

void SourceModel::printStates() {
  llvm::outs() << "Outs: " << outs->valid << " " << outs->ready << "\n";
}

TEHBModel::TEHBModel(handshake::BufferOp tehbOp,
                     mlir::DenseMap<Value, RW *> &subset)
    : OpExecutionModel<handshake::BufferOp>(tehbOp),
      ins(getState<ConsumerRW>(tehbOp.getOperand(), subset)),
      outs(getState<ProducerRW>(tehbOp.getResult(), subset)), insData(ins),
      outsData(outs), returnTEHB(insData.dataWidth) {}

void TEHBModel::reset() {
  returnTEHB.reset(ins, outs, insData.data, outsData.data);
}

void TEHBModel::exec(bool isClkRisingEdge) {
  returnTEHB.exec(isClkRisingEdge, ins, outs, insData.data, outsData.data);
}

void TEHBModel::printStates() {
  printValue<ConsumerRW, const Data>("ins", ins, insData.data);
  printValue<ProducerRW, Data>("outs", outs, outsData.data);
}

//===----------------------------------------------------------------------===//
// Arithmetic
//===----------------------------------------------------------------------===//
/// Arithmetic and generic components

TruncIModel::TruncIModel(handshake::TruncIOp trunciOp,
                         mlir::DenseMap<Value, RW *> &subset)
    : OpExecutionModel<handshake::TruncIOp>(trunciOp),
      outputWidth(trunciOp.getResult().getType().getDataBitWidth()),
      ins(getState<ChannelConsumerRW>(trunciOp.getOperand(), subset)),
      outs(getState<ChannelProducerRW>(trunciOp.getResult(), subset)) {}

void TruncIModel::reset() {
  outs->data = dataCast<const APInt>(ins->data).trunc(outputWidth);
  outs->valid = ins->valid;
  ins->ready = !ins->valid || (ins->valid && outs->ready);
}

void TruncIModel::exec(bool isClkRisingEdge) { reset(); }

void TruncIModel::printStates() {
  printValue<ConsumerRW, const Data>("ins", ins, &ins->data);
  printValue<ProducerRW, Data>("outs", outs, &outs->data);
}

SelectModel::SelectModel(handshake::SelectOp selectOp,
                         mlir::DenseMap<Value, RW *> &subset)
    : OpExecutionModel<handshake::SelectOp>(selectOp),
      condition(getState<ChannelConsumerRW>(selectOp.getCondition(), subset)),
      trueValue(getState<ChannelConsumerRW>(selectOp.getTrueValue(), subset)),
      falseValue(getState<ChannelConsumerRW>(selectOp.getFalseValue(), subset)),
      result(getState<ChannelProducerRW>(selectOp.getResult(), subset)) {}

void SelectModel::reset() {
  selectExec();
  anti.reset(falseValue->valid, trueValue->valid, kill1, kill0, g1, g0,
             antitokenStop);
}

void SelectModel::exec(bool isClkRisingEdge) {
  selectExec();
  anti.exec(isClkRisingEdge, falseValue->valid, trueValue->valid, kill1, kill0,
            g1, g0, antitokenStop);
}

void SelectModel::printStates() {
  printValue<ConsumerRW, const Data>("condition", condition, &condition->data);
  printValue<ConsumerRW, const Data>("trueValue", trueValue, &trueValue->data);
  printValue<ConsumerRW, const Data>("falseValue", falseValue,
                                     &falseValue->data);
  printValue<ProducerRW, Data>("result", result, &result->data);
}

void SelectModel::selectExec() {
  auto cond = dataCast<APInt>(condition->data).getBoolValue();

  ee = condition->valid &&
       ((!cond && falseValue->valid) || (cond && trueValue->valid));
  validInternal = ee && !antitokenStop;
  g0 = !trueValue->valid && validInternal && result->ready;
  g1 = !falseValue->valid && validInternal && result->ready;

  result->valid = validInternal;
  trueValue->ready =
      !trueValue->valid || (validInternal && result->ready) || kill0;
  falseValue->ready =
      !falseValue->valid || (validInternal && result->ready) || kill1;
  condition->ready = !condition->valid || (validInternal && result->ready);

  if (cond)
    result->data = trueValue->data;
  else
    result->data = falseValue->data;
}

template <typename Op>
GenericUnaryOpModel<Op>::GenericUnaryOpModel(
    Op op, mlir::DenseMap<Value, RW *> &subset, const UnaryCompFunc &callback)
    : OpExecutionModel<Op>(op),
      outputWidth(op.getResult().getType().getDataBitWidth()),
      callback(callback),
      ins(OpExecutionModel<Op>::template getState<ChannelConsumerRW>(
          op.getOperand(), subset)),
      outs(OpExecutionModel<Op>::template getState<ChannelProducerRW>(
          op.getResult(), subset)) {}

template <typename Op>
void GenericUnaryOpModel<Op>::reset() {
  outs->data = callback(ins->data, outputWidth);
  outs->valid = ins->valid;
  ins->ready = outs->ready;
}

template <typename Op>
void GenericUnaryOpModel<Op>::exec(bool isClkRisingEdge) {
  reset();
}

template <typename Op>
void GenericUnaryOpModel<Op>::printStates() {
  OpExecutionModel<Op>::template printValue<ConsumerRW, const Data>("ins", ins,
                                                                    &ins->data);
  OpExecutionModel<Op>::template printValue<ProducerRW, Data>("outs", outs,
                                                              &outs->data);
}

template <typename Op>
::GenericBinaryOpModel<Op>::GenericBinaryOpModel(
    Op op, mlir::DenseMap<Value, RW *> &subset, const BinaryCompFunc &callback,
    unsigned latency)
    : OpExecutionModel<Op>(op), callback(callback), latency(latency),
      bitwidth(cast<handshake::ChannelType>(op.getResult().getType())
                   .getDataBitWidth()),
      lhs(OpExecutionModel<Op>::template getState<ChannelConsumerRW>(
          op.getLhs(), subset)),
      rhs(OpExecutionModel<Op>::template getState<ChannelConsumerRW>(
          op.getRhs(), subset)),
      result(OpExecutionModel<Op>::template getState<ChannelProducerRW>(
          op.getResult(), subset)),
      insJoin({lhs, rhs}), binJoin(2) {
  if (latency > 0) {
    // RTL arithmetic units model validity as:
    // delay_buffer(size = latency - 1) + oehb_dataless.
    unsigned validDelay = latency > 0 ? latency - 1 : 0;
    validPipeline.assign(validDelay, false);
    dataPipeline.assign(latency, APInt(bitwidth, 0));
    currentCombData = APInt(bitwidth, 0);
  }
}

template <typename Op>
void ::GenericBinaryOpModel<Op>::reset() {
  if (!latency) {
    binJoin.exec(insJoin, result);
    result->data = callback(lhs->data, rhs->data);
    return;
  }

  // Arithmetic RTL units use a pipelined datapath with an OEHB-like output
  // valid register. The pipeline only advances when oehbReady is true.
  joinValid = lhs->valid && rhs->valid;
  oehbReady = !outputValid || result->ready;
  lhs->ready = oehbReady && rhs->valid;
  rhs->ready = oehbReady && lhs->valid;
  result->valid = outputValid;
  if (!dataPipeline.empty())
    result->data = dataPipeline.back();
  else
    result->data = currentCombData;
}

template <typename Op>
void ::GenericBinaryOpModel<Op>::exec(bool isClkRisingEdge) {
  if (!latency) {
    reset();
    return;
  }

  currentCombData = callback(lhs->data, rhs->data);
  reset();

  if (!isClkRisingEdge)
    return;

  bool buffValid = validPipeline.empty() ? joinValid : validPipeline.back();
  bool oldOutputValid = outputValid;

  if (oehbReady) {
    if (dataPipeline.size() > 1) {
      for (unsigned i = dataPipeline.size() - 1; i > 0; --i)
        dataPipeline[i] = dataPipeline[i - 1];
    }
    if (!dataPipeline.empty())
      dataPipeline[0] = currentCombData;

    if (validPipeline.size() > 1) {
      for (unsigned i = validPipeline.size() - 1; i > 0; --i)
        validPipeline[i] = validPipeline[i - 1];
    }
    if (!validPipeline.empty())
      validPipeline[0] = joinValid;
  }

  outputValid = buffValid || (!result->ready && oldOutputValid);
}

template <typename Op>
void ::GenericBinaryOpModel<Op>::printStates() {
  OpExecutionModel<Op>::template printValue<ConsumerRW, const Data>("lhs", lhs,
                                                                    &lhs->data);
  OpExecutionModel<Op>::template printValue<ConsumerRW, const Data>("rhs", rhs,
                                                                    &rhs->data);
  OpExecutionModel<Op>::template printValue<ProducerRW, Data>("result", result,
                                                              &result->data);
}

//===----------------------------------------------------------------------===//
// Simulator
//===----------------------------------------------------------------------===//

namespace {
static std::string trimCopy(StringRef str) {
  StringRef trimmed = str.trim();
  return trimmed.str();
}

static std::vector<std::string> splitTokens(StringRef line) {
  std::vector<std::string> tokens;
  SmallVector<StringRef> parts;
  line.split(parts, ',', -1, false);
  for (StringRef part : parts) {
    SmallVector<StringRef> subtokens;
    part.split(subtokens, ' ', -1, false);
    for (StringRef token : subtokens) {
      if (token.trim().empty())
        continue;
      tokens.push_back(token.trim().str());
    }
  }
  return tokens;
}

static uint64_t countToggles(const std::vector<uint8_t> &values) {
  if (values.empty())
    return 0;
  uint64_t toggles = 0;
  for (size_t i = 1, e = values.size(); i < e; ++i) {
    if (values[i] != values[i - 1])
      ++toggles;
  }
  return toggles;
}

static uint64_t countTogglesFromCycle(const std::vector<uint8_t> &values,
                                      unsigned startCycle) {
  if (values.size() < 2 || startCycle >= values.size())
    return 0;
  size_t begin = std::max<size_t>(1, static_cast<size_t>(startCycle) + 1);
  uint64_t toggles = 0;
  for (size_t i = begin, e = values.size(); i < e; ++i) {
    if (values[i] != values[i - 1])
      ++toggles;
  }
  return toggles;
}

static uint64_t countOnes(const std::vector<uint8_t> &values) {
  return llvm::count(values, static_cast<uint8_t>(1));
}

static uint64_t countDataBitTogglesFromCycle(const std::vector<APInt> &values,
                                             unsigned startCycle) {
  if (values.size() < 2 || startCycle >= values.size())
    return 0;
  size_t begin = std::max<size_t>(1, static_cast<size_t>(startCycle) + 1);
  uint64_t toggles = 0;
  for (size_t i = begin, e = values.size(); i < e; ++i) {
    unsigned width = std::max(values[i].getBitWidth(), values[i - 1].getBitWidth());
    APInt lhs = values[i].zextOrTrunc(width);
    APInt rhs = values[i - 1].zextOrTrunc(width);
    toggles += static_cast<uint64_t>((lhs ^ rhs).popcount());
  }
  return toggles;
}

static std::optional<unsigned> parseSingleBitIndex(StringRef token) {
  token = token.trim();
  if (!token.startswith("[") || !token.endswith("]"))
    return std::nullopt;
  token = token.drop_front().drop_back();
  if (token.contains(':'))
    return std::nullopt;
  uint64_t idx = 0;
  if (token.getAsInteger(10, idx))
    return std::nullopt;
  return static_cast<unsigned>(idx);
}

static APInt normalizeAPInt(const APInt &value, unsigned width) {
  width = std::max(1u, width);
  if (value.getBitWidth() == width)
    return value;
  return value.zextOrTrunc(width);
}

static APInt dataToAPInt(const Data &value, unsigned widthHint = 0) {
  unsigned width = std::max(1u, widthHint);
  if (value.hasValue() && value.bitwidth)
    width = std::max(width, value.bitwidth);

  if (const APInt *intValue = dataCast<APInt>(&value))
    return normalizeAPInt(*intValue, width);

  if (const APFloat *floatValue = dataCast<APFloat>(&value))
    return normalizeAPInt(floatValue->bitcastToAPInt(), width);

  return APInt(width, 0);
}

static uint64_t bitToggleDistance(const APInt &lhs, const APInt &rhs) {
  unsigned width = std::max(lhs.getBitWidth(), rhs.getBitWidth());
  APInt lhsNorm = normalizeAPInt(lhs, width);
  APInt rhsNorm = normalizeAPInt(rhs, width);
  return static_cast<uint64_t>((lhsNorm ^ rhsNorm).popcount());
}

static std::optional<uint64_t> getStaticMemrefSize(MemRefType memrefType) {
  uint64_t size = 1;
  for (int64_t dim : memrefType.getShape()) {
    if (dim < 0)
      return std::nullopt;
    size *= static_cast<uint64_t>(dim);
  }
  return size;
}

static bool parseDatFile(const std::string &path,
                         std::vector<std::vector<std::string>> &transactions) {
  std::ifstream input(path);
  if (!input.is_open())
    return false;

  transactions.clear();
  bool inRuntime = false;
  bool inTxn = false;
  std::string line;
  while (std::getline(input, line)) {
    StringRef ref(line);
    ref = ref.trim();
    if (ref.empty())
      continue;
    if (ref == "[[[runtime]]]") {
      inRuntime = true;
      continue;
    }
    if (ref == "[[[/runtime]]]") {
      inRuntime = false;
      continue;
    }
    if (!inRuntime)
      continue;

    if (ref.startswith("[[transaction]]")) {
      transactions.emplace_back();
      inTxn = true;
      continue;
    }
    if (ref == "[[/transaction]]") {
      inTxn = false;
      continue;
    }
    if (inTxn) {
      transactions.back().push_back(ref.str());
    }
  }
  return !transactions.empty();
}

static int logicBit(char c) { return c == '1' ? 1 : 0; }
} // namespace

Simulator::Simulator(handshake::FuncOp funcOp, const SimulatorOptions &options)
    : funcOp(funcOp), options(options) {
  clearFailure();
  initializeMetadata();

  for (BlockArgument arg : funcOp.getArguments())
    associateState(arg, funcOp, funcOp->getLoc());

  for (Operation &op : funcOp.getOps()) {
    for (Value res : op.getResults())
      associateState(res, &op, op.getLoc());
  }

  auto endOps = funcOp.getOps<handshake::EndOp>();
  if (endOps.empty()) {
    (void)signalFailure("No handshake.end operation found in function");
    return;
  }
  endOp = *endOps.begin();

  resValid.assign(endOp->getNumOperands(), false);
  resReady.assign(endOp->getNumOperands(), true);
  resData.resize(endOp->getNumOperands());

  for (Operation &op : funcOp.getOps())
    associateModel(&op);

  endObservers.clear();
  for (unsigned idx = 0, e = endOp->getNumOperands(); idx < e; ++idx) {
    ConsumerRW *in = consumerViews[&endOp->getOpOperand(idx)];
    EndObserver observer;
    observer.in = in;
    observer.inData = ConsumerData(in);
    observer.seen = false;
    endObservers.push_back(observer);
  }

  initializeEdgeCatalog();
  reset();
}

void Simulator::initializeMetadata() {
  argOrder.clear();
  argNames.clear();
  resNames.clear();
  argNameToIndex.clear();
  memoryImages.clear();

  auto argNamesAttr = funcOp->getAttrOfType<ArrayAttr>("argNames");
  auto resNamesAttr = funcOp->getAttrOfType<ArrayAttr>("resNames");

  for (auto [idx, arg] : llvm::enumerate(funcOp.getArguments())) {
    argOrder.push_back(arg);
    std::string argName = "arg" + std::to_string(idx);
    if (argNamesAttr && idx < argNamesAttr.size())
      argName = cast<StringAttr>(argNamesAttr[idx]).getValue().str();
    argNames.push_back(argName);
    argNameToIndex[argName] = idx;

    if (auto memref = dyn_cast<MemRefType>(arg.getType())) {
      auto memSize = getStaticMemrefSize(memref);
      if (!memSize) {
        (void)signalFailure("Dynamic memrefs are not supported by simulator");
        continue;
      }

      MemoryImage image;
      image.arg = arg;
      image.type = memref;
      image.name = argName;
      image.values.resize(*memSize);
      image.writtenMask.assign(*memSize, 0);

      Type elemType = memref.getElementType();
      if (auto intType = dyn_cast<IntegerType>(elemType)) {
        for (Data &data : image.values)
          data = APInt(intType.getWidth(), 0, intType.isSigned());
      } else if (auto floatType = dyn_cast<FloatType>(elemType)) {
        for (Data &data : image.values)
          data = APFloat::getZero(floatType.getFloatSemantics());
      } else {
        (void)signalFailure("Unsupported memref element type");
      }
      memoryImages.insert({arg, image});
    }
  }

  for (unsigned idx = 0, e = funcOp.getNumResults(); idx < e; ++idx) {
    std::string resName = "out" + std::to_string(idx);
    if (resNamesAttr && idx < resNamesAttr.size())
      resName = cast<StringAttr>(resNamesAttr[idx]).getValue().str();
    resNames.push_back(resName);
  }
}

void Simulator::clearFailure() {
  failedFlag = false;
  failureMessage.clear();
}

LogicalResult Simulator::fail(StringRef message) {
  failedFlag = true;
  failureMessage = message.str();
  return failure();
}

LogicalResult Simulator::signalFailure(StringRef message) { return fail(message); }

void Simulator::initializeEdgeCatalog() {
  edgeTraces.clear();
  for (auto &[opOperand, edge] : consumerViews) {
    EdgeTrace trace;
    trace.src = getValueProducerName(opOperand->get());
    trace.dst = getOperationName(opOperand->getOwner());
    trace.operandIndex = opOperand->getOperandNumber();
    trace.isToEnd = isa<handshake::EndOp>(opOperand->getOwner());
    trace.edge = edge;
    trace.edgeData = ConsumerData(edge);
    trace.hasData = trace.edgeData.hasValue();
    trace.dataWidth = trace.edgeData.dataWidth;
    // Capture all data-channel waves so per-channel VCD comparison can be done
    // on raw bit-vectors (not only top-level interfaces).
    trace.captureDataWave = trace.hasData;
    edgeTraces.push_back(std::move(trace));
  }

  llvm::sort(edgeTraces, [&](const EdgeTrace &lhs, const EdgeTrace &rhs) {
    if (lhs.src != rhs.src)
      return lhs.src < rhs.src;
    if (lhs.dst != rhs.dst)
      return lhs.dst < rhs.dst;
    return lhs.operandIndex < rhs.operandIndex;
  });
}

bool Simulator::sampleEdgeStates() {
  bool sawTransfer = false;
  for (EdgeTrace &trace : edgeTraces) {
    uint8_t valid = trace.edge->valid ? 1 : 0;
    uint8_t ready = trace.edge->ready ? 1 : 0;
    uint8_t transfer = (valid && ready) ? 1 : 0;
    sawTransfer = sawTransfer || transfer;
    trace.valid.push_back(valid);
    trace.ready.push_back(ready);
    trace.transfer.push_back(transfer);

    if (!trace.hasData || !trace.edgeData.data)
      continue;

    APInt curData = dataToAPInt(*trace.edgeData.data, trace.dataWidth);
    if (trace.prevData.has_value()) {
      if (curData != *trace.prevData)
        ++trace.dataWordToggles;
      trace.dataBitToggles += bitToggleDistance(curData, *trace.prevData);
    }
    trace.prevData = curData;

    if (trace.captureDataWave)
      trace.dataWave.push_back(curData);

    if (!transfer)
      continue;

    if (trace.prevTransferData.has_value()) {
      if (curData != *trace.prevTransferData)
        ++trace.transferDataWordToggles;
      trace.transferDataBitToggles +=
          bitToggleDistance(curData, *trace.prevTransferData);
    }
    trace.prevTransferData = curData;
  }
  return sawTransfer;
}

void Simulator::finalizeSwitchingStats() {
  channelWaves.clear();
  channelWaves.reserve(edgeTraces.size());
  for (const EdgeTrace &trace : edgeTraces) {
    ChannelSwitching sw;
    sw.src = trace.src;
    sw.dst = trace.dst;
    sw.operandIndex = trace.operandIndex;
    sw.isToEnd = trace.isToEnd;
    sw.hasData = trace.hasData;
    sw.dataWidth = trace.dataWidth;
    sw.valid = trace.valid;
    sw.ready = trace.ready;
    sw.transfer = trace.transfer;
    sw.dataWave = trace.dataWave;
    sw.validToggles = countToggles(sw.valid);
    sw.readyToggles = countToggles(sw.ready);
    sw.transferToggles = countToggles(sw.transfer);
    sw.dataWordToggles = trace.dataWordToggles;
    sw.dataBitToggles = trace.dataBitToggles;
    sw.transferDataWordToggles = trace.transferDataWordToggles;
    sw.transferDataBitToggles = trace.transferDataBitToggles;
    sw.transfers = countOnes(sw.transfer);
    channelWaves.push_back(std::move(sw));
  }
}

void Simulator::reset() {
  if (failedFlag)
    return;

  while (true) {
    for (auto &[_, model] : opModels)
      model->reset();
    bool fixedPoint = true;
    for (auto &[_, updater] : updaters)
      fixedPoint = fixedPoint && updater->check();
    if (fixedPoint)
      break;
    for (auto &[_, updater] : updaters)
      updater->update();
  }
  for (auto &[_, updater] : updaters)
    updater->update();
}

LogicalResult Simulator::parseInputToken(Type type, StringRef token,
                                         Data &out) const {
  token = token.trim();
  if (token.empty())
    return failure();

  if (auto channelType = dyn_cast<handshake::ChannelType>(type))
    type = channelType.getDataType();

  if (auto intType = dyn_cast<IntegerType>(type)) {
    bool negative = token.startswith("-");
    StringRef body = negative ? token.drop_front() : token;
    APInt value(intType.getWidth(), 0, intType.isSigned());
    if (body.starts_with_insensitive("0x")) {
      value = APInt(intType.getWidth(), body.drop_front(2), 16);
    } else {
      value = APInt(intType.getWidth(), body, 10);
    }
    if (negative)
      value = -value;
    out = value;
    return success();
  }

  if (auto floatType = dyn_cast<FloatType>(type)) {
    APFloat value(floatType.getFloatSemantics(), token);
    out = value;
    return success();
  }
  return failure();
}

std::string Simulator::dataToString(Type type, const Data &data) const {
  if (auto channelType = dyn_cast<handshake::ChannelType>(type))
    type = channelType.getDataType();

  if (isa<IntegerType>(type)) {
    APInt intVal = dataCast<APInt>(data);
    SmallString<64> str;
    intVal.toString(str, 16, false);
    std::string out = str.str().str();
    if (out.empty())
      out = "0";
    return "0x" + out;
  }

  if (isa<FloatType>(type)) {
    SmallVector<char> buffer;
    dataCast<APFloat>(data).toString(buffer);
    return std::string(buffer.begin(), buffer.end());
  }

  return "";
}

bool Simulator::getArgIndexByName(StringRef argName, unsigned &argIdx) const {
  auto it = argNameToIndex.find(argName.str());
  if (it == argNameToIndex.end())
    return false;
  argIdx = it->second;
  return true;
}

std::string Simulator::getOperationName(Operation *op) const {
  if (!op)
    return "";
  if (auto attr = op->getAttrOfType<StringAttr>("handshake.name"))
    return attr.getValue().str();
  if (isa<handshake::EndOp>(op))
    return "end";
  std::string fallback;
  raw_string_ostream os(fallback);
  os << op->getName().getStringRef();
  return os.str();
}

std::string Simulator::getValueProducerName(Value val) const {
  if (auto blockArg = dyn_cast<BlockArgument>(val)) {
    unsigned idx = blockArg.getArgNumber();
    if (idx < argNames.size())
      return argNames[idx];
    return "arg" + std::to_string(idx);
  }
  return getOperationName(val.getDefiningOp());
}

LogicalResult Simulator::readMemory(Value memref, const Data &addr, Data &outData,
                                    Location loc) {
  auto it = memoryImages.find(memref);
  if (it == memoryImages.end())
    return fail("Missing memory image for memref");

  uint64_t index = dataCast<APInt>(addr).getLimitedValue();
  if (index >= it->second.values.size()) {
    std::string err;
    raw_string_ostream os(err);
    os << "Out-of-bounds memory read at index " << index;
    emitError(loc) << os.str();
    return fail(os.str());
  }

  outData = it->second.values[index];
  return success();
}

LogicalResult Simulator::writeMemory(Value memref, const Data &addr,
                                     const Data &inData, Location loc) {
  auto it = memoryImages.find(memref);
  if (it == memoryImages.end())
    return fail("Missing memory image for memref");

  uint64_t index = dataCast<APInt>(addr).getLimitedValue();
  if (index >= it->second.values.size()) {
    std::string err;
    raw_string_ostream os(err);
    os << "Out-of-bounds memory write at index " << index;
    emitError(loc) << os.str();
    return fail(os.str());
  }

  it->second.values[index] = inData;
  it->second.writtenMask[index] = 1;
  return success();
}

LogicalResult Simulator::parseDatInputDir(StringRef inputVectorsDir,
                                          SimulationInputs &inputs) {
  namespace fs = std::filesystem;
  fs::path inputDir(inputVectorsDir.str());
  if (!fs::exists(inputDir) || !fs::is_directory(inputDir))
    return fail("Input vectors directory does not exist");

  for (const fs::directory_entry &entry : fs::directory_iterator(inputDir)) {
    if (!entry.is_regular_file())
      continue;
    std::string filename = entry.path().filename().string();
    if (!StringRef(filename).startswith("input_") ||
        !StringRef(filename).endswith(".dat"))
      continue;

    StringRef nameRef(filename);
    StringRef argName = nameRef.drop_front(strlen("input_")).drop_back(4);
    unsigned argIdx = 0;
    if (!getArgIndexByName(argName, argIdx))
      continue;

    std::vector<std::vector<std::string>> txns;
    if (!parseDatFile(entry.path().string(), txns))
      return fail("Failed parsing dat file " + entry.path().string());

    ArgInputTransactions argInputs;
    argInputs.provided = true;
    for (const auto &txn : txns) {
      SmallVector<std::string> values(txn.begin(), txn.end());
      argInputs.transactions.push_back(values);
    }
    inputs.byArgIndex[argIdx] = argInputs;
  }
  return success();
}

LogicalResult Simulator::parsePlainInputFile(StringRef inputArgsFile,
                                             SimulationInputs &inputs) {
  std::ifstream input(inputArgsFile.str());
  if (!input.is_open())
    return fail("Failed to open plain input file");

  std::vector<unsigned> unnamedArgs;
  for (unsigned i = 0, e = funcOp.getNumArguments(); i < e; ++i)
    unnamedArgs.push_back(i);
  unsigned unnamedCursor = 0;

  std::string line;
  while (std::getline(input, line)) {
    std::string trimmed = trimCopy(line);
    if (trimmed.empty() || StringRef(trimmed).startswith("#"))
      continue;

    unsigned argIdx = 0;
    StringRef rhs(trimmed);
    size_t eqPos = trimmed.find('=');
    if (eqPos != std::string::npos) {
      std::string lhs = trimCopy(trimmed.substr(0, eqPos));
      rhs = StringRef(trimmed).drop_front(eqPos + 1).trim();
      if (!getArgIndexByName(lhs, argIdx))
        return fail("Unknown argument name in plain input file: " + lhs);
    } else {
      if (unnamedCursor >= unnamedArgs.size())
        return fail("Too many unnamed lines in plain input file");
      argIdx = unnamedArgs[unnamedCursor++];
    }

    ArgInputTransactions parsed;
    parsed.provided = true;
    SmallVector<StringRef> txnParts;
    rhs.split(txnParts, ';', -1, false);
    if (txnParts.empty())
      txnParts.push_back(rhs);

    for (StringRef part : txnParts) {
      std::vector<std::string> vals = splitTokens(part);
      SmallVector<std::string> txnVals(vals.begin(), vals.end());
      parsed.transactions.push_back(txnVals);
    }
    inputs.byArgIndex[argIdx] = parsed;
  }

  return success();
}

LogicalResult Simulator::finalizeInputTransactions(SimulationInputs &inputs) {
  unsigned txCount = 0;
  for (auto &[argIdx, argInputs] : inputs.byArgIndex) {
    if (!argInputs.provided)
      continue;
    if (argInputs.transactions.empty())
      return fail("Provided input has no transactions");
    if (txCount == 0)
      txCount = argInputs.transactions.size();
    else if (txCount != argInputs.transactions.size())
      return fail("All provided inputs must have equal transaction count");
  }

  if (txCount == 0)
    txCount = 1;

  for (unsigned argIdx = 0, e = funcOp.getNumArguments(); argIdx < e; ++argIdx) {
    Value arg = funcOp.getArgument(argIdx);
    auto it = inputs.byArgIndex.find(argIdx);

    if (auto memref = dyn_cast<MemRefType>(arg.getType())) {
      if (it == inputs.byArgIndex.end() || !it->second.provided)
        return fail("Missing memref input for argument " + argNames[argIdx]);
      auto memSize = getStaticMemrefSize(memref);
      if (!memSize)
        return fail("Dynamic memrefs are not supported");
      for (unsigned t = 0; t < txCount; ++t) {
        if (it->second.transactions[t].size() != *memSize)
          return fail("Memref input size mismatch for " + argNames[argIdx]);
      }
      continue;
    }

    if (isa<handshake::ControlType>(arg.getType())) {
      if (it == inputs.byArgIndex.end()) {
        ArgInputTransactions autoCtrl;
        autoCtrl.provided = true;
        autoCtrl.transactions.resize(txCount);
        // Handshake controls represent tokens, not level-held enables.
        // Default to one token per transaction for all unspecified controls.
        for (unsigned t = 0; t < txCount; ++t)
          autoCtrl.transactions[t].push_back("1");
        inputs.byArgIndex[argIdx] = autoCtrl;
      } else if (it->second.transactions.size() != txCount) {
        return fail("Control input transaction count mismatch");
      }
      continue;
    }

    if (isa<handshake::ChannelType>(arg.getType())) {
      if (it == inputs.byArgIndex.end() || !it->second.provided)
        return fail("Missing channel input for argument " + argNames[argIdx]);
    }
  }

  inputs.numTransactions = txCount;
  return success();
}

LogicalResult Simulator::loadInputs(StringRef inputVectorsDir,
                                    StringRef inputArgsFile,
                                    HSInputFormat format,
                                    SimulationInputs &inputs) {
  inputs = SimulationInputs();
  if (format == HSInputFormat::Dat ||
      (format == HSInputFormat::Auto && !inputVectorsDir.empty())) {
    if (failed(parseDatInputDir(inputVectorsDir, inputs)))
      return failure();
  }

  if (format == HSInputFormat::Plain ||
      (format == HSInputFormat::Auto && !inputArgsFile.empty() &&
       inputVectorsDir.empty())) {
    if (failed(parsePlainInputFile(inputArgsFile, inputs)))
      return failure();
  }

  return finalizeInputTransactions(inputs);
}

LogicalResult Simulator::initializeMemoriesForTransaction(
    const SimulationInputs &inputs, unsigned transactionIdx) {
  for (auto &[memref, image] : memoryImages) {
    unsigned argIdx = cast<BlockArgument>(memref).getArgNumber();
    const ArgInputTransactions &argInputs = inputs.byArgIndex.lookup(argIdx);
    const SmallVector<std::string> &txnVals = argInputs.transactions[transactionIdx];
    Type elemType = image.type.getElementType();
    for (auto [idx, token] : llvm::enumerate(txnVals)) {
      Data parsed;
      if (failed(parseInputToken(elemType, token, parsed)))
        return fail("Failed parsing memref token for argument " + image.name);
      image.values[idx] = parsed;
    }
    std::fill(image.writtenMask.begin(), image.writtenMask.end(), 0);
  }
  return success();
}

LogicalResult Simulator::buildDriversForTransaction(const SimulationInputs &inputs,
                                                    unsigned transactionIdx) {
  activeDrivers.clear();

  for (unsigned argIdx = 0, e = funcOp.getNumArguments(); argIdx < e; ++argIdx) {
    Value arg = funcOp.getArgument(argIdx);
    if (!isa<handshake::ChannelType, handshake::ControlType>(arg.getType()))
      continue;

    InputDriver driver;
    driver.arg = arg;
    driver.argIndex = argIdx;
    driver.isChannel = isa<handshake::ChannelType>(arg.getType());
    driver.isControl = isa<handshake::ControlType>(arg.getType());

    const ArgInputTransactions &argInputs = inputs.byArgIndex.lookup(argIdx);
    const SmallVector<std::string> &values = argInputs.transactions[transactionIdx];

    if (driver.isChannel) {
      Type dataType = cast<handshake::ChannelType>(arg.getType()).getDataType();
      for (const std::string &token : values) {
        Data parsed;
        if (failed(parseInputToken(dataType, token, parsed)))
          return fail("Failed parsing channel token for argument " +
                      argNames[argIdx]);
        driver.channelTokens.push_back(parsed);
      }
    } else {
      for (const std::string &token : values) {
        std::string tokTrim = trimCopy(token);
        if (StringRef(tokTrim).equals_insensitive("hold") ||
            StringRef(tokTrim).equals_insensitive("level1")) {
          driver.holdHighControl = true;
          continue;
        }

        int64_t count = 1;
        if (!tokTrim.empty())
          count = std::max<int64_t>(0, std::stoll(tokTrim));
        driver.controlTokens += static_cast<uint64_t>(count);
      }
    }

    activeDrivers[argIdx] = std::move(driver);
  }

  return success();
}

void Simulator::driveInputsForCycle() {
  for (auto &[argIdx, driver] : activeDrivers) {
    if (driver.isChannel) {
      auto *producer = dyn_cast<ChannelProducerRW>(producerViews[driver.arg]);
      producer->valid = !driver.channelTokens.empty();
      if (producer->valid)
        producer->data = driver.channelTokens.front();
      updaters[driver.arg]->update();
    } else if (driver.isControl) {
      auto *producer = dyn_cast<ControlProducerRW>(producerViews[driver.arg]);
      producer->valid = driver.holdHighControl || driver.controlTokens > 0;
      updaters[driver.arg]->update();
    }
  }
}

void Simulator::consumeAcceptedInputs() {
  for (auto &[argIdx, driver] : activeDrivers) {
    ProducerRW *producer = producerViews[driver.arg];
    if (!producer->valid || !producer->ready)
      continue;

    if (driver.isChannel && !driver.channelTokens.empty())
      driver.channelTokens.pop_front();
    if (driver.isControl && !driver.holdHighControl && driver.controlTokens > 0)
      --driver.controlTokens;
  }
}

uint64_t Simulator::getPendingInputTokenCount() const {
  uint64_t pending = 0;
  for (const auto &[_, driver] : activeDrivers) {
    pending += driver.channelTokens.size();
    if (!driver.holdHighControl)
      pending += driver.controlTokens;
  }
  return pending;
}

uint64_t Simulator::getPendingMemoryOpCount() const {
  uint64_t pending = 0;
  for (const auto &[op, model] : opModels) {
    if (isa<handshake::MemoryControllerOp>(op))
      pending += static_cast<MemoryControllerModel *>(model)->getPendingCount();
    else if (isa<handshake::LSQOp>(op))
      pending += static_cast<LSQModel *>(model)->getPendingCount();
  }
  return pending;
}

std::string Simulator::gatherDeadlockDiagnostics() const {
  std::string diag;
  raw_string_ostream os(diag);
  os << "pending_tokens=" << getPendingInputTokenCount()
     << " pending_memory_ops=" << getPendingMemoryOpCount() << " stalled=[";
  bool first = true;
  for (const EdgeTrace &edge : edgeTraces) {
    if (edge.edge->valid && !edge.edge->ready) {
      if (!first)
        os << ", ";
      first = false;
      os << edge.src << "->" << edge.dst << "#" << edge.operandIndex;
    }
  }
  os << "]";
  return os.str();
}

void Simulator::resetEndObservers() {
  for (EndObserver &observer : endObservers)
    observer.seen = false;
}

bool Simulator::allEndResultsSeen() const {
  return llvm::all_of(endObservers, [](const EndObserver &observer) {
    return observer.seen;
  });
}

LogicalResult Simulator::simulate(const SimulationInputs &inputs) {
  if (failedFlag)
    return failure();

  iterNum = 0;
  results.clear();
  memories.clear();
  vcdCheck = VCDCheck();

  for (EdgeTrace &trace : edgeTraces) {
    trace.valid.clear();
    trace.ready.clear();
    trace.transfer.clear();
    trace.dataWave.clear();
    trace.dataWordToggles = 0;
    trace.dataBitToggles = 0;
    trace.transferDataWordToggles = 0;
    trace.transferDataBitToggles = 0;
    trace.prevData.reset();
    trace.prevTransferData.reset();
  }

  auto settleCombinational = [&]() {
    while (true) {
      for (auto &[_, model] : opModels)
        model->exec(false);

      bool fixedPoint = true;
      for (auto &[_, updater] : updaters)
        fixedPoint = fixedPoint && updater->check();
      if (fixedPoint)
        break;

      for (auto &[_, updater] : updaters)
        updater->update();
    }
    for (auto &[_, updater] : updaters)
      updater->update();
  };

  for (unsigned txn = 0; txn < inputs.numTransactions; ++txn) {
    reset();
    resetEndObservers();
    std::fill(resValid.begin(), resValid.end(), false);

    if (failed(initializeMemoriesForTransaction(inputs, txn)))
      return failure();
    if (failed(buildDriversForTransaction(inputs, txn)))
      return failure();

    unsigned stallCycles = 0;
    unsigned txnCycles = 0;
    while (true) {
      if (txnCycles >= options.maxCycles) {
        return fail("Simulation timed out; " + gatherDeadlockDiagnostics());
      }
      ++txnCycles;
      ++iterNum;

      driveInputsForCycle();
      settleCombinational();
      bool progress = sampleEdgeStates();

      for (auto [idx, observer] : llvm::enumerate(endObservers)) {
        if (!observer.seen && observer.in->valid && observer.in->ready) {
          observer.seen = true;
          if (observer.inData.hasValue())
            resData[idx] = *observer.inData.data;
        }
      }

      consumeAcceptedInputs();

      if (!progress) {
        ++stallCycles;
      } else {
        stallCycles = 0;
      }

      if (stallCycles > 1024) {
        return fail("Deadlock detected; " + gatherDeadlockDiagnostics());
      }

      for (auto &[_, model] : opModels)
        model->exec(true);
      for (auto &[_, updater] : updaters)
        updater->update();

      // The top-level end handshakes already encode MC/LSQ completion
      // conditions. Requiring additional internal pending-accounting here can
      // keep simulation running after architectural completion and skew
      // switching extraction.
      bool completed =
          allEndResultsSeen() && getPendingInputTokenCount() == 0;
      if (completed)
        break;
    }

    for (unsigned i = 0, e = endObservers.size(); i < e; ++i) {
      ResultRecord rec;
      rec.name =
          (i < resNames.size()) ? resNames[i] : "out" + std::to_string(i);
      if (inputs.numTransactions > 1)
        rec.name += "_txn" + std::to_string(txn);

      Value endOperand = endOp.getOperand(i);
      std::string typeStr;
      raw_string_ostream typeOS(typeStr);
      endOperand.getType().print(typeOS);
      rec.type = typeOS.str();
      rec.observed = endObservers[i].seen;
      if (endObservers[i].inData.hasValue()) {
        Type dataType = cast<handshake::ChannelType>(endOperand.getType());
        rec.value = dataToString(dataType, resData[i]);
      } else {
        rec.value = rec.observed ? "1" : "0";
      }
      results.push_back(std::move(rec));
    }

    for (auto &[_, image] : memoryImages) {
      MemoryRecord rec;
      rec.name = image.name;
      if (inputs.numTransactions > 1)
        rec.name += "_txn" + std::to_string(txn);
      std::string typeStr;
      raw_string_ostream typeOS(typeStr);
      image.type.print(typeOS);
      rec.type = typeOS.str();
      rec.values.reserve(image.values.size());
      for (const Data &data : image.values)
        rec.values.push_back(dataToString(image.type.getElementType(), data));
      rec.writtenMask = image.writtenMask;
      rec.writtenCount =
          llvm::count(image.writtenMask, static_cast<uint8_t>(1));
      memories.push_back(std::move(rec));
    }
  }

  finalizeSwitchingStats();
  return success();
}

void Simulator::simulate(llvm::ArrayRef<std::string> inputArgs) {
  SimulationInputs inputs;

  SmallVector<unsigned> channelArgs;
  for (unsigned i = 0, e = funcOp.getNumArguments(); i < e; ++i) {
    if (isa<handshake::ChannelType>(funcOp.getArgument(i).getType()))
      channelArgs.push_back(i);
  }

  if (channelArgs.empty() || inputArgs.size() % channelArgs.size() != 0) {
    (void)signalFailure(
        "Legacy positional mode expects N*<num-channel-args> scalar tokens");
    return;
  }

  unsigned txCount = inputArgs.size() / channelArgs.size();
  for (auto [idx, argIdx] : llvm::enumerate(channelArgs)) {
    ArgInputTransactions argInput;
    argInput.provided = true;
    argInput.transactions.resize(txCount);
    for (unsigned txn = 0; txn < txCount; ++txn)
      argInput.transactions[txn].push_back(inputArgs[txn * channelArgs.size() + idx]);
    inputs.byArgIndex[argIdx] = argInput;
  }

  if (failed(finalizeInputTransactions(inputs)))
    return;
  (void)simulate(inputs);
}

LogicalResult Simulator::emitJSON(StringRef filepath,
                                  const llvm::json::Value &value) const {
  std::error_code ec;
  raw_fd_ostream out(filepath, ec, sys::fs::OF_Text);
  if (ec)
    return failure();
  out << formatv("{0:2}", value);
  out << "\n";
  return success();
}

LogicalResult Simulator::dumpSwitchingJSON(StringRef filepath) const {
  llvm::json::Array channels;
  for (const ChannelSwitching &sw : channelWaves) {
    llvm::json::Array valid;
    llvm::json::Array ready;
    llvm::json::Array transfer;
    llvm::json::Array dataWave;
    llvm::json::Array transferDataWave;
    for (uint8_t value : sw.valid)
      valid.push_back(static_cast<int64_t>(value));
    for (uint8_t value : sw.ready)
      ready.push_back(static_cast<int64_t>(value));
    for (uint8_t value : sw.transfer)
      transfer.push_back(static_cast<int64_t>(value));
    if (sw.hasData) {
      for (auto [idx, value] : llvm::enumerate(sw.dataWave)) {
        SmallString<64> hex;
        value.toString(hex, 16, false);
        if (hex.empty())
          hex = "0";
        std::string str = ("0x" + hex.str()).str();
        dataWave.push_back(str);
        if (idx < sw.transfer.size() && sw.transfer[idx] != 0)
          transferDataWave.push_back(str);
      }
    }
    channels.push_back(llvm::json::Object{
        {"src", sw.src},
        {"dst", sw.dst},
        {"operand_index", static_cast<int64_t>(sw.operandIndex)},
        {"is_to_end", sw.isToEnd},
        {"has_data", sw.hasData},
        {"data_width", static_cast<int64_t>(sw.dataWidth)},
        {"valid", std::move(valid)},
        {"ready", std::move(ready)},
        {"transfer", std::move(transfer)},
        {"data", std::move(dataWave)},
        {"transfer_data", std::move(transferDataWave)},
        {"valid_toggles", static_cast<int64_t>(sw.validToggles)},
        {"ready_toggles", static_cast<int64_t>(sw.readyToggles)},
        {"transfer_toggles", static_cast<int64_t>(sw.transferToggles)},
        {"data_toggles", static_cast<int64_t>(sw.dataWordToggles)},
        {"data_bit_toggles", static_cast<int64_t>(sw.dataBitToggles)},
        {"transfer_data_toggles",
         static_cast<int64_t>(sw.transferDataWordToggles)},
        {"transfer_data_bit_toggles",
         static_cast<int64_t>(sw.transferDataBitToggles)},
        {"transfers", static_cast<int64_t>(sw.transfers)},
    });
  }

  llvm::json::Object top;
  top["cycles"] = static_cast<int64_t>(iterNum);
  top["channels"] = std::move(channels);
  if (vcdCheck.enabled) {
    llvm::json::Object check{
        {"enabled", vcdCheck.enabled},
        {"passed", vcdCheck.passed},
        {"message", vcdCheck.message},
        {"compared_cycles", static_cast<int64_t>(vcdCheck.comparedCycles)},
        {"compared_valid_samples",
         static_cast<int64_t>(vcdCheck.comparedValidSamples)},
        {"valid_mismatch_count",
         static_cast<int64_t>(vcdCheck.validMismatchCount)},
        {"compared_ready_samples",
         static_cast<int64_t>(vcdCheck.comparedReadySamples)},
        {"ready_mismatch_count",
         static_cast<int64_t>(vcdCheck.readyMismatchCount)},
        {"compared_transfer_samples",
         static_cast<int64_t>(vcdCheck.comparedTransferSamples)},
        {"transfer_mismatch_count",
         static_cast<int64_t>(vcdCheck.transferMismatchCount)},
        {"compared_data_samples",
         static_cast<int64_t>(vcdCheck.comparedDataSamples)},
        {"data_mismatch_count", static_cast<int64_t>(vcdCheck.dataMismatchCount)},
        {"data_toggle_mismatch_count",
         static_cast<int64_t>(vcdCheck.dataWordToggleMismatchCount)},
        {"data_bit_toggle_mismatch_count",
         static_cast<int64_t>(vcdCheck.dataBitToggleMismatchCount)},
        {"first_mismatch_cycle",
         static_cast<int64_t>(vcdCheck.firstMismatchCycle)},
        {"mismatch_count", static_cast<int64_t>(vcdCheck.mismatchCount)},
    };
    top["vcd_check"] = std::move(check);
  }
  return emitJSON(filepath, llvm::json::Value(std::move(top)));
}

LogicalResult Simulator::dumpWaveJSON(StringRef filepath) const {
  return dumpSwitchingJSON(filepath);
}

LogicalResult Simulator::dumpResultsJSON(StringRef filepath) const {
  llvm::json::Array resultsJson;
  for (const ResultRecord &res : results) {
    resultsJson.push_back(llvm::json::Object{
        {"name", res.name},
        {"type", res.type},
        {"value", res.value},
        {"observed", res.observed},
    });
  }

  llvm::json::Array memoriesJson;
  for (const MemoryRecord &mem : memories) {
    llvm::json::Array values;
    llvm::json::Array writtenMask;
    for (const std::string &value : mem.values)
      values.push_back(value);
    for (uint8_t bit : mem.writtenMask)
      writtenMask.push_back(static_cast<int64_t>(bit));
    memoriesJson.push_back(llvm::json::Object{
        {"name", mem.name},
        {"type", mem.type},
        {"values", std::move(values)},
        {"written_mask", std::move(writtenMask)},
        {"written_count", static_cast<int64_t>(mem.writtenCount)},
    });
  }

  llvm::json::Object top;
  top["results"] = std::move(resultsJson);
  top["memories"] = std::move(memoriesJson);

  if (vcdCheck.enabled) {
    llvm::json::Object check{
        {"enabled", vcdCheck.enabled},
        {"passed", vcdCheck.passed},
        {"message", vcdCheck.message},
        {"compared_cycles", static_cast<int64_t>(vcdCheck.comparedCycles)},
        {"compared_valid_samples",
         static_cast<int64_t>(vcdCheck.comparedValidSamples)},
        {"valid_mismatch_count",
         static_cast<int64_t>(vcdCheck.validMismatchCount)},
        {"compared_ready_samples",
         static_cast<int64_t>(vcdCheck.comparedReadySamples)},
        {"ready_mismatch_count",
         static_cast<int64_t>(vcdCheck.readyMismatchCount)},
        {"compared_transfer_samples",
         static_cast<int64_t>(vcdCheck.comparedTransferSamples)},
        {"transfer_mismatch_count",
         static_cast<int64_t>(vcdCheck.transferMismatchCount)},
        {"compared_data_samples",
         static_cast<int64_t>(vcdCheck.comparedDataSamples)},
        {"data_mismatch_count", static_cast<int64_t>(vcdCheck.dataMismatchCount)},
        {"data_toggle_mismatch_count",
         static_cast<int64_t>(vcdCheck.dataWordToggleMismatchCount)},
        {"data_bit_toggle_mismatch_count",
         static_cast<int64_t>(vcdCheck.dataBitToggleMismatchCount)},
        {"first_mismatch_cycle",
         static_cast<int64_t>(vcdCheck.firstMismatchCycle)},
        {"mismatch_count", static_cast<int64_t>(vcdCheck.mismatchCount)},
    };

    llvm::json::Array mismatches;
    for (const VCDMismatch &m : vcdCheck.mismatches) {
      mismatches.push_back(llvm::json::Object{
          {"src", m.src},
          {"dst", m.dst},
          {"operand_index", static_cast<int64_t>(m.operandIndex)},
          {"signal", m.signal},
          {"cycle", static_cast<int64_t>(m.cycle)},
          {"simulated", static_cast<int64_t>(m.simulated)},
          {"vcd", static_cast<int64_t>(m.vcd)},
      });
    }
    check["mismatches"] = std::move(mismatches);
    top["vcd_check"] = std::move(check);
  }

  return emitJSON(filepath, llvm::json::Value(std::move(top)));
}

LogicalResult Simulator::dumpSwitchingEstimationCSV(StringRef filepath) const {
  std::error_code ec;
  raw_fd_ostream out(filepath, ec, sys::fs::OF_Text);
  if (ec)
    return failure();

  struct NodeTotals {
    uint64_t data = 0;
    uint64_t valid = 0;
    uint64_t ready = 0;
  };
  std::map<std::string, NodeTotals> perNode;

  // Limit switching extraction to execution cycles, i.e., from the first
  // top-level input activity onward (profiler-style execution window).
  auto firstOneCycle = [](const std::vector<uint8_t> &sig)
      -> std::optional<unsigned> {
    for (unsigned i = 0, e = sig.size(); i < e; ++i) {
      if (sig[i] != 0)
        return i;
    }
    return std::nullopt;
  };
  unsigned executionStart = 0;
  bool foundExecutionStart = false;
  for (const ChannelSwitching &sw : channelWaves) {
    if (argNameToIndex.find(sw.src) == argNameToIndex.end())
      continue;

    std::optional<unsigned> firstTransfer = firstOneCycle(sw.transfer);
    std::optional<unsigned> firstValid = firstOneCycle(sw.valid);
    std::optional<unsigned> firstActivity = firstTransfer ? firstTransfer : firstValid;
    if (!firstActivity)
      continue;

    if (!foundExecutionStart) {
      executionStart = *firstActivity;
      foundExecutionStart = true;
    } else {
      executionStart = std::min(executionStart, *firstActivity);
    }
  }

  // Seed all named handshake nodes so zero-activity nodes are still reported.
  for (const auto &[op, _] : opModels) {
    if (!op)
      continue;
    auto nodeNameAttr = op->getAttrOfType<StringAttr>("handshake.name");
    if (!nodeNameAttr)
      continue;
    std::string nodeName = nodeNameAttr.getValue().str();
    if (nodeName.empty() || StringRef(nodeName).contains("mem_controller"))
      continue;
    perNode.try_emplace(nodeName);
  }

  for (const ChannelSwitching &sw : channelWaves) {
    // Estimation-style accounting mirrors switching_testing/vcd_parser.py:
    // valid/ready/data are accumulated from node-local channel signals that
    // carry the node prefix. For generated RTL these signals are anchored at
    // producer-side ports, so all three metrics are attributed to `src`.
    if (argNameToIndex.find(sw.src) == argNameToIndex.end() && !sw.src.empty() &&
        !StringRef(sw.src).contains("mem_controller")) {
      NodeTotals &srcTotals = perNode[sw.src];
      srcTotals.valid += countTogglesFromCycle(sw.valid, executionStart);
      srcTotals.ready += countTogglesFromCycle(sw.ready, executionStart);
      if (sw.hasData)
        srcTotals.data +=
            countDataBitTogglesFromCycle(sw.dataWave, executionStart);
    }
  }

  out << "node, data, valid, ready\n";
  for (const auto &[node, totals] : perNode) {
    out << node << "," << totals.data << "," << totals.valid << ","
        << totals.ready << "\n";
  }
  return success();
}

LogicalResult Simulator::verifyVCDExact(StringRef topVerilogPath,
                                        StringRef vcdPath) {
  vcdCheck = VCDCheck();
  vcdCheck.enabled = true;

  if (channelWaves.empty()) {
    vcdCheck.passed = false;
    vcdCheck.message = "No channel waves available for VCD verification";
    return failure();
  }

  auto verilogFile = MemoryBuffer::getFile(topVerilogPath);
  auto vcdFile = MemoryBuffer::getFile(vcdPath);
  if (!verilogFile || !vcdFile) {
    vcdCheck.passed = false;
    vcdCheck.message = "Failed to read top-verilog or VCD file";
    return failure();
  }

  std::unordered_set<std::string> requiredBaseSignals;
  std::unordered_set<std::string> requiredExactSignals;
  requiredExactSignals.insert("clk");
  requiredExactSignals.insert("clock");
  requiredExactSignals.insert("rst");
  requiredExactSignals.insert("reset");
  for (StringRef argName : argNames) {
    requiredBaseSignals.insert(argName.str());
    requiredExactSignals.insert((argName + "_valid").str());
    requiredExactSignals.insert((argName + "_ready").str());
    requiredExactSignals.insert((argName + "_data").str());
  }
  for (StringRef resName : resNames) {
    requiredBaseSignals.insert(resName.str());
    requiredExactSignals.insert((resName + "_valid").str());
    requiredExactSignals.insert((resName + "_ready").str());
    requiredExactSignals.insert((resName + "_data").str());
  }

  auto shouldTrackSignalName = [&](StringRef localName) {
    StringRef name = localName.trim();
    if (name.empty())
      return false;
    if (requiredExactSignals.count(name.str()))
      return true;
    if (requiredBaseSignals.count(name.str()))
      return true;

    std::string lower = name.lower();
    if (std::regex_search(lower, std::regex(R"((^|_)(clk|clock|rst|reset)($|_))")))
      return true;

    // Track all remaining signals so internal channel payload wires like
    // "addi0_result" (without *_data suffix) are available for mapping.
    (void)requiredBaseSignals;
    return true;
  };

  std::unordered_map<std::string, std::unordered_map<std::string, std::string>>
      instancePorts;
  {
    std::string text = std::string((*verilogFile)->getBuffer());
    std::regex instRe(
        R"((?:^|\n)\s*[A-Za-z_][A-Za-z0-9_]*\s*(?:#\s*\([\s\S]*?\))?\s*([A-Za-z_][A-Za-z0-9_]*)\s*\(([\s\S]*?)\)\s*;)");
    std::regex portRe(R"(\.([A-Za-z_][A-Za-z0-9_]*)\s*\(([^()]*)\))");
    auto begin = std::sregex_iterator(text.begin(), text.end(), instRe);
    auto end = std::sregex_iterator();
    for (auto it = begin; it != end; ++it) {
      std::string instName = (*it)[1].str();
      std::string blob = (*it)[2].str();
      std::unordered_map<std::string, std::string> ports;
      auto pBegin = std::sregex_iterator(blob.begin(), blob.end(), portRe);
      for (auto pit = pBegin; pit != end; ++pit)
        ports[(*pit)[1].str()] = trimCopy((*pit)[2].str());
      if (!ports.empty())
        instancePorts[instName] = std::move(ports);
    }
  }

  auto extractSignals = [](const std::string &expr) {
    std::vector<std::string> out;
    std::string clean = expr;
    for (char &c : clean) {
      if (c == '{' || c == '}')
        c = ' ';
    }
    std::stringstream ss(clean);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
      StringRef t(tok);
      t = t.trim();
      if (t.empty())
        continue;
      if (t.contains("'"))
        continue;
      out.push_back(t.str());
    }
    return out;
  };

  auto belongsToSrc = [](StringRef wire, StringRef src) {
    if (wire.endswith("_valid"))
      wire = wire.drop_back(strlen("_valid"));
    if (wire.endswith("_ready"))
      wire = wire.drop_back(strlen("_ready"));
    std::string pref = (src + "_").str();
    return wire == src || wire.startswith(pref);
  };

  struct SignalEvents {
    unsigned width = 1;
    std::vector<std::pair<uint64_t, char>> bitEvents;
    std::vector<std::pair<uint64_t, APInt>> vecEvents;
  };
  std::unordered_map<std::string, SignalEvents> eventsByName;
  std::unordered_map<std::string, std::vector<std::string>> idToNames;
  std::unordered_map<std::string, unsigned> signalWidthsByName;
  std::unordered_map<std::string, std::vector<std::pair<unsigned, std::string>>>
      bitSlicesByBaseName;
  std::vector<std::string> allSignalNames;

  auto parseVcdBinary = [](StringRef bits, unsigned width) {
    std::string normalized;
    normalized.reserve(bits.size());
    for (char c : bits) {
      switch (c) {
      case '1':
        normalized.push_back('1');
        break;
      case '0':
      case 'x':
      case 'X':
      case 'z':
      case 'Z':
      default:
        normalized.push_back('0');
        break;
      }
    }

    if (normalized.empty())
      normalized = "0";
    if (normalized.size() > width)
      normalized = normalized.substr(normalized.size() - width);
    if (normalized.size() < width)
      normalized = std::string(width - normalized.size(), '0') + normalized;
    return APInt(width, normalized, 2);
  };

  auto recordSignalValue = [&](const std::string &name, const APInt &rawValue,
                               uint64_t timestamp) {
    unsigned width = 1;
    if (auto it = signalWidthsByName.find(name); it != signalWidthsByName.end())
      width = std::max(1u, it->second);
    SignalEvents &signal = eventsByName[name];
    signal.width = width;
    APInt value = normalizeAPInt(rawValue, width);
    char bit = value == 0 ? '0' : '1';

    if (!signal.bitEvents.empty() && signal.bitEvents.back().first == timestamp)
      signal.bitEvents.back().second = bit;
    else
      signal.bitEvents.push_back({timestamp, bit});

    if (!signal.vecEvents.empty() && signal.vecEvents.back().first == timestamp)
      signal.vecEvents.back().second = value;
    else
      signal.vecEvents.push_back({timestamp, value});
  };

  {
    std::vector<std::string> scopeStack;
    uint64_t currentTime = 0;
    std::stringstream ss(std::string((*vcdFile)->getBuffer()));
    std::string line;
    while (std::getline(ss, line)) {
      StringRef ref(line);
      ref = ref.trim();
      if (ref.empty())
        continue;

      if (ref.startswith("$scope")) {
        SmallVector<StringRef> toks;
        ref.split(toks, ' ', -1, false);
        if (toks.size() >= 3)
          scopeStack.push_back(toks[2].str());
        continue;
      }
      if (ref.startswith("$upscope")) {
        if (!scopeStack.empty())
          scopeStack.pop_back();
        continue;
      }
      if (ref.startswith("$var")) {
        SmallVector<StringRef> toks;
        ref.split(toks, ' ', -1, false);
        if (toks.size() < 5)
          continue;
        uint64_t width = 1;
        if (toks[2].getAsInteger(10, width))
          width = 1;
        std::string id = toks[3].str();
        std::string sigBaseName = toks[4].str();
        if (!shouldTrackSignalName(sigBaseName))
          continue;
        std::optional<unsigned> bitIndex;
        if (width == 1 && toks.size() >= 6 &&
            !toks[5].startswith("$end")) {
          bitIndex = parseSingleBitIndex(toks[5]);
        }

        std::string scopePrefix;
        for (size_t i = 0; i < scopeStack.size(); ++i) {
          if (i)
            scopePrefix += ".";
          scopePrefix += scopeStack[i];
        }

        auto makeScopedName = [&](StringRef localName) {
          std::string scoped = scopePrefix;
          if (!scoped.empty())
            scoped += ".";
          scoped += localName.str();
          return scoped;
        };

        std::string fullName;
        if (bitIndex)
          fullName = makeScopedName(sigBaseName + "[" + std::to_string(*bitIndex) + "]");
        else
          fullName = makeScopedName(sigBaseName);

        idToNames[id].push_back(fullName);
        signalWidthsByName[fullName] = static_cast<unsigned>(width);
        allSignalNames.push_back(fullName);

        if (bitIndex) {
          std::string baseScopedName = makeScopedName(sigBaseName);
          bitSlicesByBaseName[baseScopedName].push_back(
              {*bitIndex, fullName});
        }
        continue;
      }
      if (ref.startswith("#")) {
        uint64_t t = 0;
        ref.drop_front().getAsInteger(10, t);
        currentTime = t;
        continue;
      }
      if (ref[0] == '0' || ref[0] == '1' || ref[0] == 'x' || ref[0] == 'X' ||
          ref[0] == 'z' || ref[0] == 'Z') {
        if (ref.size() < 2)
          continue;
        APInt value(1, ref[0] == '1' ? 1 : 0);
        std::string id = ref.drop_front().str();
        for (const std::string &name : idToNames[id])
          recordSignalValue(name, value, currentTime);
        continue;
      }
      if (ref[0] == 'b' || ref[0] == 'B') {
        SmallVector<StringRef> toks;
        ref.split(toks, ' ', -1, false);
        if (toks.size() < 2)
          continue;
        StringRef bits = toks[0].drop_front();
        std::string id = toks[1].str();
        for (const std::string &name : idToNames[id]) {
          unsigned width = 1;
          if (auto it = signalWidthsByName.find(name);
              it != signalWidthsByName.end())
            width = std::max(1u, it->second);
          APInt value = parseVcdBinary(bits, width);
          recordSignalValue(name, value, currentTime);
        }
      }
    }
  }

  for (auto &[_, slices] : bitSlicesByBaseName) {
    llvm::sort(slices, [](const auto &lhs, const auto &rhs) {
      return lhs.first < rhs.first;
    });
  }
  llvm::sort(allSignalNames);
  allSignalNames.erase(std::unique(allSignalNames.begin(), allSignalNames.end()),
                       allSignalNames.end());

  auto pickSignalByWire = [&](StringRef wire) -> std::optional<std::string> {
    if (eventsByName.count(wire.str()))
      return wire.str();
    std::string suffix = "." + wire.str();
    std::optional<std::string> best;
    for (const std::string &name : allSignalNames) {
      if (StringRef(name).endswith(suffix)) {
        if (!best || name.size() < best->size())
          best = name;
      }
    }
    return best;
  };

  auto pickPackedSignalByWire =
      [&](StringRef wire) -> std::optional<std::string> {
    if (bitSlicesByBaseName.count(wire.str()))
      return wire.str();
    std::string suffix = "." + wire.str();
    std::optional<std::string> best;
    for (const auto &[baseName, _] : bitSlicesByBaseName) {
      if (StringRef(baseName).endswith(suffix)) {
        if (!best || baseName.size() < best->size())
          best = baseName;
      }
    }
    return best;
  };

  auto packedSignalWidth = [&](StringRef baseSignal) {
    auto it = bitSlicesByBaseName.find(baseSignal.str());
    if (it == bitSlicesByBaseName.end())
      return 0u;
    unsigned width = 0;
    for (const auto &[idx, _] : it->second)
      width = std::max(width, idx + 1);
    return std::max(1u, width);
  };

  std::optional<std::string> clkSignal;
  for (const std::string &name : allSignalNames) {
    std::string lower = name;
    llvm::transform(lower, lower.begin(), ::tolower);
    if (std::regex_search(lower, std::regex(R"((^|[._])clk($|[._\[]))"))) {
      if (!clkSignal || name.size() < clkSignal->size())
        clkSignal = name;
    }
  }
  if (!clkSignal) {
    vcdCheck.passed = false;
    vcdCheck.message = "Could not detect clock signal in VCD";
    return failure();
  }

  std::optional<std::string> rstSignal;
  for (const std::string &name : allSignalNames) {
    std::string lower = name;
    llvm::transform(lower, lower.begin(), ::tolower);
    if (std::regex_search(lower,
                          std::regex(R"((^|[._])(rst|reset)($|[._\[]))"))) {
      if (!rstSignal || name.size() < rstSignal->size())
        rstSignal = name;
    }
  }

  std::vector<uint64_t> risingEdges;
  {
    const auto &events = eventsByName[*clkSignal].bitEvents;
    int prev = 0;
    for (auto [time, value] : events) {
      int cur = logicBit(value);
      if (prev == 0 && cur == 1)
        risingEdges.push_back(time);
      prev = cur;
    }
  }
  if (risingEdges.empty()) {
    vcdCheck.passed = false;
    vcdCheck.message = "Clock has no rising edges in VCD";
    return failure();
  }

  // Behavioral alignment follows profiler-style execution, which starts when
  // inputs are applied by the testbench. Keep pre-reset cycles to preserve this
  // anchor point.
  constexpr bool kTrimResetCycles = false;
  if (kTrimResetCycles && rstSignal && eventsByName.count(*rstSignal)) {
    const auto &events = eventsByName[*rstSignal].bitEvents;
    int prev = -1;
    uint64_t deassertTime = 0;
    bool found = false;
    for (auto [time, value] : events) {
      int cur = logicBit(value);
      if (prev == 1 && cur == 0) {
        deassertTime = time;
        found = true;
        break;
      }
      if (!found && cur == 0) {
        deassertTime = time;
      }
      prev = cur;
    }
    risingEdges.erase(std::remove_if(risingEdges.begin(), risingEdges.end(),
                                     [&](uint64_t t) { return t < deassertTime; }),
                      risingEdges.end());
  }

  // Sample VCD traces at each rising edge using the value that is stable
  // *before* that edge. This aligns with handshake transfer semantics
  // (valid/ready observed in the cycle leading into the edge), and avoids a
  // systematic one-cycle skew from post-edge register updates dumped at the
  // same timestamp.
  auto sampleBitsAtEdges = [&](const std::vector<std::pair<uint64_t, char>> &events,
                               unsigned count) {
    std::vector<uint8_t> samples;
    samples.reserve(count);
    size_t idx = 0;
    uint8_t cur = 0;
    for (unsigned cycle = 0; cycle < count && cycle < risingEdges.size(); ++cycle) {
      uint64_t t = risingEdges[cycle];
      while (idx < events.size() && events[idx].first < t) {
        cur = events[idx].second == '1' ? 1 : 0;
        ++idx;
      }
      samples.push_back(cur);
    }
    return samples;
  };

  auto sampleDataAtEdges = [&](const std::vector<std::pair<uint64_t, APInt>> &events,
                               unsigned width, unsigned count) {
    width = std::max(1u, width);
    std::vector<APInt> samples;
    samples.reserve(count);
    size_t idx = 0;
    APInt cur(width, 0);
    for (unsigned cycle = 0; cycle < count && cycle < risingEdges.size(); ++cycle) {
      uint64_t t = risingEdges[cycle];
      while (idx < events.size() && events[idx].first < t) {
        cur = normalizeAPInt(events[idx].second, width);
        ++idx;
      }
      samples.push_back(cur);
    }
    return samples;
  };

  auto samplePackedDataAtEdges = [&](StringRef baseSignal, unsigned widthHint,
                                     unsigned count)
      -> std::optional<std::vector<APInt>> {
    auto it = bitSlicesByBaseName.find(baseSignal.str());
    if (it == bitSlicesByBaseName.end())
      return std::nullopt;

    const auto &slices = it->second;
    if (slices.empty())
      return std::nullopt;

    unsigned width = std::max(widthHint, packedSignalWidth(baseSignal));
    width = std::max(1u, width);

    std::vector<size_t> idxBySlice(slices.size(), 0);
    std::vector<uint8_t> bitBySlice(slices.size(), 0);
    std::vector<APInt> samples;
    samples.reserve(count);

    for (unsigned cycle = 0; cycle < count && cycle < risingEdges.size();
         ++cycle) {
      uint64_t t = risingEdges[cycle];
      APInt word(width, 0);
      for (size_t s = 0, se = slices.size(); s < se; ++s) {
        unsigned bitIdx = slices[s].first;
        if (bitIdx >= width)
          continue;
        const auto evIt = eventsByName.find(slices[s].second);
        if (evIt == eventsByName.end())
          continue;
        const auto &events = evIt->second.bitEvents;
        size_t &idx = idxBySlice[s];
        while (idx < events.size() && events[idx].first < t) {
          bitBySlice[s] = events[idx].second == '1' ? 1 : 0;
          ++idx;
        }
        if (bitBySlice[s])
          word.setBit(bitIdx);
      }
      samples.push_back(std::move(word));
    }
    return samples;
  };

  struct MappedWave {
    const ChannelSwitching *sim = nullptr;
    bool hasData = false;
    unsigned dataWidth = 0;
    std::vector<uint8_t> vcdValid;
    std::vector<uint8_t> vcdReady;
    std::vector<uint8_t> vcdTransfer;
    std::vector<APInt> vcdData;
  };
  std::vector<MappedWave> mappedWaves;
  mappedWaves.reserve(channelWaves.size());

  for (const ChannelSwitching &sw : channelWaves) {
    // The switching-estimation flow reasons about internal handshake channels.
    // Exclude edges that terminate at the synthetic `handshake.end` sink, which
    // do not correspond to meaningful internal RTL node channels.
    if (sw.isToEnd)
      continue;
    // Top-level control-only arguments (e.g., `start`, `*_start`) are
    // testbench-driving artifacts and not node-local dataflow channels.
    // Excluding them prevents systematic interface-level mismatches from
    // polluting internal channel validation.
    if (argNameToIndex.find(sw.src) != argNameToIndex.end() && !sw.hasData)
      continue;

    std::string validWire;
    std::string readyWire;
    std::string dataWire;
    bool foundPair = false;
    std::vector<std::string> relatedSignals;
    unsigned sameSrcRank = 0;
    for (const ChannelSwitching &other : channelWaves) {
      if (&other == &sw)
        continue;
      if (other.src != sw.src || other.dst != sw.dst)
        continue;
      if (other.operandIndex < sw.operandIndex)
        ++sameSrcRank;
    }

    if (!foundPair) {
      auto instIt = instancePorts.find(sw.dst);
      if (instIt == instancePorts.end())
        continue;

      std::vector<std::string> allSignals;
      for (const auto &[_, expr] : instIt->second) {
        std::vector<std::string> exprSignals = extractSignals(expr);
        allSignals.insert(allSignals.end(), exprSignals.begin(), exprSignals.end());
      }
      llvm::sort(allSignals);
      allSignals.erase(std::unique(allSignals.begin(), allSignals.end()),
                       allSignals.end());
      relatedSignals = allSignals;

      std::vector<std::string> validCandidates;
      std::vector<std::string> readyCandidates;
      for (const std::string &sig : allSignals) {
        if (StringRef(sig).endswith("_valid") && belongsToSrc(sig, sw.src))
          validCandidates.push_back(sig);
        if (StringRef(sig).endswith("_ready") && belongsToSrc(sig, sw.src))
          readyCandidates.push_back(sig);
      }
      llvm::sort(validCandidates);
      llvm::sort(readyCandidates);

      struct PairCandidate {
        std::string base;
        std::string valid;
        std::string ready;
      };
      SmallVector<PairCandidate> pairCandidates;
      for (const std::string &v : validCandidates) {
        StringRef vBase = StringRef(v).drop_back(strlen("_valid"));
        for (const std::string &r : readyCandidates) {
          StringRef rBase = StringRef(r).drop_back(strlen("_ready"));
          if (vBase == rBase) {
            pairCandidates.push_back(
                PairCandidate{vBase.str(), v, r});
          }
        }
      }
      llvm::sort(pairCandidates, [](const PairCandidate &lhs,
                                    const PairCandidate &rhs) {
        return lhs.base < rhs.base;
      });
      if (!pairCandidates.empty()) {
        unsigned pick = std::min<unsigned>(sameSrcRank,
                                           pairCandidates.size() - 1);
        validWire = pairCandidates[pick].valid;
        readyWire = pairCandidates[pick].ready;
        dataWire = pairCandidates[pick].base + "_data";
        foundPair = true;
      } else if (validCandidates.size() == 1 && readyCandidates.size() == 1) {
        validWire = validCandidates.front();
        readyWire = readyCandidates.front();
        StringRef base = StringRef(validWire);
        if (base.endswith("_valid"))
          base = base.drop_back(strlen("_valid"));
        dataWire = (base + "_data").str();
        foundPair = true;
      }
    }
    if (!foundPair)
      continue;

    auto validSig = pickSignalByWire(validWire);
    auto readySig = pickSignalByWire(readyWire);
    if (!validSig || !readySig)
      continue;

    std::vector<uint8_t> vcdValid =
        sampleBitsAtEdges(eventsByName[*validSig].bitEvents,
                          static_cast<unsigned>(risingEdges.size()));
    std::vector<uint8_t> vcdReady =
        sampleBitsAtEdges(eventsByName[*readySig].bitEvents,
                          static_cast<unsigned>(risingEdges.size()));

    std::optional<std::string> dataSig;
    std::optional<std::string> packedDataSig;
    if (sw.hasData) {
      SmallVector<std::string> dataCandidates;
      if (!dataWire.empty()) {
        dataCandidates.push_back(dataWire);
        StringRef base = StringRef(dataWire);
        if (base.endswith("_data"))
          dataCandidates.push_back(base.drop_back(strlen("_data")).str());
      }
      if (!validWire.empty()) {
        StringRef base = StringRef(validWire);
        if (base.endswith("_valid")) {
          base = base.drop_back(strlen("_valid"));
          dataCandidates.push_back((base + "_data").str());
          dataCandidates.push_back(base.str());
        }
      }
      for (const std::string &sig : relatedSignals) {
        if (StringRef(sig).endswith("_valid") || StringRef(sig).endswith("_ready"))
          continue;
        if (belongsToSrc(sig, sw.src))
          dataCandidates.push_back(sig);
      }

      for (const std::string &candidate : dataCandidates) {
        if (auto packed = pickPackedSignalByWire(candidate)) {
          unsigned packedWidth = packedSignalWidth(*packed);
          if (packedWidth > 1 || sw.dataWidth > 1) {
            packedDataSig = packed;
            break;
          }
          if (!packedDataSig)
            packedDataSig = packed;
        }
        if (auto sig = pickSignalByWire(candidate)) {
          unsigned sigWidth = 1;
          if (auto it = eventsByName.find(*sig); it != eventsByName.end())
            sigWidth = std::max(1u, it->second.width);
          if (sigWidth > 1 || sw.dataWidth <= 1) {
            dataSig = sig;
            break;
          }
          if (!dataSig)
            dataSig = sig;
        }
      }
      if (!dataSig && !packedDataSig) {
        vcdCheck.passed = false;
        vcdCheck.message = "Missing data-wire mapping for channel " + sw.src +
                           " -> " + sw.dst;
        return failure();
      }
    }

    MappedWave mapped;
    mapped.sim = &sw;
    mapped.hasData = sw.hasData;
    mapped.dataWidth = sw.dataWidth;
    mapped.vcdValid = std::move(vcdValid);
    mapped.vcdReady = std::move(vcdReady);
    mapped.vcdTransfer.reserve(
        std::min(mapped.vcdValid.size(), mapped.vcdReady.size()));
    for (unsigned i = 0, e = std::min(mapped.vcdValid.size(), mapped.vcdReady.size());
         i < e; ++i)
      mapped.vcdTransfer.push_back((mapped.vcdValid[i] && mapped.vcdReady[i]) ? 1
                                                                               : 0);

    if (mapped.hasData) {
      unsigned width = mapped.dataWidth;
      if (width == 0 && dataSig)
        width = eventsByName[*dataSig].width;
      if (width == 0 && packedDataSig)
        width = packedSignalWidth(*packedDataSig);
      width = std::max(1u, width);
      mapped.dataWidth = width;

      if (packedDataSig && (!dataSig || width > 1)) {
        std::optional<std::vector<APInt>> packedSamples =
            samplePackedDataAtEdges(*packedDataSig, width,
                                    static_cast<unsigned>(risingEdges.size()));
        if (!packedSamples) {
          vcdCheck.passed = false;
          vcdCheck.message =
              "Failed to sample packed data for channel " + sw.src + " -> " +
              sw.dst;
          return failure();
        }
        mapped.vcdData = std::move(*packedSamples);
      } else if (dataSig) {
        mapped.vcdData =
            sampleDataAtEdges(eventsByName[*dataSig].vecEvents, width,
                              static_cast<unsigned>(risingEdges.size()));
      }
    }
    mappedWaves.push_back(std::move(mapped));
  }

  if (mappedWaves.empty()) {
    vcdCheck.passed = false;
    vcdCheck.message = "No mappable handshake channels between simulator and VCD";
    return failure();
  }

  auto firstHighCycle = [](const std::vector<uint8_t> &sig)
      -> std::optional<unsigned> {
    for (unsigned i = 0, e = sig.size(); i < e; ++i)
      if (sig[i])
        return i;
    return std::nullopt;
  };

  // Align comparison to profiler-style execution start:
  // 1) Prefer top-level input channels (application point of testbench inputs).
  // 2) Prefer first accepted transfer, fallback to first valid assertion.
  auto findAlignedStart = [&](bool topLevelOnly, bool useTransfer)
      -> std::optional<std::pair<unsigned, unsigned>> {
    bool found = false;
    unsigned simMin = 0;
    unsigned vcdMin = 0;

    for (const MappedWave &mapped : mappedWaves) {
      bool isTopLevelInput = argNameToIndex.find(mapped.sim->src) != argNameToIndex.end();
      if (topLevelOnly && !isTopLevelInput)
        continue;

      const std::vector<uint8_t> &simSig =
          useTransfer ? mapped.sim->transfer : mapped.sim->valid;
      const std::vector<uint8_t> &vcdSig =
          useTransfer ? mapped.vcdTransfer : mapped.vcdValid;

      std::optional<unsigned> simFirst = firstHighCycle(simSig);
      std::optional<unsigned> vcdFirst = firstHighCycle(vcdSig);
      if (!simFirst || !vcdFirst)
        continue;

      if (!found) {
        simMin = *simFirst;
        vcdMin = *vcdFirst;
        found = true;
      } else {
        simMin = std::min(simMin, *simFirst);
        vcdMin = std::min(vcdMin, *vcdFirst);
      }
    }

    if (!found)
      return std::nullopt;
    return std::make_pair(simMin, vcdMin);
  };

  unsigned simStart = 0;
  unsigned vcdStart = 0;
  if (auto start = findAlignedStart(/*topLevelOnly=*/true, /*useTransfer=*/true)) {
    simStart = start->first;
    vcdStart = start->second;
  } else if (auto start =
                 findAlignedStart(/*topLevelOnly=*/true, /*useTransfer=*/false)) {
    simStart = start->first;
    vcdStart = start->second;
  } else if (auto start =
                 findAlignedStart(/*topLevelOnly=*/false, /*useTransfer=*/true)) {
    simStart = start->first;
    vcdStart = start->second;
  } else if (auto start =
                 findAlignedStart(/*topLevelOnly=*/false, /*useTransfer=*/false)) {
    simStart = start->first;
    vcdStart = start->second;
  }

  unsigned comparedCycles = 0;
  if (iterNum > simStart && risingEdges.size() > vcdStart)
    comparedCycles =
        std::min<unsigned>(iterNum - simStart,
                           static_cast<unsigned>(risingEdges.size()) - vcdStart);
  vcdCheck.comparedCycles = comparedCycles;
  auto countTogglesInWindow = [](const std::vector<uint8_t> &sig,
                                 unsigned start, unsigned span) -> uint64_t {
    if (start >= sig.size() || span < 2)
      return 0;
    unsigned end = std::min<unsigned>(sig.size(), start + span);
    if (end <= start + 1)
      return 0;
    uint64_t toggles = 0;
    for (unsigned i = start + 1; i < end; ++i) {
      if (sig[i] != sig[i - 1])
        ++toggles;
    }
    return toggles;
  };

  uint64_t mismatchCount = 0;
  uint64_t comparedValidSamples = 0;
  uint64_t validMismatchCount = 0;
  uint64_t comparedReadySamples = 0;
  uint64_t readyMismatchCount = 0;
  uint64_t comparedTransferSamples = 0;
  uint64_t transferMismatchCount = 0;
  uint64_t comparedDataSamples = 0;
  uint64_t dataMismatchCount = 0;
  uint64_t dataWordToggleMismatchCount = 0;
  uint64_t dataBitToggleMismatchCount = 0;
  unsigned firstMismatch = 0;
  bool seenMismatch = false;
  for (const MappedWave &mapped : mappedWaves) {
    const ChannelSwitching &sw = *mapped.sim;
    uint64_t simValidToggles =
        countTogglesInWindow(sw.valid, simStart, comparedCycles);
    uint64_t vcdValidToggles =
        countTogglesInWindow(mapped.vcdValid, vcdStart, comparedCycles);
    ++comparedValidSamples;
    if (simValidToggles != vcdValidToggles) {
      ++mismatchCount;
      ++validMismatchCount;
      if (!seenMismatch) {
        firstMismatch = 0;
        seenMismatch = true;
      }
      if (vcdCheck.mismatches.size() < 10000) {
        vcdCheck.mismatches.push_back(
            {sw.src, sw.dst, sw.operandIndex, "valid_toggles", 0,
             simValidToggles, vcdValidToggles});
      }
    }

    uint64_t simReadyToggles =
        countTogglesInWindow(sw.ready, simStart, comparedCycles);
    uint64_t vcdReadyToggles =
        countTogglesInWindow(mapped.vcdReady, vcdStart, comparedCycles);
    ++comparedReadySamples;
    if (simReadyToggles != vcdReadyToggles) {
      ++mismatchCount;
      ++readyMismatchCount;
      if (!seenMismatch) {
        firstMismatch = 0;
        seenMismatch = true;
      }
      if (vcdCheck.mismatches.size() < 10000) {
        vcdCheck.mismatches.push_back(
            {sw.src, sw.dst, sw.operandIndex, "ready_toggles", 0,
             simReadyToggles, vcdReadyToggles});
      }
    }

    uint64_t simTransferToggles =
        countTogglesInWindow(sw.transfer, simStart, comparedCycles);
    uint64_t vcdTransferToggles =
        countTogglesInWindow(mapped.vcdTransfer, vcdStart, comparedCycles);
    ++comparedTransferSamples;
    if (simTransferToggles != vcdTransferToggles) {
      ++mismatchCount;
      ++transferMismatchCount;
      if (!seenMismatch) {
        firstMismatch = 0;
        seenMismatch = true;
      }
      if (vcdCheck.mismatches.size() < 10000) {
        vcdCheck.mismatches.push_back(
            {sw.src, sw.dst, sw.operandIndex, "transfer_toggles", 0,
             simTransferToggles, vcdTransferToggles});
      }
    }

    if (!mapped.hasData || sw.dataWave.empty() || mapped.vcdData.empty())
      continue;

    auto firstOneInWindow = [](const std::vector<uint8_t> &sig, unsigned start,
                               unsigned span) -> std::optional<unsigned> {
      if (start >= sig.size() || span == 0)
        return std::nullopt;
      unsigned end = std::min<unsigned>(sig.size(), start + span);
      for (unsigned i = start; i < end; ++i) {
        if (sig[i] != 0)
          return i - start;
      }
      return std::nullopt;
    };
    auto lastOneInWindow = [](const std::vector<uint8_t> &sig, unsigned start,
                              unsigned span) -> std::optional<unsigned> {
      if (start >= sig.size() || span == 0)
        return std::nullopt;
      unsigned end = std::min<unsigned>(sig.size(), start + span);
      for (unsigned i = end; i > start; --i) {
        if (sig[i - 1] != 0)
          return (i - 1) - start;
      }
      return std::nullopt;
    };

    unsigned dataSimStart = simStart;
    unsigned dataVcdStart = vcdStart;
    std::optional<unsigned> simFirstTransfer =
        firstOneInWindow(sw.transfer, simStart, comparedCycles);
    std::optional<unsigned> vcdFirstTransfer =
        firstOneInWindow(mapped.vcdTransfer, vcdStart, comparedCycles);
    if (simFirstTransfer && vcdFirstTransfer) {
      dataSimStart += *simFirstTransfer;
      dataVcdStart += *vcdFirstTransfer;
    } else {
      std::optional<unsigned> simFirstValid =
          firstOneInWindow(sw.valid, simStart, comparedCycles);
      std::optional<unsigned> vcdFirstValid =
          firstOneInWindow(mapped.vcdValid, vcdStart, comparedCycles);
      if (simFirstValid && vcdFirstValid) {
        dataSimStart += *simFirstValid;
        dataVcdStart += *vcdFirstValid;
      }
    }

    if (dataSimStart >= sw.dataWave.size() || dataVcdStart >= mapped.vcdData.size())
      continue;

    unsigned simShift = dataSimStart - simStart;
    unsigned vcdShift = dataVcdStart - vcdStart;
    unsigned simWindowCycles =
        comparedCycles > simShift ? comparedCycles - simShift : 0;
    unsigned vcdWindowCycles =
        comparedCycles > vcdShift ? comparedCycles - vcdShift : 0;

    unsigned dataCycles =
        std::min<unsigned>(simWindowCycles, sw.dataWave.size() - dataSimStart);
    dataCycles =
        std::min<unsigned>(dataCycles, mapped.vcdData.size() - dataVcdStart);
    dataCycles = std::min<unsigned>(dataCycles, vcdWindowCycles);
    if (!dataCycles)
      continue;

    std::optional<unsigned> simLastTransfer =
        lastOneInWindow(sw.transfer, dataSimStart, dataCycles);
    std::optional<unsigned> vcdLastTransfer =
        lastOneInWindow(mapped.vcdTransfer, dataVcdStart, dataCycles);
    if (simLastTransfer && vcdLastTransfer) {
      dataCycles = std::min<unsigned>(dataCycles,
                                      std::min(*simLastTransfer, *vcdLastTransfer) + 1);
    } else {
      std::optional<unsigned> simLastValid =
          lastOneInWindow(sw.valid, dataSimStart, dataCycles);
      std::optional<unsigned> vcdLastValid =
          lastOneInWindow(mapped.vcdValid, dataVcdStart, dataCycles);
      if (simLastValid && vcdLastValid) {
        dataCycles =
            std::min<unsigned>(dataCycles, std::min(*simLastValid, *vcdLastValid) + 1);
      }
    }
    if (!dataCycles)
      continue;

    SmallVector<APInt> simDataWindow;
    SmallVector<APInt> vcdDataWindow;
    simDataWindow.reserve(dataCycles);
    vcdDataWindow.reserve(dataCycles);
    for (unsigned cycle = 0; cycle < dataCycles; ++cycle) {
      simDataWindow.push_back(
          normalizeAPInt(sw.dataWave[dataSimStart + cycle], mapped.dataWidth));
      vcdDataWindow.push_back(
          normalizeAPInt(mapped.vcdData[dataVcdStart + cycle], mapped.dataWidth));
    }
    SmallVector<APInt> simTransferDataWindow;
    SmallVector<APInt> vcdTransferDataWindow;
    simTransferDataWindow.reserve(dataCycles);
    vcdTransferDataWindow.reserve(dataCycles);
    for (unsigned cycle = 0; cycle < dataCycles; ++cycle) {
      unsigned simCycle = dataSimStart + cycle;
      unsigned vcdCycle = dataVcdStart + cycle;
      bool simXfer = simCycle < sw.transfer.size() && sw.transfer[simCycle] != 0;
      bool vcdXfer =
          vcdCycle < mapped.vcdTransfer.size() && mapped.vcdTransfer[vcdCycle] != 0;
      if (simXfer)
        simTransferDataWindow.push_back(simDataWindow[cycle]);
      if (vcdXfer)
        vcdTransferDataWindow.push_back(vcdDataWindow[cycle]);
    }

    unsigned comparedPayloads = std::min<unsigned>(simTransferDataWindow.size(),
                                                   vcdTransferDataWindow.size());
    comparedDataSamples += comparedPayloads;
    for (unsigned i = 0; i < comparedPayloads; ++i) {
      APInt simData = simTransferDataWindow[i];
      APInt vcdData = vcdTransferDataWindow[i];
      if (simData == vcdData)
        continue;
      ++mismatchCount;
      ++dataMismatchCount;
      if (!seenMismatch) {
        firstMismatch = i;
        seenMismatch = true;
      }
      if (vcdCheck.mismatches.size() < 10000) {
        vcdCheck.mismatches.push_back(
            {sw.src, sw.dst, sw.operandIndex, "data", i,
             simData.getLimitedValue(), vcdData.getLimitedValue()});
      }
    }
    if (simTransferDataWindow.size() != vcdTransferDataWindow.size()) {
      ++mismatchCount;
      ++dataMismatchCount;
      if (!seenMismatch) {
        firstMismatch = 0;
        seenMismatch = true;
      }
      if (vcdCheck.mismatches.size() < 10000) {
        vcdCheck.mismatches.push_back(
            {sw.src, sw.dst, sw.operandIndex, "data_transfer_count", 0,
             simTransferDataWindow.size(), vcdTransferDataWindow.size()});
      }
    }

    uint64_t simDataWordToggles = 0;
    uint64_t vcdDataWordToggles = 0;
    uint64_t simDataBitToggles = 0;
    uint64_t vcdDataBitToggles = 0;
    for (size_t i = 1, e = simTransferDataWindow.size(); i < e; ++i) {
      if (simTransferDataWindow[i] != simTransferDataWindow[i - 1])
        ++simDataWordToggles;
      simDataBitToggles +=
          bitToggleDistance(simTransferDataWindow[i],
                            simTransferDataWindow[i - 1]);
    }
    for (size_t i = 1, e = vcdTransferDataWindow.size(); i < e; ++i) {
      if (vcdTransferDataWindow[i] != vcdTransferDataWindow[i - 1])
        ++vcdDataWordToggles;
      vcdDataBitToggles +=
          bitToggleDistance(vcdTransferDataWindow[i],
                            vcdTransferDataWindow[i - 1]);
    }

    unsigned mismatchCycleRef = 0;
    if (simDataWordToggles != vcdDataWordToggles) {
      ++mismatchCount;
      ++dataWordToggleMismatchCount;
      if (!seenMismatch) {
        firstMismatch = mismatchCycleRef;
        seenMismatch = true;
      }
      if (vcdCheck.mismatches.size() < 10000) {
        vcdCheck.mismatches.push_back({sw.src, sw.dst, sw.operandIndex,
                                       "data_toggles", mismatchCycleRef,
                                       simDataWordToggles, vcdDataWordToggles});
      }
    }
    if (simDataBitToggles != vcdDataBitToggles) {
      ++mismatchCount;
      ++dataBitToggleMismatchCount;
      if (!seenMismatch) {
        firstMismatch = mismatchCycleRef;
        seenMismatch = true;
      }
      if (vcdCheck.mismatches.size() < 10000) {
        vcdCheck.mismatches.push_back(
            {sw.src, sw.dst, sw.operandIndex, "data_bit_toggles",
             mismatchCycleRef, simDataBitToggles, vcdDataBitToggles});
      }
    }
  }

  vcdCheck.comparedValidSamples = comparedValidSamples;
  vcdCheck.validMismatchCount = validMismatchCount;
  vcdCheck.comparedReadySamples = comparedReadySamples;
  vcdCheck.readyMismatchCount = readyMismatchCount;
  vcdCheck.comparedTransferSamples = comparedTransferSamples;
  vcdCheck.transferMismatchCount = transferMismatchCount;
  vcdCheck.comparedDataSamples = comparedDataSamples;
  vcdCheck.dataMismatchCount = dataMismatchCount;
  vcdCheck.dataWordToggleMismatchCount = dataWordToggleMismatchCount;
  vcdCheck.dataBitToggleMismatchCount = dataBitToggleMismatchCount;
  vcdCheck.mismatchCount = mismatchCount;
  vcdCheck.firstMismatchCycle = firstMismatch;
  bool validPass = validMismatchCount == 0;
  bool readyPass = readyMismatchCount == 0;
  bool transferPass = transferMismatchCount == 0;
  bool dataPayloadPass = dataMismatchCount == 0;
  bool dataTogglePass =
      (dataWordToggleMismatchCount == 0) && (dataBitToggleMismatchCount == 0);
  bool dataPass = dataPayloadPass && dataTogglePass;
  vcdCheck.passed = validPass && readyPass && transferPass && dataPass;
  {
    std::string msg;
    raw_string_ostream os(msg);
    os << (vcdCheck.passed ? "switching_and_trace_match"
                           : "switching_or_trace_mismatch")
       << " (valid_toggle_mismatches=" << validMismatchCount
       << ", ready_toggle_mismatches=" << readyMismatchCount
       << ", transfer_toggle_mismatches=" << transferMismatchCount
       << ", data_trace_mismatches=" << dataMismatchCount
       << ", data_payload_pass=" << (dataPayloadPass ? "true" : "false")
       << ", data_toggle_mismatches="
       << (dataWordToggleMismatchCount + dataBitToggleMismatchCount)
       << ", data_toggle_pass=" << (dataTogglePass ? "true" : "false")
       << ")";
    vcdCheck.message = os.str();
  }

  return vcdCheck.passed ? success() : failure();
}

void Simulator::printResults() {
  llvm::outs() << "Results\n";
  for (const ResultRecord &res : results) {
    llvm::outs() << res.name << " (" << res.type << "): " << res.value
                 << " observed=" << res.observed << "\n";
  }
  llvm::outs() << "Number of iterations: " << iterNum << "\n";
}

void Simulator::printModelStates() {
  std::map<std::string, ExecutionModel *> names;
  for (auto &[op, model] : opModels) {
    names.insert({getOperationName(op), model});
  }
  for (auto &[name, model] : names) {
    llvm::outs() << "=================== ===================\n";
    llvm::outs() << name << "\n";
    model->printStates();
  }
}

Simulator::~Simulator() {
  for (auto &[_, model] : opModels)
    delete model;
  for (auto &[_, state] : oldValuesStates)
    delete state;
  for (auto &[_, state] : newValuesStates)
    delete state;
  for (auto &[_, state] : updaters)
    delete state;
  for (auto &[_, rw] : consumerViews)
    delete rw;
  for (auto &[_, rw] : producerViews)
    delete rw;
}

template <typename Model, typename Op, typename... Args>
void Simulator::registerModel(Op op, Args &&...modelArgs) {
  mlir::DenseMap<Value, RW *> subset;
  for (OpOperand &oprd : op->getOpOperands()) {
    auto it = consumerViews.find(&oprd);
    if (it != consumerViews.end())
      subset.insert({oprd.get(), it->second});
  }
  for (Value result : op->getResults()) {
    auto it = producerViews.find(result);
    if (it != producerViews.end())
      subset.insert({result, it->second});
  }

  ExecutionModel *model =
      new Model(op, subset, std::forward<Args>(modelArgs)...);
  opModels.insert({op, model});
}

void Simulator::associateModel(Operation *op) {
  llvm::TypeSwitch<Operation *>(op)
      .Case<handshake::BranchOp>(
          [&](handshake::BranchOp branchOp) { registerModel<BranchModel>(branchOp); })
      .Case<handshake::ConditionalBranchOp>([&](handshake::ConditionalBranchOp op) {
        registerModel<CondBranchModel>(op);
      })
      .Case<handshake::ConstantOp>(
          [&](handshake::ConstantOp op) { registerModel<ConstantModel>(op); })
      .Case<handshake::ControlMergeOp>([&](handshake::ControlMergeOp op) {
        registerModel<ControlMergeModel>(op);
      })
      .Case<handshake::ForkOp>(
          [&](handshake::ForkOp op) { registerModel<ForkModel>(op); })
      .Case<handshake::JoinOp>(
          [&](handshake::JoinOp op) { registerModel<JoinModel>(op); })
      .Case<handshake::LazyForkOp>(
          [&](handshake::LazyForkOp op) { registerModel<LazyForkModel>(op); })
      .Case<handshake::MergeOp>(
          [&](handshake::MergeOp op) { registerModel<MergeModel>(op); })
      .Case<handshake::MuxOp>(
          [&](handshake::MuxOp op) { registerModel<MuxModel>(op); })
      .Case<handshake::LoadOp>(
          [&](handshake::LoadOp op) { registerModel<LoadModel>(op); })
      .Case<handshake::StoreOp>(
          [&](handshake::StoreOp op) { registerModel<StoreModel>(op); })
      .Case<handshake::MemoryControllerOp>([&](handshake::MemoryControllerOp op) {
        registerModel<MemoryControllerModel>(op, *this);
      })
      .Case<handshake::LSQOp>(
          [&](handshake::LSQOp op) { registerModel<LSQModel>(op, *this); })
      .Case<handshake::NotOp>([&](handshake::NotOp notOp) {
        UnaryCompFunc callback = [](const Data &lhs, unsigned) {
          return ~dataCast<APInt>(lhs);
        };
        registerModel<GenericUnaryOpModel<handshake::NotOp>>(notOp, callback);
      })
      .Case<handshake::BufferOp>([&](handshake::BufferOp bufferOp) {
        // Model mapping mirrors generated RTL buffer generators:
        // - ONE_SLOT_BREAK_DV  -> oehb
        // - ONE_SLOT_BREAK_R   -> tehb
        // - FIFO/shift-register families -> queue model.
        handshake::BufferType bufferType = bufferOp.getBufferType();
        if (bufferType == handshake::BufferType::ONE_SLOT_BREAK_DV) {
          registerModel<OEHBModel>(bufferOp);
        } else if (bufferType == handshake::BufferType::ONE_SLOT_BREAK_R) {
          registerModel<TEHBModel>(bufferOp);
        } else if (bufferType == handshake::BufferType::FIFO_BREAK_NONE ||
                   bufferType == handshake::BufferType::FIFO_BREAK_DV ||
                   bufferType == handshake::BufferType::SHIFT_REG_BREAK_DV ||
                   bufferType == handshake::BufferType::ONE_SLOT_BREAK_DVR ||
                   bufferOp.getNumSlots() > 1) {
          registerModel<FIFOBufferModel>(bufferOp);
        } else {
          registerModel<FIFOBufferModel>(bufferOp);
        }
      })
      .Case<handshake::SinkOp>(
          [&](handshake::SinkOp op) { registerModel<SinkModel>(op); })
      .Case<handshake::SourceOp>(
          [&](handshake::SourceOp op) { registerModel<SourceModel>(op); })
      .Case<handshake::AddFOp>([&](handshake::AddFOp op) {
        BinaryCompFunc callback = [](Data lhs, Data rhs) {
          return dataCast<APFloat>(lhs) + dataCast<APFloat>(rhs);
        };
        registerModel<GenericBinaryOpModel<handshake::AddFOp>>(op, callback, 9);
      })
      .Case<handshake::AddIOp>([&](handshake::AddIOp op) {
        BinaryCompFunc callback = [](Data lhs, Data rhs) {
          return dataCast<APInt>(lhs) + dataCast<APInt>(rhs);
        };
        registerModel<GenericBinaryOpModel<handshake::AddIOp>>(op, callback, 0);
      })
      .Case<handshake::AndIOp>([&](handshake::AndIOp op) {
        BinaryCompFunc callback = [](Data lhs, Data rhs) {
          return dataCast<APInt>(lhs) & dataCast<APInt>(rhs);
        };
        registerModel<GenericBinaryOpModel<handshake::AndIOp>>(op, callback, 0);
      })
      .Case<handshake::CmpFOp>([&](handshake::CmpFOp op) {
        BinaryCompFunc callback = [](const Data &, const Data &) {
          return APInt(1, 0);
        };
        registerModel<GenericBinaryOpModel<handshake::CmpFOp>>(op, callback, 4);
      })
      .Case<handshake::CmpIOp>([&](handshake::CmpIOp op) {
        handshake::CmpIPredicate pred = op.getPredicate();
        BinaryCompFunc callback = [pred](Data lhs, Data rhs) {
          APInt l = dataCast<APInt>(lhs), r = dataCast<APInt>(rhs);
          bool value = false;
          switch (pred) {
          case handshake::CmpIPredicate::eq:
            value = l == r;
            break;
          case handshake::CmpIPredicate::ne:
            value = l != r;
            break;
          case handshake::CmpIPredicate::uge:
            value = l.uge(r);
            break;
          case handshake::CmpIPredicate::ugt:
            value = l.ugt(r);
            break;
          case handshake::CmpIPredicate::ule:
            value = l.ule(r);
            break;
          case handshake::CmpIPredicate::ult:
            value = l.ult(r);
            break;
          case handshake::CmpIPredicate::sge:
            value = l.sge(r);
            break;
          case handshake::CmpIPredicate::sgt:
            value = l.sgt(r);
            break;
          case handshake::CmpIPredicate::sle:
            value = l.sle(r);
            break;
          case handshake::CmpIPredicate::slt:
            value = l.slt(r);
            break;
          }
          return APInt(1, value);
        };
        registerModel<GenericBinaryOpModel<handshake::CmpIOp>>(op, callback, 0);
      })
      .Case<handshake::DivFOp>([&](handshake::DivFOp op) {
        BinaryCompFunc callback = [](Data lhs, Data rhs) {
          return dataCast<APFloat>(lhs) / dataCast<APFloat>(rhs);
        };
        registerModel<GenericBinaryOpModel<handshake::DivFOp>>(op, callback, 29);
      })
      .Case<handshake::DivSIOp>([&](handshake::DivSIOp op) {
        BinaryCompFunc callback = [](Data lhs, Data rhs) {
          return dataCast<APInt>(lhs).sdiv(dataCast<APInt>(rhs));
        };
        registerModel<GenericBinaryOpModel<handshake::DivSIOp>>(op, callback, 36);
      })
      .Case<handshake::DivUIOp>([&](handshake::DivUIOp op) {
        BinaryCompFunc callback = [](Data lhs, Data rhs) {
          return dataCast<APInt>(lhs).udiv(dataCast<APInt>(rhs));
        };
        registerModel<GenericBinaryOpModel<handshake::DivUIOp>>(op, callback, 36);
      })
      .Case<handshake::ExtSIOp>([&](handshake::ExtSIOp op) {
        UnaryCompFunc callback = [](const Data &lhs, unsigned outWidth) {
          return dataCast<APInt>(lhs).sext(outWidth);
        };
        registerModel<GenericUnaryOpModel<handshake::ExtSIOp>>(op, callback);
      })
      .Case<handshake::ExtUIOp>([&](handshake::ExtUIOp op) {
        UnaryCompFunc callback = [](const Data &lhs, unsigned outWidth) {
          return dataCast<APInt>(lhs).zext(outWidth);
        };
        registerModel<GenericUnaryOpModel<handshake::ExtUIOp>>(op, callback);
      })
      .Case<handshake::MaximumFOp>([&](handshake::MaximumFOp op) {
        BinaryCompFunc callback = [](Data lhs, Data rhs) {
          return maxnum(dataCast<APFloat>(lhs), dataCast<APFloat>(rhs));
        };
        registerModel<GenericBinaryOpModel<handshake::MaximumFOp>>(op, callback, 2);
      })
      .Case<handshake::MinimumFOp>([&](handshake::MinimumFOp op) {
        BinaryCompFunc callback = [](Data lhs, Data rhs) {
          return minnum(dataCast<APFloat>(lhs), dataCast<APFloat>(rhs));
        };
        registerModel<GenericBinaryOpModel<handshake::MinimumFOp>>(op, callback, 2);
      })
      .Case<handshake::MulFOp>([&](handshake::MulFOp op) {
        BinaryCompFunc callback = [](Data lhs, Data rhs) {
          return dataCast<APFloat>(lhs) * dataCast<APFloat>(rhs);
        };
        registerModel<GenericBinaryOpModel<handshake::MulFOp>>(op, callback, 4);
      })
      .Case<handshake::MulIOp>([&](handshake::MulIOp op) {
        BinaryCompFunc callback = [](Data lhs, Data rhs) {
          return dataCast<APInt>(lhs) * dataCast<APInt>(rhs);
        };
        registerModel<GenericBinaryOpModel<handshake::MulIOp>>(op, callback, 4);
      })
      .Case<handshake::NegFOp>([&](handshake::NegFOp op) {
        UnaryCompFunc callback = [](const Data &lhs, unsigned) {
          return neg(dataCast<APFloat>(lhs));
        };
        registerModel<GenericUnaryOpModel<handshake::NegFOp>>(op, callback);
      })
      .Case<handshake::OrIOp>([&](handshake::OrIOp op) {
        BinaryCompFunc callback = [](Data lhs, Data rhs) {
          return dataCast<APInt>(lhs) | dataCast<APInt>(rhs);
        };
        registerModel<GenericBinaryOpModel<handshake::OrIOp>>(op, callback, 0);
      })
      .Case<handshake::SelectOp>(
          [&](handshake::SelectOp op) { registerModel<SelectModel>(op); })
      .Case<handshake::ShLIOp>([&](handshake::ShLIOp op) {
        BinaryCompFunc callback = [](Data lhs, Data rhs) {
          return dataCast<APInt>(lhs) << dataCast<APInt>(rhs);
        };
        registerModel<GenericBinaryOpModel<handshake::ShLIOp>>(op, callback, 0);
      })
      .Case<handshake::ShRSIOp>([&](handshake::ShRSIOp op) {
        BinaryCompFunc callback = [](Data lhs, Data rhs) {
          return dataCast<APInt>(lhs).ashr(dataCast<APInt>(rhs));
        };
        registerModel<GenericBinaryOpModel<handshake::ShRSIOp>>(op, callback, 0);
      })
      .Case<handshake::ShRUIOp>([&](handshake::ShRUIOp op) {
        BinaryCompFunc callback = [](Data lhs, Data rhs) {
          return dataCast<APInt>(lhs).lshr(dataCast<APInt>(rhs));
        };
        registerModel<GenericBinaryOpModel<handshake::ShRUIOp>>(op, callback, 0);
      })
      .Case<handshake::SubFOp>([&](handshake::SubFOp op) {
        BinaryCompFunc callback = [](Data lhs, Data rhs) {
          return dataCast<APFloat>(lhs) - dataCast<APFloat>(rhs);
        };
        registerModel<GenericBinaryOpModel<handshake::SubFOp>>(op, callback, 9);
      })
      .Case<handshake::SubIOp>([&](handshake::SubIOp op) {
        BinaryCompFunc callback = [](Data lhs, Data rhs) {
          return dataCast<APInt>(lhs) - dataCast<APInt>(rhs);
        };
        registerModel<GenericBinaryOpModel<handshake::SubIOp>>(op, callback, 0);
      })
      .Case<handshake::TruncIOp>(
          [&](handshake::TruncIOp op) { registerModel<TruncIModel>(op); })
      .Case<handshake::XOrIOp>([&](handshake::XOrIOp op) {
        BinaryCompFunc callback = [](Data lhs, Data rhs) {
          return dataCast<APInt>(lhs) ^ dataCast<APInt>(rhs);
        };
        registerModel<GenericBinaryOpModel<handshake::XOrIOp>>(op, callback, 0);
      })
      .Case<handshake::EndOp>([&](handshake::EndOp op) {
        registerModel<EndModel>(op, resValid, resReady, resData);
      })
      .Default([&](Operation *unsupportedOp) {
        std::string err;
        raw_string_ostream os(err);
        os << "Unsupported operation in simulator: " << unsupportedOp->getName();
        (void)signalFailure(os.str());
      });
}

template <typename State, typename Updater, typename Producer, typename Consumer,
          typename Ty>
void Simulator::registerState(Value val, Operation *, Ty type) {
  TypedValue<Ty> typedVal = cast<TypedValue<Ty>>(val);

  State *oldState = new State(typedVal);
  State *newState = new State(typedVal);
  Updater *upd = new Updater(*oldState, *newState);
  Producer *producerRW = new Producer(*oldState, *newState);

  oldValuesStates.insert({val, oldState});
  newValuesStates.insert({val, newState});
  updaters.insert({val, upd});
  producerViews.insert({val, producerRW});

  for (OpOperand &use : val.getUses()) {
    Consumer *consumerRW = new Consumer(*oldState, *newState);
    consumerViews.insert({&use, consumerRW});
  }
}

void Simulator::associateState(Value val, Operation *producerOp, Location loc) {
  llvm::TypeSwitch<Type>(val.getType())
      .Case<handshake::ChannelType>([&](handshake::ChannelType channelType) {
        registerState<ChannelState, ChannelUpdater, ChannelProducerRW,
                      ChannelConsumerRW>(val, producerOp, channelType);
      })
      .Case<handshake::ControlType>([&](handshake::ControlType controlType) {
        registerState<ControlState, ControlUpdater, ControlProducerRW,
                      ControlConsumerRW>(val, producerOp, controlType);
      })
      .Case<MemRefType>([&](MemRefType) {
        // Memrefs are modeled as memory images and do not carry handshake state.
      })
      .Default([&](Type) {
        emitError(loc) << "Value " << val << " has unsupported type";
        (void)signalFailure("Unsupported value type in simulator state map");
      });
}
