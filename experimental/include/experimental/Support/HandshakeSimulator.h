#ifndef EXPERIMENTAL_SUPPORT_HANDSHAKE_SIMULATOR_H
#define EXPERIMENTAL_SUPPORT_HANDSHAKE_SIMULATOR_H
#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Dialect/Handshake/HandshakeTypes.h"
#include "dynamatic/Support/JSON/JSON.h"
#include "dynamatic/Support/LLVM.h"
#include "dynamatic/Transforms/HandshakeMaterialize.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/Any.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include <cstdint>
#include <deque>
#include <functional>
#include <map>
#include <optional>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

using namespace llvm;
using namespace mlir;
using namespace dynamatic;

namespace dynamatic {
namespace experimental {

enum class HSInputFormat { Auto, Dat, Plain };

struct ArgInputTransactions {
  llvm::SmallVector<llvm::SmallVector<std::string>> transactions;
  bool provided = false;
};

struct SimulationInputs {
  mlir::DenseMap<unsigned, ArgInputTransactions> byArgIndex;
  unsigned numTransactions = 0;
};

struct SimulatorOptions {
  unsigned maxCycles = 1000000;
  std::string lsqConfigDir;
};

class Simulator;

// The wrapper for llvm::Any that lets the user check if the value was changed
// It's required for updaters to stop the internal loop
struct Data {
  llvm::Any data;
  llvm::hash_code hash;
  unsigned bitwidth;

  Data() = default;

  Data(const APInt &value);

  Data(const APFloat &value);

  Data &operator=(const APInt &value);

  Data &operator=(const APFloat &value);

  bool hasValue() const;
};

template <class T>
T dataCast(const Data &value) {
  return llvm::any_cast<T>(value.data);
}

template <class T>
T dataCast(Data &value) {
  return llvm::any_cast<T>(value.data);
}

template <class T>
T dataCast(Data &&value) {
  return llvm::any_cast<T>(value.data);
}

template <class T>
const T *dataCast(const Data *value) {
  return llvm::any_cast<T>(&value->data);
}

template <class T>
T *dataCast(Data *value) {
  return llvm::any_cast<T>(&value->data);
}

//===----------------------------------------------------------------------===//
// States
//===----------------------------------------------------------------------===//
/// Classes that store states.
/// The simulator defines 2 maps - newValueStates and oldValueStates - to
/// represent the previous (used during the signals' collection to prevent "data
/// race conditions") and current (where we collect new values to). The map
/// updaters (inside the Simulator class as well), in fact, containing
/// references to corresponding pair of new and old valueStates, assign the
/// newly collected values from newValueStates to oldValueStates when the
/// collection is finished.

// The state in circuit's value states
class ValueState {
public:
  ValueState(Value val);

  virtual ~ValueState() = default;

protected:
  Value val;
};

/// The state, templated with concrete hasndshake value type
template <typename Ty>
class TypedValueState : public ValueState {
public:
  TypedValueState(TypedValue<Ty> val);
};

class ChannelState : public TypedValueState<handshake::ChannelType> {
  // Give the corresponding RW API an access to data members
  friend struct ChannelConsumerRW;
  friend struct ChannelProducerRW;
  // Give the corresponding Updater an access to data members
  friend class ChannelUpdater;

public:
  using TypedValueState<handshake::ChannelType>::TypedValueState;

  ChannelState(TypedValue<handshake::ChannelType> channel);

protected:
  bool valid = false;
  bool ready = false;
  Data data = {};
  // Maybe some additional data signals here
};

class ControlState : public TypedValueState<handshake::ControlType> {
  // Give the corresponding RW API an access to data members
  friend struct ControlConsumerRW;
  friend struct ControlProducerRW;
  // Give the corresponding Updater an access to data members
  friend class ControlUpdater;

public:
  using TypedValueState<handshake::ControlType>::TypedValueState;

  ControlState(TypedValue<handshake::ControlType> control);

protected:
  bool valid = false;
  bool ready = false;
};

//===----------------------------------------------------------------------===//
// Updater
//===----------------------------------------------------------------------===//
/// Classes to update old values states with new ones after finishing the data
/// collection.
/// The map updaters (defined inside the Simulator class) contains references to
/// the corresponding pair of new and old valueStates, assigns the newly
/// collected values from newValueStates to oldValueStates when the collection
/// is finished.

/// Base Updater class
class Updater {
public:
  virtual ~Updater() = default;
  // Check if, for some oldValueState, newValueState, oldValueState ==
  // newValueState
  virtual bool check() = 0;
  // Update oldValueState with the values of newValueState
  virtual void update() = 0;
  // Make valid signal of the state true (For example, when need manual
  // management for circuit's external inputs)
  virtual void setValid() = 0;
  // For the particular valueState, remove valid if ready is set
  virtual void resetValid() = 0;
};

/// Templated updater class that contains pair <oldValueState, newValueState>
template <typename State>
class DoubleUpdater : public Updater {
protected:
  State &oldState;
  State &newState;

public:
  DoubleUpdater(State &oldState, State &newState);
};

/// Update valueStates with "Channel type": valid, ready, data
class ChannelUpdater : public DoubleUpdater<ChannelState> {
public:
  ChannelUpdater(ChannelState &oldState, ChannelState &newState);
  bool check() override;
  void setValid() override;
  void resetValid() override;
  void update() override;
};

/// Update valueStates with "Control type": valid, ready
class ControlUpdater : public DoubleUpdater<ControlState> {
public:
  ControlUpdater(ControlState &oldState, ControlState &newState);
  bool check() override;
  void setValid() override;
  void resetValid() override;
  void update() override;
};

//===----------------------------------------------------------------------===//
// Readers & writers
//===----------------------------------------------------------------------===//

/// Classes with references to oldValueStates and newValueStates to represent
/// readers/writers API (that is, opened the user's access). The map
/// <value, ProducerRW*> producerViews (inside the Simulator class) stores such
/// API's for values, data & valid signals of which can be changed. The map
/// <OpOperand*, ConsumerRW*> consumerViews contains pointers to "consumer
/// views" for all uses of the particular value. Keys in these maps are not the
/// same because multiple users may share the same value. The case when there're
/// several OpOperands connecting 2 components within one value is considered to
/// be a just one OpOperand.

/// Base RW
class RW {
public:
  RW() = default;
  virtual ~RW() = default;
};

class ProducerRW : public RW {
protected:
  enum ProducerDescendants { D_ChannelProducerRW, D_ControlProducerRW };

public:
  bool &valid;
  const bool &ready;

public:
  ProducerDescendants getType() const;

  ProducerRW(bool &valid, const bool &ready,
             ProducerDescendants p = D_ControlProducerRW);

  ProducerRW(ProducerRW &p);

  virtual ~ProducerRW() = default;

private:
  // LLVM RTTI is used here for convenient conversion
  const ProducerDescendants prod;
};

class ConsumerRW : public RW {
public:
  const bool &valid;
  bool &ready;

protected:
  enum ConsumerDescendants { D_ChannelConsumerRW, D_ControlConsumerRW };

public:
  ConsumerDescendants getType() const;

  ConsumerRW(const bool &valid, bool &ready,
             ConsumerDescendants c = D_ControlConsumerRW);

  ConsumerRW(ConsumerRW &c);

  virtual ~ConsumerRW() = default;

private:
  // LLVM RTTI is used here for convenient conversion
  const ConsumerDescendants cons;
};

////--- Control (no data)

/// In this case the user can change the ready signal, but has
/// ReadOnly access to the valid one.
struct ControlConsumerRW : public ConsumerRW {
  ControlConsumerRW(ControlState &reader, ControlState &writer);

  static bool classof(const ConsumerRW *c);
};

/// In this case the user can change the valid signal, but has
/// ReadOnly access to the ready one.
struct ControlProducerRW : public ProducerRW {
  ControlProducerRW(ControlState &reader, ControlState &writer);

  static bool classof(const ProducerRW *c);
};

////--- Channel (data)

/// In this case the user can change the valid signal, but has ReadOnly access
/// to the valid and data ones.
struct ChannelConsumerRW : public ConsumerRW {
  const Data &data;

  ChannelConsumerRW(ChannelState &reader, ChannelState &writer);

  static bool classof(const ConsumerRW *c);
};

/// In this case the user can change the valid and data signals, but has
/// ReadOnly access to the ready one.
struct ChannelProducerRW : public ProducerRW {
  Data &data;

  ChannelProducerRW(ChannelState &reader, ChannelState &writer);

  static bool classof(const ProducerRW *c);
};

//===----------------------------------------------------------------------===//
// Datafull & Dataless
//===----------------------------------------------------------------------===//
/// The following structures are used to represent components that can be
/// either datafull or dataless. They store a pointer to Data and maybe some
/// userful members and functions (e.g. hasValue() or the width).

/// A struct to represent consumer's data
struct ConsumerData {
  Data const *data = nullptr;
  unsigned dataWidth = 0;

  ConsumerData(ConsumerRW *ins);

  bool hasValue() const;
};

/// A struct to represent producer's data
struct ProducerData {
  Data *data = nullptr;
  unsigned dataWidth = 0;

  ProducerData(ProducerRW *outs);

  ProducerData &operator=(const ConsumerData &value);

  bool hasValue() const;
};

//===----------------------------------------------------------------------===//
// Execution Model
//===----------------------------------------------------------------------===//
/// Classes to represent execution models

/// Base class
class ExecutionModel {

public:
  ExecutionModel(Operation *op);
  virtual void reset() = 0;
  virtual void exec(bool isClkRisingEdge) = 0;
  virtual void printStates() = 0;

  virtual ~ExecutionModel() = default;

protected:
  Operation *op;
};

/// Typed execution model
template <typename Op>
class OpExecutionModel : public ExecutionModel {
public:
  OpExecutionModel(Op op);

protected:
  Op getOperation();

  // Get exact RW state type (ChannelProducerRW / ChannelConsumerRW /
  // ControlProducerRW / ControlConsumerRW)
  template <typename State>
  static State *getState(Value val, mlir::DenseMap<Value, RW *> &rws);

  // Just a temporary function to print models' states
  template <typename State, typename Data>
  void printValue(const std::string &name, State *ins, Data *insData);
};

template <typename Op>
OpExecutionModel<Op>::OpExecutionModel(Op op)
    : ExecutionModel(op.getOperation()) {}

template <typename Op>
Op OpExecutionModel<Op>::getOperation() {
  return cast<Op>(op);
}

template <typename Op>
template <typename State>
State *OpExecutionModel<Op>::getState(Value val,
                                      mlir::DenseMap<Value, RW *> &rws) {
  return static_cast<State *>(rws[val]);
}

template <typename Op>
template <typename State, typename DataTy>
void OpExecutionModel<Op>::printValue(const std::string &name, State *ins,
                                      DataTy *insData) {
  llvm::outs() << name << ": ";
  if (insData) {
    APInt t = dataCast<APInt>(*insData);
    auto val = t.reverseBits().getZExtValue();
    for (unsigned i = 0; i < t.getBitWidth(); ++i) {
      llvm::outs() << (val & 1);
      val >>= 1;
    }
    llvm::outs() << " ";
  }

  llvm::outs() << ins->valid << " " << ins->ready << "\n";
}

//===----------------------------------------------------------------------===//
// Support
//===----------------------------------------------------------------------===//
/// Components required for the internal state.

class Antotokens {
public:
  Antotokens() = default;

  void reset(const bool &pvalid1, const bool &pvalid0, bool &kill1, bool &kill0,
             const bool &generateAt1, const bool &generateAt0, bool &stopValid);

  void exec(bool isClkRisingEdge, const bool &pvalid1, const bool &pvalid0,
            bool &kill1, bool &kill0, const bool &generateAt1,
            const bool &generateAt0, bool &stopValid);

private:
  bool regIn0 = false, regIn1 = false, regOut0 = false, regOut1 = false;
};

class ForkSupport {
public:
  ForkSupport(unsigned size, unsigned datawidth = 0);

  void resetDataless(ConsumerRW *ins, std::vector<ProducerRW *> &outs);

  void execDataless(bool isClkRisingEdge, ConsumerRW *ins,
                    std::vector<ProducerRW *> &outs);
  void reset(ConsumerRW *ins, std::vector<ProducerRW *> &outs,
             const ConsumerData &insData, std::vector<ProducerData> &outsData);

  void exec(bool isClkRisingEdge, ConsumerRW *ins,
            std::vector<ProducerRW *> &outs, const ConsumerData &insData,
            std::vector<ProducerData> &outsData);

private:
  unsigned size;
  // eager fork
  std::vector<bool> transmitValue, keepValue;
  // fork dataless
  std::vector<bool> blockStopArray;
  bool anyBlockStop = false, backpressure = false;

public:
  unsigned datawidth;
};

class JoinSupport {
public:
  JoinSupport(unsigned size);

  void exec(std::vector<ConsumerRW *> &ins, ProducerRW *outs);

private:
  unsigned size;
};

class OEHBSupport {
public:
  OEHBSupport(unsigned datawidth = 0);

  void resetDataless(ConsumerRW *ins, ProducerRW *outs);

  void execDataless(bool isClkRisingEdge, ConsumerRW *ins, ProducerRW *outs);

  void reset(ConsumerRW *ins, ProducerRW *outs, const Data *insData,
             Data *outsData);

  void exec(bool isClkRisingEdge, ConsumerRW *ins, ProducerRW *outs,
            const Data *insData, Data *outsData);

  unsigned datawidth;

private:
  // oehb dataless
  bool outputValid = false;
  // oehb datafull
  bool regEn = false;
};

class TEHBSupport {
public:
  TEHBSupport(unsigned datawidth = 0);
  void resetDataless(ConsumerRW *ins, ProducerRW *outs);
  void execDataless(bool isClkRisingEdge, ConsumerRW *ins, ProducerRW *outs);

  void reset(ConsumerRW *ins, ProducerRW *outs, const Data *insData,
             Data *outsData);
  void exec(bool isClkRisingEdge, ConsumerRW *ins, ProducerRW *outs,
            const Data *insData, Data *outsData);
  unsigned datawidth = 0;

private:
  // tehb dataless
  bool fullReg = false, outputValid = false;
  // tehb datafull
  bool regNotFull = false, regEnable = false;
  Data dataReg;

  void resetDataFull(ConsumerRW *ins, ProducerRW *outs, const Data *insData,
                     Data *outsData);

  void execDataFull(bool isClkRisingEdge, ConsumerRW *ins, ProducerRW *outs,
                    const Data *insData, Data *outsData);
};

//===----------------------------------------------------------------------===//
// Handshake
//===----------------------------------------------------------------------===//
/// Handshake components. Some of them can represent both datafull and dataless
/// options.

/// Example of a model that can be initialized with o without data.
class BranchModel : public OpExecutionModel<handshake::BranchOp> {
public:
  using OpExecutionModel<handshake::BranchOp>::OpExecutionModel;
  BranchModel(handshake::BranchOp branchOp,
              mlir::DenseMap<Value, RW *> &subset);
  void reset() override;

  void exec(bool isClkRisingEdge) override;

  void printStates() override;

private:
  // ports
  ConsumerRW *ins;
  ProducerRW *outs;

  ConsumerData insData;
  ProducerData outsData;
};

class CondBranchModel
    : public OpExecutionModel<handshake::ConditionalBranchOp> {
public:
  using OpExecutionModel<handshake::ConditionalBranchOp>::OpExecutionModel;
  CondBranchModel(handshake::ConditionalBranchOp condBranchOp,
                  mlir::DenseMap<Value, RW *> &subset);

  void reset() override;
  void exec(bool isClkRisingEdge) override;

  void printStates() override;

private:
  // ports
  ConsumerRW *data;
  ChannelConsumerRW *condition;
  ProducerRW *trueOut, *falseOut;

  ConsumerData dataData;
  ProducerData trueOutData, falseOutData;

  // cond br dataless
  bool brInpValid = false;

  // internal components
  JoinSupport condBrJoin;
};

class ConstantModel : public OpExecutionModel<handshake::ConstantOp> {
public:
  using OpExecutionModel<handshake::ConstantOp>::OpExecutionModel;
  ConstantModel(handshake::ConstantOp constOp,
                mlir::DenseMap<Value, RW *> &subset);

  void reset() override;

  void exec(bool isClkRisingEdge) override;

  void printStates() override;

private:
  // ports
  Data value;
  ControlConsumerRW *ctrl;
  ChannelProducerRW *outs;
};

class ControlMergeModel : public OpExecutionModel<handshake::ControlMergeOp> {
public:
  using OpExecutionModel<handshake::ControlMergeOp>::OpExecutionModel;
  ControlMergeModel(handshake::ControlMergeOp cMergeOp,
                    mlir::DenseMap<Value, RW *> &subset);

  void reset() override;

  void exec(bool isClkRisingEdge) override;

  void printStates() override;

private:
  // parameters
  unsigned size, indexWidth;

  // internal components
  TEHBSupport cMergeTEHB;
  ForkSupport cMergeFork;

  // ports
  std::vector<ConsumerRW *> ins;
  ProducerRW *outs;
  ChannelProducerRW *index;

  std::vector<ConsumerData> insData;
  ProducerData outsData;

  // control merge dataless
  Data indexTEHB = APInt(indexWidth, 0);
  bool dataAvailable = false, readyToFork = false, tehbOutValid = false,
       tehbOutReady = false;
  ConsumerRW insTEHB;
  ProducerRW outsTEHB;
  ConsumerRW insFork;
  std::vector<ProducerRW *> outsFork;
};

class ForkModel : public OpExecutionModel<handshake::ForkOp> {
public:
  using OpExecutionModel<handshake::ForkOp>::OpExecutionModel;
  ForkModel(handshake::ForkOp forkOp, mlir::DenseMap<Value, RW *> &subset);

  void reset() override;

  void exec(bool isClkRisingEdge) override;

  void printStates() override;

private:
  // parameters
  unsigned size;
  ForkSupport forkSupport;

  // ports
  ConsumerRW *ins;
  std::vector<ProducerRW *> outs;

  ConsumerData insData;
  std::vector<ProducerData> outsData;
};

// Never used but let it be
class JoinModel : public OpExecutionModel<handshake::JoinOp> {
public:
  using OpExecutionModel<handshake::JoinOp>::OpExecutionModel;
  JoinModel(handshake::JoinOp joinOp, mlir::DenseMap<Value, RW *> &subset);

  void reset() override;

  void exec(bool isClkRisingEdge) override;

  void printStates() override;

private:
  // ports
  std::vector<ConsumerRW *> ins;
  ProducerRW *outs;

  // internal components
  JoinSupport join;
};

class LazyForkModel : public OpExecutionModel<handshake::LazyForkOp> {
public:
  using OpExecutionModel<handshake::LazyForkOp>::OpExecutionModel;
  LazyForkModel(handshake::LazyForkOp lazyForkOp,
                mlir::DenseMap<Value, RW *> &subset);

  void reset() override;
  void exec(bool isClkRisingEdge) override;

  void printStates() override;

private:
  // parameters
  unsigned size;

  // ports
  ConsumerRW *ins;
  std::vector<ProducerRW *> outs;

  ConsumerData insData;
  std::vector<ProducerData> outsData;
};

class MergeModel : public OpExecutionModel<handshake::MergeOp> {
public:
  using OpExecutionModel<handshake::MergeOp>::OpExecutionModel;
  MergeModel(handshake::MergeOp mergeOp, mlir::DenseMap<Value, RW *> &subset);

  void reset() override;

  void exec(bool isClkRisingEdge) override;

  void printStates() override;

private:
  // parameters
  unsigned size;

  // ports
  std::vector<ConsumerRW *> ins;
  ProducerRW *outs;

  std::vector<ConsumerData> insData;
  ProducerData outsData;

  // merge dataless
  bool tehbValid = false, tehbReady = false;
  ConsumerRW insTEHB;
  // merge datafull
  Data tehbDataIn;

  // internal components
  TEHBSupport mergeTEHB;

  void execDataless();

  void execDataFull();
};

class MuxModel : public OpExecutionModel<handshake::MuxOp> {
public:
  using OpExecutionModel<handshake::MuxOp>::OpExecutionModel;
  MuxModel(handshake::MuxOp muxOp, mlir::DenseMap<Value, RW *> &subset);

  void reset() override;

  void exec(bool isClkRisingEdge) override;

  void printStates() override;

private:
  // parameters
  unsigned size, selectWidth;
  unsigned indexNum = 0;

  // ports
  std::vector<ConsumerRW *> ins;
  ChannelConsumerRW *index;
  ProducerRW *outs;

  std::vector<ConsumerData> insData;
  ProducerData outsData;

  void execDataless();

  void execDataFull();
};

class OEHBModel : public OpExecutionModel<handshake::BufferOp> {
public:
  using OpExecutionModel<handshake::BufferOp>::OpExecutionModel;

  OEHBModel(handshake::BufferOp oehbOp, mlir::DenseMap<Value, RW *> &subset);

  void reset() override;

  void exec(bool isClkRisingEdge) override;

  void printStates() override;

private:
  // ports
  ConsumerRW *ins;
  ProducerRW *outs;

  ConsumerData insData;
  ProducerData outsData;

  // internal components
  OEHBSupport oehbDl;
};

class SinkModel : public OpExecutionModel<handshake::SinkOp> {
public:
  using OpExecutionModel<handshake::SinkOp>::OpExecutionModel;
  SinkModel(handshake::SinkOp sinkOp, mlir::DenseMap<Value, RW *> &subset);

  void reset() override;

  void exec(bool isClkRisingEdge) override;

  void printStates() override;

private:
  // ports
  ConsumerRW *ins;
  ConsumerData insData;
};

class SourceModel : public OpExecutionModel<handshake::SourceOp> {
public:
  using OpExecutionModel<handshake::SourceOp>::OpExecutionModel;
  SourceModel(handshake::SourceOp sourceOp,
              mlir::DenseMap<Value, RW *> &subset);

  void reset() override;

  void exec(bool isClkRisingEdge) override;

  void printStates() override;

private:
  // ports
  ProducerRW *outs;
};

class TEHBModel : public OpExecutionModel<handshake::BufferOp> {
public:
  using OpExecutionModel<handshake::BufferOp>::OpExecutionModel;

  TEHBModel(handshake::BufferOp tehbOp, mlir::DenseMap<Value, RW *> &subset);

  void reset() override;

  void exec(bool isClkRisingEdge) override;

  void printStates() override;

private:
  // ports
  ConsumerRW *ins;
  ProducerRW *outs;

  ConsumerData insData;
  ProducerData outsData;

  // internal components
  TEHBSupport returnTEHB;
};

/// Multi-slot elastic buffer model for `tfifo`/`elastic_fifo_inner` style units.
/// It implements exact bypass and queue handshaking equations used in RTL.
class FIFOBufferModel : public OpExecutionModel<handshake::BufferOp> {
public:
  using OpExecutionModel<handshake::BufferOp>::OpExecutionModel;
  FIFOBufferModel(handshake::BufferOp bufferOp,
                  mlir::DenseMap<Value, RW *> &subset);

  void reset() override;

  void exec(bool isClkRisingEdge) override;

  void printStates() override;

private:
  ConsumerRW *ins;
  ProducerRW *outs;
  ConsumerData insData;
  ProducerData outsData;

  unsigned capacity = 1;
  bool bypass = false;
  bool hasData = false;
  unsigned occupancy = 0;
  std::deque<Data> payloads;
  bool readEnable = false;
  bool writeEnable = false;

  void updateCombinational();
  void updateSequential();
};

class LoadModel : public OpExecutionModel<handshake::LoadOp> {
public:
  using OpExecutionModel<handshake::LoadOp>::OpExecutionModel;
  LoadModel(handshake::LoadOp loadOp, mlir::DenseMap<Value, RW *> &subset);

  void reset() override;

  void exec(bool isClkRisingEdge) override;

  void printStates() override;

private:
  ChannelConsumerRW *addrIn;
  ChannelConsumerRW *dataFromMem;
  ChannelProducerRW *addrOut;
  ChannelProducerRW *dataOut;

  TEHBSupport addrTEHB;
  TEHBSupport dataTEHB;
};

class StoreModel : public OpExecutionModel<handshake::StoreOp> {
public:
  using OpExecutionModel<handshake::StoreOp>::OpExecutionModel;
  StoreModel(handshake::StoreOp storeOp, mlir::DenseMap<Value, RW *> &subset);

  void reset() override;

  void exec(bool isClkRisingEdge) override;

  void printStates() override;

private:
  ChannelConsumerRW *addrIn;
  ChannelConsumerRW *dataIn;
  ChannelProducerRW *addrOut;
  ChannelProducerRW *dataOut;
};

class MemoryControllerModel : public OpExecutionModel<handshake::MemoryControllerOp> {
public:
  using OpExecutionModel<handshake::MemoryControllerOp>::OpExecutionModel;
  MemoryControllerModel(handshake::MemoryControllerOp mcOp,
                        mlir::DenseMap<Value, RW *> &subset,
                        Simulator &sim);

  void reset() override;

  void exec(bool isClkRisingEdge) override;

  void printStates() override;

  uint64_t getPendingCount() const;

private:
  struct MCPortLoad {
    ChannelConsumerRW *addrIn = nullptr;
    ChannelProducerRW *dataOut = nullptr;
  };

  struct MCPortStore {
    ChannelConsumerRW *addrIn = nullptr;
    ChannelConsumerRW *dataIn = nullptr;
  };

  handshake::MemoryControllerOp mcOp;
  Simulator &sim;
  Value memref;

  ControlConsumerRW *memStart = nullptr;
  ControlConsumerRW *ctrlEnd = nullptr;
  ControlProducerRW *memEnd = nullptr;

  std::vector<ChannelConsumerRW *> ctrlInputs;
  std::vector<MCPortLoad> loadPorts;
  std::vector<MCPortStore> storePorts;

  bool running = false;
  bool noMoreRequests = false;
  uint64_t pendingStores = 0;

  struct PendingLoad {
    unsigned portIdx = 0;
    Data data;
    unsigned remainingCycles = 1;
  };
  std::deque<PendingLoad> pendingLoads;
  std::vector<uint8_t> loadValidRegs;
  std::vector<Data> loadDataRegs;

  std::optional<unsigned> selectedLoadPort;
  std::optional<unsigned> selectedStorePort;
  std::optional<Data> selectedStoreAddr;
  std::optional<Data> selectedStoreData;

  void updateSequential();
  void updateCombinational();
};

class LSQModel : public OpExecutionModel<handshake::LSQOp> {
public:
  using OpExecutionModel<handshake::LSQOp>::OpExecutionModel;
  LSQModel(handshake::LSQOp lsqOp, mlir::DenseMap<Value, RW *> &subset,
           Simulator &sim);

  void reset() override;

  void exec(bool isClkRisingEdge) override;

  void printStates() override;

  uint64_t getPendingCount() const;

private:
  enum class AccessKind { Load, Store };

  struct Access {
    AccessKind kind;
    unsigned groupIdx = 0;
    unsigned groupAccessIndex = 0;
    unsigned requiredStoresBefore = 0;
    ChannelConsumerRW *addrIn = nullptr;
    ChannelConsumerRW *dataIn = nullptr;
    ChannelProducerRW *dataOut = nullptr;
  };

  struct GroupFrame {
    unsigned issuedCount = 0;
    unsigned acceptedStores = 0;
    std::vector<uint8_t> issued;
  };

  struct GroupState {
    ControlConsumerRW *ctrl = nullptr;
    std::vector<unsigned> accesses;
    unsigned numLoads = 0;
    unsigned numStores = 0;
    std::deque<GroupFrame> frames;
  };

  handshake::LSQOp lsqOp;
  Simulator &sim;
  bool isMaster = false;
  Value memref;

  ControlConsumerRW *memStart = nullptr;
  ControlConsumerRW *ctrlEnd = nullptr;
  ControlProducerRW *memEnd = nullptr;

  ChannelConsumerRW *ldDataFromMC = nullptr;
  ChannelProducerRW *ldAddrToMC = nullptr;
  ChannelProducerRW *stAddrToMC = nullptr;
  ChannelProducerRW *stDataToMC = nullptr;

  std::vector<GroupState> groups;
  std::vector<Access> accesses;
  std::vector<uint64_t> addrSlotsByAccess;
  std::vector<uint64_t> storeDataSlotsByAccess;
  std::vector<std::deque<Data>> pendingAddrByAccess;
  std::vector<std::deque<Data>> pendingStoreDataByAccess;
  std::vector<std::deque<Data>> loadResponsesByAccess;
  std::deque<std::tuple<unsigned, Data, unsigned>> pendingMasterLoads;
  std::optional<unsigned> selectedLoadAccess;
  std::optional<unsigned> selectedStoreAccess;
  Data selectedLoadAddr;
  Data selectedStoreAddr;
  Data selectedStoreData;
  /// When false, only one group can be accepted/issued at a time. This mirrors
  /// the LSQ `groupMulti=0` configuration and preserves per-group ordering.
  bool allowMultipleActiveGroups = true;
  unsigned loadQueueDepth = 16;
  unsigned storeQueueDepth = 16;

  bool running = false;
  bool memStartReadyReg = true;
  bool ctrlEndReadyReg = false;
  bool memEndValidReg = false;
  bool lsqTempGenMem = false;
  std::deque<unsigned> pendingLoadAccesses;
  std::deque<std::pair<unsigned, Data>> pendingMasterLoadData;
  Data lastObservedLoadDataGlobal;
  bool hasLastObservedLoadDataGlobal = false;
  std::vector<Data> lastObservedLoadDataByAccess;
  std::vector<uint8_t> hasLastObservedLoadDataByAccess;

  bool loadInterfaceBusy = false;
  bool storeInterfaceBusy = false;
  uint64_t allocatedLoadEntries = 0;
  uint64_t allocatedStoreEntries = 0;

  void updateSequential();
  void updateCombinational();
};

class EndModel : public OpExecutionModel<handshake::EndOp> {
public:
  using OpExecutionModel<handshake::EndOp>::OpExecutionModel;
  EndModel(handshake::EndOp endOp, mlir::DenseMap<Value, RW *> &subset,
           std::vector<bool> &resValid, const std::vector<bool> &resReady,
           std::vector<Data> &resData);

  void reset() override;

  void exec(bool isClkRisingEdge) override;

  void printStates() override;

private:
  std::vector<ConsumerRW *> ins;
  std::vector<ConsumerData> insData;
  std::vector<bool> &resValid;
  const std::vector<bool> &resReady;
  std::vector<Data> &resData;
};

//===----------------------------------------------------------------------===//
// Arithmetic
//===----------------------------------------------------------------------===//
/// Arithmetic and generic components

class TruncIModel : public OpExecutionModel<handshake::TruncIOp> {
public:
  using OpExecutionModel<handshake::TruncIOp>::OpExecutionModel;
  TruncIModel(handshake::TruncIOp trunciOp,
              mlir::DenseMap<Value, RW *> &subset);

  void reset() override;

  void exec(bool isClkRisingEdge) override;

  void printStates() override;

private:
  // parameters
  unsigned outputWidth;

  // ports
  ChannelConsumerRW *ins;
  ChannelProducerRW *outs;
};

class SelectModel : public OpExecutionModel<handshake::SelectOp> {
public:
  using OpExecutionModel<handshake::SelectOp>::OpExecutionModel;
  SelectModel(handshake::SelectOp selectOp,
              mlir::DenseMap<Value, RW *> &subset);

  void reset() override;

  void exec(bool isClkRisingEdge) override;

  void printStates() override;

private:
  // ports
  ChannelConsumerRW *condition, *trueValue, *falseValue;
  ChannelProducerRW *result;

  bool ee = false, validInternal = false, kill0 = false, kill1 = false,
       antitokenStop = false, g0 = false, g1 = false;
  // internal components
  Antotokens anti;

  void selectExec();
};

using UnaryCompFunc = std::function<Data(const Data &, unsigned)>;

/// Class to represent ExtSI, ExtUI, Negf, Not
template <typename Op>
class GenericUnaryOpModel : public OpExecutionModel<Op> {
public:
  using OpExecutionModel<Op>::OpExecutionModel;
  GenericUnaryOpModel(Op op, mlir::DenseMap<Value, RW *> &subset,
                      const UnaryCompFunc &callback);
  void reset() override;

  void exec(bool isClkRisingEdge) override;

  void printStates() override;

private:
  // parameters
  unsigned outputWidth;
  UnaryCompFunc callback;

  // ports
  ChannelConsumerRW *ins;
  ChannelProducerRW *outs;
};

using BinaryCompFunc = std::function<Data(const Data &, const Data &)>;

/// Mutual component for binary operations: AddF, AddI, AndI, CmpF, CmpI, DivF,
/// DivSI, DivUI, Maximumf, Minimumf, MulF, MulI, OrI, ShlI, ShrSI, ShrUI, SubF,
/// SubI, XorI
template <typename Op>
class GenericBinaryOpModel : public OpExecutionModel<Op> {
public:
  using OpExecutionModel<Op>::OpExecutionModel;
  GenericBinaryOpModel(Op op, mlir::DenseMap<Value, RW *> &subset,
                       const BinaryCompFunc &callback, unsigned latency = 0);

  void reset() override;

  void exec(bool isClkRisingEdge) override;

  void printStates() override;

private:
  // parameters
  BinaryCompFunc callback;
  unsigned latency = 0;
  unsigned bitwidth;

  // ports
  ChannelConsumerRW *lhs, *rhs;
  ChannelProducerRW *result;

  // handshake/pipeline state
  std::vector<ConsumerRW *> insJoin;
  bool joinValid = false;
  bool oehbReady = false;
  bool outputValid = false;
  std::vector<bool> validPipeline;
  std::vector<Data> dataPipeline;
  Data currentCombData;

  // internal components
  JoinSupport binJoin;
};

//===----------------------------------------------------------------------===//
// Simulator
//===----------------------------------------------------------------------===//

class Simulator {
public:
  struct ChannelSwitching {
    std::string src;
    std::string dst;
    unsigned operandIndex = 0;
    bool isToEnd = false;
    bool hasData = false;
    unsigned dataWidth = 0;

    std::vector<uint8_t> valid;
    std::vector<uint8_t> ready;
    std::vector<uint8_t> transfer;
    std::vector<llvm::APInt> dataWave;

    uint64_t validToggles = 0;
    uint64_t readyToggles = 0;
    uint64_t transferToggles = 0;
    uint64_t dataWordToggles = 0;
    uint64_t dataBitToggles = 0;
    uint64_t transferDataWordToggles = 0;
    uint64_t transferDataBitToggles = 0;
    uint64_t transfers = 0;
  };

  struct ResultRecord {
    std::string name;
    std::string type;
    std::string value;
    bool observed = false;
  };

  struct MemoryRecord {
    std::string name;
    std::string type;
    std::vector<std::string> values;
    std::vector<uint8_t> writtenMask;
    uint64_t writtenCount = 0;
  };

  struct VCDMismatch {
    std::string src;
    std::string dst;
    unsigned operandIndex = 0;
    std::string signal;
    unsigned cycle = 0;
    uint64_t simulated = 0;
    uint64_t vcd = 0;
  };

  struct VCDCheck {
    bool enabled = false;
    bool passed = false;
    std::string message;
    unsigned comparedCycles = 0;
    unsigned firstMismatchCycle = 0;
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
    std::vector<VCDMismatch> mismatches;
  };

  Simulator(handshake::FuncOp funcOp, const SimulatorOptions &options = {});

  void reset();

  /// Legacy positional-input behavior (plain scalar channels only).
  void simulate(llvm::ArrayRef<std::string> inputArgs);

  /// Full simulation entrypoint.
  LogicalResult simulate(const SimulationInputs &inputs);

  LogicalResult loadInputs(StringRef inputVectorsDir, StringRef inputArgsFile,
                           HSInputFormat format, SimulationInputs &inputs);

  LogicalResult dumpSwitchingJSON(StringRef filepath) const;

  LogicalResult dumpWaveJSON(StringRef filepath) const;

  LogicalResult dumpResultsJSON(StringRef filepath) const;

  /// Dump node-level switching in switching-estimation CSV format:
  ///   node, data, valid, ready
  LogicalResult dumpSwitchingEstimationCSV(StringRef filepath) const;

  LogicalResult verifyVCDExact(StringRef topVerilogPath, StringRef vcdPath);

  unsigned getCyclesExecuted() const { return iterNum; }

  bool hasFailed() const { return failedFlag; }

  StringRef getFailureMessage() const { return failureMessage; }

  LogicalResult signalFailure(StringRef message);

  const std::vector<ChannelSwitching> &getSwitching() const {
    return channelWaves;
  }

  const std::vector<ResultRecord> &getResultRecords() const { return results; }

  const std::vector<MemoryRecord> &getMemoryRecords() const { return memories; }

  const VCDCheck &getVCDCheck() const { return vcdCheck; }

  const SimulatorOptions &getOptions() const { return options; }

  LogicalResult readMemory(Value memref, const Data &addr, Data &outData,
                           Location loc);

  LogicalResult writeMemory(Value memref, const Data &addr, const Data &inData,
                            Location loc);

  // Just a temporary function to print the results of the simulation to
  // standart output
  void printResults();

  // A temporary function
  void printModelStates();

  ~Simulator();

private:
  struct MemoryImage {
    Value arg;
    MemRefType type;
    std::string name;
    std::vector<Data> values;
    std::vector<uint8_t> writtenMask;
  };

  struct InputDriver {
    Value arg;
    unsigned argIndex = 0;
    bool isChannel = false;
    bool isControl = false;
    bool holdHighControl = false;
    std::deque<Data> channelTokens;
    uint64_t controlTokens = 0;
  };

  struct EndObserver {
    ConsumerRW *in = nullptr;
    ConsumerData inData = ConsumerData(nullptr);
    bool seen = false;
  };

  struct EdgeTrace {
    std::string src;
    std::string dst;
    unsigned operandIndex = 0;
    bool isToEnd = false;
    ConsumerRW *edge = nullptr;
    ConsumerData edgeData = ConsumerData(nullptr);
    bool hasData = false;
    unsigned dataWidth = 0;
    bool captureDataWave = false;
    std::vector<uint8_t> valid;
    std::vector<uint8_t> ready;
    std::vector<uint8_t> transfer;
    std::vector<llvm::APInt> dataWave;
    uint64_t dataWordToggles = 0;
    uint64_t dataBitToggles = 0;
    uint64_t transferDataWordToggles = 0;
    uint64_t transferDataBitToggles = 0;
    std::optional<llvm::APInt> prevData;
    std::optional<llvm::APInt> prevTransferData;
  };

  handshake::FuncOp funcOp;
  SimulatorOptions options;

  std::vector<bool> resValid;
  std::vector<bool> resReady;
  std::vector<Data> resData;
  handshake::EndOp endOp;
  unsigned iterNum = 0;
  bool failedFlag = false;
  std::string failureMessage;

  mlir::DenseMap<Operation *, ExecutionModel *> opModels;
  mlir::DenseMap<Value, ValueState *> oldValuesStates;
  mlir::DenseMap<Value, ValueState *> newValuesStates;
  mlir::DenseMap<Value, Updater *> updaters;

  mlir::DenseMap<OpOperand *, ConsumerRW *> consumerViews;
  mlir::DenseMap<Value, ProducerRW *> producerViews;

  std::vector<EndObserver> endObservers;
  std::vector<EdgeTrace> edgeTraces;
  std::vector<ChannelSwitching> channelWaves;
  std::vector<ResultRecord> results;
  std::vector<MemoryRecord> memories;
  VCDCheck vcdCheck;

  std::vector<Value> argOrder;
  std::vector<std::string> argNames;
  std::vector<std::string> resNames;
  std::unordered_map<unsigned, InputDriver> activeDrivers;
  std::unordered_map<std::string, unsigned> argNameToIndex;
  mlir::DenseMap<Value, MemoryImage> memoryImages;

  void clearFailure();
  LogicalResult fail(StringRef message);

  void initializeMetadata();

  void initializeEdgeCatalog();

  bool sampleEdgeStates();

  void finalizeSwitchingStats();

  LogicalResult buildDriversForTransaction(const SimulationInputs &inputs,
                                           unsigned transactionIdx);

  LogicalResult initializeMemoriesForTransaction(const SimulationInputs &inputs,
                                                 unsigned transactionIdx);

  void driveInputsForCycle();

  void consumeAcceptedInputs();

  bool allEndResultsSeen() const;

  void resetEndObservers();

  LogicalResult parseInputToken(Type type, StringRef token, Data &out) const;

  std::string dataToString(Type type, const Data &data) const;

  uint64_t getPendingInputTokenCount() const;

  uint64_t getPendingMemoryOpCount() const;

  std::string gatherDeadlockDiagnostics() const;

  LogicalResult parseDatInputDir(StringRef inputVectorsDir,
                                 SimulationInputs &inputs);

  LogicalResult parsePlainInputFile(StringRef inputArgsFile,
                                    SimulationInputs &inputs);

  LogicalResult finalizeInputTransactions(SimulationInputs &inputs);

  bool getArgIndexByName(StringRef argName, unsigned &argIdx) const;

  std::string getValueProducerName(Value val) const;

  std::string getOperationName(Operation *op) const;

  LogicalResult emitJSON(StringRef filepath, const llvm::json::Value &value) const;

  template <typename Model, typename Op, typename... Args>
  void registerModel(Op op, Args &&...modelArgs);

  void associateModel(Operation *op);

  template <typename State, typename Updater, typename Producer,
            typename Consumer, typename Ty>
  void registerState(Value val, Operation *producerOp, Ty type);

  void associateState(Value val, Operation *producerOp, Location loc);
};

} // namespace experimental
} // namespace dynamatic

#endif
