#include "experimental/Support/HandshakeSimulator.h"
#include "dynamatic/Support/JSON/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include <filesystem>
#include <limits>

using namespace llvm;
using namespace mlir;
using namespace dynamatic;
using namespace dynamatic::experimental;

FIFOBufferModel::FIFOBufferModel(handshake::BufferOp bufferOp,
                                 mlir::DenseMap<Value, RW *> &subset)
    : OpExecutionModel<handshake::BufferOp>(bufferOp),
      ins(getState<ConsumerRW>(bufferOp.getOperand(), subset)),
      outs(getState<ProducerRW>(bufferOp.getResult(), subset)), insData(ins),
      outsData(outs), capacity(static_cast<unsigned>(bufferOp.getNumSlots())),
      // `FIFO_BREAK_NONE` is the only FIFO family with transparent bypass.
      // `FIFO_BREAK_DV`/`SHIFT_REG_BREAK_DV`/`ONE_SLOT_BREAK_DVR` are
      // non-bypass storage elements.
      bypass(bufferOp.getBufferType() == handshake::BufferType::FIFO_BREAK_NONE),
      hasData(insData.hasValue()) {}

void FIFOBufferModel::reset() {
  occupancy = 0;
  payloads.clear();
  readEnable = false;
  writeEnable = false;
  updateCombinational();
}

void FIFOBufferModel::updateCombinational() {
  const bool fifoValid = occupancy > 0;
  const bool fifoReady = (occupancy < capacity) || outs->ready;

  if (bypass) {
    // tfifo:
    // outs_valid = ins_valid || fifo_valid
    // ins_ready  = fifo_ready || outs_ready
    // fifo_pvalid = ins_valid && (!outs_ready || fifo_valid)
    outs->valid = ins->valid || fifoValid;
    ins->ready = fifoReady || outs->ready;

    if (hasData && outsData.data) {
      if (fifoValid) {
        *outsData.data = payloads.front();
      } else if (insData.hasValue()) {
        outsData = insData;
      }
    }

    readEnable = outs->ready && fifoValid;
    writeEnable = (ins->valid && (!outs->ready || fifoValid)) && fifoReady;
    return;
  }

  // elastic_fifo_inner (no transparent bypass path).
  outs->valid = fifoValid;
  ins->ready = fifoReady;

  if (hasData && outsData.data && fifoValid)
    *outsData.data = payloads.front();

  readEnable = outs->ready && fifoValid;
  writeEnable = ins->valid && fifoReady;
}

void FIFOBufferModel::updateSequential() {
  if (readEnable && occupancy > 0) {
    --occupancy;
    if (hasData)
      payloads.pop_front();
  }

  if (writeEnable) {
    ++occupancy;
    if (hasData && insData.hasValue())
      payloads.push_back(*insData.data);
  }
}

void FIFOBufferModel::exec(bool isClkRisingEdge) {
  updateCombinational();
  if (isClkRisingEdge)
    updateSequential();
}

void FIFOBufferModel::printStates() {
  llvm::outs() << "occupancy=" << occupancy << " capacity=" << capacity
               << " bypass=" << bypass << "\n";
  printValue<ConsumerRW, const Data>("ins", ins, insData.data);
  printValue<ProducerRW, Data>("outs", outs, outsData.data);
}

LoadModel::LoadModel(handshake::LoadOp loadOp, mlir::DenseMap<Value, RW *> &subset)
    : OpExecutionModel<handshake::LoadOp>(loadOp),
      addrIn(getState<ChannelConsumerRW>(loadOp.getAddress(), subset)),
      dataFromMem(getState<ChannelConsumerRW>(loadOp.getData(), subset)),
      addrOut(getState<ChannelProducerRW>(loadOp.getAddressResult(), subset)),
      dataOut(getState<ChannelProducerRW>(loadOp.getDataResult(), subset)),
      addrTEHB(addrOut->data.bitwidth), dataTEHB(dataOut->data.bitwidth) {}

void LoadModel::reset() {
  addrTEHB.reset(addrIn, addrOut, &addrIn->data, &addrOut->data);
  dataTEHB.reset(dataFromMem, dataOut, &dataFromMem->data, &dataOut->data);
}

void LoadModel::exec(bool isClkRisingEdge) {
  addrTEHB.exec(isClkRisingEdge, addrIn, addrOut, &addrIn->data, &addrOut->data);
  dataTEHB.exec(isClkRisingEdge, dataFromMem, dataOut, &dataFromMem->data,
                &dataOut->data);
}

void LoadModel::printStates() {
  printValue<ChannelConsumerRW, const Data>("addrIn", addrIn, &addrIn->data);
  printValue<ChannelProducerRW, Data>("addrOut", addrOut, &addrOut->data);
  printValue<ChannelConsumerRW, const Data>("dataFromMem", dataFromMem,
                                            &dataFromMem->data);
  printValue<ChannelProducerRW, Data>("dataOut", dataOut, &dataOut->data);
}

StoreModel::StoreModel(handshake::StoreOp storeOp,
                       mlir::DenseMap<Value, RW *> &subset)
    : OpExecutionModel<handshake::StoreOp>(storeOp),
      addrIn(getState<ChannelConsumerRW>(storeOp.getAddress(), subset)),
      dataIn(getState<ChannelConsumerRW>(storeOp.getData(), subset)),
      addrOut(getState<ChannelProducerRW>(storeOp.getAddressResult(), subset)),
      dataOut(getState<ChannelProducerRW>(storeOp.getDataResult(), subset)) {}

void StoreModel::reset() {
  addrOut->valid = addrIn->valid;
  addrOut->data = addrIn->data;
  addrIn->ready = addrOut->ready;

  dataOut->valid = dataIn->valid;
  dataOut->data = dataIn->data;
  dataIn->ready = dataOut->ready;
}

void StoreModel::exec(bool isClkRisingEdge) { reset(); }

void StoreModel::printStates() {
  printValue<ChannelConsumerRW, const Data>("addrIn", addrIn, &addrIn->data);
  printValue<ChannelConsumerRW, const Data>("dataIn", dataIn, &dataIn->data);
  printValue<ChannelProducerRW, Data>("addrOut", addrOut, &addrOut->data);
  printValue<ChannelProducerRW, Data>("dataOut", dataOut, &dataOut->data);
}

MemoryControllerModel::MemoryControllerModel(
    handshake::MemoryControllerOp mcOp, mlir::DenseMap<Value, RW *> &subset,
    Simulator &sim)
    : OpExecutionModel<handshake::MemoryControllerOp>(mcOp), mcOp(mcOp),
      sim(sim), memref(mcOp.getMemRef()),
      memStart(getState<ControlConsumerRW>(mcOp.getMemStart(), subset)),
      ctrlEnd(getState<ControlConsumerRW>(mcOp.getCtrlEnd(), subset)),
      memEnd(getState<ControlProducerRW>(mcOp.getMemEnd(), subset)) {
  MCPorts ports = mcOp.getPorts();

  for (MCBlock block : ports.getBlocks()) {
    if (block->ctrlPort) {
      unsigned idx = block->ctrlPort->getCtrlInputIndex();
      ctrlInputs.push_back(
          getState<ChannelConsumerRW>(mcOp.getOperand(idx), subset));
    }

    for (const MemoryPort &port : block->accessPorts) {
      if (auto load = dyn_cast<dynamatic::LoadPort>(port)) {
        loadPorts.push_back({
            getState<ChannelConsumerRW>(mcOp.getOperand(load->getAddrInputIndex()),
                                        subset),
            getState<ChannelProducerRW>(
                mcOp.getResult(load->getDataOutputIndex()), subset),
        });
      } else if (auto store = dyn_cast<dynamatic::StorePort>(port)) {
        storePorts.push_back({
            getState<ChannelConsumerRW>(
                mcOp.getOperand(store->getAddrInputIndex()), subset),
            getState<ChannelConsumerRW>(
                mcOp.getOperand(store->getDataInputIndex()), subset),
        });
      }
    }
  }

  if (ports.connectsToLSQ()) {
    LSQLoadStorePort lsqPort = ports.getLSQPort();
    loadPorts.push_back({
        getState<ChannelConsumerRW>(
            mcOp.getOperand(lsqPort.getLoadAddrInputIndex()), subset),
        getState<ChannelProducerRW>(
            mcOp.getResult(lsqPort.getLoadDataOutputIndex()), subset),
    });
    storePorts.push_back({
        getState<ChannelConsumerRW>(
            mcOp.getOperand(lsqPort.getStoreAddrInputIndex()), subset),
        getState<ChannelConsumerRW>(
            mcOp.getOperand(lsqPort.getStoreDataInputIndex()), subset),
    });
  }
  loadValidRegs.assign(loadPorts.size(), 0);
  loadDataRegs.resize(loadPorts.size());
  for (auto [idx, loadPort] : llvm::enumerate(loadPorts))
    loadDataRegs[idx] = loadPort.dataOut->data;
}

void MemoryControllerModel::reset() {
  running = false;
  noMoreRequests = false;
  pendingStores = 0;
  pendingLoads.clear();
  std::fill(loadValidRegs.begin(), loadValidRegs.end(), 0);
  for (auto [idx, loadPort] : llvm::enumerate(loadPorts))
    loadDataRegs[idx] = loadPort.dataOut->data;
  selectedLoadPort.reset();
  selectedStorePort.reset();
  selectedStoreAddr.reset();
  selectedStoreData.reset();
  updateCombinational();
}

void MemoryControllerModel::exec(bool isClkRisingEdge) {
  updateCombinational();
  if (isClkRisingEdge)
    updateSequential();
}

void MemoryControllerModel::printStates() {
  uint64_t pendingReadResponses = 0;
  for (uint8_t valid : loadValidRegs)
    pendingReadResponses += valid ? 1 : 0;
  llvm::outs() << "MC running=" << running << " noMoreRequests="
               << noMoreRequests << " pendingStores=" << pendingStores
               << " pendingLoads=" << pendingLoads.size()
               << " pendingReadResponses=" << pendingReadResponses << "\n";
}

uint64_t MemoryControllerModel::getPendingCount() const {
  uint64_t pending = pendingStores + pendingLoads.size();
  for (uint8_t valid : loadValidRegs)
    pending += valid ? 1 : 0;
  return pending;
}

void MemoryControllerModel::updateCombinational() {
  selectedLoadPort.reset();
  selectedStorePort.reset();
  selectedStoreAddr.reset();
  selectedStoreData.reset();

  memStart->ready = !running;
  ctrlEnd->ready = !noMoreRequests;

  bool anyCtrlValid = false;
  for (ChannelConsumerRW *ctrlInput : ctrlInputs) {
    ctrlInput->ready = true;
    anyCtrlValid = anyCtrlValid || ctrlInput->valid;
  }

  for (auto [idx, loadPort] : llvm::enumerate(loadPorts)) {
    loadPort.addrIn->ready = false;
    if (loadValidRegs[idx]) {
      loadPort.dataOut->valid = true;
      loadPort.dataOut->data = loadDataRegs[idx];
    } else {
      loadPort.dataOut->valid = false;
    }
  }

  for (MCPortStore &storePort : storePorts) {
    storePort.addrIn->ready = false;
    storePort.dataIn->ready = false;
  }

  // Lowest-index arbitration, following RTL implementation.
  for (auto [idx, loadPort] : llvm::enumerate(loadPorts)) {
    // read_memory_arbiter gates address acceptance by both address validity and
    // corresponding load-data consumer readiness (nReady).
    if (loadPort.addrIn->valid && loadPort.dataOut->ready) {
      loadPort.addrIn->ready = true;
      selectedLoadPort = idx;
      break;
    }
  }

  for (auto [idx, storePort] : llvm::enumerate(storePorts)) {
    if (storePort.addrIn->valid && storePort.dataIn->valid) {
      storePort.addrIn->ready = true;
      storePort.dataIn->ready = true;
      selectedStorePort = idx;
      selectedStoreAddr = storePort.addrIn->data;
      selectedStoreData = storePort.dataIn->data;
      break;
    }
  }

  bool allLoadResponsesDrained =
      llvm::all_of(loadValidRegs, [](uint8_t valid) { return valid == 0; });
  memEnd->valid = running && noMoreRequests && !anyCtrlValid &&
                  pendingStores == 0 && pendingLoads.empty() &&
                  allLoadResponsesDrained;
}

void MemoryControllerModel::updateSequential() {
  if (memStart->valid && memStart->ready)
    running = true;

  bool functionReturn = memEnd->valid && memEnd->ready;
  if (!functionReturn && ctrlEnd->valid && ctrlEnd->ready)
    noMoreRequests = true;

  for (ChannelConsumerRW *ctrlInput : ctrlInputs) {
    if (!ctrlInput->valid || !ctrlInput->ready)
      continue;

    uint64_t storesToAdd = dataCast<APInt>(ctrlInput->data).getLimitedValue();
    pendingStores += storesToAdd;
  }

  if (selectedLoadPort) {
    Data loadedData;
    if (succeeded(
            sim.readMemory(memref, loadPorts[*selectedLoadPort].addrIn->data,
                           loadedData, op->getLoc()))) {
      pendingLoads.push_back({*selectedLoadPort, loadedData, 1});
    }
  }

  for (PendingLoad &pending : pendingLoads) {
    if (pending.remainingCycles > 0)
      --pending.remainingCycles;
  }
  while (!pendingLoads.empty() && pendingLoads.front().remainingCycles == 0) {
    PendingLoad pending = pendingLoads.front();
    pendingLoads.pop_front();
    if (pending.portIdx < loadDataRegs.size())
      loadDataRegs[pending.portIdx] = pending.data;
  }

  // read_data_signals valid behavior:
  // - selected port valid is set.
  // - otherwise valid clears only when the corresponding nReady is high.
  for (auto [idx, loadPort] : llvm::enumerate(loadPorts)) {
    bool selected = selectedLoadPort && *selectedLoadPort == idx;
    if (selected) {
      loadValidRegs[idx] = 1;
    } else if (loadPort.dataOut->ready) {
      loadValidRegs[idx] = 0;
    }
  }

  if (selectedStorePort && selectedStoreAddr && selectedStoreData) {
    (void)sim.writeMemory(memref, *selectedStoreAddr, *selectedStoreData,
                          op->getLoc());
    if (pendingStores > 0)
      --pendingStores;
  }

  if (functionReturn) {
    running = false;
    noMoreRequests = false;
  }
}

namespace {
struct ParsedLSQConfig {
  std::vector<std::vector<unsigned>> ldOrder;
  unsigned groupMulti = 0;
  unsigned fifoDepth = 16;
  unsigned fifoDepthL = 16;
  unsigned fifoDepthS = 16;
};

static bool parseUnsignedMatrix(const llvm::json::Object &object, StringRef key,
                                std::vector<std::vector<unsigned>> &out) {
  const llvm::json::Array *rows = object.getArray(key);
  if (!rows)
    return false;

  out.clear();
  for (const llvm::json::Value &rowVal : *rows) {
    const llvm::json::Array *row = rowVal.getAsArray();
    if (!row)
      return false;
    out.emplace_back();
    for (const llvm::json::Value &elem : *row) {
      std::optional<uint64_t> number = elem.getAsUINT64();
      if (!number)
        return false;
      out.back().push_back(static_cast<unsigned>(*number));
    }
  }
  return true;
}

static bool parseUnsignedScalar(const llvm::json::Object &object, StringRef key,
                                unsigned &out) {
  std::optional<int64_t> number = object.getInteger(key);
  if (!number || *number < 0 ||
      *number > static_cast<int64_t>(std::numeric_limits<unsigned>::max()))
    return false;
  out = static_cast<unsigned>(*number);
  return true;
}

static std::optional<ParsedLSQConfig> loadLSQConfig(StringRef dirPath,
                                                    StringRef lsqName) {
  SmallString<256> jsonPath(dirPath);
  sys::path::append(jsonPath, "handshake_lsq_" + lsqName.str() + ".json");
  auto fileOrErr = MemoryBuffer::getFile(jsonPath);
  if (!fileOrErr)
    return std::nullopt;

  auto parsed = llvm::json::parse((*fileOrErr)->getBuffer());
  if (!parsed)
    return std::nullopt;

  const llvm::json::Object *obj = parsed->getAsObject();
  if (!obj)
    return std::nullopt;

  ParsedLSQConfig config;
  if (!parseUnsignedMatrix(*obj, "ldOrder", config.ldOrder))
    return std::nullopt;
  if (!parseUnsignedScalar(*obj, "groupMulti", config.groupMulti))
    return std::nullopt;
  (void)parseUnsignedScalar(*obj, "fifoDepth", config.fifoDepth);
  if (!parseUnsignedScalar(*obj, "fifoDepth_L", config.fifoDepthL))
    config.fifoDepthL = config.fifoDepth;
  if (!parseUnsignedScalar(*obj, "fifoDepth_S", config.fifoDepthS))
    config.fifoDepthS = config.fifoDepth;
  return config;
}
} // namespace

LSQModel::LSQModel(handshake::LSQOp lsqOp, mlir::DenseMap<Value, RW *> &subset,
                   Simulator &sim)
    : OpExecutionModel<handshake::LSQOp>(lsqOp), lsqOp(lsqOp), sim(sim),
      isMaster(!lsqOp.isConnectedToMC()) {
  if (isMaster) {
    memref = lsqOp.getInputs().front();
    memStart = getState<ControlConsumerRW>(lsqOp.getInputs()[1], subset);
    ctrlEnd = getState<ControlConsumerRW>(lsqOp.getInputs().back(), subset);
    memEnd = getState<ControlProducerRW>(lsqOp.getResults().back(), subset);
  }

  std::vector<std::vector<unsigned>> ldOrder;
  if (!sim.getOptions().lsqConfigDir.empty()) {
    StringRef lsqName =
        cast<StringAttr>(lsqOp->getAttr("handshake.name")).getValue();
    auto config = loadLSQConfig(sim.getOptions().lsqConfigDir, lsqName);
    if (!config) {
      (void)sim.signalFailure(
          ("Missing or malformed LSQ config for " + lsqName).str());
    } else {
      ldOrder = std::move(config->ldOrder);
      allowMultipleActiveGroups = config->groupMulti != 0;
      loadQueueDepth = std::max(1u, config->fifoDepthL);
      storeQueueDepth = std::max(1u, config->fifoDepthS);
    }
  } else {
    (void)sim.signalFailure(
        "LSQ found in design but --lsq-config-dir was not provided");
  }

  LSQPorts ports = lsqOp.getPorts();
  for (LSQGroup group : ports.getGroups()) {
    GroupState groupState;
    if (group->ctrlPort) {
      groupState.ctrl = getState<ControlConsumerRW>(
          lsqOp.getOperand(group->ctrlPort->getCtrlInputIndex()), subset);
    }

    unsigned storesSeen = 0;
    unsigned loadsSeen = 0;
    for (auto [groupAccessIdx, accessPort] :
         llvm::enumerate(group->accessPorts)) {
      Access access;
      access.groupIdx = group.groupID;
      access.groupAccessIndex = groupAccessIdx;
      access.requiredStoresBefore = storesSeen;

      if (auto load = dyn_cast<LoadPort>(accessPort)) {
        access.kind = AccessKind::Load;
        access.addrIn = getState<ChannelConsumerRW>(
            lsqOp.getOperand(load->getAddrInputIndex()), subset);
        access.dataOut = getState<ChannelProducerRW>(
            lsqOp.getResult(load->getDataOutputIndex()), subset);
        if (group.groupID < ldOrder.size() && loadsSeen < ldOrder[group.groupID].size())
          access.requiredStoresBefore = ldOrder[group.groupID][loadsSeen];
        ++groupState.numLoads;
        ++loadsSeen;
      } else if (auto store = dyn_cast<StorePort>(accessPort)) {
        access.kind = AccessKind::Store;
        access.addrIn = getState<ChannelConsumerRW>(
            lsqOp.getOperand(store->getAddrInputIndex()), subset);
        access.dataIn = getState<ChannelConsumerRW>(
            lsqOp.getOperand(store->getDataInputIndex()), subset);
        ++groupState.numStores;
        ++storesSeen;
      } else {
        continue;
      }

      groupState.accesses.push_back(accesses.size());
      accesses.push_back(access);
    }
    groups.push_back(groupState);
  }

  pendingAddrByAccess.resize(accesses.size());
  pendingStoreDataByAccess.resize(accesses.size());
  loadResponsesByAccess.resize(accesses.size());
  addrSlotsByAccess.assign(accesses.size(), 0);
  storeDataSlotsByAccess.assign(accesses.size(), 0);
  lastObservedLoadDataByAccess.resize(accesses.size());
  hasLastObservedLoadDataByAccess.assign(accesses.size(), 0);

  if (ports.connectsToMC()) {
    MCLoadStorePort mcPort = ports.getMCPort();
    ldDataFromMC = getState<ChannelConsumerRW>(
        lsqOp.getOperand(mcPort.getLoadDataInputIndex()), subset);
    ldAddrToMC = getState<ChannelProducerRW>(
        lsqOp.getResult(mcPort.getLoadAddrOutputIndex()), subset);
    stAddrToMC = getState<ChannelProducerRW>(
        lsqOp.getResult(mcPort.getStoreAddrOutputIndex()), subset);
    stDataToMC = getState<ChannelProducerRW>(
        lsqOp.getResult(mcPort.getStoreDataOutputIndex()), subset);
  }

  for (auto [idx, access] : llvm::enumerate(accesses)) {
    if (access.dataOut) {
      lastObservedLoadDataByAccess[idx] = access.dataOut->data;
      hasLastObservedLoadDataByAccess[idx] = 1;
      if (!hasLastObservedLoadDataGlobal) {
        lastObservedLoadDataGlobal = access.dataOut->data;
        hasLastObservedLoadDataGlobal = true;
      }
    }
  }
}

void LSQModel::reset() {
  running = false;
  memStartReadyReg = true;
  ctrlEndReadyReg = false;
  memEndValidReg = false;
  lsqTempGenMem = false;
  allocatedLoadEntries = 0;
  allocatedStoreEntries = 0;
  pendingLoadAccesses.clear();
  pendingMasterLoads.clear();
  pendingMasterLoadData.clear();
  hasLastObservedLoadDataGlobal = false;
  std::fill(hasLastObservedLoadDataByAccess.begin(),
            hasLastObservedLoadDataByAccess.end(), 0);
  loadInterfaceBusy = false;
  storeInterfaceBusy = false;
  selectedLoadAccess.reset();
  selectedStoreAccess.reset();
  std::fill(addrSlotsByAccess.begin(), addrSlotsByAccess.end(), 0);
  std::fill(storeDataSlotsByAccess.begin(), storeDataSlotsByAccess.end(), 0);
  for (GroupState &group : groups) {
    group.frames.clear();
  }
  for (auto &queue : pendingAddrByAccess)
    queue.clear();
  for (auto &queue : pendingStoreDataByAccess)
    queue.clear();
  for (auto &queue : loadResponsesByAccess)
    queue.clear();
  for (auto [idx, access] : llvm::enumerate(accesses)) {
    if (access.dataOut) {
      lastObservedLoadDataByAccess[idx] = access.dataOut->data;
      hasLastObservedLoadDataByAccess[idx] = 1;
      if (!hasLastObservedLoadDataGlobal) {
        lastObservedLoadDataGlobal = access.dataOut->data;
        hasLastObservedLoadDataGlobal = true;
      }
    }
  }
  while (!pendingMasterLoadData.empty()) {
    auto [accessIdx, data] = pendingMasterLoadData.front();
    pendingMasterLoadData.pop_front();
    if (accessIdx < loadResponsesByAccess.size())
      loadResponsesByAccess[accessIdx].push_back(data);
  }
  updateCombinational();
}

void LSQModel::exec(bool isClkRisingEdge) {
  updateCombinational();
  if (isClkRisingEdge)
    updateSequential();
}

void LSQModel::printStates() {
  uint64_t pendingAddr = 0;
  uint64_t pendingStoreData = 0;
  uint64_t pendingFrames = 0;
  uint64_t addrSlots = 0;
  uint64_t dataSlots = 0;
  for (const auto &queue : pendingAddrByAccess)
    pendingAddr += queue.size();
  for (const auto &queue : pendingStoreDataByAccess)
    pendingStoreData += queue.size();
  for (uint64_t slots : addrSlotsByAccess)
    addrSlots += slots;
  for (uint64_t slots : storeDataSlotsByAccess)
    dataSlots += slots;
  for (const GroupState &group : groups)
    pendingFrames += group.frames.size();

  llvm::outs() << "LSQ running=" << running
               << " pendingReq=" << pendingLoadAccesses.size()
               << " pendingMaster=" << pendingMasterLoads.size()
               << " pendingFrames=" << pendingFrames
               << " pendingAddr=" << pendingAddr
               << " pendingStoreData=" << pendingStoreData
               << " addrSlots=" << addrSlots
               << " dataSlots=" << dataSlots
               << " allocLd=" << allocatedLoadEntries
               << " allocSt=" << allocatedStoreEntries
               << " memStartReady=" << memStartReadyReg
               << " ctrlEndReady=" << ctrlEndReadyReg
               << " memEndValid=" << memEndValidReg
               << " allowMultiGroups=" << allowMultipleActiveGroups << "\n";
}

uint64_t LSQModel::getPendingCount() const {
  uint64_t pending = pendingLoadAccesses.size() + pendingMasterLoads.size() +
                     allocatedLoadEntries + allocatedStoreEntries;
  for (const auto &queue : pendingAddrByAccess)
    pending += queue.size();
  for (const auto &queue : pendingStoreDataByAccess)
    pending += queue.size();
  for (const GroupState &group : groups)
    pending += group.frames.size();
  for (const auto &queue : loadResponsesByAccess)
    pending += queue.size();
  return pending;
}

void LSQModel::updateCombinational() {
  selectedLoadAccess.reset();
  selectedStoreAccess.reset();

  if (memStart)
    memStart->ready = memStartReadyReg;
  if (ctrlEnd)
    ctrlEnd->ready = ctrlEndReadyReg;

  if (ldAddrToMC)
    ldAddrToMC->valid = false;
  if (stAddrToMC)
    stAddrToMC->valid = false;
  if (stDataToMC)
    stDataToMC->valid = false;
  if (ldDataFromMC)
    ldDataFromMC->ready = false;

  auto firstActiveGroup = [&]() -> std::optional<unsigned> {
    for (auto [groupIdx, group] : llvm::enumerate(groups)) {
      if (!group.frames.empty()) {
        return groupIdx;
      }
    }
    return std::nullopt;
  };
  std::optional<unsigned> activeGroup = firstActiveGroup();
  for (GroupState &group : groups) {
    if (group.ctrl)
      group.ctrl->ready = false;
  }

  for (auto [idx, access] : llvm::enumerate(accesses)) {
    if (access.kind == AccessKind::Load) {
      if (access.addrIn) {
        bool hasAllocatedSlot =
            pendingAddrByAccess[idx].size() < addrSlotsByAccess[idx];
        bool hasQueueSpace = pendingAddrByAccess[idx].size() < loadQueueDepth;
        access.addrIn->ready = hasAllocatedSlot && hasQueueSpace;
      }
    } else {
      if (access.addrIn) {
        bool hasAllocatedSlot =
            pendingAddrByAccess[idx].size() < addrSlotsByAccess[idx];
        bool hasQueueSpace = pendingAddrByAccess[idx].size() < storeQueueDepth;
        access.addrIn->ready = hasAllocatedSlot && hasQueueSpace;
      }
      if (access.dataIn) {
        bool hasAllocatedSlot = pendingStoreDataByAccess[idx].size() <
                                storeDataSlotsByAccess[idx];
        bool hasQueueSpace =
            pendingStoreDataByAccess[idx].size() < storeQueueDepth;
        access.dataIn->ready = hasAllocatedSlot && hasQueueSpace;
      }
    }
    if (access.dataOut) {
      if (!loadResponsesByAccess[idx].empty()) {
        access.dataOut->valid = true;
        access.dataOut->data = loadResponsesByAccess[idx].front();
      } else {
        access.dataOut->valid = false;
        // When downstream backpressures, LSQ output buses in RTL may still
        // reflect shared response-bus activity; otherwise keep a stable
        // per-access hold value.
        if (!access.dataOut->ready && hasLastObservedLoadDataGlobal)
          access.dataOut->data = lastObservedLoadDataGlobal;
        else if (hasLastObservedLoadDataByAccess[idx] != 0)
          access.dataOut->data = lastObservedLoadDataByAccess[idx];
      }
    }
  }

  // Candidate selection follows deterministic priority:
  // lowest group index, then lowest access index within the group.
  std::optional<unsigned> loadCandidate;
  std::optional<unsigned> storeCandidate;
  bool hasSelectedLoadAddr = false;
  bool hasSelectedStorePayload = false;
  auto considerGroup = [&](unsigned groupIdx) {
    GroupState &group = groups[groupIdx];
    if (group.frames.empty())
      return;
    const GroupFrame &frame = group.frames.front();

    for (auto [localIdx, accessIdx] : llvm::enumerate(group.accesses)) {
      if (localIdx < frame.issued.size() && frame.issued[localIdx])
        continue;

      Access &access = accesses[accessIdx];
      if (access.kind == AccessKind::Load) {
        if (loadCandidate)
          continue;
        if (pendingAddrByAccess[accessIdx].empty())
          continue;
        if (frame.acceptedStores < access.requiredStoresBefore)
          continue;
        loadCandidate = accessIdx;
        selectedLoadAddr = pendingAddrByAccess[accessIdx].front();
        hasSelectedLoadAddr = true;
      } else {
        if (storeCandidate)
          continue;
        if (pendingAddrByAccess[accessIdx].empty() ||
            pendingStoreDataByAccess[accessIdx].empty())
          continue;
        storeCandidate = accessIdx;
        selectedStoreAddr = pendingAddrByAccess[accessIdx].front();
        selectedStoreData = pendingStoreDataByAccess[accessIdx].front();
        hasSelectedStorePayload = true;
      }

      if (loadCandidate && storeCandidate)
        return;
    }
  };

  if (!allowMultipleActiveGroups && activeGroup) {
    considerGroup(*activeGroup);
  } else {
    for (auto [groupIdx, group] : llvm::enumerate(groups)) {
      (void)group;
      considerGroup(groupIdx);
      if (loadCandidate && storeCandidate)
        break;
    }
  }

  if (loadCandidate) {
    if (isMaster) {
      selectedLoadAccess = *loadCandidate;
    } else if (ldAddrToMC && hasSelectedLoadAddr) {
      ldAddrToMC->valid = true;
      ldAddrToMC->data = selectedLoadAddr;
      if (ldAddrToMC->ready)
        selectedLoadAccess = *loadCandidate;
    }
  }

  if (storeCandidate) {
    if (isMaster) {
      selectedStoreAccess = *storeCandidate;
    } else if (stAddrToMC && stDataToMC && hasSelectedStorePayload) {
      stAddrToMC->valid = true;
      stAddrToMC->data = selectedStoreAddr;
      stDataToMC->valid = true;
      stDataToMC->data = selectedStoreData;
      if (stAddrToMC->ready && stDataToMC->ready)
        selectedStoreAccess = *storeCandidate;
    }
  }

  // Refine per-access readiness using strict slot accounting:
  // issuing an access consumes one allocated slot, so we do not grant extra
  // slot credit combinationally from same-cycle issue.
  for (auto [idx, access] : llvm::enumerate(accesses)) {
    if (access.addrIn) {
      uint64_t slotCredits = addrSlotsByAccess[idx];
      uint64_t queueDepth =
          (access.kind == AccessKind::Load) ? loadQueueDepth : storeQueueDepth;
      bool hasAllocatedSlot = pendingAddrByAccess[idx].size() < slotCredits;
      bool hasQueueSpace = pendingAddrByAccess[idx].size() < queueDepth;
      access.addrIn->ready = hasAllocatedSlot && hasQueueSpace;
    }

    if (access.kind == AccessKind::Store && access.dataIn) {
      uint64_t dataCredits = storeDataSlotsByAccess[idx];
      bool hasAllocatedSlot =
          pendingStoreDataByAccess[idx].size() < dataCredits;
      bool hasQueueSpace =
          pendingStoreDataByAccess[idx].size() < storeQueueDepth;
      access.dataIn->ready = hasAllocatedSlot && hasQueueSpace;
    }
  }

  // Group-init admission is based on queue occupancy at the start of the
  // cycle. We intentionally do not add same-cycle release credit here so LSQ
  // control acceptance matches queue-state-driven RTL behavior.
  uint64_t availableLoadEntries =
      (allocatedLoadEntries >= loadQueueDepth)
          ? 0
          : (loadQueueDepth - allocatedLoadEntries);
  uint64_t availableStoreEntries =
      (allocatedStoreEntries >= storeQueueDepth)
          ? 0
          : (storeQueueDepth - allocatedStoreEntries);

  for (GroupState &group : groups) {
    if (!group.ctrl)
      continue;
    bool canAccept =
        static_cast<uint64_t>(group.numLoads) <= availableLoadEntries &&
        static_cast<uint64_t>(group.numStores) <= availableStoreEntries;
    group.ctrl->ready = canAccept;
    if (canAccept && group.ctrl->valid) {
      availableLoadEntries -= group.numLoads;
      availableStoreEntries -= group.numStores;
    }
  }

  // Keep LSQ ready for MC load-data responses. Gating this by pendingLoadAccesses
  // creates a circular wait with MC request acceptance (MC load-addr acceptance is
  // conditioned on corresponding data-out readiness).
  if (ldDataFromMC)
    ldDataFromMC->ready = true;

  bool allGroupsIdle = llvm::all_of(groups, [](const GroupState &group) {
    return group.frames.empty();
  });
  bool anyCtrlValid = llvm::any_of(groups, [](const GroupState &group) {
    return group.ctrl && group.ctrl->valid;
  });
  bool anyPendingInput = false;
  for (const auto &queue : pendingAddrByAccess) {
    if (!queue.empty()) {
      anyPendingInput = true;
      break;
    }
  }
  if (!anyPendingInput) {
    for (const auto &queue : pendingStoreDataByAccess) {
      if (!queue.empty()) {
        anyPendingInput = true;
        break;
      }
    }
  }
  bool anyPendingResponses = false;
  for (const auto &queue : loadResponsesByAccess) {
    if (!queue.empty()) {
      anyPendingResponses = true;
      break;
    }
  }

  bool allRequestsDone = allGroupsIdle && pendingLoadAccesses.empty() &&
                         pendingMasterLoads.empty() &&
                         pendingMasterLoadData.empty() && !anyPendingInput &&
                         !anyPendingResponses && allocatedLoadEntries == 0 &&
                         allocatedStoreEntries == 0;

  if (memEnd)
    memEnd->valid = memEndValidReg;

  // RTL `TEMP_GEN_MEM`: ctrlEnd is valid while there are no remaining queue
  // entries, no active requests, and no concurrent new group-init valid.
  lsqTempGenMem = ctrlEnd && ctrlEnd->valid && allRequestsDone && !anyCtrlValid;
}

void LSQModel::updateSequential() {
  // Model the LSQ top-level control interface as in generated RTL:
  // memStart_ready is latched, ctrlEnd_ready is raised only when TEMP_GEN_MEM
  // becomes true, and memEnd_valid latches once completion is detected.
  const bool memStartHs = memStart && memStart->valid && memStartReadyReg;
  const bool functionReturn = memEnd && memEndValidReg && memEnd->ready && running;

  if (memStartHs)
    running = true;
  if (functionReturn)
    running = false;

  if (functionReturn) {
    memStartReadyReg = true;
  } else if (memStartHs) {
    memStartReadyReg = false;
  }

  if (memEndValidReg || lsqTempGenMem)
    memEndValidReg = true;

  if (ctrlEnd) {
    const bool ctrlHandshake = ctrlEnd->valid && ctrlEndReadyReg;
    ctrlEndReadyReg = (!ctrlHandshake) && (lsqTempGenMem || ctrlEndReadyReg);
  }

  for (GroupState &group : groups) {
    if (!group.ctrl)
      continue;
    if (group.ctrl->valid && group.ctrl->ready) {
      GroupFrame frame;
      frame.issuedCount = 0;
      frame.acceptedStores = 0;
      frame.issued.assign(group.accesses.size(), 0);
      group.frames.push_back(std::move(frame));
      allocatedLoadEntries += group.numLoads;
      allocatedStoreEntries += group.numStores;
      for (unsigned accessIdx : group.accesses) {
        ++addrSlotsByAccess[accessIdx];
        if (accesses[accessIdx].kind == AccessKind::Store)
          ++storeDataSlotsByAccess[accessIdx];
      }
    }
  }

  for (auto [idx, access] : llvm::enumerate(accesses)) {
    if (access.addrIn && access.addrIn->valid && access.addrIn->ready)
      pendingAddrByAccess[idx].push_back(access.addrIn->data);
    if (access.dataIn && access.dataIn->valid && access.dataIn->ready)
      pendingStoreDataByAccess[idx].push_back(access.dataIn->data);
  }

  if (selectedStoreAccess) {
    unsigned accessIdx = *selectedStoreAccess;
    Access &access = accesses[accessIdx];
    Data addrPayload;
    Data dataPayload;
    if (!pendingAddrByAccess[accessIdx].empty()) {
      addrPayload = pendingAddrByAccess[accessIdx].front();
      pendingAddrByAccess[accessIdx].pop_front();
    } else {
      addrPayload = selectedStoreAddr;
    }
    if (!pendingStoreDataByAccess[accessIdx].empty()) {
      dataPayload = pendingStoreDataByAccess[accessIdx].front();
      pendingStoreDataByAccess[accessIdx].pop_front();
    } else {
      dataPayload = selectedStoreData;
    }

    if (isMaster) {
      (void)sim.writeMemory(memref, addrPayload, dataPayload, op->getLoc());
    }
    if (addrSlotsByAccess[accessIdx] > 0)
      --addrSlotsByAccess[accessIdx];
    if (storeDataSlotsByAccess[accessIdx] > 0)
      --storeDataSlotsByAccess[accessIdx];
    if (allocatedStoreEntries > 0)
      --allocatedStoreEntries;

    GroupState &group = groups[access.groupIdx];
    if (!group.frames.empty()) {
      GroupFrame &frame = group.frames.front();
      ++frame.acceptedStores;
      if (access.groupAccessIndex < frame.issued.size() &&
          !frame.issued[access.groupAccessIndex]) {
        frame.issued[access.groupAccessIndex] = 1;
        ++frame.issuedCount;
      }
    }
  }

  if (selectedLoadAccess) {
    unsigned accessIdx = *selectedLoadAccess;
    Access &access = accesses[accessIdx];
    Data addrPayload;
    if (!pendingAddrByAccess[accessIdx].empty()) {
      addrPayload = pendingAddrByAccess[accessIdx].front();
      pendingAddrByAccess[accessIdx].pop_front();
    } else {
      addrPayload = selectedLoadAddr;
    }

    if (isMaster) {
      Data loadedData;
      if (succeeded(sim.readMemory(memref, addrPayload, loadedData, op->getLoc())))
        pendingMasterLoads.emplace_back(accessIdx, loadedData, 1);
    } else {
      pendingLoadAccesses.push_back(accessIdx);
    }
    if (addrSlotsByAccess[accessIdx] > 0)
      --addrSlotsByAccess[accessIdx];

    GroupState &group = groups[access.groupIdx];
    if (!group.frames.empty()) {
      GroupFrame &frame = group.frames.front();
      if (access.groupAccessIndex < frame.issued.size() &&
          !frame.issued[access.groupAccessIndex]) {
        frame.issued[access.groupAccessIndex] = 1;
        ++frame.issuedCount;
      }
    }
  }

  for (auto [idx, access] : llvm::enumerate(accesses)) {
    if (!access.dataOut)
      continue;
    if (!loadResponsesByAccess[idx].empty() && access.dataOut->valid &&
        access.dataOut->ready) {
      loadResponsesByAccess[idx].pop_front();
      if (allocatedLoadEntries > 0)
        --allocatedLoadEntries;
    }
  }

  if (ldDataFromMC && ldDataFromMC->valid && ldDataFromMC->ready &&
      !pendingLoadAccesses.empty()) {
    unsigned accessIdx = pendingLoadAccesses.front();
    pendingLoadAccesses.pop_front();
    if (accessIdx < loadResponsesByAccess.size()) {
      loadResponsesByAccess[accessIdx].push_back(ldDataFromMC->data);
      lastObservedLoadDataGlobal = ldDataFromMC->data;
      hasLastObservedLoadDataGlobal = true;
      lastObservedLoadDataByAccess[accessIdx] = ldDataFromMC->data;
      hasLastObservedLoadDataByAccess[accessIdx] = 1;
    }
  }

  for (auto &pending : pendingMasterLoads) {
    if (std::get<2>(pending) > 0)
      --std::get<2>(pending);
  }
  while (!pendingMasterLoads.empty() && std::get<2>(pendingMasterLoads.front()) == 0) {
    auto [accessIdx, data, _] = pendingMasterLoads.front();
    pendingMasterLoads.pop_front();
    pendingMasterLoadData.emplace_back(accessIdx, data);
  }
  while (!pendingMasterLoadData.empty()) {
    auto [accessIdx, data] = pendingMasterLoadData.front();
    pendingMasterLoadData.pop_front();
    if (accessIdx < loadResponsesByAccess.size()) {
      loadResponsesByAccess[accessIdx].push_back(data);
      lastObservedLoadDataGlobal = data;
      hasLastObservedLoadDataGlobal = true;
      lastObservedLoadDataByAccess[accessIdx] = data;
      hasLastObservedLoadDataByAccess[accessIdx] = 1;
    }
  }

  for (GroupState &group : groups) {
    while (!group.frames.empty()) {
      const GroupFrame &frame = group.frames.front();
      if (frame.issuedCount < group.accesses.size())
        break;
      group.frames.pop_front();
    }
  }

  (void)functionReturn;
}

EndModel::EndModel(handshake::EndOp endOp, mlir::DenseMap<Value, RW *> &subset,
                   std::vector<bool> &resValid,
                   const std::vector<bool> &resReady,
                   std::vector<Data> &resData)
    : OpExecutionModel<handshake::EndOp>(endOp),
      resValid(resValid), resReady(resReady), resData(resData) {
  ins.reserve(endOp->getNumOperands());
  insData.reserve(endOp->getNumOperands());

  for (auto [idx, inVal] : llvm::enumerate(endOp.getInputs())) {
    ConsumerRW *in = getState<ConsumerRW>(inVal, subset);
    ins.push_back(in);
    insData.emplace_back(in);
    (void)idx;
  }
}

void EndModel::reset() {
  for (auto [idx, in] : llvm::enumerate(ins)) {
    resValid[idx] = in->valid;
    // The end sink drives static backpressure and must not gate ready with
    // incoming valid, otherwise cyclic paths can deadlock.
    in->ready = resReady[idx];
    if (insData[idx].hasValue())
      resData[idx] = *insData[idx].data;
  }
}

void EndModel::exec(bool isClkRisingEdge) { reset(); }

void EndModel::printStates() {
  for (auto [idx, in] : llvm::enumerate(ins)) {
    printValue<ConsumerRW, const Data>("in", in, insData[idx].data);
    llvm::outs() << "out: " << resValid[idx] << " " << resReady[idx] << "\n";
  }
}
