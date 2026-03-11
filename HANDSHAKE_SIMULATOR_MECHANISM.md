# Handshake Simulator Mechanism (branch `origin/handshake_simulator`)

## Scope and Snapshot

- Branch analyzed: `origin/handshake_simulator`
- Commit: `bfbf585fdf69325b82bda67fdd49c6f41de68b5e`
- Commit date: **February 26, 2026**
- Commit message: `[HS] Initial commit for the in progress handshake simulator`
- This write-up is based on source under:
  - `experimental/include/experimental/Support/HandshakeSimulator.h`
  - `experimental/lib/Support/HandshakeSimulator.cpp`
  - `experimental/lib/Support/HandshakeUnitModels.cpp`
  - `experimental/tools/handshake-simulator/handshake-simulator.cpp`

## 1) Core Simulation Method

The simulator is **cycle/tick based** with a two-phase cycle:

1. **Combinational settle phase**
2. **Clock-edge (rising-edge) sequential update phase**

This is visible in:
- `ExecutionModel::exec(bool isClkRisingEdge)` interface (`HandshakeSimulator.h`, around line 366)
- Main loop in `Simulator::simulate` (`HandshakeSimulator.cpp`, around lines 1977-2041)

### 1.1 State representation and fixed-point settling

The simulator keeps two copies of handshake value state:
- `oldValuesStates`
- `newValuesStates`

with per-value `Updater`s (`HandshakeSimulator.h`, state/updater sections and member maps around lines 1418-1423).

Each cycle, combinational logic is evaluated repeatedly until no state changes:

- run all models with `exec(false)`
- test `updater->check()` for all values
- if not stable, commit updates and iterate

(`HandshakeSimulator.cpp`, lines ~1977-1993)

This is effectively a software "delta-cycle settle" inside one clock cycle.

### 1.2 Per-cycle algorithm

For each transaction (`HandshakeSimulator.cpp`, lines ~1995 onward):

1. `driveInputsForCycle()`
2. `settleCombinational()` (fixed point)
3. `sampleEdgeStates()` (collect valid/ready/transfer/data for switching)
4. capture top-level `end` handshakes
5. `consumeAcceptedInputs()`
6. deadlock bookkeeping (`stallCycles`)
7. rising-edge update: all models `exec(true)` then updater commit
8. completion test: all end results observed and no pending external tokens

Timeout and deadlock handling:
- per-transaction `maxCycles` (default 1,000,000)
- deadlock if no progress for >1024 cycles
- diagnostics include pending token/memory op counts and stalled edges

(`HandshakeSimulator.cpp`, around lines 2007-2036 and 1926-1941)

## 2) Signal/State Access Model

Each SSA value gets:
- a producer view (`ProducerRW`) that can drive `valid` (+ `data` for channel)
- one consumer view (`ConsumerRW`) per use (`OpOperand`) that observes `valid/data` and drives `ready`

(`HandshakeSimulator.h`, RW section and comments; wiring in `registerState` in `HandshakeSimulator.cpp`, lines ~3783-3799)

This gives each operation model an RTL-like port interface.

## 3) Unit Modeling Strategy

Operation-to-model mapping is centralized in `Simulator::associateModel` (`HandshakeSimulator.cpp`, lines ~3538-3778).

### 3.1 Combinational handshake units

These are modeled primarily as combinational equations (`exec` mostly calls `reset`):

- `BranchModel`
- `CondBranchModel`
- `ConstantModel`
- `JoinModel`
- `LazyForkModel`
- `MergeModel` (priority first-valid semantics)
- `MuxModel` (selected input + index handshake)
- `TruncIModel`
- `GenericUnaryOpModel` (e.g., `not`, `ext*`, `negf`)
- `GenericBinaryOpModel` when latency=0

Examples:
- `BranchModel` forwards `valid/ready/data` directly (`HandshakeSimulator.cpp`, ~476-482)
- `MergeModel` grants `ready` only to first valid input (`~761-797`)
- `MuxModel` only selected lane handshakes (`~835-870`)

### 3.2 Buffer family mapping

`handshake::BufferOp` is mapped by buffer type (`HandshakeSimulator.cpp`, ~3575-3593):

- `ONE_SLOT_BREAK_DV` -> `OEHBModel`
- `ONE_SLOT_BREAK_R` -> `TEHBModel`
- FIFO/shift-register families -> `FIFOBufferModel`

`FIFOBufferModel` behavior (`HandshakeUnitModels.cpp`):
- explicit occupancy/payload queue
- supports bypass mode only for `FIFO_BREAK_NONE`
- computes `readEnable/writeEnable` combinationally, updates queue on rising edge

(see ~13-87)

### 3.3 Arithmetic units and latencies

Arithmetic ops are modeled with callbacks over `APInt`/`APFloat`.

- zero-latency integer/logical ops use join + combinational compute
- latency-carrying ops use a pipeline (`validPipeline`, `dataPipeline`) with OEHB-like output validity

(`HandshakeSimulator.cpp`, `GenericBinaryOpModel`, ~1044-1122)

Configured latencies in this branch include:
- `addf`: 9
- `subf`: 9
- `mulf`: 4
- `divf`: 29
- `divsi/divui`: 36
- `maximumf/minimumf`: 2

Note: `CmpFOp` callback is currently a placeholder returning zero (`HandshakeSimulator.cpp`, ~3617-3622).

### 3.4 Memory path models

Implemented in `HandshakeUnitModels.cpp`:

- `LoadModel`: TEHB on address and response channels (~96-121)
- `StoreModel`: pass-through handshakes for addr/data legs (~123-148)
- `MemoryControllerModel`:
  - arbitration among load/store ports (lowest index)
  - read/write through simulator memory image
  - tracks pending stores/loads and output-valid registers
  - memStart/ctrlEnd/memEnd control protocol
  (~150-364)

### 3.5 LSQ model

`LSQModel` supports:
- grouped accesses with per-group frames
- configurable load-order constraints (`ldOrder`)
- `groupMulti` behavior (single active group vs multi-group)
- queue depth limits for load/store queues
- master mode (direct memory) or MC-connected mode (channelized requests)

(`HandshakeUnitModels.cpp`, constructor and update methods ~436-1071)

LSQ requires JSON config from `--lsq-config-dir`, otherwise simulation fails (`~447-464`).

### 3.6 End model

`EndModel` drives sink-like readiness using static `resReady` and records observed outputs (`HandshakeUnitModels.cpp`, ~1073-1102).

## 4) Input, Transactions, and Memory Images

Input handling supports:
- `.dat` directory mode (`input_<arg>.dat`)
- plain text mode
- legacy positional mode for scalar channels

(`HandshakeSimulator.cpp`, `loadInputs`, `parseDatInputDir`, `parsePlainInputFile`, `simulate(ArrayRef<string>)`, around ~1644-1807 and ~2098-2125)

Rules worth noting:
- all provided inputs must share the same number of transactions
- memref args are mandatory and must be statically shaped
- unspecified control args default to one token per transaction
- control tokens can use `hold`/`level1` for level-high drive

(`finalizeInputTransactions` and `buildDriversForTransaction`, around ~1729-1874)

Memory model:
- each memref argument maps to a `MemoryImage`
- simulator reads/writes by index
- tracks per-location write mask

(`initializeMetadata`, `readMemory`, `writeMemory`, around ~1340-1383 and ~1605-1642)

## 5) Switching and Wave Extraction

The simulator tracks every channel edge (src, dst, operand index), then samples per-cycle:
- `valid`
- `ready`
- `transfer = valid & ready`
- optional data wave and data toggles

(`initializeEdgeCatalog`, `sampleEdgeStates`, `finalizeSwitchingStats`, around ~1407-1498)

Outputs:
- switching JSON / wave JSON
- results JSON (outputs + memories + optional VCD check results)
- node-level switching CSV

(`dumpSwitchingJSON`, `dumpResultsJSON`, `dumpSwitchingEstimationCSV`, around ~2139 onward)

## 6) VCD "exact" Verification Flow

`verifyVCDExact(topVerilog, vcd)` compares simulator traces with VCD:

1. parse top-level Verilog instances/ports for candidate wire mapping
2. parse VCD signal events
3. detect clock rising edges
4. sample values **before** each rising edge (pre-edge semantics)
5. map channel valid/ready/data wires to simulator channel traces
6. align simulation/VCD start points (prefer top-level input activity)
7. compare toggles and payload traces; record mismatches

(`HandshakeSimulator.cpp`, ~2392-3480)

This is not just waveform equality; it compares switching/toggle metrics and transfer-aligned payload traces.

## 7) Concrete Cycle Examples

### Example A: FIFO bypass behavior (`FIFO_BREAK_NONE`)

Given:
- occupancy = 0
- `ins.valid = 1`, `outs.ready = 1`
- bypass enabled

Combinational (`FIFOBufferModel::updateCombinational`):
- `outs.valid = ins.valid || fifoValid = 1`
- `ins.ready = fifoReady || outs.ready = 1`
- `outs.data = ins.data` (because FIFO empty)
- `writeEnable = ins.valid && (!outs.ready || fifoValid) && fifoReady = 0`

Clock edge:
- no enqueue (`writeEnable=0`), occupancy remains 0
- token effectively bypasses the queue in same cycle

If `outs.ready=0`, same setup enqueues instead (`writeEnable=1`), occupancy increments on edge.

### Example B: Priority merge with two simultaneous valids

For a 2-input merge:
- `in0.valid=1`, `in1.valid=1`, `outs.ready=1`

`MergeModel::execDataFull/execDataless` picks first valid input:
- `outs.valid=1`
- `in0.ready=1`, `in1.ready=0`
- output data = `in0.data` (dataful case)

So the model is deterministic and priority-based.

### Example C: Latency pipeline (e.g., `AddFOp`, latency 9)

At cycle `t`, if both operands are valid and output can progress (`oehbReady`):
- combinationally compute `currentCombData = lhs + rhs`
- on rising edge, shift data/valid pipelines and inject new values

Output behavior:
- `result.valid` reflects `outputValid`
- data appears after pipeline delay
- backpressure keeps `outputValid` asserted until downstream accepts

(`GenericBinaryOpModel`, ~1076-1122)

### Example D: MC load handshake and response

When a load port has:
- `addrIn.valid=1`
- corresponding `dataOut.ready=1`

`MemoryControllerModel` selects it, then on edge:
- reads memory image at address
- pushes pending load with remainingCycles=1
- decrements countdown to 0 and commits `loadDataRegs`
- sets per-port `loadValidRegs[idx]=1`

Next combinational cycle:
- `dataOut.valid=1`
- `dataOut.data` is returned value

(`HandshakeUnitModels.cpp`, ~275-349)

### Example E: LSQ enforces store-before-load order inside a group

Assume one group has:
- access 0: store
- access 1: load
- `requiredStoresBefore(load)=1` from `ldOrder`

Flow:
- group control token arrives, frame allocated (`GroupFrame` created)
- even if load address is already queued, load is **not** selected until `acceptedStores >= 1`
- once store handshakes, LSQ increments `acceptedStores`
- load becomes eligible and can issue afterward

This comes from candidate gating in `LSQModel::updateCombinational` (`requiredStoresBefore` check around `HandshakeUnitModels.cpp`, ~733-739 and store accounting ~972-980).

## 8) CLI and Execution Surface

Tool: `experimental/tools/handshake-simulator/handshake-simulator.cpp`

Main knobs:
- `--input-format=auto|dat|plain`
- `--input-vectors-dir`, `--input-args-file`
- `--max-cycles`
- `--lsq-config-dir`
- `--dump-switching-json`
- `--dump-results-json`
- `--dump-switching-est-csv`
- `--verify-vcd=exact --top-verilog ... --vcd ...`

(see option definitions around lines 36-94 and execution flow around 152-257)

## 9) Current Status (as of March 3, 2026 in this workspace)

1. The branch is explicitly marked as in-progress (commit message).
2. It is under `experimental/` and built into `DynamaticExperimentalSupport` (`experimental/lib/Support/CMakeLists.txt`).
3. Dedicated CLI tool target exists: `handshake-simulator`.
4. Operation coverage is broad (control/dataflow/arithmetic/buffers/memory/LSQ), with unsupported ops producing simulator failure.
5. At least one model is clearly provisional: `CmpFOp` currently returns constant false.
6. LSQ simulation depends on external JSON config; missing config is treated as failure.
7. Test coverage in this branch appears minimal for simulator behavior:
   - `experimental/test/tools/handshake-simulator/example.mlir` is a very small structural test.
   - No comprehensive golden cycle-by-cycle regression suite is visible in this branch snapshot.
8. In this workspace, the currently checked-out branch in `dynamatic-mlir` is `sa_final`; `handshake_simulator` exists as remote branch `origin/handshake_simulator`.

## 10) File Pointers for Fast Navigation

- Main simulator loop and cycle scheduling:
  - `experimental/lib/Support/HandshakeSimulator.cpp` (~1955-2051)
- State maps and updater mechanism:
  - `experimental/include/experimental/Support/HandshakeSimulator.h` (state/updater sections)
  - `experimental/lib/Support/HandshakeSimulator.cpp` (~117-168)
- Model mapping:
  - `experimental/lib/Support/HandshakeSimulator.cpp` (~3538-3778)
- Buffer/Load/Store/MC/LSQ/End models:
  - `experimental/lib/Support/HandshakeUnitModels.cpp`
- VCD exact comparison:
  - `experimental/lib/Support/HandshakeSimulator.cpp` (~2392-3480)
- CLI front-end:
  - `experimental/tools/handshake-simulator/handshake-simulator.cpp`
