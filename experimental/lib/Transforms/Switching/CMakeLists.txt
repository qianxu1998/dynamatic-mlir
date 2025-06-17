add_dynamatic_library(DynamaticExperimentalSwitchingEstimation
  SwitchingEstimation.cpp
  SwitchingSupport.cpp
  ProfilingAnalyzer.cpp
  ExecModel.cpp
  DataChannelCal.cpp
  HandShakeChannelCal.cpp
  NodeId.cpp
  NameTable.cpp
  NodeInfo.cpp
  GraphModel.cpp


  DEPENDS
  DynamaticExperimentalTransformsPassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRMemRefDialect
  MLIRFuncDialect
  MLIRSupport
  MLIRTransformUtils
  DynamaticHandshake
  DynamaticSupport
  DynamaticExperimentalSupport
)