add_dynamatic_library(DynamaticSwitchingEstimation
  Debug.cpp
  SwitchingEstimation.cpp
  GraphModel.cpp
  SwitchingSupport.cpp
  ProfilingAnalyzer.cpp
  NodeModels.cpp
  DataChannelCal.cpp
  HandShakeChannelCal.cpp

  DEPENDS
  DynamaticExperimentalAnalysisPassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRMemRefDialect
  MLIRFuncDialect
  MLIRSupport
  MLIRTransformUtils
  DynamaticHandshake
  DynamaticBufferPlacement
  DynamaticSupport
  DynamaticExperimentalSupport
  )
