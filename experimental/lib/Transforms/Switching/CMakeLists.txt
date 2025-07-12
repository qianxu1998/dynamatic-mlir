add_dynamatic_library(DynamaticExperimentalSwitchingEstimation
  SwitchingEstimation.cpp
  SwitchingSupport.cpp
  ProfilingAnalyzer.cpp
  SwitchingNodeModels/SwitchingNodeModels.cpp
  SwitchingNodeModels/DataNodes.cpp
  SwitchingNodeModels/ControlNodes.cpp

  DataChannelCal.cpp
  HandShakeChannelCal.cpp
  NodeId.cpp
  NameTable.cpp
  NodeInfo.cpp

  GraphModel.cpp
  DataGlitches.cpp  
  HandShakeBufferAnalysis.cpp
  DataChannelPropagation.cpp

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