add_dynamatic_library(DynamaticSwitchingEstimation
  SwitchingEstimation.cpp
  GraphModel.cpp
  SwitchingSupport.cpp
  NodeModels.cpp

  DEPENDS
  DynamaticExperimentalAnalysisPassIncGen

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

