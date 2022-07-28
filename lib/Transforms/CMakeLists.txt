add_mlir_library(SODALoopPipelining
  SchedulingUtils.cpp
  ForwardResults.cpp
  IfConversion.cpp
  LoopSchedule.cpp
  LoopScheduleBuilder.cpp
  
  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/soda/Transforms

  LINK_LIBS PUBLIC
  MLIRAffineDialect
  MLIRAnalysis
  MLIRAffineAnalysis
  MLIRSCFDialect
  MLIRPass
  MLIRRewrite
  )
