add_mlir_library(SODATestTransforms
  TestForwardResults.cpp
  TestIfConversion.cpp
  TestDataFlowGraph.cpp
  TestLoopSchedulingPass.cpp

  EXCLUDE_FROM_LIBMLIR

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Transforms

  LINK_LIBS PUBLIC
  MLIRAnalysis
  MLIRInferIntRangeInterface
  MLIRTestDialect
  MLIRTransforms
  MLIRTransformUtils
  SODALoopPipelining
  SODAAnalysis
  )

target_include_directories(SODATestTransforms
  PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../Dialect/Test
  ${CMAKE_CURRENT_BINARY_DIR}/../Dialect/Test
  )