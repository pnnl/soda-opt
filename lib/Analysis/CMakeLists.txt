add_mlir_library(SODAAnalysis
  DataFlowGraph.cpp
  
  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Analysis

  DEPENDS
  mlir-headers

  LINK_LIBS PUBLIC
  MLIRCallInterfaces
  MLIRControlFlowInterfaces
  MLIRDataLayoutInterfaces
  MLIRInferIntRangeInterface
  MLIRInferTypeOpInterface
  MLIRLoopLikeInterface
  MLIRSideEffectInterfaces
  MLIRViewLikeInterface
  )