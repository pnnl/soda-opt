add_mlir_dialect_library(MLIRSODAAffineTransformOps
  AffineTransformOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJ_INCLUDE_DIR}/soda/Dialect/Transform/TransformOps

  DEPENDS
  MLIRSODAAffineTransformOpsIncGen

  LINK_LIBS PUBLIC
  MLIRAffineDialect
  MLIRFuncDialect
  MLIRIR
  MLIRLoopLikeInterface
  MLIRSCFDialect
  MLIRSCFToControlFlow
  MLIRSCFTransforms
  MLIRSCFUtils
  MLIRTransformDialect
  MLIRVectorDialect
)
