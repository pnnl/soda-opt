add_mlir_dialect_library(SODAAffineTransformOps
  AffineTransformOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJ_INCLUDE_DIR}/soda/Dialect/Transform/TransformOps

  DEPENDS
  SODAAffineTransformOpsIncGen

  LINK_LIBS PUBLIC
  MLIRAffineDialect
  MLIRFuncDialect
  MLIRIR
  MLIRLoopLikeInterface
  MLIRAffineUtils
  MLIRSCFTransforms
  MLIRSCFUtils
  MLIRTransformDialect
)
