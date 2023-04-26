add_mlir_dialect_library(SODATransformTransforms
  Interpreter.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJ_INCLUDE_DIR}/soda/Dialect/Transform

  DEPENDS
  SODATransformDialectTransformsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRPDLDialect
  MLIRTransformDialect
)
