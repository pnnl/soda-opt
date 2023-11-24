add_mlir_dialect_library(SODAAccelDialect
  AccelOps.cpp
  AccelDialect.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/soda/Dialect/Accel

  DEPENDS
  MLIRAccelOpsIncGen

  LINK_LIBS PUBLIC
  MLIRDialect
  MLIRIR
)
