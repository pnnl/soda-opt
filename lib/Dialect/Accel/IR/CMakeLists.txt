add_mlir_dialect_library(SODAAccelDialect
  AccelAttributes.cpp
  AccelDialect.cpp
  AccelOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/soda/Dialect/Accel

  DEPENDS
  MLIRAccelOpsIncGen
  SODAAccelEnumsIncGen
  SODAAccelAttributesIncGen

  LINK_LIBS PUBLIC
  MLIRAnalysis
  MLIRArithDialect
  MLIRDialect
  MLIRIR
  MLIRLinalgDialect
)
