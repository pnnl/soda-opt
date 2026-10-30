add_mlir_conversion_library(SODALinalgToAccel
  LinalgToAccel.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/soda/Conversion

  DEPENDS
  SODAAccelConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRAffineDialect
  MLIRArithDialect
  MLIRAsmParser
  MLIRFuncDialect
  MLIRIR
  MLIRLinalgDialect
  MLIRLinalgUtils
  MLIRMemRefDialect
  MLIRPass
  MLIRSCFDialect
  SODAAccelDialect
)
