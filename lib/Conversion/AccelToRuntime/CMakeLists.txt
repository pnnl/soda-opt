add_mlir_conversion_library(SODAAccelToRuntime
  AccelToAXIRuntime.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/soda/Conversion

  DEPENDS
  SODAAccelConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRArithUtils
  MLIRFuncDialect
  MLIRIR
  MLIRMemRefDialect
  MLIRPass
  MLIRSideEffectInterfaces
  SODAAccelDialect
)
