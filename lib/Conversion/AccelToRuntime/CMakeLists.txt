add_mlir_dialect_library(SODAAccelToRuntime
  AccelToAXI4MLIR.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJ_INCLUDE_DIR}/soda/Conversion/AccelToRuntime

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
)
 