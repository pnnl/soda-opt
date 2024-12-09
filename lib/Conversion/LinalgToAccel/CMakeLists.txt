add_mlir_conversion_library(SODALinalgToAccel
  LinalgGenericToAccel.cpp
  AXI4MLIRUtils.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJ_INCLUDE_DIR}/soda/Conversion/LinalgToAccel

  DEPENDS
  SODAConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRArithDialect
  # WIP: AXI4MLIR integration
  #SODAAccelDialect
  MLIRMemRefDialect
  MLIRTransforms
  )
