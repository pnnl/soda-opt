add_mlir_dialect_library(SODALinalgTransforms
  Tiling.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJ_INCLUDE_DIR}/soda/Dialect/Linalg

  DEPENDS
  SODALinalgPassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRSupport
)
