add_mlir_dialect_library(SODAAffineTransforms
  AffineDataCopyGen.cpp
  AffineLoopTiling.cpp
  AffineLoopPermutation.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJ_INCLUDE_DIR}/soda/Dialect/Affine

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRSupport
)
