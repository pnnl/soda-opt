add_mlir_dialect_library(SODASparseTensorUtils
  BufferizationOptions.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJ_INCLUDE_DIR}/soda/Dialect/SparseTensor/Utils

  LINK_LIBS PUBLIC
  MLIRBufferizationDialect
)
