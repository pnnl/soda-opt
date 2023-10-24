add_mlir_dialect_library(SODASparseTensorTransforms
    DenseBufferization.cpp
    InsertTensorCopies.cpp
    SparseTensorMemrefDebug.cpp

    ADDITIONAL_HEADER_DIRS
    ${PROJ_INCLUDE_DIR}/soda/Dialect/SparseTensor/Transforms

    DEPENDS
    SODASparseTensorTransformsIncGen

    LINK_LIBS PUBLIC
    MLIRMemRefDialect
    MLIRBufferizationDialect
    MLIRIR
    MLIRPass
    SODASparseTensorUtils
)
