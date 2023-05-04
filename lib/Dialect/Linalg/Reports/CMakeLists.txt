add_mlir_dialect_library(SODALinalgReports
    GenerateLinalgSummary.cpp

    ADDITIONAL_HEADER_DIRS
    ${PROJ_INCLUDE_DIR}/soda/Dialect/Linalg/Reports

    DEPENDS
    SODALinalgReportsPassIncGen

    LINK_LIBS PUBLIC
    MLIRIR
    MLIRPass
)
