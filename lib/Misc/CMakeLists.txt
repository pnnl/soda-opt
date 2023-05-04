add_mlir_dialect_library(SODAMisc
    MiscPasses.cpp
    MemrefEraseDealloc.cpp
    ForwardMemrefAlloc.cpp
    ForwardLinalgOps.cpp

    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/

    DEPENDS
    MLIRSODAMiscPassIncGen

    LINK_LIBS PUBLIC
    MLIRIR
    MLIRPass
)
