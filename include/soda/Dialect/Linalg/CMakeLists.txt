add_subdirectory(Reports)

set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name SODALinalg)
mlir_tablegen(Passes.capi.h.inc -gen-pass-capi-header --prefix SODALinalg)
mlir_tablegen(Passes.capi.cpp.inc -gen-pass-capi-impl --prefix SODALinalg)
add_public_tablegen_target(SODALinalgPassIncGen)

add_mlir_doc(Passes SODALinalgPasses ./ -gen-pass-doc)
