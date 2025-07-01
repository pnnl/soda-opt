set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name Transform)
add_public_tablegen_target(SODATransformDialectTransformsIncGen)

add_mlir_doc(Passes TransformPasses ./ -gen-pass-doc)


set(LLVM_TARGET_DEFINITIONS AffineTransformOps.td)
mlir_tablegen(AffineTransformOps.h.inc -gen-op-decls)
mlir_tablegen(AffineTransformOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(MLIRAffineTransformOpsIncGen)

add_mlir_doc(AffineTransformOps AffineLoopTransformOps Dialects/ -gen-op-doc)