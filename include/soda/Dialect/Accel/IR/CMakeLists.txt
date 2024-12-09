add_mlir_dialect(AccelOps accel)
add_mlir_doc(AccelOps AccelOps Dialects/ -gen-dialect-doc)

set(LLVM_TARGET_DEFINITIONS AccelAttributes.td)
mlir_tablegen(AccelAttributes.h.inc -gen-attrdef-decls)
mlir_tablegen(AccelAttributes.cpp.inc -gen-attrdef-defs)
add_public_tablegen_target(SODAAccelAttributeIncGen)
