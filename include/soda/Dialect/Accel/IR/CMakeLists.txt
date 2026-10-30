add_mlir_dialect(AccelOps accel)
add_mlir_doc(AccelOps AccelOps Dialects/ -gen-dialect-doc)

set(LLVM_TARGET_DEFINITIONS AccelEnums.td)
mlir_tablegen(AccelEnums.h.inc -gen-enum-decls)
mlir_tablegen(AccelEnums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(SODAAccelEnumsIncGen)

set(LLVM_TARGET_DEFINITIONS AccelAttributes.td)
mlir_tablegen(AccelAttributes.h.inc -gen-attrdef-decls -attrdefs-dialect=accel)
mlir_tablegen(AccelAttributes.cpp.inc -gen-attrdef-defs -attrdefs-dialect=accel)
add_public_tablegen_target(SODAAccelAttributesIncGen)
