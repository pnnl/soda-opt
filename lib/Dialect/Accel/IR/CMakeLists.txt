add_mlir_dialect_library(SODAAccelDialect
  AccelAttributes.cpp
  AccelOps.cpp
  AccelDialect.cpp
  OpcodeAttr/OpcodeList.cpp
  OpcodeAttr/OpcodeMap.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/soda/Dialect/Accel

  DEPENDS
  MLIRAccelOpsIncGen
  SODAAccelAttributeIncGen

  LINK_LIBS PUBLIC
  MLIRDialect
  MLIRIR
)
