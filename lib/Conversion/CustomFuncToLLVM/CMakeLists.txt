add_mlir_dialect_library(SODAFuncToLLVM
  FuncToLLVM.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJ_INCLUDE_DIR}/soda/Conversion/CustomFuncToLLVM

  DEPENDS
  MLIRSODAMiscPassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
)
