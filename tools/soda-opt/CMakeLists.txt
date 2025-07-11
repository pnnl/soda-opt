# This includes several libs that extends the linking time
# only the necessary libs are included instead
get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

message(PUBLIC "${dialect_libs}")
# message(PUBLIC "${conversion_libs}")
set(LIBS

  # ${dialect_libs}
  # ${conversion_libs}

  # Core
  MLIRIR
  MLIRTransforms
  MLIROptLib

  # Dialects
  MLIRAsyncDialect
  MLIRBufferizationDialect
  MLIRLLVMDialect
  MLIRMLProgramDialect
  # MLIRTosaDialect
  # MLIRTensorDialect
  MLIRLinalgDialect
  MLIRMemRefDialect
  MLIRShapeDialect
  MLIRIndexDialect
  MLIRAffineDialect
  MLIRArithDialect
  MLIRArithValueBoundsOpInterfaceImpl
  MLIRMathDialect
  MLIRFuncDialect
  MLIRSCFDialect
  MLIRControlFlowDialect
  MLIRTransformDialect
  MLIRPDLDialect
  MLIRPDLInterpDialect
  MLIRTilingInterfaceTestPasses
  
  # TransformOps 
  MLIRFuncTransformOps
  MLIRLinalgTransformOps
  MLIRSCFTransformOps
  
  # Utils
  MLIRLinalgUtils
  MLIRSCFUtils

  # Transforms
  MLIRFuncTransforms
  MLIRLinalgTransforms
  MLIRAffineTransforms
  MLIRArithTransforms
  MLIRShapeOpsTransforms
  MLIRSCFTransforms
  MLIRReconcileUnrealizedCasts
  MLIRMemRefTransforms
  MLIRBufferizationTransformOps
  MLIRControlFlowTransforms
  MLIRMLProgramTransforms
  MLIRTransformDebugExtension
  MLIRTransformLoopExtension
  MLIRTransformDialectTransforms

  # SODATransforms
  SODAAffineTransformOps

  # Test Passes
  MLIRLinalgTestPasses
  MLIRAffineTransformsTestPasses

  # Conversion
  MLIRAffineToStandard
  MLIRComplexToStandard
  MLIRComplexToLLVM
  MLIRSCFToControlFlow
  MLIRMemRefToLLVM
  MLIRMathToLLVM
  MLIRMathToLibm
  MLIRArithToLLVM
  MLIRFuncToLLVM
  MLIRLinalgToStandard
  MLIRVectorToLLVMPass

  # Conversion SODA
  SODAKERNELToSODA
  SODAFuncToLLVM
  # SODALinalgToAccel

  # SODA
  SODALinalgTransforms
  SODAAffineTransforms
  SODATransformTransforms
  SODALinalgReports
  SODAMisc
  MLIRSODAOps
  MLIRSODATransforms

  # SNN
  # WIP Remove SNN
  # MLIRSNNOps
  # MLIRSNNTransforms

  # ACCEL
  # WIP: AXI4MLIR integration
  #SODAAccelDialect
  #SODAAccelToRuntime
)

set(SOURCES
  soda-opt.cpp
)

add_llvm_executable(soda-opt ${SOURCES})

llvm_update_compile_flags(soda-opt)
target_link_libraries(soda-opt PRIVATE ${LIBS})
llvm_update_compile_flags(soda-opt)

mlir_check_link_libraries(soda-opt)

install(TARGETS soda-opt)
