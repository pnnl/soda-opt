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
  MLIRSparseTensorDialect
  MLIRTensorDialect
  MLIRLinalgDialect
  MLIRMemRefDialect
  # MLIRShapeDialect
  # MLIRIndexDialect
  MLIRAffineDialect
  MLIRArithDialect
  MLIRMathDialect
  MLIRFuncDialect
  MLIRSCFDialect
  MLIRTransformDialect
  MLIRPDLDialect
  MLIRPDLInterpDialect
  MLIRTilingInterfaceTestPasses
  MLIRTensorInferTypeOpInterfaceImpl
  
  # TransformOps 
  MLIRShapeOpsTransforms
  MLIRLinalgTransformOps
  MLIRSCFTransformOps
  
  # Utils
  MLIRLinalgUtils
  MLIRSCFUtils

  # Transforms
  MLIRFuncTransforms
  MLIRLinalgTransforms
  MLIRAffineTransforms
  MLIRSCFTransforms
  MLIRReconcileUnrealizedCasts
  MLIRMemRefTransforms

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
  MLIRLinalgToLLVM
  MLIRLinalgToStandard
  MLIRSparseTensorPipelines

  # Conversion SODA
  SODAKERNELToSODA
  SODAFuncToLLVM

  # SODA
  SODALinalgTransforms
  SODAAffineTransforms
  SODATransformTransforms
  SODASparseTensorPipelines
  SODALinalgReports
  SODAMisc
  MLIRSODAOps
  MLIRSODATransforms

  # SNN
  MLIRSNNOps
  MLIRSNNTransforms

  # Runtime
  soda_runner_ext
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
