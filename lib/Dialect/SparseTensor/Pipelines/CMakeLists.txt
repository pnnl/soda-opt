add_mlir_dialect_library(SODASparseTensorPipelines
  SODASparseTensorPipelines.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJ_INCLUDE_DIR}/soda/Dialect/SparseTensor/Pipelines

  LINK_LIBS PUBLIC
  MLIRArithTransforms
  MLIRAffineToStandard
  MLIRBufferizationTransforms
  MLIRComplexToLLVM
  MLIRComplexToLibm
  MLIRComplexToStandard
  MLIRFuncTransforms
  MLIRGPUToNVVMTransforms
  MLIRGPUTransforms
  MLIRLinalgTransforms
  MLIRMathToLibm
  MLIRMathToLLVM
  MLIRMemRefToLLVM
  MLIROpenMPToLLVM
  MLIRPass
  MLIRReconcileUnrealizedCasts
  MLIRSCFToControlFlow
  MLIRSCFToOpenMP
  MLIRSparseTensorDialect
  MLIRSparseTensorTransforms
  MLIRTensorTransforms
  MLIRVectorToLLVM
  MLIRVectorTransforms
  SODASparseTensorTransforms
)

if(MLIR_ENABLE_CUDA_RUNNER)
   # Enable gpu-to-cubin pass.
  target_compile_definitions(obj.MLIRSparseTensorPipelines
    PRIVATE
    MLIR_GPU_TO_CUBIN_PASS_ENABLE=1
  )
endif()
