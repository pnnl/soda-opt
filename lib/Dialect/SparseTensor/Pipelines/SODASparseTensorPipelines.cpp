//===- SODASparseTensorPipelines.cpp - Pipelines for sparse tensor code in SODA-OPT -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file was modified based on:
//   llvm-project/mlir/lib/Dialect/SparseTensor/Pipelines/SparseTensorPipelines.cpp
// Implements the pipeline by splitting the SparsificationAndBufferization pass,
// adding debugging, and adding OpenMP support.
//
//===----------------------------------------------------------------------===//


#include "mlir/Conversion/GPUToNVVM/GPUToNVVMPass.h"
#include "mlir/Conversion/SCFToOpenMP/SCFToOpenMP.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/Bufferize.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"
#include "mlir/Dialect/SparseTensor/Transforms/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

#include "soda/Dialect/SparseTensor/Pipelines/Passes.h"
#include "soda/Dialect/SparseTensor/Transforms/Passes.h"

using namespace mlir;
using namespace mlir::sparse_tensor;
using namespace mlir::soda;

//===----------------------------------------------------------------------===//
// Pipeline implementation.
//===----------------------------------------------------------------------===//
namespace mlir {
namespace soda {

void buildSODASparseCompiler(
    OpPassManager &pm, const SODASparseCompilerOptions &options) {

  pm.addNestedPass<func::FuncOp>(createLinalgGeneralizationPass());
  
  // Splitting the SparsificationAndBufferization pass:
  // 1. Enabling passes
  pm.addPass(createPreSparsificationRewritePass());
  pm.addNestedPass<func::FuncOp>(
    bufferization::createEmptyTensorToAllocTensorPass());

  // 2. Insert tensor copies
  pm.addPass(createInsertTensorCopiesPass(options.testBufferizationAnalysisOnly));

  // 3. Sparsification
  pm.addPass(createSparsificationPass(options.sparsificationOptions()));
  pm.addPass(createPostSparsificationRewritePass(options.enableRuntimeLibrary));
  if (options.enableMemrefPrints) {
    pm.addPass(createSparseTensorMemrefDebugPass());
  }

  if (options.vectorLength > 0) {
    pm.addPass(mlir::createLoopInvariantCodeMotionPass());
    pm.addPass(createSparseVectorizationPass(
      options.vectorLength, /*enableVLAVectorization=*/options.armSVE, 
      /*enableSIMDIndex32=*/options.force32BitVectorIndices));
  }

  if (options.enableRuntimeLibrary) {
    pm.addPass(
      createSparseTensorConversionPass(options.sparseTensorConversionOptions()));
  } else {
    pm.addPass(createSparseTensorCodegenPass(options.enableBufferInitialization));
    pm.addPass(createSparseBufferRewritePass(options.enableBufferInitialization));
    // pm.addPass(createStorageSpecifierToLLVMPass());
  }

  // 4. Dense bufferization
  pm.addPass(createDenseBufferizationPass(options.testBufferizationAnalysisOnly));

  // Continuing with the original sparse-compiler pipeline:
  pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
  pm.addNestedPass<func::FuncOp>(
      mlir::bufferization::createFinalizingBufferizePass());

  // GPU code generation.
  // const bool gpuCodegen = options.gpuTriple.hasValue();
  // if (gpuCodegen) {
  //   pm.addPass(createSparseGPUCodegenPass());
  //   pm.addNestedPass<gpu::GPUModuleOp>(createStripDebugInfoPass());
  //   pm.addNestedPass<gpu::GPUModuleOp>(createConvertSCFToCFPass());
  //   pm.addNestedPass<gpu::GPUModuleOp>(createLowerGpuOpsToNVVMOpsPass());
  // }

  // TODO(springerm): Add sparse support to the BufferDeallocation pass and add
  // it to this pipeline.
  pm.addNestedPass<func::FuncOp>(createConvertLinalgToLoopsPass());
  pm.addNestedPass<func::FuncOp>(createConvertVectorToSCFPass());

  /*
   * NEW -> Adding OpenMP lowering *before* the rest of the SCF-to passes
   * to capture scf.parallel, scf.reduce, etc. before they're lost to the
   * CF and LLVM dialects.
   */
  if (options.enableOpenMP) {
    pm.addPass(createConvertSCFToOpenMPPass());
    pm.addNestedPass<func::FuncOp>(createCanonicalizerPass()); 
  }

  pm.addNestedPass<func::FuncOp>(createConvertSCFToCFPass());
  pm.addPass(memref::createExpandStridedMetadataPass());
  pm.addPass(createLowerAffinePass());
  
  pm.addPass(createConvertVectorToLLVMPass(options.lowerVectorToLLVMOptions()));
  pm.addPass(createMemRefToLLVMConversionPass());
  pm.addNestedPass<func::FuncOp>(createConvertComplexToStandardPass());
  pm.addNestedPass<func::FuncOp>(arith::createArithExpandOpsPass());
  pm.addNestedPass<func::FuncOp>(createConvertMathToLLVMPass());
  pm.addPass(createConvertMathToLibmPass());
  pm.addPass(createConvertComplexToLibmPass());

  // Repeat convert-vector-to-llvm.
  pm.addPass(createConvertVectorToLLVMPass(options.lowerVectorToLLVMOptions()));
  pm.addPass(createConvertComplexToLLVMPass());
  pm.addPass(createConvertVectorToLLVMPass(options.lowerVectorToLLVMOptions()));

  /*
   * NEW -> Adding OpenMP lowering *after* the rest of the to-LLVM passes
   * so that all other dialects are not treated as "illegal" by the OpenMP
   * conversion pass. TODO: Confirm.
   */
  if (options.enableOpenMP) {
    pm.addPass(createConvertOpenMPToLLVMPass());
  }

  pm.addPass(createConvertFuncToLLVMPass());

//   // Finalize GPU code generation.
//   if (gpuCodegen) {
// #if MLIR_GPU_TO_CUBIN_PASS_ENABLE
//     pm.addNestedPass<gpu::GPUModuleOp>(createGpuSerializeToCubinPass(
//         options.gpuTriple, options.gpuChip, options.gpuFeatures));
// #endif
//     pm.addPass(createGpuToLLVMConversionPass());
//   }

  pm.addPass(createReconcileUnrealizedCastsPass());
}

//===----------------------------------------------------------------------===//
// Pipeline registration.
//===----------------------------------------------------------------------===//

void registerSODASparseTensorPipelines() {
  PassPipelineRegistration<SODASparseCompilerOptions>(
      "soda-sparse-compiler",
      "The modified pipeline for SODA-OPT that takes sparsity-agnostic "
      " IR using thesparse-tensor type, and lowers it to LLVM IR with "
      " concrete representations and algorithms for sparse tensors.",
      buildSODASparseCompiler);
}

} // namespace soda
} // namespace mlir