//===- DenseBufferization.cpp - Lower Dense Tensor via 'bufferization' -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file was extracted and modified based on:
//   llvm-project/mlir/lib/Dialect/SparseTensor/Transforms/SparsificationAndBufferization.cpp
// Separating out the functionality that generates bufferizes only dense tensors.
//
// See extracted documentation below for details.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/Bufferize.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotModuleBufferize.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/Transforms.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"
#include "mlir/Dialect/SparseTensor/Transforms/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

#include "soda/Dialect/SparseTensor/Transforms/Passes.h"
#include "soda/Dialect/SparseTensor/Utils/BufferizationOptions.h"

namespace mlir::soda {
#define GEN_PASS_DEF_DENSEBUFFERIZATION
#include "soda/Dialect/SparseTensor/Transforms/Passes.h.inc"
} // namespace mlir::soda

using namespace mlir;
using namespace mlir::func;
using namespace mlir::sparse_tensor;
using namespace mlir::soda;

#define DEBUG_TYPE "soda-sparse-compiler-dense-bufferization"

namespace {

/// Return `true` if one of the given types is a sparse tensor type.
static bool containsSparseTensor(TypeRange types) {
  for (Type t : types)
    if (getSparseTensorEncoding(t))
      return true;
  return false;
}

/// A pass that lowers dense tensor ops to memref ops:
///
/// * Dense tensor ops are lowered through BufferizableOpInterface
///   implementations.
///
class DenseBufferizationPass
    : public mlir::soda::impl::DenseBufferizationBase<DenseBufferizationPass> {
public:

  DenseBufferizationPass() = default;
  DenseBufferizationPass(bool testBufferizationAnalysisOnly) {
    bufferizationOptions = mlir::soda::getBufferizationOptions(/*analysisOnly=*/testBufferizationAnalysisOnly);
  }

  void getDependentDialects(::mlir::DialectRegistry &registry) const override {
    registry.insert<bufferization::BufferizationDialect>();
  }

  /// Bufferize all dense ops. This assumes that no further analysis is needed
  /// and that all required buffer copies were already inserted by
  /// `insertTensorCopies` in the form of `bufferization.alloc_tensor` ops.
  LogicalResult runDenseBufferization() {
    bufferization::OpFilter denseOpFilter;
    denseOpFilter.allowOperation([&](Operation *op) {
      if (containsSparseTensor(TypeRange(op->getResults())) ||
          containsSparseTensor(TypeRange(op->getOperands())))
        return false;
      if (auto funcOp = dyn_cast<func::FuncOp>(op)) {
        FunctionType funcType = funcOp.getFunctionType();
        if (containsSparseTensor(funcType.getInputs()) ||
            containsSparseTensor(funcType.getResults()))
          return false;
      }
      return true;
    });

    if (failed(bufferization::bufferizeOp(getOperation(), bufferizationOptions,
                                          /*copyBeforeWrite=*/false,
                                          &denseOpFilter)))
      return failure();

    bufferization::removeBufferizationAttributesInModule(getOperation());
    return success();
  }

  void runOnOperation() override {
    // Bufferize all dense ops.
    if (failed(runDenseBufferization()))
      signalPassFailure();
  }

private:
  bufferization::OneShotBufferizationOptions bufferizationOptions;
};

} // namespace

std::unique_ptr<Pass> mlir::soda::createDenseBufferizationPass() {
  return std::make_unique<DenseBufferizationPass>();
}

std::unique_ptr<Pass> mlir::soda::createDenseBufferizationPass(bool testBufferizationAnalysisOnly) {
  return std::make_unique<DenseBufferizationPass>(testBufferizationAnalysisOnly);
}