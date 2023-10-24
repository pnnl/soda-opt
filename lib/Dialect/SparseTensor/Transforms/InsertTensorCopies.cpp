//===- InsertTensorCopies.cpp - Tensor Copy Insertion via 'bufferization' Dialect -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file was extracted and modified based on:
//   llvm-project/mlir/lib/Dialect/SparseTensor/Transforms/SparsificationAndBufferization.cpp
// Separating out the functionality that generates tensor copies 
// using One-Shot Analysis in the form of bufferization.alloc_tensor ops.
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

namespace mlir {
namespace soda {
#define GEN_PASS_DEF_INSERTTENSORCOPIES
#include "soda/Dialect/SparseTensor/Transforms/Passes.h.inc"
} // namespace soda
} // namespace mlir

using namespace mlir;
using namespace mlir::soda;

#define DEBUG_TYPE "soda-sparse-compiler-insert-tensor-copies"

namespace {

/// A pass that inserts tensor copies.
///
/// One-Shot Analysis is used to detect RaW conflicts and to insert buffer
/// copies of the tensor level (`insertTensorCopies`).
///
class InsertTensorCopiesPass
    : public mlir::soda::impl::InsertTensorCopiesBase<
          InsertTensorCopiesPass> {
public:

  InsertTensorCopiesPass() = default;
  InsertTensorCopiesPass(bool testBufferizationAnalysisOnly) {
    bufferizationOptions = mlir::soda::getBufferizationOptions(/*analysisOnly=*/testBufferizationAnalysisOnly);
  }

  void getDependentDialects(::mlir::DialectRegistry &registry) const override {
    registry.insert<bufferization::BufferizationDialect>();
  }

  void runOnOperation() override {
    // Insert tensor copies. This step runs One-Shot Analysis (which analyzes
    // SSA use-def chains of tensor IR) and decides where buffer copies are
    // needed and where buffers can be written to in-place. These decisions are
    // materialized in the IR in the form of `bufferization.alloc_tensor` ops.
    //
    // Note: All following steps in this pass must be careful not to modify the
    // structure of the IR (i.e., tensor use-def chains), as that could
    // invalidate the results of the analysis. From now on, only small and
    // localized rewrites are allowed, such as replacing a tensor op with its
    // memref equivalent.
    if (failed(bufferization::insertTensorCopies(getOperation(),
                                                 bufferizationOptions)))
      return signalPassFailure();

    // `testAnalysisOnly` is a debug/testing flag. If set, the results of
    // OneShotAnalysis are added to the IR via attributes. In that case, do not
    // continue with the remaining pipeline.
    if (bufferizationOptions.testAnalysisOnly)
      return;
  }

private:
  bufferization::OneShotBufferizationOptions bufferizationOptions;
};

} // namespace

std::unique_ptr<Pass> mlir::soda::createInsertTensorCopiesPass() {
  return std::make_unique<InsertTensorCopiesPass>();
}

std::unique_ptr<Pass> mlir::soda::createInsertTensorCopiesPass(bool testBufferizationAnalysisOnly) {
  return std::make_unique<InsertTensorCopiesPass>(testBufferizationAnalysisOnly);
}