//===- AccelToAXI4MLIR.h - Convert Accel to AXI4MLIR calls ----*- C++ -*-===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SODA_CONVERSION_ACCELTORUNTIME_ACCELTOAXI4MLIR_H_
#define SODA_CONVERSION_ACCELTORUNTIME_ACCELTOAXI4MLIR_H_

#include "mlir/IR/PatternMatch.h"

namespace mlir {
class MLIRContext;
class Pass;
class RewritePatternSet;
class ModuleOp;
template <typename T>
class OperationPass;

struct AccelToAXI4MLIROptions {
  /// Accelerator Tile Size information
  unsigned tileSize = 1;

  /// DMA Information
  unsigned dmaAddress = 0;
  unsigned dmaInputAddress = 0;
  unsigned dmaInputBufferSize = 100000;
  unsigned dmaOutputAddress = 100000;
  unsigned dmaOutputBufferSize = 100000;

  /// Flow information
  bool flowCpuAcc = false;
  unsigned numberOfCaches = false;
  ArrayRef<unsigned> cacheSizes;
  ArrayRef<unsigned> tileSizes;
  unsigned elementSize = false;
};

/// Populate the given list with patterns that convert from Accel to AXI4MLIR
/// runtime calls.
void populateAccelToAXI4MLIRConversionPatterns(RewritePatternSet &patterns);

/// Populate the given list with patterns that convert from Accel to AXI4MLIR
/// runtime calls.
void populateAccelToAXI4MLIRConversionPatternsWithOptions(
    RewritePatternSet &patterns,
    const AccelToAXI4MLIROptions &options = AccelToAXI4MLIROptions());

/// Create the pass to convert accel operations to axi4mlir calls
std::unique_ptr<OperationPass<ModuleOp>> createConvertAccelToAXI4MLIRPass();

std::unique_ptr<OperationPass<ModuleOp>>
createConvertAccelToAXI4MLIRPass(const AccelToAXI4MLIROptions &options);

} // namespace mlir

#endif // SODA_CONVERSION_ACCELTORUNTIME_ACCELTOAXI4MLIR_H_
