//===- LinalgGenericToAccel.h - Convert linalg to AXI4MLIR calls ----*- C++
//-*-===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SODA_CONVERSION_LINALGTOACCEL_LINALGGENERICTOACCEL_H_
#define SODA_CONVERSION_LINALGTOACCEL_LINALGGENERICTOACCEL_H_

#include "soda/Conversion/LinalgToAccel/AXI4MLIRUtils.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir {
class MLIRContext;
class Pass;
class RewritePatternSet;
class ModuleOp;
template <typename T>
class OperationPass;

/// Populate the list with patterns that convert from LinalgOps to AccelOps
void populateLinalgGenericToAccelConversionPatternsWithOptions(
    RewritePatternSet &patterns,
    const AccelTransformationOptions &options = AccelTransformationOptions());
void populateLinalgGenericToAccelConversionPatterns(
    RewritePatternSet &patterns);

/// Create the pass to convert from LinalgOps to AccelOps
std::unique_ptr<OperationPass<ModuleOp>>
createConvertLinalgGenericToAccelPass();

std::unique_ptr<OperationPass<ModuleOp>> createConvertLinalgGenericToAccelPass(
    const AccelTransformationOptions &options);

} // namespace mlir

#endif // SODA_CONVERSION_LINALGTOACCEL_LINALGGENERICTOACCEL_H_
