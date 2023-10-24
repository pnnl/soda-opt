//===- Merger.cpp - Implementation of iteration lattices ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "soda/Dialect/SparseTensor/Utils/BufferizationOptions.h"

namespace mlir {
namespace soda {

/// Return configuration options for One-Shot Bufferize.
bufferization::OneShotBufferizationOptions
getBufferizationOptions(bool analysisOnly) {
  using namespace bufferization;
  OneShotBufferizationOptions options;
  options.bufferizeFunctionBoundaries = true;
  // TODO(springerm): To spot memory leaks more easily, returning dense allocs
  // should be disallowed.
  options.allowReturnAllocs = true;
  options.functionBoundaryTypeConversion = LayoutMapOption::IdentityLayoutMap;
  options.unknownTypeConverterFn = [](Value value, Attribute memorySpace,
                                      const BufferizationOptions &options) {
    return getMemRefTypeWithStaticIdentityLayout(
        cast<TensorType>(value.getType()), memorySpace);
  };
  if (analysisOnly) {
    options.testAnalysisOnly = true;
    options.printConflicts = true;
  }
  return options;
}

} // namespace soda
} // namespace mlir