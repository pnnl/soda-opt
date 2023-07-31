//===- BufferizationOptions.h - Utilities for creating bufferization objects ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines utilities to create bufferization options objects.
//
//===----------------------------------------------------------------------===//

#ifndef SODA_SPARSETENSOR_UTILS_BUFFERIZATIONOPTIONS_H
#define SODA_SPARSETENSOR_UTILS_BUFFERIZATIONOPTIONS_H

#include "mlir/Dialect/Bufferization/Transforms/Bufferize.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"

namespace mlir {
namespace soda {

/// Return configuration options for One-Shot Bufferize.
bufferization::OneShotBufferizationOptions
getBufferizationOptions(bool analysisOnly);

} // namespace soda
} // namespace mlir


#endif // SODA_SPARSETENSOR_UTILS_BUFFERIZATIONOPTIONS_H
