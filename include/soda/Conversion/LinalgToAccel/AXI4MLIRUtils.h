//===- Utils.h - Function and method used by axi4mlir passes ----*- C++ -*-===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SODA_CONVERSION_LINALGTOAXI4MLIR_UTILS_H_
#define SODA_CONVERSION_LINALGTOAXI4MLIR_UTILS_H_

#include "mlir/IR/PatternMatch.h"

namespace mlir {
class MLIRContext;
class Pass;
class RewritePatternSet;
class PatternRewriter;
class ModuleOp;
namespace func {
class FuncOp;
} // namespace func

struct AccelTransformationOptions {
  /// Accelerator Tile Size information
  unsigned accelSize = 1;
  ArrayRef<unsigned> accelSizes;

  /// DMA Information
  unsigned dmaAddress = 0;
  unsigned dmaInputAddress = 0;
  unsigned dmaInputBufferSize = 100000;
  unsigned dmaOutputAddress = 100000;
  unsigned dmaOutputBufferSize = 100000;

  /// Flow information

  /// IDs of opcodes that should be accumulated on the CPU
  ArrayRef<unsigned> accOnCpu;
  bool flowCpuAcc = false;
  unsigned numberOfCaches = false;
  ArrayRef<unsigned> cacheSizes;
  ArrayRef<unsigned> tileSizes;
  unsigned elementSize = false;
  ArrayRef<unsigned> loopPermutation;

  /// Anchor
  std::string anchorFuncName;
  std::string anchorOpName;
  std::string anchorFilterName;

  /// Opcode information
  std::string opcodeMap;
  std::string initFlow;
  std::string opcodeFlow;

public:
  /// Utility to print members of the struct
  void dump() const;
};

/// Apply tiling patterns to matmul operations with the correct attribute
void applyPatterns(func::FuncOp funcOp, const AccelTransformationOptions &options);

/// Populates patterns that implement a FSM of modifications.
/// Changhing the kLinalgTransformMarker
/// GENERALIZE -> INTERCHANGE -> MEM(TILE) L3(TILE) -> L2(TILE) -> L1(TILE) ->
/// ACCEL
void populateCommonLinalgTransformationPatterns(
    RewritePatternSet &patterns, const AccelTransformationOptions &options);

} // namespace mlir

#endif // SODA_CONVERSION_LINALGTOAXI4MLIR_UTILS_H_
