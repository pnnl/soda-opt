//===-- Passes.td - Transform dialect pass definitions -----*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SODA_DIALECT_TRANSFORM_TRANSFORMS_PASSES
#define SODA_DIALECT_TRANSFORM_TRANSFORMS_PASSES

include "mlir/Dialect/Transform/IR/TransformDialect.td"
include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.td"
include "mlir/Dialect/PDL/IR/PDLTypes.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Pass/PassBase.td"

def TransformDialectEraseSchedule : Pass<"soda-transform-erase-schedule", "ModuleOp"> {
  let summary = "Erase the schedule of a transform dialect program";
  let description = [{
    Erase the schedule of a transform dialect program. This pass is used to
    prepare a program for lowering to LLVM.
  }];
}

#endif // SODA_DIALECT_TRANSFORM_TRANSFORMS_PASSES
