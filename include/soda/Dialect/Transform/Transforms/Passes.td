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
include "mlir/Dialect/Transform/IR/TransformEffects.td"
include "mlir/Dialect/Transform/IR/TransformInterfaces.td"
include "mlir/Dialect/PDL/IR/PDLTypes.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Dialect/PDL/IR/PDLTypes.td"
include "mlir/IR/FunctionInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Pass/PassBase.td"

def TransformDialectInterpreter : Pass<"soda-transform-interpreter", "ModuleOp"> {
  let summary = "Interpret a transform dialect program";
  let description = [{
    Interpret a transform dialect program. This pass is used to test the
    interpreter.
  }];
  
  let options = [
    Option<"enableExpensiveChecks", "enable-expensive-checks", "bool",
           /*default=*/"false",
           "Enable expensive checks in the pass">,
  ];
}

def TransformDialectEraseSchedule : Pass<"soda-transform-erase-schedule", "ModuleOp"> {
  let summary = "Erase the schedule of a transform dialect program";
  let description = [{
    Erase the schedule of a transform dialect program. This pass is used to
    prepare a program for lowering to LLVM.
  }];
}

#endif // SODA_DIALECT_TRANSFORM_TRANSFORMS_PASSES
