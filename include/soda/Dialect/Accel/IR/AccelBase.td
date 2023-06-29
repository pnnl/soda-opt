//===- AccelBase.td - Base definitions for accel dialect ----*- tablegen -*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef ACCEL_BASE
#define ACCEL_BASE
include "mlir/IR/OpBase.td"
def Accel_Dialect : Dialect {
  let name = "accel";
  let cppNamespace = "::mlir::accel";
  let description = [{
    The accel dialect is intended to hold accel operations that abstract
    AXI4MLIR DMA communciations.
  }];
}
#endif // ACCEL_BASE
