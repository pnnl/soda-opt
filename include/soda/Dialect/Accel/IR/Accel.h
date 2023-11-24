//===- Accel.h - Accel dialect ------------------------------------*- C++-*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SODA_DIALECT_ACCEL_IR_ACCEL_H_
#define SODA_DIALECT_ACCEL_IR_ACCEL_H_

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/VectorInterfaces.h"

//===----------------------------------------------------------------------===//
// Accel Dialect
//===----------------------------------------------------------------------===//

#include "soda/Dialect/Accel/IR/AccelOpsDialect.h.inc"

//===----------------------------------------------------------------------===//
// Accel Dialect Operations
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "soda/Dialect/Accel/IR/AccelOps.h.inc"

#endif // SODA_DIALECT_ACCEL_IR_ACCEL_H_
