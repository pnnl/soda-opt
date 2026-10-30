//===- AccelPasses.h - Accel conversion passes ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Passes that lower linalg ops to the accel dialect and the accel dialect to
// the AXI DMA runtime library.
//
//===----------------------------------------------------------------------===//

#ifndef SODA_CONVERSION_ACCELPASSES_H
#define SODA_CONVERSION_ACCELPASSES_H

#include "mlir/Pass/Pass.h"

namespace mlir {
class ModuleOp;

namespace affine {
class AffineDialect;
} // namespace affine

namespace arith {
class ArithDialect;
} // namespace arith

namespace func {
class FuncDialect;
} // namespace func

namespace memref {
class MemRefDialect;
} // namespace memref

namespace scf {
class SCFDialect;
} // namespace scf

namespace accel {
class AccelDialect;

#define GEN_PASS_DECL
#include "soda/Conversion/AccelPasses.h.inc"

#define GEN_PASS_REGISTRATION
#include "soda/Conversion/AccelPasses.h.inc"

} // namespace accel
} // namespace mlir

#endif // SODA_CONVERSION_ACCELPASSES_H
