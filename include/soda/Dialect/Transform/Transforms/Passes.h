//===- CheckUses.h - Expensive transform value validity checks --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SODA_DIALECT_TRANSFORM_TRANSFORMS_PASSES_H
#define SODA_DIALECT_TRANSFORM_TRANSFORMS_PASSES_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
class Pass;
namespace soda {

namespace trans {
#define GEN_PASS_DECL
#include "soda/Dialect/Transform/Transforms/Passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "soda/Dialect/Transform/Transforms/Passes.h.inc"
} // namespace transform
}
} // namespace mlir

#endif // SODA_DIALECT_TRANSFORM_TRANSFORMS_PASSES_H
