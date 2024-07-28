//===- AccelDialect.cpp - MLIR dialect for Accel implementation -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Transforms/InliningUtils.h"
#include "soda/Dialect/Accel/IR/Accel.h"

using namespace mlir;
using namespace mlir::accel;

#include "soda/Dialect/Accel/IR/AccelOpsDialect.cpp.inc"

namespace {
/// This class defines the interface for handling inlining with accel
/// operations.
struct AccelInlinerInterface : public DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;

  /// All operations within accel ops can be inlined.
  bool isLegalToInline(Operation *, Region *, bool, IRMapping &) const final {
    return true;
  }
};
} // namespace

void mlir::accel::AccelDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "soda/Dialect/Accel/IR/AccelOps.cpp.inc"
      >();
  addInterfaces<AccelInlinerInterface>();
}
