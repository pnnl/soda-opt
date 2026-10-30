//===- AccelOps.cpp - MLIR operations for accel implementation ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Builders.h"
#include "soda/Dialect/Accel/IR/Accel.h"

using namespace mlir;
using namespace mlir::accel;

template <typename OpTy>
static LogicalResult verifyTileInfoOp(OpTy op) {
  int64_t rank = op.getTile().getType().getRank();
  if (static_cast<int64_t>(op.getDim()) >= rank)
    return op.emitOpError() << "dimension " << op.getDim()
                            << " is out of range for a memref of rank "
                            << rank;
  return success();
}

LogicalResult SendDimOp::verify() { return verifyTileInfoOp(*this); }

LogicalResult SendIdxOp::verify() { return verifyTileInfoOp(*this); }

LogicalResult RecvOp::verify() {
  Type elementType = getDst().getType().getElementType();
  if (isAccumulate() && !isa<IntegerType, FloatType>(elementType))
    return emitOpError() << "accumulate mode needs an integer or float "
                            "element type, got "
                         << elementType;
  return success();
}

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "soda/Dialect/Accel/IR/AccelOps.cpp.inc"
