//===- OpcodeExprDetail.h - MLIR Opcode Expr storage details ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This holds implementation details of OpcodeExpr. Ideally it would not be
// exposed and would be kept local to OpcodeExpr.cpp however, MLIRContext.cpp
// needs to know the sizes for placement-new style Allocation.
//
//===----------------------------------------------------------------------===//
#ifndef MLIR_IR_OPCODEEXPRDETAIL_H_
#define MLIR_IR_OPCODEEXPRDETAIL_H_

#include "mlir/IR/MLIRContext.h"
#include "soda/Dialect/Accel/IR/OpcodeExpr.h"
#include "mlir/Support/StorageUniquer.h"

namespace mlir {

class MLIRContext;

namespace detail {

/// Base storage class appearing in an affine expression.
struct OpcodeExprStorage : public StorageUniquer::BaseStorage {
  MLIRContext *context;
  OpcodeExprKind kind;
};

/// A send(id) or recv(id) expression appearing in an opcode expression.
struct OpcodeSendRecvIdExprStorage : public OpcodeExprStorage {
  using KeyTy = std::tuple<unsigned, unsigned>;

  bool operator==(const KeyTy &key) const {
    return static_cast<OpcodeExprKind>(std::get<0>(key)) == kind &&
           std::get<1>(key) == id;
  }

  static OpcodeSendRecvIdExprStorage *
  construct(StorageUniquer::StorageAllocator &allocator, const KeyTy &key) {
    auto *result = allocator.allocate<OpcodeSendRecvIdExprStorage>();
    result->kind = static_cast<OpcodeExprKind>(std::get<0>(key));
    result->id = std::get<1>(key);
    return result;
  }

  unsigned id;
};

/// A send_*(id, pos) expression appearing in an opcode expression.
/// This supports: send_idx(id, pos), send_dim(id, pos)
struct OpcodeSendIdPosExprStorage : public OpcodeExprStorage {
  using KeyTy = std::tuple<unsigned, unsigned, unsigned>;

  bool operator==(const KeyTy &key) const {
    return static_cast<OpcodeExprKind>(std::get<0>(key)) == kind &&
           std::get<1>(key) == id && std::get<2>(key) == pos;
  }

  static OpcodeSendIdPosExprStorage *
  construct(StorageUniquer::StorageAllocator &allocator, const KeyTy &key) {
    auto *result = allocator.allocate<OpcodeSendIdPosExprStorage>();
    result->kind = static_cast<OpcodeExprKind>(std::get<0>(key));
    result->id = std::get<1>(key);
    result->pos = std::get<2>(key);
    return result;
  }

  unsigned id;
  unsigned pos;
};

/// A send_literal(integer_literal) expression appearing in an opcode
/// expression. This supports: send_literal(v: i32)
/// TODO: Make it support literals of different bitwidths
struct OpcodeSendLiteralExprStorage : public OpcodeExprStorage {
  using KeyTy = std::tuple<unsigned, int>;

  bool operator==(const KeyTy &key) const {
    return static_cast<OpcodeExprKind>(std::get<0>(key)) == kind &&
           std::get<1>(key) == value;
  }

  static OpcodeSendLiteralExprStorage *
  construct(StorageUniquer::StorageAllocator &allocator, const KeyTy &key) {
    auto *result = allocator.allocate<OpcodeSendLiteralExprStorage>();
    result->kind = static_cast<OpcodeExprKind>(std::get<0>(key));
    result->value = std::get<1>(key);
    return result;
  }

  int value;
};

/// A binary operation appearing in an affine expression.
struct OpcodeBinaryOpExprStorage : public OpcodeExprStorage {
  using KeyTy = std::tuple<unsigned, OpcodeExpr, OpcodeExpr>;

  bool operator==(const KeyTy &key) const {
    return static_cast<OpcodeExprKind>(std::get<0>(key)) == kind &&
           std::get<1>(key) == lhs && std::get<2>(key) == rhs;
  }

  static OpcodeBinaryOpExprStorage *
  construct(StorageUniquer::StorageAllocator &allocator, const KeyTy &key) {
    auto *result = allocator.allocate<OpcodeBinaryOpExprStorage>();
    result->kind = static_cast<OpcodeExprKind>(std::get<0>(key));
    result->lhs = std::get<1>(key);
    result->rhs = std::get<2>(key);
    result->context = result->lhs.getContext();
    return result;
  }

  OpcodeExpr lhs;
  OpcodeExpr rhs;
};

/// A dimensional or symbolic identifier appearing in an affine expression.
struct OpcodeDimExprStorage : public OpcodeExprStorage {
  using KeyTy = std::pair<unsigned, unsigned>;

  bool operator==(const KeyTy &key) const {
    return kind == static_cast<OpcodeExprKind>(key.first) &&
           position == key.second;
  }

  static OpcodeDimExprStorage *
  construct(StorageUniquer::StorageAllocator &allocator, const KeyTy &key) {
    auto *result = allocator.allocate<OpcodeDimExprStorage>();
    result->kind = static_cast<OpcodeExprKind>(key.first);
    result->position = key.second;
    return result;
  }

  /// Position of this identifier in the argument list.
  unsigned position;
};

/// An integer constant appearing in affine expression.
struct OpcodeConstantExprStorage : public OpcodeExprStorage {
  using KeyTy = int64_t;

  bool operator==(const KeyTy &key) const { return constant == key; }

  static OpcodeConstantExprStorage *
  construct(StorageUniquer::StorageAllocator &allocator, const KeyTy &key) {
    auto *result = allocator.allocate<OpcodeConstantExprStorage>();
    result->kind = OpcodeExprKind::Constant;
    result->constant = key;
    return result;
  }

  // The constant.
  int64_t constant;
};

} // namespace detail
} // namespace mlir
#endif // MLIR_IR_OPCODEEXPRDETAIL_H_
