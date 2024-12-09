//===- OpcodeExpr.h - MLIR Opcode Expr Class --------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// An affine expression is an affine combination of dimension identifiers and
// symbols, including ceildiv/floordiv/mod by a constant integer.
//
//===----------------------------------------------------------------------===//

#ifndef SODA_DIALECT_ACCEL_IR_OPCODEEXPR_H
#define SODA_DIALECT_ACCEL_IR_OPCODEEXPR_H

#include "mlir/Support/LLVM.h"
#include "llvm/ADT/DenseMapInfo.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/Support/Casting.h"
#include <functional>
#include <type_traits>

namespace mlir {

class MLIRContext;
class OpcodeMap;
class IntegerSet;

namespace detail {

struct OpcodeExprStorage;
struct OpcodeBinaryOpExprStorage;
struct OpcodeDimExprStorage;
struct OpcodeSymbolExprStorage;
struct OpcodeConstantExprStorage;
struct OpcodeSendRecvIdExprStorage;
struct OpcodeSendIdPosExprStorage;
struct OpcodeSendLiteralExprStorage;

} // namespace detail

enum class OpcodeExprKind {
  Add,
  /// RHS of mul is always a constant or a symbolic expression.
  Mul,
  /// RHS of mod is always a constant or a symbolic expression with a positive
  /// value.
  Mod,
  /// RHS of floordiv is always a constant or a symbolic expression.
  FloorDiv,
  /// RHS of ceildiv is always a constant or a symbolic expression.
  CeilDiv,

  /// This is a marker for the last affine binary op. The range of binary
  /// op's is expected to be this element and earlier.
  LAST_OPCODE_BINARY_OP = CeilDiv,

  /// Constant integer.
  Constant,
  /// Dimensional identifier.
  DimId,
  /// Symbolic identifier.
  SymbolId,

  Send,
  SendLiteral,
  SendDim,
  SendIdx,
  Recv
};

/// Base type for opcode expression.
/// OpcodeExpr's are immutable value types with intuitive operators to
/// operate on chainable, lightweight compositions.
/// An OpcodeExpr is an interface to the underlying storage type pointer.
class OpcodeExpr {
public:
  using ImplType = detail::OpcodeExprStorage;

  constexpr OpcodeExpr() {}
  /* implicit */ OpcodeExpr(const ImplType *expr)
      : expr(const_cast<ImplType *>(expr)) {}

  bool operator==(OpcodeExpr other) const { return expr == other.expr; }
  bool operator!=(OpcodeExpr other) const { return !(*this == other); }
  bool operator==(int64_t v) const;
  bool operator!=(int64_t v) const { return !(*this == v); }
  explicit operator bool() const { return expr; }

  bool operator!() const { return expr == nullptr; }

  template <typename U>
  bool isa() const;
  template <typename U>
  U dyn_cast() const;
  template <typename U>
  U dyn_cast_or_null() const;
  template <typename U>
  U cast() const;

  MLIRContext *getContext() const;

  /// Return the classification for this type.
  OpcodeExprKind getKind() const;

  void print(raw_ostream &os) const;
  void dump() const;

  /// Returns true if this expression is made out of only symbols and
  /// constants, i.e., it does not involve dimensional identifiers.
  bool isSymbolicOrConstant() const;

  /// Returns true if this is a pure opcode expression, i.e., multiplication,
  /// floordiv, ceildiv, and mod is only allowed w.r.t constants.
  bool isPureOpcode() const;

  /// Returns the greatest known integral divisor of this opcode expression. The
  /// result is always positive.
  int64_t getLargestKnownDivisor() const;

  /// Return true if the opcode expression is a multiple of 'factor'.
  bool isMultipleOf(int64_t factor) const;

  /// Return true if the opcode expression involves OpcodeDimExpr `position`.
  bool isFunctionOfDim(unsigned position) const;

  /// Return true if the opcode expression involves OpcodeSymbolExpr `position`.
  bool isFunctionOfSymbol(unsigned position) const;

  /// Walk all of the OpcodeExpr's in this expression in postorder.
  void walk(std::function<void(OpcodeExpr)> callback) const;

  /// This method substitutes any uses of dimensions and symbols (e.g.
  /// dim#0 with dimReplacements[0]) and returns the modified expression tree.
  /// This is a dense replacement method: a replacement must be specified for
  /// every single dim and symbol.
  OpcodeExpr replaceDimsAndSymbols(ArrayRef<OpcodeExpr> dimReplacements,
                                   ArrayRef<OpcodeExpr> symReplacements) const;

  /// Dim-only version of replaceDimsAndSymbols.
  OpcodeExpr replaceDims(ArrayRef<OpcodeExpr> dimReplacements) const;

  /// Symbol-only version of replaceDimsAndSymbols.
  OpcodeExpr replaceSymbols(ArrayRef<OpcodeExpr> symReplacements) const;

  /// Sparse replace method. Replace `expr` by `replacement` and return the
  /// modified expression tree.
  OpcodeExpr replace(OpcodeExpr expr, OpcodeExpr replacement) const;

  /// Sparse replace method. If `*this` appears in `map` replaces it by
  /// `map[*this]` and return the modified expression tree. Otherwise traverse
  /// `*this` and apply replace with `map` on its subexpressions.
  OpcodeExpr replace(const DenseMap<OpcodeExpr, OpcodeExpr> &map) const;

  /// Replace dims[offset ... numDims)
  /// by dims[offset + shift ... shift + numDims).
  OpcodeExpr shiftDims(unsigned numDims, unsigned shift,
                       unsigned offset = 0) const;

  /// Replace symbols[offset ... numSymbols)
  /// by symbols[offset + shift ... shift + numSymbols).
  OpcodeExpr shiftSymbols(unsigned numSymbols, unsigned shift,
                          unsigned offset = 0) const;

  OpcodeExpr operator+(int64_t v) const;
  OpcodeExpr operator+(OpcodeExpr other) const;
  OpcodeExpr operator-() const;
  OpcodeExpr operator-(int64_t v) const;
  OpcodeExpr operator-(OpcodeExpr other) const;
  OpcodeExpr operator*(int64_t v) const;
  OpcodeExpr operator*(OpcodeExpr other) const;
  OpcodeExpr floorDiv(uint64_t v) const;
  OpcodeExpr floorDiv(OpcodeExpr other) const;
  OpcodeExpr ceilDiv(uint64_t v) const;
  OpcodeExpr ceilDiv(OpcodeExpr other) const;
  OpcodeExpr operator%(uint64_t v) const;
  OpcodeExpr operator%(OpcodeExpr other) const;

  /// Compose with an OpcodeMap.
  /// Returns the composition of this OpcodeExpr with `map`.
  ///
  /// Prerequisites:
  /// `this` and `map` are composable, i.e. that the number of OpcodeDimExpr of
  /// `this` is smaller than the number of results of `map`. If a result of a
  /// map does not have a corresponding OpcodeDimExpr, that result simply does
  /// not appear in the produced OpcodeExpr.
  ///
  /// Example:
  ///   expr: `d0 + d2`
  ///   map:  `(d0, d1, d2)[s0, s1] -> (d0 + s1, d1 + s0, d0 + d1 + d2)`
  ///   returned expr: `d0 * 2 + d1 + d2 + s1`
  OpcodeExpr compose(OpcodeMap map) const;

  friend ::llvm::hash_code hash_value(OpcodeExpr arg);

  /// Methods supporting C API.
  const void *getAsOpaquePointer() const {
    return static_cast<const void *>(expr);
  }
  static OpcodeExpr getFromOpaquePointer(const void *pointer) {
    return OpcodeExpr(
        reinterpret_cast<ImplType *>(const_cast<void *>(pointer)));
  }

protected:
  ImplType *expr{nullptr};
};

/// Opcode binary operation expression. An affine binary operation could be an
/// add, mul, floordiv, ceildiv, or a modulo operation. (Subtraction is
/// represented through a multiply by -1 and add.) These expressions are always
/// constructed in a simplified form. For eg., the LHS and RHS operands can't
/// both be constants. There are additional canonicalizing rules depending on
/// the op type: see checks in the constructor.
class OpcodeBinaryOpExpr : public OpcodeExpr {
public:
  using ImplType = detail::OpcodeBinaryOpExprStorage;
  /* implicit */ OpcodeBinaryOpExpr(OpcodeExpr::ImplType *ptr);
  OpcodeExpr getLHS() const;
  OpcodeExpr getRHS() const;
};

/// A dimensional identifier appearing in an affine expression.
class OpcodeDimExpr : public OpcodeExpr {
public:
  using ImplType = detail::OpcodeDimExprStorage;
  /* implicit */ OpcodeDimExpr(OpcodeExpr::ImplType *ptr);
  unsigned getPosition() const;
};

/// A symbolic identifier appearing in an opcode expression.
class OpcodeSymbolExpr : public OpcodeExpr {
public:
  using ImplType = detail::OpcodeDimExprStorage;
  /* implicit */ OpcodeSymbolExpr(OpcodeExpr::ImplType *ptr);
  unsigned getPosition() const;
};

/// A symbolic identifier appearing in an opcode expression.
class OpcodeSendIdExpr : public OpcodeExpr {
public:
  using ImplType = detail::OpcodeSendRecvIdExprStorage;
  /* implicit */ OpcodeSendIdExpr(OpcodeExpr::ImplType *ptr = nullptr);
  unsigned getId() const;
};

/// A symbolic identifier appearing in an opcode expression.
class OpcodeRecvIdExpr : public OpcodeExpr {
public:
  using ImplType = detail::OpcodeSendRecvIdExprStorage;
  /* implicit */ OpcodeRecvIdExpr(OpcodeExpr::ImplType *ptr = nullptr);
  unsigned getId() const;
};

/// A symbolic identifier appearing in an opcode expression.
class OpcodeSendDimExpr : public OpcodeExpr {
public:
  using ImplType = detail::OpcodeSendIdPosExprStorage;
  /* implicit */ OpcodeSendDimExpr(OpcodeExpr::ImplType *ptr = nullptr);
  unsigned getId() const;
  unsigned getPosition() const;
};

/// A symbolic identifier appearing in an opcode expression.
class OpcodeSendIdxExpr : public OpcodeExpr {
public:
  using ImplType = detail::OpcodeSendIdPosExprStorage;
  /* implicit */ OpcodeSendIdxExpr(OpcodeExpr::ImplType *ptr = nullptr);
  unsigned getId() const;
  unsigned getPosition() const;
};

/// An integer constant appearing in opcode expression.
class OpcodeSendLiteralExpr : public OpcodeExpr {
public:
  using ImplType = detail::OpcodeSendLiteralExprStorage;
  /* implicit */ OpcodeSendLiteralExpr(OpcodeExpr::ImplType *ptr = nullptr);
  int getValue() const;
};

/// An integer constant appearing in affine expression.
class OpcodeConstantExpr : public OpcodeExpr {
public:
  using ImplType = detail::OpcodeConstantExprStorage;
  /* implicit */ OpcodeConstantExpr(OpcodeExpr::ImplType *ptr = nullptr);
  int64_t getValue() const;
};

/// Make OpcodeExpr hashable.
inline ::llvm::hash_code hash_value(OpcodeExpr arg) {
  return ::llvm::hash_value(arg.expr);
}

inline OpcodeExpr operator+(int64_t val, OpcodeExpr expr) { return expr + val; }
inline OpcodeExpr operator*(int64_t val, OpcodeExpr expr) { return expr * val; }
inline OpcodeExpr operator-(int64_t val, OpcodeExpr expr) {
  return expr * (-1) + val;
}

/// These free functions allow clients of the API to not use classes in detail.
OpcodeExpr getOpcodeDimExpr(unsigned position, MLIRContext *context);
OpcodeExpr getOpcodeSymbolExpr(unsigned position, MLIRContext *context);
OpcodeExpr getOpcodeConstantExpr(int64_t constant, MLIRContext *context);
OpcodeExpr getOpcodeBinaryOpExpr(OpcodeExprKind kind, OpcodeExpr lhs,
                                 OpcodeExpr rhs);

OpcodeExpr getOpcodeSendExpr(unsigned id, MLIRContext *context);
OpcodeExpr getOpcodeRecvExpr(unsigned id, MLIRContext *context);
OpcodeExpr getOpcodeSendLiteralExpr(int literal, MLIRContext *context);
OpcodeExpr getOpcodeSendDimExpr(unsigned id, unsigned position,
                                MLIRContext *context);
OpcodeExpr getOpcodeSendIdxExpr(unsigned id, unsigned position,
                                MLIRContext *context);

/// Constructs an affine expression from a flat ArrayRef. If there are local
/// identifiers (neither dimensional nor symbolic) that appear in the sum of
/// products expression, 'localExprs' is expected to have the OpcodeExpr
/// for it, and is substituted into. The ArrayRef 'eq' is expected to be in the
/// format [dims, symbols, locals, constant term].
OpcodeExpr getOpcodeExprFromFlatForm(ArrayRef<int64_t> flatExprs,
                                     unsigned numDims, unsigned numSymbols,
                                     ArrayRef<OpcodeExpr> localExprs,
                                     MLIRContext *context);

raw_ostream &operator<<(raw_ostream &os, OpcodeExpr expr);

template <typename U>
bool OpcodeExpr::isa() const {
  if (std::is_same<U, OpcodeBinaryOpExpr>::value)
    return getKind() <= OpcodeExprKind::LAST_OPCODE_BINARY_OP;
  if (std::is_same<U, OpcodeDimExpr>::value)
    return getKind() == OpcodeExprKind::DimId;
  if (std::is_same<U, OpcodeSymbolExpr>::value)
    return getKind() == OpcodeExprKind::SymbolId;
  if (std::is_same<U, OpcodeSymbolExpr>::value)
    return getKind() == OpcodeExprKind::SymbolId;
  if (std::is_same<U, OpcodeConstantExpr>::value)
    return getKind() == OpcodeExprKind::Constant;
  if (std::is_same<U, OpcodeSendIdExpr>::value)
    return getKind() == OpcodeExprKind::Send;
  if (std::is_same<U, OpcodeRecvIdExpr>::value)
    return getKind() == OpcodeExprKind::Recv;
  if (std::is_same<U, OpcodeSendDimExpr>::value)
    return getKind() == OpcodeExprKind::SendDim;
  if (std::is_same<U, OpcodeSendIdxExpr>::value)
    return getKind() == OpcodeExprKind::SendIdx;
  if (std::is_same<U, OpcodeSendLiteralExpr>::value)
    return getKind() == OpcodeExprKind::SendLiteral;
}
template <typename U>
U OpcodeExpr::dyn_cast() const {
  if (isa<U>())
    return U(expr);
  return U(nullptr);
}
template <typename U>
U OpcodeExpr::dyn_cast_or_null() const {
  return (!*this || !isa<U>()) ? U(nullptr) : U(expr);
}
template <typename U>
U OpcodeExpr::cast() const {
  assert(isa<U>());
  return U(expr);
}

/// Simplify an affine expression by flattening and some amount of simple
/// analysis. This has complexity linear in the number of nodes in 'expr'.
/// Returns the simplified expression, which is the same as the input expression
/// if it can't be simplified. When `expr` is semi-affine, a simplified
/// semi-affine expression is constructed in the sorted order of dimension and
/// symbol positions.
OpcodeExpr simplifyOpcodeExpr(OpcodeExpr expr, unsigned numDims,
                              unsigned numSymbols);

// namespace detail {
// template <int N>
// void bindDims(MLIRContext *ctx) {}

// template <int N, typename OpcodeExprTy, typename... OpcodeExprTy2>
// void bindDims(MLIRContext *ctx, OpcodeExprTy &e, OpcodeExprTy2 &...exprs) {
//   e = getOpcodeDimExpr(N, ctx);
//   bindDims<N + 1, OpcodeExprTy2 &...>(ctx, exprs...);
// }

// template <int N>
// void bindSymbols(MLIRContext *ctx) {}

// template <int N, typename OpcodeExprTy, typename... OpcodeExprTy2>
// void bindSymbols(MLIRContext *ctx, OpcodeExprTy &e, OpcodeExprTy2 &...exprs)
// {
//   e = getOpcodeSymbolExpr(N, ctx);
//   bindSymbols<N + 1, OpcodeExprTy2 &...>(ctx, exprs...);
// }
// } // namespace detail

// /// Bind a list of OpcodeExpr references to DimExpr at positions:
// ///   [0 .. sizeof...(exprs)]
// template <typename... OpcodeExprTy>
// void bindDims(MLIRContext *ctx, OpcodeExprTy &...exprs) {
//   detail::bindDims<0>(ctx, exprs...);
// }

// /// Bind a list of OpcodeExpr references to SymbolExpr at positions:
// ///   [0 .. sizeof...(exprs)]
// template <typename... OpcodeExprTy>
// void bindSymbols(MLIRContext *ctx, OpcodeExprTy &...exprs) {
//   detail::bindSymbols<0>(ctx, exprs...);
// }

} // namespace mlir

namespace llvm {

// OpcodeExpr hash just like pointers
template <>
struct DenseMapInfo<mlir::OpcodeExpr> {
  static mlir::OpcodeExpr getEmptyKey() {
    auto *pointer = llvm::DenseMapInfo<void *>::getEmptyKey();
    return mlir::OpcodeExpr(static_cast<mlir::OpcodeExpr::ImplType *>(pointer));
  }
  static mlir::OpcodeExpr getTombstoneKey() {
    auto *pointer = llvm::DenseMapInfo<void *>::getTombstoneKey();
    return mlir::OpcodeExpr(static_cast<mlir::OpcodeExpr::ImplType *>(pointer));
  }
  static unsigned getHashValue(mlir::OpcodeExpr val) {
    return mlir::hash_value(val);
  }
  static bool isEqual(mlir::OpcodeExpr LHS, mlir::OpcodeExpr RHS) {
    return LHS == RHS;
  }
};

} // namespace llvm

#endif // SODA_DIALECT_ACCEL_IR_OPCODEEXPR_H
