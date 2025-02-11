//===- OpcodeMap.h - MLIR Opcode Map Class ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Opcode maps are mathematical functions which map a list of dimension
// identifiers and symbols, to multidimensional opcode expressions.
//
//===----------------------------------------------------------------------===//

#ifndef SODA_DIALECT_ACCEL_IR_OPCODEMAP_H
#define SODA_DIALECT_ACCEL_IR_OPCODEMAP_H

#include "soda/Dialect/Accel/IR/OpcodeExpr.h"
#include "soda/Dialect/Accel/IR/OpcodeList.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMapInfo.h"

namespace llvm {
class SmallBitVector;
} // namespace llvm

namespace mlir {

namespace detail {
struct OpcodeMapStorage;
} // namespace detail

class Attribute;
class MLIRContext;

/// A dictionary of opcodes
class OpcodeMap {
public:
  using ImplType = detail::OpcodeMapStorage;

  constexpr OpcodeMap() = default;
  explicit OpcodeMap(ImplType *map) : map(map) {}

  /// Returns a zero result opcode map with no dimensions or symbols: () -> ().
  static OpcodeMap get(MLIRContext *context);

  /// Returns an opcode map with `dimCount` dimensions and `symbolCount` mapping
  /// to the given results.
  static OpcodeMap get(unsigned opcodeCount,
                       ArrayRef<std::tuple<StringRef, OpcodeList>> opcodes,
                       MLIRContext *context);

  /// Returns a vector of OpcodeMaps; each with as many results as
  /// `exprs.size()`, as many dims as the largest dim in `exprs` and as many
  /// symbols as the largest symbol in `exprs`.
  static SmallVector<OpcodeMap, 4>
  inferFromExprList(ArrayRef<ArrayRef<OpcodeExpr>> exprsList);
  static SmallVector<OpcodeMap, 4>
  inferFromExprList(ArrayRef<SmallVector<OpcodeExpr, 4>> exprsList);

  MLIRContext *getContext() const;

  explicit operator bool() const { return map != nullptr; }
  bool operator==(OpcodeMap other) const { return other.map == map; }
  bool operator!=(OpcodeMap other) const { return !(other.map == map); }

  /// Returns true if this opcode map is an empty map, i.e., () -> ().
  bool isEmpty() const;

  // Prints opcode map to 'os'.
  void print(raw_ostream &os) const;
  void dump() const;

  unsigned getNumOpcodes() const;
  ArrayRef<std::tuple<StringRef, OpcodeList>> getOpcodes() const;
  std::tuple<StringRef, OpcodeList> getOpcode(unsigned idx) const;
  OpcodeList getOpcodeList(StringRef key) const;
  unsigned getOpcodeListPosition(StringRef key) const;

  void walkExprs(llvm::function_ref<void(OpcodeExpr)> callback) const;

  /// Sparse replace method. Apply OpcodeExpr::replace(`expr`, `replacement`) to
  /// each of the results and return a new OpcodeMap with the new results and
  /// with the specified number of dims and symbols.
  OpcodeMap replace(OpcodeExpr expr, OpcodeExpr replacement,
                    unsigned numResultDims, unsigned numResultSyms) const;

  /// Returns the OpcodeMap resulting from composing `this` with `map`.
  OpcodeMap compose(OpcodeMap map) const;

  friend ::llvm::hash_code hash_value(OpcodeMap arg);

  /// Methods supporting C API.
  const void *getAsOpaquePointer() const {
    return static_cast<const void *>(map);
  }
  static OpcodeMap getFromOpaquePointer(const void *pointer) {
    return OpcodeMap(reinterpret_cast<ImplType *>(const_cast<void *>(pointer)));
  }

private:
  ImplType *map{nullptr};

  static OpcodeMap getImpl(unsigned dimCount, unsigned symbolCount,
                           ArrayRef<OpcodeExpr> results, MLIRContext *context);

  static OpcodeMap getImpl(unsigned opcodeCount, ArrayRef<StringRef> opcodes,
                           ArrayRef<OpcodeList> actions, MLIRContext *context);

  static OpcodeMap getImpl(unsigned opcodeCount,
                           ArrayRef<std::tuple<StringRef, OpcodeList>> opcodes,
                           MLIRContext *context);
};

// Make OpcodeExpr hashable.
inline ::llvm::hash_code hash_value(OpcodeMap arg) {
  return ::llvm::hash_value(arg.map);
}

// /// A mutable opcode map. Its opcode expressions are however unique.
// struct MutableOpcodeMap {
// public:
//   MutableOpcodeMap() = default;
//   MutableOpcodeMap(OpcodeMap map);

//   ArrayRef<OpcodeExpr> getResults() const { return results; }
//   OpcodeExpr getResult(unsigned idx) const { return results[idx]; }
//   void setResult(unsigned idx, OpcodeExpr result) { results[idx] = result; }
//   unsigned getNumResults() const { return results.size(); }
//   unsigned getNumDims() const { return numDims; }
//   void setNumDims(unsigned d) { numDims = d; }
//   unsigned getNumSymbols() const { return numSymbols; }
//   void setNumSymbols(unsigned d) { numSymbols = d; }
//   MLIRContext *getContext() const { return context; }

//   /// Returns true if the idx'th result expression is a multiple of factor.
//   bool isMultipleOf(unsigned idx, int64_t factor) const;

//   /// Resets this MutableOpcodeMap with 'map'.
//   void reset(OpcodeMap map);

//   /// Simplify the (result) expressions in this map using analysis (used by
//   //-simplify-opcode-expr pass).
//   void simplify();
//   /// Get the OpcodeMap corresponding to this MutableOpcodeMap. Note that an
//   /// OpcodeMap will be uniqued and stored in context, while a mutable one
//   /// isn't.
//   OpcodeMap getOpcodeMap() const;

// private:
//   // Same meaning as OpcodeMap's fields.
//   SmallVector<OpcodeExpr, 8> results;
//   unsigned numDims = 0;
//   unsigned numSymbols = 0;
//   /// A pointer to the IR's context to store all newly created
//   /// OpcodeExprStorage's.
//   MLIRContext *context = nullptr;
// };

/// Concatenates a list of `maps` into a single OpcodeMap, stepping over
/// potentially empty maps.
OpcodeMap concatOpcodeMaps(ArrayRef<OpcodeMap> maps);

inline raw_ostream &operator<<(raw_ostream &os, OpcodeMap map) {
  map.print(os);
  return os;
}
} // namespace mlir

namespace llvm {

// OpcodeExpr hash just like pointers
template <>
struct DenseMapInfo<mlir::OpcodeMap> {
  static mlir::OpcodeMap getEmptyKey() {
    auto *pointer = llvm::DenseMapInfo<void *>::getEmptyKey();
    return mlir::OpcodeMap(static_cast<mlir::OpcodeMap::ImplType *>(pointer));
  }
  static mlir::OpcodeMap getTombstoneKey() {
    auto *pointer = llvm::DenseMapInfo<void *>::getTombstoneKey();
    return mlir::OpcodeMap(static_cast<mlir::OpcodeMap::ImplType *>(pointer));
  }
  static unsigned getHashValue(mlir::OpcodeMap val) {
    return mlir::hash_value(val);
  }
  static bool isEqual(mlir::OpcodeMap LHS, mlir::OpcodeMap RHS) {
    return LHS == RHS;
  }
};

} // namespace llvm

#endif // SODA_DIALECT_ACCEL_IR_OPCODEMAP_H
