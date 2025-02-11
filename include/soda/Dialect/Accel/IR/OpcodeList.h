//===- OpcodeList.h - MLIR Opcode List Class -------------------*- C++ -*--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Opcode lists are mathematical functions which list a list of dimension
// identifiers and symbols, to multidimensional opcode expressions.
//
//===----------------------------------------------------------------------===//

#ifndef SODA_DIALECT_ACCEL_IR_OPCODELIST_H
#define SODA_DIALECT_ACCEL_IR_OPCODELIST_H

#include "soda/Dialect/Accel/IR/OpcodeExpr.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMapInfo.h"

namespace llvm {
class SmallBitVector;
} // namespace llvm

namespace mlir {

namespace detail {
struct OpcodeListStorage;
} // namespace detail

class Attribute;
class MLIRContext;

/// A dictionary of opcodes
class OpcodeList {
public:
  using ImplType = detail::OpcodeListStorage;

  using value_type = ArrayRef<OpcodeExpr>;
  using pointer = value_type *;
  using const_pointer = const value_type *;
  using reference = value_type &;
  using const_reference = const value_type &;
  using iterator = pointer;
  using const_iterator = const_pointer;
  using reverse_iterator = std::reverse_iterator<iterator>;
  using const_reverse_iterator = std::reverse_iterator<const_iterator>;

  constexpr OpcodeList() = default;
  explicit OpcodeList(ImplType *list) : list(list) {}

  /// Returns an empty opcode list
  static OpcodeList get(MLIRContext *context);

  /// Returns an opcode list with `actionsCount` internal expressions
  static OpcodeList get(unsigned actionsCount, ArrayRef<OpcodeExpr> exprs,
                        MLIRContext *context);

  MLIRContext *getContext() const;

  explicit operator bool() const { return list != nullptr; }
  bool operator==(OpcodeList other) const { return other.list == list; }
  bool operator!=(OpcodeList other) const { return !(other.list == list); }

  /// Returns true if this opcode list is an empty list, i.e., () -> ().
  bool isEmpty() const;

  // Prints opcode list to 'os'.
  void print(raw_ostream &os) const;
  void dump() const;

  unsigned getNumActions() const;

  ArrayRef<OpcodeExpr> getActions() const;
  OpcodeExpr getAction(unsigned idx) const;

  /// Walk all of the OpcodeExpr's in this listping. Each node in an expression
  /// tree is visited in postorder.
  void walkExprs(llvm::function_ref<void(OpcodeExpr)> callback) const;

  // /// Sparse replace method. Apply OpcodeExpr::replace(`expr`, `replacement`)
  // to
  // /// each of the results and return a new OpcodeList with the new results
  // and
  // /// with the specified number of dims and symbols.
  // OpcodeList replace(OpcodeExpr expr, OpcodeExpr replacement,
  //                   unsigned numResultDims, unsigned numResultSyms) const;

  // /// Sparse replace method. Apply OpcodeExpr::replace(`list`) to each of the
  // /// results and return a new OpcodeList with the new results and with
  // inferred
  // /// number of dims and symbols.
  // OpcodeList replace(const DenseMap<OpcodeExpr, OpcodeExpr> &list) const;

  // /// Sparse replace method. Apply OpcodeExpr::replace(`list`) to each of the
  // /// results and return a new OpcodeList with the new results and with the
  // /// specified number of dims and symbols.
  // OpcodeList replace(const DenseMap<OpcodeExpr, OpcodeExpr> &list,
  //                   unsigned numResultDims, unsigned numResultSyms) const;

  friend ::llvm::hash_code hash_value(OpcodeList arg);

  iterator begin() const;
  iterator end() const;

  reverse_iterator rbegin() const;
  reverse_iterator rend() const;

  /// Methods supporting C API.
  const void *getAsOpaquePointer() const {
    return static_cast<const void *>(list);
  }
  static OpcodeList getFromOpaquePointer(const void *pointer) {
    return OpcodeList(
        reinterpret_cast<ImplType *>(const_cast<void *>(pointer)));
  }

private:
  ImplType *list{nullptr};

  static OpcodeList getImpl(unsigned actionCount, ArrayRef<OpcodeExpr> exprs,
                            MLIRContext *context);
};

// Make OpcodeExpr hashable.
inline ::llvm::hash_code hash_value(OpcodeList arg) {
  return ::llvm::hash_value(arg.list);
}

// /// A mutable opcode list. Its opcode expressions are however unique.
// struct MutableOpcodeList {
// public:
//   MutableOpcodeList() = default;
//   MutableOpcodeList(OpcodeList list);

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

//   /// Resets this MutableOpcodeList with 'list'.
//   void reset(OpcodeList list);

//   /// Simplify the (result) expressions in this list using analysis (used by
//   //-simplify-opcode-expr pass).
//   void simplify();
//   /// Get the OpcodeList corresponding to this MutableOpcodeList. Note that
//   an
//   /// OpcodeList will be uniqued and stored in context, while a mutable one
//   /// isn't.
//   OpcodeList getOpcodeList() const;

// private:
//   // Same meaning as OpcodeList's fields.
//   SmallVector<OpcodeExpr, 8> results;
//   unsigned numDims = 0;
//   unsigned numSymbols = 0;
//   /// A pointer to the IR's context to store all newly created
//   /// OpcodeExprStorage's.
//   MLIRContext *context = nullptr;
// };

/// Concatenates a list of `lists` into a single OpcodeList, stepping over
/// potentially empty lists. Assumes each of the underlying list has 0 symbols.
/// The resulting list has a number of dims equal to the max of `lists`' dims
/// and the concatenated results as its results. Returns an empty list if all
/// input `lists` are empty.
///
/// Example:
/// When applied to the following list of 3 opcode lists,
///
/// ```mlir
///    {
///      (i, j, k) -> (i, k),
///      (i, j, k) -> (k, j),
///      (i, j, k) -> (i, j)
///    }
/// ```
///
/// Returns the list:
///
/// ```mlir
///     (i, j, k) -> (i, k, k, j, i, j)
/// ```
OpcodeList concatOpcodeLists(ArrayRef<OpcodeList> lists);

inline raw_ostream &operator<<(raw_ostream &os, OpcodeList list) {
  list.print(os);
  return os;
}
} // namespace mlir

namespace llvm {

// OpcodeExpr hash just like pointers
template <>
struct DenseMapInfo<mlir::OpcodeList> {
  static mlir::OpcodeList getEmptyKey() {
    auto *pointer = llvm::DenseMapInfo<void *>::getEmptyKey();
    return mlir::OpcodeList(static_cast<mlir::OpcodeList::ImplType *>(pointer));
  }
  static mlir::OpcodeList getTombstoneKey() {
    auto *pointer = llvm::DenseMapInfo<void *>::getTombstoneKey();
    return mlir::OpcodeList(static_cast<mlir::OpcodeList::ImplType *>(pointer));
  }
  static unsigned getHashValue(mlir::OpcodeList val) {
    return mlir::hash_value(val);
  }
  static bool isEqual(mlir::OpcodeList LHS, mlir::OpcodeList RHS) {
    return LHS == RHS;
  }
};

} // namespace llvm

#endif // SODA_DIALECT_ACCEL_IR_OPCODELIST_H
