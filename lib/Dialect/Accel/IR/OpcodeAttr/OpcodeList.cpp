//===- OpcodeList.cpp - MLIR Opcode List Classes --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "soda/Dialect/Accel/IR/OpcodeList.h"
#include "../OpcodeListDetail.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Support/MathExtras.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

MLIRContext *OpcodeList::getContext() const { return list->context; }

bool OpcodeList::isEmpty() const { return getNumActions() == 0; }

OpcodeList::iterator OpcodeList::begin() const {
  // Warning: is this alive long enough?
  auto actions = list->results();
  return &actions;
}
OpcodeList::iterator OpcodeList::end() const {
  auto actions = list->results();
  return &actions + list->numActions;
}

OpcodeList::reverse_iterator OpcodeList::rbegin() const {
  return OpcodeList::reverse_iterator(end());
}
OpcodeList::reverse_iterator OpcodeList::rend() const {
  return OpcodeList::reverse_iterator(begin());
}

unsigned OpcodeList::getNumActions() const {
  assert(list && "uninitialized list storage");
  return list->numActions;
}

ArrayRef<OpcodeExpr> OpcodeList::getActions() const {
  assert(list && "uninitialized list storage");
  return list->results(); // TODO: Results should be renamed
}
OpcodeExpr OpcodeList::getAction(unsigned idx) const {
  return getActions()[idx];
}

/// Walk all of the OpcodeExpr's in this listping. Each node in an expression
/// tree is visited in postorder.
void OpcodeList::walkExprs(
    llvm::function_ref<void(OpcodeExpr)> callback) const {
  for (auto expr : getActions())
    expr.walk(callback);
}

OpcodeList mlir::concatOpcodeLists(ArrayRef<OpcodeList> lists) {
  // TODO: Implement this
  // unsigned newNumResults = 0, numActions = 0;
  // for (auto m : lists)
  //   newNumResults += m.getNumActions();
  // SmallVector<OpcodeExpr, 8> results;
  // results.reserve(newNumResults);
  // for (auto m : lists) {
  //   for (auto res : m.getActions()) {
  //     // TODO: must push_back the arriving expressions saved in the lists
  //     // TODO: this requires a different function other than shiftSymbols
  //     results.push_back(res.shiftSymbols(0, numActions));
  //   }
  // }
  return OpcodeList::get(lists.front().getContext());
}

// //===----------------------------------------------------------------------===//
// // MutableOpcodeList.
// //===----------------------------------------------------------------------===//

// MutableOpcodeList::MutableOpcodeList(OpcodeList list)
//     : numDims(list.getNumDims()), numSymbols(list.getNumSymbols()),
//       context(list.getContext()) {
//   for (auto result : list.getResults())
//     results.push_back(result);
// }

// void MutableOpcodeList::reset(OpcodeList list) {
//   results.clear();
//   numDims = list.getNumDims();
//   numSymbols = list.getNumSymbols();
//   context = list.getContext();
//   for (auto result : list.getResults())
//     results.push_back(result);
// }

// bool MutableOpcodeList::isMultipleOf(unsigned idx, int64_t factor) const {
//   if (results[idx].isMultipleOf(factor))
//     return true;

//   // TODO: use simplifyOpcodeExpr and FlatOpcodeConstraints to
//   // complete this (for a more powerful analysis).
//   return false;
// }

// // Simplifies the result affine expressions of this list. The expressions
// have to
// // be pure for the simplification implemented.
// void MutableOpcodeList::simplify() {
//   // Simplify each of the results if possible.
//   // TODO: functional-style list
//   for (unsigned i = 0, e = getNumResults(); i < e; i++) {
//     results[i] = simplifyOpcodeExpr(getResult(i), numDims, numSymbols);
//   }
// }

// OpcodeList MutableOpcodeList::getOpcodeList() const {
//   return OpcodeList::get(numDims, numSymbols, results, context);
// }
