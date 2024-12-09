//===- OpcodeMap.cpp - MLIR Opcode Map Classes ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "soda/Dialect/Accel/IR/OpcodeMap.h"
#include "../OpcodeMapDetail.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Support/MathExtras.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

template <typename OpcodeExprContainer>
static SmallVector<OpcodeMap, 4>
inferFromExprList(ArrayRef<OpcodeExprContainer> exprsList) {
  assert(!exprsList.empty());
  assert(!exprsList[0].empty());
  // auto context = exprsList[0][0].getContext();
  // int64_t maxDim = -1, maxSym = -1;
  // getAMaxDimAndSymbol(exprsList, maxDim, maxSym);
  SmallVector<OpcodeMap, 4> maps;
  // maps.reserve(exprsList.size());
  // for (const auto &exprs : exprsList)
  //   maps.push_back(OpcodeMap::get(/*dimCount=*/maxDim + 1,
  //                                 /*symbolCount=*/maxSym + 1, exprs,
  //                                 context));
  return maps;
}

SmallVector<OpcodeMap, 4>
OpcodeMap::inferFromExprList(ArrayRef<ArrayRef<OpcodeExpr>> exprsList) {
  return ::inferFromExprList(exprsList);
}

SmallVector<OpcodeMap, 4>
OpcodeMap::inferFromExprList(ArrayRef<SmallVector<OpcodeExpr, 4>> exprsList) {
  return ::inferFromExprList(exprsList);
}

MLIRContext *OpcodeMap::getContext() const { return map->context; }

bool OpcodeMap::isEmpty() const { return getNumOpcodes() == 0; }

unsigned OpcodeMap::getNumOpcodes() const { return getOpcodes().size(); }

ArrayRef<std::tuple<StringRef, OpcodeList>> OpcodeMap::getOpcodes() const {
  assert(map && "uninitialized map storage");
  return map->opcodes();
}
std::tuple<StringRef, OpcodeList> OpcodeMap::getOpcode(unsigned idx) const {
  return getOpcodes()[idx];
}

OpcodeList OpcodeMap::getOpcodeList(StringRef key) const {
  for (auto kv : getOpcodes())
    if (std::get<0>(kv) == key)
      return std::get<1>(kv);
  return {};
}

unsigned OpcodeMap::getOpcodeListPosition(StringRef key) const {
  unsigned idx = 0;
  for (auto kv : getOpcodes()) {
    if (std::get<0>(kv) == key)
      return idx;
    ++idx;
  }
  return -1;
}

/// Walk all of the OpcodeExpr's in this mapping.
void OpcodeMap::walkExprs(llvm::function_ref<void(OpcodeExpr)> callback) const {
  for (auto kv : getOpcodes())
    for (auto list : std::get<1>(kv))
      for (auto expr : list)
        callback(expr);
}

/// Sparse replace method. Apply OpcodeExpr::replace(`expr`, `replacement`) to
/// each of the results and return a new OpcodeMap with the new results and
/// with the specified number of dims and symbols.
// TODO: This should replace the OpcodeList of a opcode (StringRef)
OpcodeMap OpcodeMap::replace(OpcodeExpr expr, OpcodeExpr replacement,
                             unsigned numResultDims,
                             unsigned numResultSyms) const {
  // SmallVector<OpcodeExpr, 4> newOpcodes;
  // newOpcodes.reserve(getNumOpcodes());
  // for (OpcodeExpr e : getOpcodes())
  //   newOpcodes.push_back(e.replace(expr, replacement));
  // return OpcodeMap::get(numResultDims, numResultSyms, newOpcodes,
  // getContext());
  return OpcodeMap::get(getContext());
}

OpcodeMap OpcodeMap::compose(OpcodeMap map) const {
  // assert(getNumDims() == map.getNumResults() && "Number of results
  // mismatch");
  // // Prepare `map` by concatenating the symbols and rewriting its exprs.
  // unsigned numDims = map.getNumDims();
  // unsigned numSymbolsThisMap = getNumSymbols();
  // unsigned numSymbols = numSymbolsThisMap + map.getNumSymbols();
  // SmallVector<OpcodeExpr, 8> newDims(numDims);
  // for (unsigned idx = 0; idx < numDims; ++idx) {
  //   newDims[idx] = getOpcodeDimExpr(idx, getContext());
  // }
  // SmallVector<OpcodeExpr, 8> newSymbols(numSymbols - numSymbolsThisMap);
  // for (unsigned idx = numSymbolsThisMap; idx < numSymbols; ++idx) {
  //   newSymbols[idx - numSymbolsThisMap] =
  //       getOpcodeSymbolExpr(idx, getContext());
  // }
  // auto newMap =
  //     map.replaceDimsAndSymbols(newDims, newSymbols, numDims, numSymbols);
  // SmallVector<OpcodeExpr, 8> exprs;
  // exprs.reserve(getResults().size());
  // for (auto expr : getResults())
  //   exprs.push_back(expr.compose(newMap));
  // return OpcodeMap::get(numDims, numSymbols, exprs, map.getContext());
  return OpcodeMap::get(map.getContext());
}

OpcodeMap mlir::concatOpcodeMaps(ArrayRef<OpcodeMap> maps) {
  // TODO: Implement this
  // unsigned numResults = 0, numDims = 0, numSymbols = 0;
  // for (auto m : maps)
  //   numResults += m.getNumResults();
  // SmallVector<OpcodeExpr, 8> results;
  // results.reserve(numResults);
  // for (auto m : maps) {
  //   for (auto res : m.getResults())
  //     results.push_back(res.shiftSymbols(m.getNumSymbols(), numSymbols));

  //   numSymbols += m.getNumSymbols();
  //   numDims = std::max(m.getNumDims(), numDims);
  // }
  return OpcodeMap::get(maps.front().getContext());
}

// //===----------------------------------------------------------------------===//
// // MutableOpcodeMap.
// //===----------------------------------------------------------------------===//

// MutableOpcodeMap::MutableOpcodeMap(OpcodeMap map)
//     : numDims(map.getNumDims()), numSymbols(map.getNumSymbols()),
//       context(map.getContext()) {
//   for (auto result : map.getResults())
//     results.push_back(result);
// }

// void MutableOpcodeMap::reset(OpcodeMap map) {
//   results.clear();
//   numDims = map.getNumDims();
//   numSymbols = map.getNumSymbols();
//   context = map.getContext();
//   for (auto result : map.getResults())
//     results.push_back(result);
// }

// bool MutableOpcodeMap::isMultipleOf(unsigned idx, int64_t factor) const {
//   if (results[idx].isMultipleOf(factor))
//     return true;

//   // TODO: use simplifyOpcodeExpr and FlatOpcodeConstraints to
//   // complete this (for a more powerful analysis).
//   return false;
// }

// // Simplifies the result affine expressions of this map. The expressions
// // have to  be pure for the simplification implemented.
// void MutableOpcodeMap::simplify() {
//   // Simplify each of the results if possible.
//   // TODO: functional-style map
//   for (unsigned i = 0, e = getNumResults(); i < e; i++) {
//     results[i] = simplifyOpcodeExpr(getResult(i), numDims, numSymbols);
//   }
// }

// OpcodeMap MutableOpcodeMap::getOpcodeMap() const {
//   return OpcodeMap::get(numDims, numSymbols, results, context);
// }
