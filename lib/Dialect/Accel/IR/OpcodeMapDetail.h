//===- OpcodeMapDetail.h - MLIR Opcode Map details Class --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This holds implementation details of OpcodeMap.
//
//===----------------------------------------------------------------------===//

#ifndef OPCODEMAPDETAIL_H_
#define OPCODEMAPDETAIL_H_

#include "soda/Dialect/Accel/IR/OpcodeExpr.h"
#include "soda/Dialect/Accel/IR/OpcodeMap.h"
#include "mlir/Support/StorageUniquer.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/TrailingObjects.h"

namespace mlir {
namespace detail {

using OpcodeKV = std::tuple<StringRef, OpcodeList>;
using OpcodeDictArrayRef = ArrayRef<OpcodeKV>;

struct OpcodeMapStorage final
    : public StorageUniquer::BaseStorage,
      public llvm::TrailingObjects<OpcodeMapStorage, OpcodeKV> {
  /// The hash key used for uniquing.
  using KeyTy = std::tuple<unsigned, OpcodeDictArrayRef>;

  unsigned numOpcodes; // TODO:: Using numDims as numOpcodes

  MLIRContext *context;

  /// The opcode expressions for this opcode map.
  OpcodeDictArrayRef opcodes() const {
    return {getTrailingObjects<OpcodeKV>(), numOpcodes};
  }

  bool operator==(const KeyTy &key) const {
    return std::get<0>(key) == numOpcodes && std::get<1>(key) == opcodes();
  }

  // Constructs an OpcodeMapStorage from a key. The context must be set by the
  // caller.
  static OpcodeMapStorage *
  construct(StorageUniquer::StorageAllocator &allocator, const KeyTy &key) {
    auto opcodes = std::get<1>(key);
    auto byteSize =
        OpcodeMapStorage::totalSizeToAlloc<OpcodeKV>(opcodes.size());
    auto *rawMem = allocator.allocate(byteSize, alignof(OpcodeMapStorage));
    auto *res = new (rawMem) OpcodeMapStorage();
    res->numOpcodes = opcodes.size();
    std::uninitialized_copy(opcodes.begin(), opcodes.end(),
                            res->getTrailingObjects<OpcodeKV>());
    return res;
  }
};

} // namespace detail
} // namespace mlir

#endif // OPCODEMAPDETAIL_H_
