//===- AccelAttributes.h - Accel dialect attributes -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SODA_DIALECT_ACCEL_IR_ACCELATTRIBUTES_H_
#define SODA_DIALECT_ACCEL_IR_ACCELATTRIBUTES_H_

#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/Hashing.h"

#include "soda/Dialect/Accel/IR/AccelEnums.h.inc"

namespace mlir {
namespace accel {

/// One entry of an `#accel.opcode_map`: an opcode name and its actions
/// (an ArrayAttr of OpcodeActionAttr).
struct OpcodeMapEntry {
  StringAttr name;
  ArrayAttr actions;

  bool operator==(const OpcodeMapEntry &other) const {
    return name == other.name && actions == other.actions;
  }
};

inline llvm::hash_code hash_value(const OpcodeMapEntry &entry) {
  return llvm::hash_combine(entry.name, entry.actions);
}

/// One nesting level of an `#accel.opcode_flow`: the opcode names placed
/// before and after the nested level (ArrayAttrs of StringAttr).
struct OpcodeFlowLevel {
  ArrayAttr pre;
  ArrayAttr post;

  bool operator==(const OpcodeFlowLevel &other) const {
    return pre == other.pre && post == other.post;
  }
};

inline llvm::hash_code hash_value(const OpcodeFlowLevel &level) {
  return llvm::hash_combine(level.pre, level.post);
}

} // namespace accel
} // namespace mlir

#define GET_ATTRDEF_CLASSES
#include "soda/Dialect/Accel/IR/AccelAttributes.h.inc"

#endif // SODA_DIALECT_ACCEL_IR_ACCELATTRIBUTES_H_
