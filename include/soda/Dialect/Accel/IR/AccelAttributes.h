//===- AccelAttributes.h - Accel attribute declarations  --------*- C++ -*-===//
//===----------------------------------------------------------------------===//
//
// This file declares Accel dialect specific attributes.
//
//===----------------------------------------------------------------------===//

#ifndef SODA_DIALECT_ACCEL_IR_ACCELATTRIBUTES_H
#define SODA_DIALECT_ACCEL_IR_ACCELATTRIBUTES_H

#include "mlir/IR/BuiltinAttributes.h"
#include "soda/Dialect/Accel/IR/OpcodeExpr.h"
#include "soda/Dialect/Accel/IR/OpcodeList.h"
#include "soda/Dialect/Accel/IR/OpcodeMap.h"
#include "llvm/ADT/TypeSwitch.h"


#define GET_ATTRDEF_CLASSES
#include "soda/Dialect/Accel/IR/AccelAttributes.h.inc"

namespace mlir {
namespace accel {


namespace detail {
// struct AccelInterfaceAttrStorage;
} // namespace detail

} // namespace accel
} // namespace soda

#endif // SODA_DIALECT_ACCEL_IR_ACCELATTRIBUTES_H