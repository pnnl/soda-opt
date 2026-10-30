//===- AccelEnums.td - Accel enum definitions --------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ACCEL_ENUMS
#define ACCEL_ENUMS

include "mlir/IR/EnumAttr.td"

def Accel_OpcodeActionKind : I32EnumAttr<"OpcodeActionKind",
    "kind of action performed by an opcode", [
      I32EnumAttrCase<"send", 0>,
      I32EnumAttrCase<"send_literal", 1>,
      I32EnumAttrCase<"send_dim", 2>,
      I32EnumAttrCase<"send_idx", 3>,
      I32EnumAttrCase<"recv", 4>
    ]> {
  let cppNamespace = "::mlir::accel";
  let genSpecializedAttr = 0;
}

def Accel_RecvMode : I32EnumAttr<"RecvMode",
    "how received data is written into the destination", [
      I32EnumAttrCase<"replace", 0>,
      I32EnumAttrCase<"accumulate", 1>
    ]> {
  let cppNamespace = "::mlir::accel";
}

#endif // ACCEL_ENUMS
