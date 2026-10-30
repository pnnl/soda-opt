//===- AccelBase.td - Base definitions for accel dialect ----*- tablegen -*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ACCEL_BASE
#define ACCEL_BASE

include "mlir/IR/OpBase.td"

def Accel_Dialect : Dialect {
  let name = "accel";
  let cppNamespace = "::mlir::accel";
  let summary = "Host-side operations to drive AXI-stream accelerators";
  let description = [{
    The accel dialect implements the AXI4MLIR flow (CGO'24). Its operations
    move data and opcodes between host memory and the input/output buffers of
    a DMA engine that streams them to a custom accelerator.

    The dialect also defines the attributes that describe how a
    `linalg.generic` (or named linalg op) drives an accelerator. They are
    attached as discardable attributes:

    * `accel.opcode_map`: `#accel.opcode_map<...>`, the valid opcodes and the
      actions each one performs.
    * `accel.opcode_flow`: `#accel.opcode_flow<...>`, the order of opcodes
      and the loop level of each one.
    * `accel.init_opcodes`: `#accel.opcode_flow<...>`, opcodes sent once,
      before the kernel.
    * `accel.dma_init_config`: `#accel.dma_config<...>`, the DMA settings.
    * `accel.accel_dim`: an `affine_map` giving the accelerator tile size of
      every loop (0 means the loop is not tiled).
    * `accel.permutation_map`: an `affine_map` permutation giving the order
      of the tiled loops, outermost first.
  }];

  let useDefaultAttributePrinterParser = 1;
  let hasOperationAttrVerify = 1;

  let extraClassDeclaration = [{
    void registerAttributes();

    static constexpr ::llvm::StringLiteral kOpcodeMapAttrName =
        "accel.opcode_map";
    static constexpr ::llvm::StringLiteral kOpcodeFlowAttrName =
        "accel.opcode_flow";
    static constexpr ::llvm::StringLiteral kInitOpcodesAttrName =
        "accel.init_opcodes";
    static constexpr ::llvm::StringLiteral kDmaInitConfigAttrName =
        "accel.dma_init_config";
    static constexpr ::llvm::StringLiteral kAccelDimAttrName =
        "accel.accel_dim";
    static constexpr ::llvm::StringLiteral kPermutationMapAttrName =
        "accel.permutation_map";
  }];
}

#endif // ACCEL_BASE
