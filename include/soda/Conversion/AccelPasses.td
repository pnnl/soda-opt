//===-- AccelPasses.td - Accel conversion passes ------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SODA_CONVERSION_ACCELPASSES
#define SODA_CONVERSION_ACCELPASSES

include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// AccelToAXIRuntime
//===----------------------------------------------------------------------===//

def ConvertAccelToAXIRuntime
    : Pass<"convert-accel-to-axi-runtime", "ModuleOp"> {
  let summary = "Lower accel ops to calls into the AXI DMA runtime library";
  let description = [{
    Replaces accel operations with calls to the AXI runtime library
    (`soda/ExecutionEngine/axi/AxiRuntime.h`):

    * `accel.init_dma` becomes `dma_init`, and `dma_free` is called before
      every return of the function.
    * `accel.send` copies its memref into the DMA input buffer
      (`copy_to_inbuffer_<type>`). `accel.send_literal`, `accel.send_dim` and
      `accel.send_idx` copy one word (`copy_literal_to_inbuffer_i32`).
    * Send-like ops chained through their offsets are sent as one DMA
      transaction (`dma_start_send` + `dma_wait_send`) after the last op of
      the chain. Pending sends are always started before a receive, a loop or
      any other op with side effects.
    * `accel.recv` starts and waits for a receive (`dma_start_recv` +
      `dma_wait_recv`) and copies the data out of the DMA output buffer
      (`copy_from_outbuffer_<type>`, or `copy_from_outbuffer_accumulate_<type>`
      in accumulate mode).

    Element types `i32` and `f32` are supported; the DMA buffers hold 32-bit
    words.
  }];
  let dependentDialects = [
    "arith::ArithDialect",
    "func::FuncDialect",
    "memref::MemRefDialect",
  ];
  let options = [
    Option<"coalesceSends", "coalesce-sends", "bool", /*default=*/"true",
           "Send chained send-like ops as a single DMA transaction">,
  ];
}

//===----------------------------------------------------------------------===//
// LinalgToAccel
//===----------------------------------------------------------------------===//

def ConvertLinalgToAccel : Pass<"convert-linalg-to-accel", "func::FuncOp"> {
  let summary = "Lower linalg ops with AXI4MLIR attributes to accel ops";
  let description = [{
    Replaces every linalg op that has an `accel.opcode_flow` attribute with a
    loop nest that drives the accelerator described by its `accel.*`
    attributes (see the accel dialect):

    * The loops of the op with a non-zero `accel.accel_dim` are tiled by
      that size, in the order of `accel.permutation_map`. Other loops are not
      tiled: the accelerator handles their whole range.
    * The groups of `accel.opcode_flow` are placed on the loop nest: the
      innermost group in the innermost loop body, and each enclosing group one
      loop further out, before and after its inner loop.
    * Each opcode emits its actions as accel ops. `send(i)` and `recv(i)`
      move the slice of operand `i` that the loops placed so far select; the
      loops nested inside contribute their whole range (a stationary tile).
      `send_dim(i, d)` and `send_idx(i, d)` send the size and the tile index
      of that slice along dimension `d`.
    * `accel.init_opcodes` are sent once, before the loop nest, and
      `accel.init_dma` is created at the start of the function.

    The options attach missing attributes to the ops named by `anchor-op`,
    so that the flow can be chosen from the command line. Attribute strings
    may omit the `#accel.` prefix, e.g.
    `opcode-flow="opcode_flow<(sA (sB rC))>"`.

    The op must have buffer semantics, and the tiled loops must have static
    sizes divisible by their accelerator tile size.
  }];
  let dependentDialects = [
    "accel::AccelDialect",
    "affine::AffineDialect",
    "arith::ArithDialect",
    "memref::MemRefDialect",
    "scf::SCFDialect",
  ];
  let options = [
    Option<"anchorFuncName", "anchor-func", "std::string", /*default=*/"\"\"",
           "Only convert ops in this function">,
    Option<"anchorOpName", "anchor-op", "std::string", /*default=*/"\"\"",
           "Attach the attributes given by options to ops with this name "
           "(e.g. linalg.matmul)">,
    Option<"opcodeMap", "opcode-map", "std::string", /*default=*/"\"\"",
           "accel.opcode_map of anchored ops">,
    Option<"opcodeFlow", "opcode-flow", "std::string", /*default=*/"\"\"",
           "accel.opcode_flow of anchored ops">,
    Option<"initOpcodes", "init-opcodes", "std::string", /*default=*/"\"\"",
           "accel.init_opcodes of anchored ops">,
    ListOption<"accelDim", "accel-dim", "int64_t",
               "accel.accel_dim of anchored ops: one tile size per loop">,
    ListOption<"permutation", "permutation", "unsigned",
               "accel.permutation_map of anchored ops: the loop order">,
    Option<"dmaAddress", "dma-address", "unsigned", /*default=*/"0x40400000",
           "DMA engine address, for ops without accel.dma_init_config">,
    Option<"dmaInputAddress", "dma-input-address", "unsigned",
           /*default=*/"0x16000000", "DMA input buffer address">,
    Option<"dmaInputSize", "dma-input-size", "unsigned", /*default=*/"65536",
           "DMA input buffer size in words">,
    Option<"dmaOutputAddress", "dma-output-address", "unsigned",
           /*default=*/"0x16400000", "DMA output buffer address">,
    Option<"dmaOutputSize", "dma-output-size", "unsigned", /*default=*/"65536",
           "DMA output buffer size in words">,
    Option<"strictPlacement", "strict-placement", "bool", /*default=*/"false",
           "Reject sends and receives of slices that span loops nested "
           "inside them, instead of moving the whole span">,
  ];
}

#endif // SODA_CONVERSION_ACCELPASSES
