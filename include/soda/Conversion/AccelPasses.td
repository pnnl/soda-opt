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

#endif // SODA_CONVERSION_ACCELPASSES
