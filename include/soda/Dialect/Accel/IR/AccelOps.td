//===- AccelOps.td - Accel op definitions ------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ACCEL_OPS
#define ACCEL_OPS

include "soda/Dialect/Accel/IR/AccelBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

// Base class for accel dialect ops. 
class Accel_Op<string mnemonic, list<Trait> traits = []> :
    Op<Accel_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// dmaInitOp
//===----------------------------------------------------------------------===//

def Accel_InitDMAOp : Accel_Op<"init_dma"> {
  let summary = "initializes the DMA";
  let description = [{
    During lowering to AXI4MLIR calls, this op is lowered to a call to
    `dma_init` at the beginning of the operation scope/basic block, and
    a call to `dma_free` at the end of the operation scope/basic block.
    ```
  }];
  let arguments = (ins SignlessIntegerLike:$dmaAddress,
                       SignlessIntegerLike:$dmaInputAddress,
                       SignlessIntegerLike:$dmaInputBufferSize,
                       SignlessIntegerLike:$dmaOutputAddress,
                       SignlessIntegerLike:$dmaOutputBufferSize);


 // let results = (outs SignlessIntegerLike:$result);

  let assemblyFormat = [{
  $dmaAddress `,`
  $dmaInputAddress `,`
  $dmaInputBufferSize `,`
  $dmaOutputAddress `,`
  $dmaOutputBufferSize
  attr-dict `:`
  `(`
  type($dmaAddress) `,`
  type($dmaInputAddress) `,`
  type($dmaInputBufferSize) `,`
  type($dmaOutputAddress) `,`
  type($dmaOutputBufferSize) 
  `)`
  }];
}

def Accel_SendOp : Accel_Op<"send"> {
  let summary = "send MemRef to DMA region";
  let description = [{
    TODO
  }];
  let arguments = (ins AnyMemRef:$input,
                       Optional<I32>:$offset_value);


  let results = (outs I32:$out_offset);

  let assemblyFormat = [{
    $input (`,` $offset_value^)?  attr-dict `:` 
    `(` type($input) (`,` type($offset_value)^)? `)` `->` type($out_offset)
  }];
}

def Accel_SendLiteralOp : Accel_Op<"sendLiteral"> {
  let summary = "send Literal to DMA region";
  let description = [{
    Used to send a literal value to the DMA region.
    The literal value is considered an opcode.
  }];
  let arguments = (ins SignlessIntegerLike:$opcode,
                       Optional<I32>:$offset_value);


  let results = (outs I32:$out_offset);

  let assemblyFormat = [{
    $opcode (`,` $offset_value^)?  attr-dict `:` 
    `(` type($opcode) (`,` type($offset_value)^)? `)` `->` type($out_offset)
  }];
}

def Accel_RecvOp : Accel_Op<"recv"> {
  let summary = "receive data from the DMA region into the MemRef";
  let description = [{
    TODO
  }];
  let arguments = (ins AnyMemRef:$dst,
                       Optional<I32>:$offset_value);


  let results = (outs I32:$out_offset);

  let assemblyFormat = [{
    $dst (`,` $offset_value^)?  attr-dict `:` 
    `(` type($dst) (`,` type($offset_value)^)? `)` `->` type($out_offset)
  }];
}

#endif // ACCEL_OPS
