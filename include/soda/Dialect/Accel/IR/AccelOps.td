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
include "soda/Dialect/Accel/IR/AccelEnums.td"

// Base class for accel dialect ops.
class Accel_Op<string mnemonic, list<Trait> traits = []> :
    Op<Accel_Dialect, mnemonic, traits>;

// Common description of the transfer offsets.
defvar Accel_OffsetDescription = [{
    Offsets are in elements of the DMA buffer and default to 0. Every transfer
    op returns `offset + n`, where `n` is the number of elements it moves. Ops
    chained through their offsets form a single DMA transaction: the lowering
    copies all their data into the buffer and starts one transfer.
}];

//===----------------------------------------------------------------------===//
// InitDMAOp
//===----------------------------------------------------------------------===//

def Accel_InitDMAOp : Accel_Op<"init_dma"> {
  let summary = "Initializes the DMA engine";
  let description = [{
    Configures the DMA engine at `dmaAddress`, with an input buffer and an
    output buffer given by their address and size in elements. It lowers to
    `dma_init`, and `dma_free` is called before the function returns.

    ```mlir
    accel.init_dma %addr, %in, %in_size, %out, %out_size
        : (i32, i32, i32, i32, i32)
    ```
  }];
  let arguments = (ins SignlessIntegerLike:$dmaAddress,
                       SignlessIntegerLike:$dmaInputAddress,
                       SignlessIntegerLike:$dmaInputBufferSize,
                       SignlessIntegerLike:$dmaOutputAddress,
                       SignlessIntegerLike:$dmaOutputBufferSize);

  let assemblyFormat = [{
    $dmaAddress `,` $dmaInputAddress `,` $dmaInputBufferSize `,`
    $dmaOutputAddress `,` $dmaOutputBufferSize attr-dict `:`
    `(` type($dmaAddress) `,` type($dmaInputAddress) `,`
        type($dmaInputBufferSize) `,` type($dmaOutputAddress) `,`
        type($dmaOutputBufferSize) `)`
  }];
}

//===----------------------------------------------------------------------===//
// Send ops
//===----------------------------------------------------------------------===//

def Accel_SendOp : Accel_Op<"send"> {
  let summary = "Sends a memref to the accelerator";
  let description = [{
    Copies `input` in row-major order into the DMA input buffer at `offset`
    and sends it.
  }] # Accel_OffsetDescription # [{

    ```mlir
    %o = accel.send %tile, %offset : (memref<4x4xi32, strided<[16, 1], offset: ?>>, i32) -> i32
    ```
  }];
  let arguments = (ins AnyMemRef:$input, Optional<I32>:$offset_value);
  let results = (outs I32:$out_offset);

  let assemblyFormat = [{
    $input (`,` $offset_value^)? attr-dict `:`
    `(` type($input) (`,` type($offset_value)^)? `)` `->` type($out_offset)
  }];
}

def Accel_SendLiteralOp : Accel_Op<"send_literal"> {
  let summary = "Sends a 32-bit word to the accelerator";
  let description = [{
    Sends `literal`, usually an opcode, as one element.
  }] # Accel_OffsetDescription # [{

    ```mlir
    %o = accel.send_literal %c34, %offset : (i32, i32) -> i32
    ```
  }];
  let arguments = (ins I32:$literal, Optional<I32>:$offset_value);
  let results = (outs I32:$out_offset);

  let assemblyFormat = [{
    $literal (`,` $offset_value^)? attr-dict `:`
    `(` type($literal) (`,` type($offset_value)^)? `)` `->` type($out_offset)
  }];
}

class Accel_SendTileInfoOp<string mnemonic>
    : Accel_Op<mnemonic> {
  let arguments = (ins AnyMemRef:$tile,
                       ConfinedAttr<I64Attr, [IntNonNegative]>:$dim,
                       Optional<I32>:$offset_value);
  let results = (outs I32:$out_offset);

  let assemblyFormat = [{
    $tile `[` $dim `]` (`,` $offset_value^)? attr-dict `:`
    `(` type($tile) (`,` type($offset_value)^)? `)` `->` type($out_offset)
  }];
  let hasVerifier = 1;
}

def Accel_SendDimOp : Accel_SendTileInfoOp<"send_dim"> {
  let summary = "Sends the size of a tile along one dimension";
  let description = [{
    Sends the size of `tile` along dimension `dim` as one element.
  }] # Accel_OffsetDescription # [{

    ```mlir
    %o = accel.send_dim %tile[1] : (memref<10x20xi32, strided<[80, 1], offset: ?>>) -> i32
    ```
  }];
}

def Accel_SendIdxOp : Accel_SendTileInfoOp<"send_idx"> {
  let summary = "Sends the index of a tile along one dimension";
  let description = [{
    Sends the index of `tile` along dimension `dim` as one element. The index
    is the tile's offset in its source memref divided by the tile size along
    `dim`. With tiles of size (10, 20), the tile at offset (10, 80) is tile
    (1, 4), so `send_idx %tile[1]` sends 4. When `tile` is not a
    `memref.subview` it is a whole buffer and the index is 0.
  }] # Accel_OffsetDescription # [{

    ```mlir
    %o = accel.send_idx %tile[0] : (memref<10x20xi32, strided<[80, 1], offset: ?>>) -> i32
    ```
  }];
}

//===----------------------------------------------------------------------===//
// RecvOp
//===----------------------------------------------------------------------===//

def Accel_RecvOp : Accel_Op<"recv"> {
  let summary = "Receives data from the accelerator into a memref";
  let description = [{
    Receives as many elements as `dst` holds from the DMA output buffer at
    `offset` and writes them into `dst` in row-major order. With
    `mode = accumulate`, the received values are added to `dst` instead.
  }] # Accel_OffsetDescription # [{

    ```mlir
    %o = accel.recv %tile : (memref<4x4xi32>) -> i32
    %o = accel.recv %tile, %offset mode = accumulate : (memref<4x4xf32>, i32) -> i32
    ```
  }];
  let arguments = (ins AnyMemRef:$dst, Optional<I32>:$offset_value,
                       OptionalAttr<Accel_RecvMode>:$mode);
  let results = (outs I32:$out_offset);

  let assemblyFormat = [{
    $dst (`,` $offset_value^)? (`mode` `=` $mode^)? attr-dict `:`
    `(` type($dst) (`,` type($offset_value)^)? `)` `->` type($out_offset)
  }];
  let hasVerifier = 1;

  let extraClassDeclaration = [{
    bool isAccumulate() {
      return getMode() && *getMode() == RecvMode::accumulate;
    }
  }];
}

#endif // ACCEL_OPS
