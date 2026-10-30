//===- AccelAttributes.td - Accel attribute definitions ----*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ACCEL_ATTRIBUTES
#define ACCEL_ATTRIBUTES

include "soda/Dialect/Accel/IR/AccelBase.td"
include "soda/Dialect/Accel/IR/AccelEnums.td"
include "mlir/IR/AttrTypeBase.td"

class Accel_Attr<string name, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<Accel_Dialect, name, traits> {
  let mnemonic = attrMnemonic;
}

//===----------------------------------------------------------------------===//
// OpcodeActionAttr
//===----------------------------------------------------------------------===//

def Accel_OpcodeActionAttr : Accel_Attr<"OpcodeAction", "action"> {
  let summary = "One action performed by an opcode";
  let description = [{
    An action moves one item between the host and the accelerator. Operand
    ids refer to the operands of the linalg op that carries the attribute
    (for a matmul, 0 = A, 1 = B, 2 = C).

    ```
    action ::= `send` `(` operand `)`
             | `send_literal` `(` integer `)`
             | `send_dim` `(` operand `,` dim `)`
             | `send_idx` `(` operand `,` dim `)`
             | `recv` `(` operand (`,` (`accumulate` | `replace`))? `)`
    ```

    * `send(i)`: sends the current tile of operand `i`.
    * `send_literal(x)`: sends the 32-bit word `x` (usually an opcode).
    * `send_dim(i, d)`: sends the size of the current tile of operand `i`
      along dimension `d`.
    * `send_idx(i, d)`: sends the index of the current tile of operand `i`
      along dimension `d`, i.e. its offset divided by the tile size. With
      tiles of size (10, 20), tile (1, 4) gives `send_idx(i, 0)` = 1,
      `send_idx(i, 1)` = 4 and `send_dim(i, 1)` = 20.
    * `recv(i)`: receives the current tile of operand `i`, replacing its
      contents. With `accumulate`, the received values are added to it.

    The standalone form is `#accel.action<send(0)>`. Inside an
    `#accel.opcode_map`, actions are written without the wrapper.
  }];

  let parameters = (ins
    EnumParameter<Accel_OpcodeActionKind>:$kind,
    "int64_t":$operand,
    "int64_t":$value,
    EnumParameter<Accel_RecvMode>:$mode
  );
  let hasCustomAssemblyFormat = 1;
  let genVerifyDecl = 1;

  let extraClassDeclaration = [{
    static OpcodeActionAttr getSend(::mlir::MLIRContext *ctx, int64_t operand);
    static OpcodeActionAttr getSendLiteral(::mlir::MLIRContext *ctx,
                                           int64_t literal);
    static OpcodeActionAttr getSendDim(::mlir::MLIRContext *ctx,
                                       int64_t operand, int64_t dim);
    static OpcodeActionAttr getSendIdx(::mlir::MLIRContext *ctx,
                                       int64_t operand, int64_t dim);
    static OpcodeActionAttr getRecv(::mlir::MLIRContext *ctx, int64_t operand,
                                    RecvMode mode = RecvMode::replace);

    /// True if the action uses `getOperand()` (all kinds but send_literal).
    bool hasOperand() const {
      return getKind() != OpcodeActionKind::send_literal;
    }
    /// The tile dimension of a send_dim or send_idx action.
    int64_t getDim() const { return getValue(); }
    /// The literal of a send_literal action.
    int64_t getLiteral() const { return getValue(); }

    /// Parses and prints the action without the `#accel.action<>` wrapper.
    static ::mlir::FailureOr<OpcodeActionAttr>
    parseBody(::mlir::AsmParser &parser);
    void printBody(::mlir::AsmPrinter &printer) const;
  }];
}

//===----------------------------------------------------------------------===//
// OpcodeMapAttr
//===----------------------------------------------------------------------===//

def Accel_OpcodeMapEntryParam :
    AttrOrTypeParameter<"::mlir::accel::OpcodeMapEntry", "opcode map entry">;

def Accel_OpcodeMapAttr : Accel_Attr<"OpcodeMap", "opcode_map"> {
  let summary = "The opcodes an accelerator accepts";
  let description = [{
    Maps opcode names to the list of actions each one performs, in order.

    ```mlir
    #accel.opcode_map<
      sA     = [send_literal(0x22), send(0)],
      sB     = [send_literal(0x23), send(1)],
      sBcCrC = [send_literal(0x25), send(1), recv(2, accumulate)],
      reset  = [send_literal(0xFF)]>
    ```

    Names are bare identifiers or string literals.
  }];

  let parameters = (ins ArrayRefParameter<"::mlir::accel::OpcodeMapEntry",
                                          "opcode entries">:$entries);
  let hasCustomAssemblyFormat = 1;
  let genVerifyDecl = 1;

  let extraClassDeclaration = [{
    /// Returns the actions of opcode `name`, or nullptr if it is not defined.
    ::mlir::ArrayAttr lookup(::llvm::StringRef name) const;
  }];
}

//===----------------------------------------------------------------------===//
// OpcodeFlowAttr
//===----------------------------------------------------------------------===//

def Accel_OpcodeFlowAttr : Accel_Attr<"OpcodeFlow", "opcode_flow"> {
  let summary = "The order and loop placement of opcodes";
  let description = [{
    A chain of nested groups of opcode names:

    ```
    flow  ::= `<` group `>`
    group ::= `(` name* group? name* `)`
    ```

    The innermost group runs in the body of the innermost tiled loop. A group
    `k` levels further out runs in the body of the loop `k` levels further
    out: names before its nested group are placed just before the inner loop,
    names after it just after. A group holds at most one nested group, and
    groups may be empty to skip a loop level.

    ```mlir
    #accel.opcode_flow<(sA sB cC rC)>   // nothing stationary
    #accel.opcode_flow<(sA (sBcCrC))>   // A stationary
    #accel.opcode_flow<((sA sB cC) rC)> // C stationary
    ```

    Levels are stored outermost first. For the innermost level, every name is
    in `pre` and `post` is empty.
  }];

  let parameters = (ins ArrayRefParameter<"::mlir::accel::OpcodeFlowLevel",
                                          "flow levels">:$levels);
  let hasCustomAssemblyFormat = 1;
  let genVerifyDecl = 1;

  let extraClassDeclaration = [{
    /// Number of nested groups.
    unsigned getDepth() const { return getLevels().size(); }
    /// Calls `fn` for every opcode name in the flow.
    void walkOpcodes(::llvm::function_ref<void(::mlir::StringAttr)> fn) const;
  }];
}

//===----------------------------------------------------------------------===//
// DmaConfigAttr
//===----------------------------------------------------------------------===//

def Accel_DmaConfigAttr : Accel_Attr<"DmaConfig", "dma_config"> {
  let summary = "DMA engine configuration";
  let description = [{
    Physical address of the DMA engine and the address and size (in 32-bit
    words) of its input and output buffers.

    ```mlir
    #accel.dma_config<address = 0x40400000, input_address = 0x16000000,
                      input_size = 65536, output_address = 0x16400000,
                      output_size = 65536>
    ```
  }];

  let parameters = (ins
    "uint32_t":$address,
    "uint32_t":$input_address,
    "uint32_t":$input_size,
    "uint32_t":$output_address,
    "uint32_t":$output_size
  );
  let assemblyFormat = "`<` struct(params) `>`";
}

#endif // ACCEL_ATTRIBUTES
