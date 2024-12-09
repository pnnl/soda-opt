//===-- AccelAttr.td - Accel attributes --------------------*- tablegen -*-===//
//
// This file implements support for the following attributes:

//   // Map of opcode names to opcode sequences.
//   opcode_map = opcode_map < // Valid Opcodes
//     sA       = [send_literal(0x22), send(0)],
//     sB       = [send_literal(0x23), send(1)],
//     cC       = [send_literal(0xF0)],
//     rC       = [send_literal(0x24), recv(2)],
//     sBcCrC   = [send_literal(0x25), send(1), recv(2)],
//     reset    = [send_literal(0xFF)] >,

//   // Opcodes sent once. Tokens defined in opcode_map.
//   init_opcodes = init_opcodes < (reset) >,

//   // Flow to implement. Tokens defined in opcode_map.
//   opcode_flow = opcode_flow < (sA (sBcCrC)) >   // As
//   opcode_flow = opcode_flow < ((sA sB cC) rC) > // Cs
//   opcode_flow = opcode_flow < (sB sA cC rC) >   // Ns
//===----------------------------------------------------------------------===//

#ifndef ACCELATTR_TD
#define ACCELATTR_TD

include "mlir/IR/AttrTypeBase.td"
include "soda/Dialect/Accel/IR/AccelBase.td"

class Accel_Attr<string name, list<Trait> traits = [],
                 string baseCppClass = "::mlir::Attribute">
    : AttrDef<Accel_Dialect, name, traits, baseCppClass> {
  let mnemonic = ?;
}

// Accel attributes

// opcode_flow attribute
// Implements the following syntax:
// opcode_flow_entry ::= `opcode_flow` `<` flow_expr > 
// flow_expr ::= `(` flow_expr `)` | bare_id (` ` bare_id)*

// def Accel_OpcodeFlowAttr : Accel_Attr<"OpcodeFlow"> {
//   // let parser = [{
//   //   if (parser.parseLess())
//   //     return failure();
//   //   SmallVector<Attribute, 4> flow;
//   //   do {
//   //     Attribute attr;
//   //     if (parser.parseAttribute(attr))
//   //       return failure();
//   //     flow.push_back(attr);
//   //   } while (succeeded(parser.parseOptionalComma()));
//   //   if (parser.parseGreater())
//   //     return failure();
//   //   return success(OpflowAttr::get(flow, getContext()));
//   // }];
// }

// opcode_map attribute

// Implements the following syntax:

def Accel_OpcodeMapAttr : Accel_Attr<"OpcodeMap"> {
  let summary = "An Attribute containing an OpcodeMap object";
  let description = [{
    Syntax:

    ```
    opcode_dict  ::= `opcode_map` `<` opcode-entry (`,` opcode-entry)* `>`
    
    opcode_entry ::= (bare-id | string-literal) `=` opcode_list
    
    opcode_list  ::= `[` opcode_expr (`,` opcode_expr)* `]`
    
    opcode_expr  ::= op_send(bare-id)
                   | op_send_literal(integer-literal)
                   | op_send_dim(bare-id)
                   | op_send_idx(bare-id)
                   | op_recv(bare-id)
    ```
  }];
  // let parameters = (ins ArrayRefParameter<"NamedAttribute", "">:$value);
  // let builders = [
  //   AttrBuilder<(ins CArg<"ArrayRef<NamedAttribute>", "std::nullopt">:$value)>
  // ];
  let parameters = (ins "OpcodeMap":$value);
  let builders = [
    AttrBuilderWithInferredContext<(ins "OpcodeMap":$value), [{
      return $_get(value.getContext(), value);
    }]>
  ];
  let extraClassDeclaration = [{
    // using ValueType = ArrayRef<NamedAttribute>;
    using ValueType = OpcodeMap;
    OpcodeMap getOpcodeMap() const { return getValue(); }
  
  private:
    static OpcodeMapAttr getEmpty(MLIRContext *context);
    friend MLIRContext;
  public:
  }];
  let skipDefaultBuilders = 1;
}

#endif // ACCELATTR_TD