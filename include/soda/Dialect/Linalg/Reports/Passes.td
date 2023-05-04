//===- Passes.td - Linalg Reports for SODA -*- tablegen -*-----------------===//
//===----------------------------------------------------------------------===//
//
// Defines language-specific pattern match optimizations for SODA using
// Declarative Rewrite Rules (DRR) specified using TableGen records.
//
//===----------------------------------------------------------------------===//

#ifndef SODA_LINALG_REPORTS_PASSES
#define SODA_LINALG_REPORTS_PASSES

include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// Basic analysis passes
//===----------------------------------------------------------------------===//

def GenerateLinalgSummary : Pass<"soda-generate-linalg-summary", "ModuleOp"> {
  let summary = "Generate a summary of linalg operations in the module";
  let description = [{
      Generate a summary of linalg operations in the module.
  }];
  let constructor = "mlir::soda::linalg::reports::createGenerateLinalgSummary()";
  let options = [
    Option<"writeToTerminal", "write-to-terminal", "bool", /*default=*/"false",
           "If set, write to terminal instead of a file (default: false)">
  ];
}

#endif // SODA_LINALG_REPORTS_PASSES
