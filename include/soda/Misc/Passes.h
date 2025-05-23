//===- Passes.h - soda-opt pass entry points --------------------*- C++ -*-===//
//
// This header file defines prototypes that expose pass constructors.
//
//===----------------------------------------------------------------------===//

#ifndef SODA_MISC_PASSES_H
#define SODA_MISC_PASSES_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include <memory>

namespace mlir {
class Pass;

namespace soda {

//===----------------------------------------------------------------------===//
// Misc
//===----------------------------------------------------------------------===//
std::unique_ptr<mlir::Pass> createTestPrintOpNestingPass();
std::unique_ptr<mlir::Pass> createTestArgumentsToXMLPass();
std::unique_ptr<mlir::Pass> createTestArgumentsToCTestbenchPass();

std::unique_ptr<mlir::Pass> createEraseMemrefDeallocPass();
std::unique_ptr<mlir::Pass> createForwardMemrefAllocPass();
std::unique_ptr<mlir::Pass> createForwardLinalgFillPass();
std::unique_ptr<mlir::Pass> createForwardMemrefCopyPass();
void populateEraseMemrefDeallocPattern(RewritePatternSet &patterns);

//===----------------------------------------------------------------------===//
// Register passes
//===----------------------------------------------------------------------===//

/// Include the auto-generated definitions for passes
// TODO: only the registration call is necessary. Move pass class decls to
// another file
#define GEN_PASS_CLASSES
#include "soda/Misc/Passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "soda/Misc/Passes.h.inc"

} // namespace soda
} // namespace mlir

#endif // SODA_MISC_PASSES_H
