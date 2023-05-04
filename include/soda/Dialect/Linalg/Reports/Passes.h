//===- Passes.h - Linalg Reports pass entry points --------------*- C++ -*-===//

#ifndef SODA_LINALG_REPORTS_PASSES_H
#define SODA_LINALG_REPORTS_PASSES_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
class Pass;

namespace soda {
namespace linalg {
namespace reports {

#define GEN_PASS_DECL
#include "soda/Dialect/Linalg/Reports/Passes.h.inc"

//===----------------------------------------------------------------------===//
// Analysis
//===----------------------------------------------------------------------===//
std::unique_ptr<mlir::Pass> createGenerateLinalgSummary();

//===----------------------------------------------------------------------===//
// Register passes
//===----------------------------------------------------------------------===//
#define GEN_PASS_REGISTRATION
#include "soda/Dialect/Linalg/Reports/Passes.h.inc"

} // namespace reports
} // namespace linalg
} // namespace soda
} // namespace mlir

#endif // SODA_LINALG_REPORTS_PASSES_H
