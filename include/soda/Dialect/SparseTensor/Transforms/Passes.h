//===- Passes.h - SparseTensor Transforms pass entry points --------------*- C++ -*-===//

#ifndef SODA_SPARSETENSOR_TRANSFORMS_PASSES_H
#define SODA_SPARSETENSOR_TRANSFORMS_PASSES_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
class Pass;

namespace soda {

#define GEN_PASS_DECL
#include "soda/Dialect/SparseTensor/Transforms/Passes.h.inc"

std::unique_ptr<Pass> createDenseBufferizationPass(bool testBufferizationAnalysisOnly);
std::unique_ptr<Pass> createDenseBufferizationPass();

std::unique_ptr<Pass> createInsertTensorCopiesPass(bool testBufferizationAnalysisOnly);
std::unique_ptr<Pass> createInsertTensorCopiesPass();

std::unique_ptr<Pass> createSparseTensorMemrefDebugPass();

//===----------------------------------------------------------------------===//
// Register passes
//===----------------------------------------------------------------------===//
#define GEN_PASS_REGISTRATION
#include "soda/Dialect/SparseTensor/Transforms/Passes.h.inc"

} // namespace soda
} // namespace mlir

#endif // SODA_SPARSETENSOR_TRANSFORMS_PASSES_H
