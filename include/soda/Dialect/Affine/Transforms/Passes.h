//===- Passes.h - soda-opt pass entry points --------------------*- C++ -*-===//
//
// This header file defines prototypes that expose pass constructors.
//
//===----------------------------------------------------------------------===//

#ifndef SODA_AFFINE_TRANSFORMS_PASSES_H
#define SODA_AFFINE_TRANSFORMS_PASSES_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include <memory>

namespace mlir {
class Pass;
} // namespace mlir

namespace mlir {
class ModuleOp;
namespace memref {
class DeallocOp;
class AllocOp;
class AllocaOp;
class CopyOp;
} // namespace memref
namespace linalg {
class FillOp;
}
} // namespace mlir

namespace mlir {
namespace soda {

//===----------------------------------------------------------------------===//
// Optimizations
//===----------------------------------------------------------------------===//

/// Performs packing (or explicit copying) of accessed memref regions into
/// buffers in the specified faster memory space through either pointwise copies
/// or DMA operations.
std::unique_ptr<OperationPass<func::FuncOp>> createAffineDataCopyGenPass(
    unsigned slowMemorySpace, unsigned fastMemorySpace,
    unsigned tagMemorySpace = 0, int minDmaTransferSize = 1024,
    uint64_t fastMemCapacityBytes = std::numeric_limits<uint64_t>::max(),
    bool generateDma = false);

/// Expose affine loop tiling creation with explicit tileSize selection
std::unique_ptr<OperationPass<func::FuncOp>>
createAffineLoopTilingPass(unsigned tileSize);

std::unique_ptr<OperationPass<func::FuncOp>>
createAffineLoopPermutationPass(const ArrayRef<unsigned> &permList);

} // namespace soda
} // namespace mlir

#endif // SODA_AFFINE_TRANSFORMS_PASSES_H
