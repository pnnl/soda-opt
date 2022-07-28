#include "mlir/Dialect/Affine/IR/AffineOps.h"

#include "mlir/IR/Operation.h"
#include <vector>

#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/AffineStructures.h"
#include "mlir/Dialect/Affine/Analysis/Utils.h"

namespace mlir {

void forwardResults(AffineForOp *forOp);

void forwardResult(AffineForOp *forOp, Operation *store, Operation *load);

} // namespace mlir