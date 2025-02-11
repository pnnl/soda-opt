//===----------------------------------------------------------------------===//
//
// This file implements a linalg Tiling pass using the impl from dialect/linalg.
// 
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Linalg/Transforms/TilingInterfaceImpl.h"

#include "mlir/Dialect/Transform/IR/TransformTypes.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"

#include "soda/Dialect/Transform/Transforms/Passes.h"

namespace mlir {
namespace soda {
namespace trans {
#define GEN_PASS_DEF_TRANSFORMDIALECTERASESCHEDULE
#include "soda/Dialect/Transform/Transforms/Passes.h.inc"
} // namespace transform
} // namespace soda
} // namespace mlir

#define DEBUG_TYPE "soda-trans-interpreter"

using namespace mlir;
using namespace mlir::soda;

namespace {

struct EraseSchedulePass
    : public mlir::soda::trans::impl::TransformDialectEraseScheduleBase<
          EraseSchedulePass> {
  EraseSchedulePass() = default;

  void runOnOperation() override {
    getOperation()->walk<WalkOrder::PreOrder>([&](Operation *nestedOp) {
      if (isa<mlir::transform::TransformOpInterface>(nestedOp)) {
        nestedOp->erase();
        return WalkResult::skip();
      }
      return WalkResult::advance();
    });
  }
};

} // namespace
