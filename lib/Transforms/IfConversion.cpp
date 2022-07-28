#include "mlir/IR/AffineMap.h"
#include "mlir/IR/IntegerSet.h"
#include "soda/Transforms/IfConversion.h"

void mlir::convertAffineIf(AffineIfOp *ifOp) {
  MLIRContext *context = ifOp->getOperation()->getContext();
  OpBuilder builder(ifOp->getOperation());

  // Create affine map from the integer set.
  auto map = AffineMap::get(ifOp->getIntegerSet().getNumDims(),
                            ifOp->getIntegerSet().getNumSymbols(),
                            ifOp->getIntegerSet().getConstraints(), context);

  // Create affine apply.
  SmallVector<Value, 8> applyOperands{ifOp->getOperands()};
  auto applyOperation = builder.create<AffineApplyOp>(
      ifOp->getOperation()->getLoc(), map, applyOperands);

  // Build constraint using compare operation and zero constant.
  Operation *zeroOperation =
      builder.create<mlir::arith::ConstantIndexOp>(ifOp->getOperation()->getLoc(), 0);
  Operation *compareOperation = builder.create<mlir::arith::CmpIOp>(
      ifOp->getLoc(),
      ifOp->getIntegerSet().getEqFlags()[0] ? arith::CmpIPredicate::eq
                                            : arith::CmpIPredicate::sge,
      applyOperation->getResult(0), zeroOperation->getResult(0));

  // Create select operation.
  Operation *selectOperation = builder.create<arith::SelectOp>(
      ifOp->getLoc(), compareOperation->getResult(0),
      ifOp->getThenBlock()->getTerminator()->getOperand(0),
      ifOp->getElseBlock()->getTerminator()->getOperand(0));

  // Move all operations from then and else blocks to the parent region.
  ifOp->walk([&](Operation *operation) {
    if (isa<AffineYieldOp>(operation)) {
      return;
    }
    operation->moveBefore(applyOperation);
  });

  ifOp->getResult(0).replaceAllUsesWith(selectOperation->getResult(0));
  ifOp->getResult(0).dropAllUses();
  ifOp->erase();
}