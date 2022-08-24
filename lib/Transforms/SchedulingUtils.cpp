#include "soda/Transforms/SchedulingUtils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/Operation.h"

using namespace mlir;

AffineForOp mlir::extractInnermostLoop(func::FuncOp &Function, int loop) {
  Operation *forOp;
  int count = 0;
  Function.walk([&](Operation *op) {
    int hasNestedLoops = false;
    op->walk([&](Operation *nestedOp) {
      if (isa<AffineForOp>(nestedOp) && nestedOp != op) {
        hasNestedLoops = true;
      }
    });
    if (isa<AffineForOp>(op) && !hasNestedLoops &&
        !(isa<AffineIfOp>(op->getParentOp()) &&
          dyn_cast<AffineIfOp>(op->getParentOp()).getElseBlock() ==
              op->getBlock()) &&
        count < loop) {
      forOp = op;
      count++;
    }
  });

  return dyn_cast<AffineForOp>(forOp);
}

void mlir::conditionLoopExecution(AffineForOp *forOp,
                                  LoopSchedule &loopSchedule) {
  
  AffineMap mapL = forOp->getLowerBoundMap();
  AffineMap mapU = forOp->getUpperBoundMap();
  int difference;
  AffineExpr mapExpression;
  unsigned int mapDims;
  unsigned int mapSyms;
  AffineExpr setExpression;
  SmallVector<Value> range;

  if(mapL.getNumDims() == 0 && mapL.getNumSymbols() == 0) {
      // upper bound cannot be a constant, otherwise we would not be here
      difference = - 1 - loopSchedule.getPrologue().size() -
                   forOp->getConstantLowerBound();
      mapExpression = mapU.getResult(0);
      mapDims = mapU.getNumDims();
      mapSyms = mapU.getNumSymbols();
      setExpression = getAffineBinaryOpExpr(
        AffineExprKind::Add, mapExpression,
        getAffineConstantExpr(difference, forOp->getContext()));
      range = forOp->getUpperBound().getOperands();
  }
  if(mapU.getNumDims() == 0 && mapU.getNumSymbols() == 0) {
      // lower bound cannot be a constant, otherwise we would not be here
      difference = forOp->getConstantUpperBound() - 1 
            - loopSchedule.getPrologue().size();
      mapExpression = mapL.getResult(0);
      mapDims = mapL.getNumDims();
      mapSyms = mapL.getNumSymbols();
      AffineExpr invert = - mapExpression;
      setExpression = getAffineBinaryOpExpr(
        AffineExprKind::Add,
        getAffineConstantExpr(difference, forOp->getContext()), invert);
      range = forOp->getLowerBound().getOperands();
  }

  SmallVector<AffineExpr, 1> constraints{setExpression};
  SmallVector<bool, 1> eqFlags{false};
  IntegerSet set = IntegerSet::get(mapDims, mapSyms,
                                   constraints, eqFlags);

  OpBuilder builder(forOp->getOperation());
  AffineIfOp affineIf = builder.create<AffineIfOp>(
      forOp->getLoc(), set, range,
      /*withElseRegion=*/true);

  auto *thenBlock = affineIf.getThenBlock();
  forOp->getOperation()->moveBefore(thenBlock, thenBlock->begin());

  if (forOp->getNumResults() > 0) {
    for (auto user: forOp->getResult(0).getUsers()) {
      user->moveAfter(forOp->getOperation());
    }
  }

  auto clonedForOp = affineIf.getElseBodyBuilder().clone(*forOp->getOperation());

  if (forOp->getNumResults() > 0) {
    for (auto user: forOp->getResult(0).getUsers()) {
      auto clonedUser = affineIf.getElseBodyBuilder().clone(*user);
      clonedUser->setOperand(0, clonedForOp->getResult(0));
    }
  }

}
