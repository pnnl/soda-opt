#include "soda/Transforms/ForwardResults.h"

void mlir::forwardResults(AffineForOp *forOp) {
  std::vector<Operation *> operations;
  forOp->walk([&](Operation *operation) { operations.push_back(operation); });

  for (auto sourceOperation : operations) {
    if (!isa<AffineLoadOp>(sourceOperation))
      continue;

    MemRefAccess sourceAccess(sourceOperation);
    for (auto destinationOperation : operations) {
      if (!isa<AffineStoreOp>(destinationOperation))
        continue;

      MemRefAccess destinationAccess(destinationOperation);

      if (sourceAccess == destinationAccess) {
        bool dependsOnInductionVar = false;
        for (auto sourceOperand : sourceOperation->getOperands()) {
          if (sourceOperand == forOp->getInductionVar()) {
            dependsOnInductionVar = true;
          }
        }

        if (!dependsOnInductionVar) {
          forwardResult(forOp, destinationOperation, sourceOperation);
        }
      }
    }
  }
}

void mlir::forwardResult(AffineForOp *forOp, Operation *store,
                         Operation *load) {
  // Move load outside the loop
  load->moveBefore(forOp->getOperation());

  forOp->getBody()->addArgument(load->getResult(0).getType(), forOp->getLoc());

  unsigned numArguments = forOp->getBody()->getNumArguments();
  load->getResult(0).replaceAllUsesWith(
      forOp->getRegion().getArgument(numArguments - 1));

  unsigned numOperands = forOp->getNumOperands();
  forOp->getOperation()->insertOperands(numOperands, {load->getResult(0)});

  OpBuilder builder(forOp->getOperation()->getContext());

  builder.setInsertionPointAfter(forOp->getOperation());
  auto newLoop = builder.create<AffineForOp>(
      forOp->getLoc(), forOp->getLowerBoundOperands(),
      forOp->getLowerBoundMap(), forOp->getUpperBoundOperands(),
      forOp->getUpperBoundMap(), forOp->getStep(), forOp->getIterOperands(),
      [&](OpBuilder &nestedBuilder, Location nestedLoc, Value iv,
          ValueRange itrArgs) {
        nestedBuilder.create<AffineYieldOp>(nestedLoc);
      });

  forOp->walk([&](Operation *operation) {
    if (isa<AffineYieldOp, AffineForOp>(operation) ||
        operation->getParentOp() != forOp->getOperation() || operation == store)
      return;
    operation->moveBefore(newLoop.getBody()->getTerminator());
  });

  // Add a final store after the loop
  SmallVector<Value, 8> indexOperands;
  int nonIndexOperands = 2;

  for (auto operand = store->operand_begin();
       operand != store->operand_end(); operand++) {
    if (nonIndexOperands-- > 0)
      continue;

    indexOperands.append({*operand});
  }

  // Fix for the case when indices refer to the same value
  auto memrefType = store->getOperand(1).getType().cast<MemRefType>();
  int64_t rank = memrefType.getRank();
  if (rank != indexOperands.size()) {
    indexOperands.append({store->getOperand(store->getNumOperands() - 1)});
  }

  builder.create<AffineStoreOp>(store->getLoc(), newLoop->getResult(0),
                                      store->getOperand(1), indexOperands);

  std::vector<Value> operands;
  operands.push_back(store->getOperand(0));
  newLoop.getBody()->getTerminator()->setOperands(operands);

  int index = 1;
  for (auto argument : forOp->getRegionIterArgs()) {
    argument.replaceAllUsesWith(newLoop.getBody()->getArgument(index++));
  }
  forOp->getInductionVar().replaceAllUsesWith(newLoop.getInductionVar());

  store->erase();
  forOp->erase();
}