#include <vector>

//#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/Support/LLVM.h"
#include "soda/Transforms/LoopScheduleBuilder.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

using namespace mlir;

llvm::StringRef createdByScheduling = "isCreatedByScheduling";

// Temporary workaround because BlockAndValueMapping does not have a getInverse method anymore
mlir::BlockAndValueMapping getInverseMapping(mlir::BlockAndValueMapping* toInvert){
  BlockAndValueMapping result;
    for (const auto &pair : toInvert->getValueMap())
      result.map(pair.second, pair.first);
    return result;
}

void mlir::LoopScheduleBuilder::mapResults(int originalIteration,
                                           Operation *original,
                                           Operation *cloned) {
  for (unsigned i = 0; i < original->getNumResults(); i++) {
    originalIterationMappings[originalIteration]->map(original->getResult(i),
                                                      cloned->getResult(i));
  }
}

Operation *mlir::LoopScheduleBuilder::createPrologueIndexVariableReplacement(
    int difference) {
  if (originalForOp->hasConstantLowerBound()) {
    Operation *constantOperation = builder.create<arith::ConstantIndexOp>(
        originalForOp->getLoc(),
        originalForOp->getConstantLowerBound() + difference);
    constantOperation->setAttr(createdByScheduling, builder.getStringAttr("true"));
    return constantOperation;
  }

  auto applyOperation = builder.create<AffineApplyOp>(
      originalForOp->getLoc(), originalForOp->getLowerBoundMap(),
      originalForOp->getLowerBoundOperands());
  applyOperation->setAttr(createdByScheduling, builder.getStringAttr("true"));

  if (difference == 0) {
    return applyOperation;
  } else {
    auto map = AffineMap::get(
        1, 0,
        getAffineBinaryOpExpr(AffineExprKind::Add, getAffineDimExpr(0, context),
                              getAffineConstantExpr(difference, context)));

    SmallVector<Value, 8> applyOperands{applyOperation->getResult(0)};
    auto additionalApplyOperation = builder.create<AffineApplyOp>(
        originalForOp->getLoc(), map, applyOperands);
    additionalApplyOperation->setAttr(createdByScheduling, builder.getStringAttr("true"));
    return additionalApplyOperation;
  }
}

Operation *mlir::LoopScheduleBuilder::createEpilogueIndexVariableReplacement(
    int difference) {
  // TODO: Explain this
  int adjustedDifference = difference - 1 - schedule.getPrologue().size();
  if (originalForOp->hasConstantUpperBound()) {
    Operation *constantOperation = builder.create<mlir::arith::ConstantIndexOp>(
        originalForOp->getLoc(),
        originalForOp->getConstantUpperBound() + adjustedDifference);
    constantOperation->setAttr(createdByScheduling, builder.getStringAttr("true"));
    return constantOperation;
  }

  auto applyOperation = builder.create<AffineApplyOp>(
      originalForOp->getLoc(), originalForOp->getUpperBoundMap(),
      originalForOp->getUpperBoundOperands());
  applyOperation->setAttr(createdByScheduling, builder.getStringAttr("true"));

  if (difference == 0) {
    return applyOperation;
  } else {
    auto map =
        AffineMap::get(1, 0,
                       getAffineBinaryOpExpr(
                           AffineExprKind::Add, getAffineDimExpr(0, context),
                           getAffineConstantExpr(adjustedDifference, context)));

    SmallVector<Value, 8> applyOperands{applyOperation->getResult(0)};
    auto additionalApplyOperation = builder.create<AffineApplyOp>(
        originalForOp->getLoc(), map, applyOperands);
    additionalApplyOperation->setAttr(createdByScheduling, builder.getStringAttr("true"));
    return additionalApplyOperation;
  }
}

void mlir::LoopScheduleBuilder::preprocessLoop() {
  if (originalForOp->getNumIterOperands() > 0) {
    originalIterationOperandIndex = originalForOp->getNumOperands() - 1;

    originalIterationArgument =
        originalLoopBody->getArgument(originalLoopBody->getNumArguments() - 1);
    originalYieldOperand = originalLoopBody->getTerminator()->getOperand(0);

    originalIterationMappings[0]->map(
        originalIterationArgument,
        originalForOp->getOperand(originalIterationOperandIndex));

    forwardedLoop = true;
  }
}

void mlir::LoopScheduleBuilder::addPrologue() {
  auto indexVariable = originalForOp->getInductionVar();

  for (auto prologueIteration : schedule.getPrologue()) {

    for (auto cycle : prologueIteration.getCycles()) {

      for (auto scheduledOperation : cycle.getOperations()) {

        auto operands = scheduledOperation.operation->getOperands();

        // Check if some of the operands depend on the iteration variable
        // and if so add additional operations to define constant indexes
        for (auto operand : operands) {
          // If the mapping has already been done for this operand, skip
          // other checks
          if (originalIterationMappings[scheduledOperation.originalIteration]
                  ->contains(operand)) {
            continue;
          }

          // Create constant operation if the operand refers to the
          // iteration variable. The constant operation will have the value
          // of the original iteration from which current operation is taken
          if (operand == indexVariable) {
            auto replacementOperation = createPrologueIndexVariableReplacement(
                scheduledOperation.originalIteration);
            originalIterationMappings[scheduledOperation.originalIteration]
                ->map(operand, replacementOperation->getResult(0));
          }
        }

        // Clone the operation and position it
        Operation *cloned = builder.clone(
            *scheduledOperation.operation,
            *originalIterationMappings[scheduledOperation.originalIteration]);
        originalIterationMappings[scheduledOperation.originalIteration]->erase(
            scheduledOperation.operation->getResult(0));
        cloned->moveBefore(originalForOp->getOperation());
        cloned->dropAllUses();
        cloned->setAttr(createdByScheduling, builder.getStringAttr("true"));
        mapResults(scheduledOperation.originalIteration,
                   scheduledOperation.operation, cloned);

        // Needed for pass-only arguments
        prologueOperations.push_back(cloned);
        prologueOperationsOriginalIterations.push_back(
            scheduledOperation.originalIteration);
        prologueOriginalOperations.push_back(scheduledOperation.operation);

        if (forwardedLoop && originalYieldOperand ==
                                 scheduledOperation.operation->getResult(0)) {
          originalIterationMappings[scheduledOperation.originalIteration + 1]
              ->map(originalIterationArgument, cloned->getResult(0));
        }
      }
    }
  }
}

void mlir::LoopScheduleBuilder::updateLoopLowerBound() {
  if (originalForOp->hasConstantLowerBound()) {
    originalForOp->setConstantLowerBound(
        originalForOp->getConstantLowerBound() + schedule.getPrologue().size());
  } else {
    AffineMap map = originalForOp->getLowerBoundMap();
    std::vector<AffineExpr> results;
    for (auto result : map.getResults()) {
      results.push_back(getAffineBinaryOpExpr(
          AffineExprKind::Add, result,
          getAffineConstantExpr(schedule.getPrologue().size(), context)));
    }

    ArrayRef<AffineExpr> ref{results};

    AffineMap newMap =
        AffineMap::get(map.getNumDims(), map.getNumSymbols(), ref, context);
    originalForOp->setLowerBoundMap(newMap);
  }
}

void mlir::LoopScheduleBuilder::rebuildIteration() {
  auto indexVariable = originalForOp->getInductionVar();

  updateLoopLowerBound();

  std::vector<Operation *> newLoopOperations;
  std::vector<LoopSchedule::ScheduledOperation> scheduledOperations;
  std::vector<int> argumentsOriginalIterations;

  if (forwardedLoop) {
    originalIterationMappings[0]->map(
        originalIterationArgument,
        originalIterationArgument);
    argumentsOriginalIterations.push_back(0);
  }

  for (auto cycle : schedule.getIteration().getCycles()) {

    for (auto scheduledOperation : cycle.getOperations()) {

      builder.setInsertionPointAfter(scheduledOperation.operation);

      // Clone the operation
      Operation *cloned = builder.clone(
          *scheduledOperation.operation,
          *originalIterationMappings[scheduledOperation.originalIteration]);
      cloned->dropAllUses();
      originalIterationMappings[scheduledOperation.originalIteration]->erase(
          scheduledOperation.operation->getResult(0));

      // Go through the operands and add operations where needed
      for (auto operand : cloned->getOperands()) {
        if (originalIterationMappings[scheduledOperation.originalIteration]
                ->contains(operand) ||
            operand == indexVariable || operand.getDefiningOp() == nullptr) {
          continue;
        }

        Operation *definingOperation = operand.getDefiningOp();

        // If operand refers to a constant from the prologue, this means it
        // should refer to an affine apply in the iteration
        if (isa<arith::ConstantIndexOp>(definingOperation) &&
            definingOperation->hasAttr(createdByScheduling)) {
          auto constantOp = cast<arith::ConstantIndexOp>(definingOperation);
          // Create affine map
          auto map = AffineMap::get(
              1, 0,
              getAffineBinaryOpExpr(
                  AffineExprKind::Add, getAffineDimExpr(0, context),
                  getAffineConstantExpr(
                      constantOp.value() -
                          originalForOp->getConstantLowerBound(),
                      context)));

          // Create affine apply operation
          SmallVector<Value, 8> applyOperands{indexVariable};
          auto applyOperation = builder.create<AffineApplyOp>(
              cloned->getLoc(), map, applyOperands);
          newLoopOperations.push_back(applyOperation);
          mapResults(scheduledOperation.originalIteration, definingOperation,
                     applyOperation.getOperation());

          // Replace cloned operation
          Operation *oldCloned = cloned;
          cloned = builder.clone(
              *oldCloned,
              *originalIterationMappings[scheduledOperation.originalIteration]);
          originalIterationMappings[scheduledOperation.originalIteration]
              ->erase(oldCloned->getResult(0));
          oldCloned->erase();
          continue;
        }

        // With non constant lower bound, there is an index op instead of a constant
        // We need again an affine apply in the iteration
        if(operand.getType().isa<IndexType>() &&
            definingOperation->hasAttr(createdByScheduling)){
          
        auto map = AffineMap::get(
              1, 0,
              getAffineBinaryOpExpr(
                  AffineExprKind::Add, getAffineDimExpr(0, context),
                  getAffineConstantExpr(
                      0 - schedule.getPrologue().size(),
                      context)));

        // Create affine apply operation
          SmallVector<Value, 8> applyOperands{indexVariable};
          auto applyOperation = builder.create<AffineApplyOp>(
              cloned->getLoc(), map, applyOperands);
          newLoopOperations.push_back(applyOperation);
          mapResults(scheduledOperation.originalIteration, definingOperation,
                     applyOperation.getOperation());

          // Replace cloned operation
          Operation *oldCloned = cloned;
          cloned = builder.clone(
              *oldCloned,
              *originalIterationMappings[scheduledOperation.originalIteration]);
          originalIterationMappings[scheduledOperation.originalIteration]
              ->erase(oldCloned->getResult(0));
          oldCloned->erase();
          continue;
        }

        // If operand refers to a result from the prologue, it needs to be
        // added as an operand of the iteration
        if (definingOperation->hasAttr(createdByScheduling) &&
            definingOperation->getBlock() != originalLoopBody) {

          // Add operand
          unsigned numOperands = originalForOp->getNumOperands();
          originalForOp->getOperation()->insertOperands(numOperands, {operand});

          // Add argument
          originalLoopBody->addArgument(operand.getType(), originalForOp->getLoc());
          unsigned numArguments = originalLoopBody->getNumArguments();
          originalIterationMappings[scheduledOperation.originalIteration]->map(
              operand,
              originalForOp->getRegion().getArgument(numArguments - 1));

          // Needed for the yield operation generation
          argumentsOriginalIterations.push_back(
              scheduledOperation.originalIteration);
        }

        // Replace cloned operation
        Operation *oldCloned = cloned;
        cloned = builder.clone(
            *oldCloned,
            *originalIterationMappings[scheduledOperation.originalIteration]);
        originalIterationMappings[scheduledOperation.originalIteration]->erase(
            oldCloned->getResult(0));
        oldCloned->erase();
      }

      cloned->setAttr(createdByScheduling, builder.getStringAttr("true"));
      mapResults(scheduledOperation.originalIteration,
                 scheduledOperation.operation, cloned);
      scheduledOperations.push_back(scheduledOperation);
      newLoopOperations.push_back(cloned);

      if (forwardedLoop &&
          originalIterationMappings.size() >
              scheduledOperation.originalIteration + 1 &&
          originalYieldOperand == scheduledOperation.operation->getResult(0)) {
        originalIterationMappings[scheduledOperation.originalIteration + 1]
            ->map(originalIterationArgument, cloned->getResult(0));
      }
    }
  }

  // Add pass-only arguments
  BlockAndValueMapping additionalMapping;
  for (int i = 0; i < prologueOperations.size(); i++) {
    if (prologueOperations[i]->use_empty() && prologueOperations[i]->getNumResults() > 0) {
      // Add operand
      unsigned numOperands = originalForOp->getNumOperands();
      originalForOp->getOperation()->insertOperands(
          numOperands, {prologueOperations[i]->getResult(0)});

      // Add argument
      originalLoopBody->addArgument(
          prologueOperations[i]->getResult(0).getType(), originalForOp->getLoc());

      argumentsOriginalIterations.push_back(
          prologueOperationsOriginalIterations[i]);

      additionalMapping.map(prologueOperations[i]->getResult(0),
                            originalLoopBody->getArgument(
                                originalLoopBody->getNumArguments() - 1));

      originalIterationMappings[prologueOperationsOriginalIterations[i]]->map(
          prologueOperations[i]->getResult(0),
          originalLoopBody->getArgument(originalLoopBody->getNumArguments() -
                                        1));
    }
  }

  // Yield results following the argument order
  std::vector<Value> yieldOperands;
  std::vector<Value> originalYieldValues;
  int loopResultCount = 0;

  for (auto operand :
       originalForOp->getIterOperands()) {

    if (loopResultCount == 0 && forwardedLoop) {
      originalYieldValues.push_back(originalIterationArgument);
      yieldOperands.push_back(originalIterationMappings[1]->lookupOrNull(originalIterationArgument));
      loopResultCount++;
      continue;
    }

// Original implementation:
//    BlockAndValueMapping inverseMap =
//        originalIterationMappings[argumentsOriginalIterations[loopResultCount]]
//            ->getInverse();
// New (temporary?) implementation:
    BlockAndValueMapping inverseMap =
      getInverseMapping(originalIterationMappings[argumentsOriginalIterations[loopResultCount]]);
    auto originalValue = inverseMap.lookupOrNull(operand);

    auto valueToYield =
        originalIterationMappings[argumentsOriginalIterations[loopResultCount] +
                                  1]
            ->lookupOrNull(originalValue);

    if (valueToYield.getDefiningOp()->getParentOp() !=
        originalForOp->getOperation()) {
      valueToYield = additionalMapping.lookupOrNull(valueToYield);
    }

    originalYieldValues.push_back(originalValue);
    yieldOperands.push_back(valueToYield);
    loopResultCount++;
  }

  originalLoopBody->getTerminator()->setOperands(yieldOperands);


  // Create a new loop
  builder.setInsertionPointAfter(originalForOp->getOperation());
  auto newLoop = builder.create<AffineForOp>(
      originalForOp->getLoc(), originalForOp->getLowerBoundOperands(),
      originalForOp->getLowerBoundMap(), originalForOp->getUpperBoundOperands(),
      originalForOp->getUpperBoundMap(), originalForOp->getStep(),
      originalForOp->getIterOperands(),
      [&](OpBuilder &nestedBuilder, Location nestedLoc, Value iv,
          ValueRange itrArgs) {
        nestedBuilder.create<AffineYieldOp>(nestedLoc);
      });

  // Move loop operations to the new loop
  for (auto operation : newLoopOperations) {
    operation->moveBefore(newLoop.getBody()->getTerminator());
  }

  SmallPtrSet<Operation *, 8> originalOperations;
  for (auto cycle : schedule.getIteration().getCycles()) {
    for (auto scheduledOperation : cycle.getOperations()) {
      originalOperations.insert(scheduledOperation.operation);
    }
  }

  // Use new loop arguments
  int index = 1;
  for (auto argument :
       originalForOp->getRegionIterArgs()) {
    argument.replaceAllUsesExcept(newLoop.getBody()->getArgument(index++),
                                  originalOperations);
  }
  originalForOp->getInductionVar().replaceAllUsesWith(
      newLoop.getInductionVar());

  newLoop.getBody()->getTerminator()->setOperands(
      originalLoopBody->getTerminator()->getOperands());

  index = 0;
  for (auto operand : originalYieldValues) {
    // Map new loop result
    originalIterationMappings[argumentsOriginalIterations[index] + 1]->map(
        operand, newLoop.getOperation()->getResult(index++));
  }

  // Needed for building epilogue
  newForOp = newLoop.getOperation();
  newForBlock = newLoop.getBody();
}

Operation *mlir::LoopScheduleBuilder::replaceAffineLoad(Operation *affineLoad) {
  SmallVector<Value, 8> indexOperands;
  int nonIndexOperands = 1;
  for (auto operand = affineLoad->operand_begin();
       operand != affineLoad->operand_end(); operand++) {
    if (nonIndexOperands-- > 0)
      continue;
    indexOperands.append({*operand});
  }

  auto stdLoad = builder.create<memref::LoadOp>(
      affineLoad->getLoc(), affineLoad->getOperand(0), indexOperands);
  affineLoad->erase();

  return stdLoad;
}

Operation *
mlir::LoopScheduleBuilder::replaceAffineStore(Operation *affineStore) {
  SmallVector<Value, 8> indexOperands;
  int nonIndexOperands = 2;
  for (auto operand = affineStore->operand_begin();
       operand != affineStore->operand_end(); operand++) {
    if (nonIndexOperands-- > 0)
      continue;
    indexOperands.append({*operand});
  }

  auto stdStore =
      builder.create<memref::StoreOp>(affineStore->getLoc(), affineStore->getOperand(0),
                              affineStore->getOperand(1), indexOperands);
  affineStore->erase();

  return stdStore;
}

void mlir::LoopScheduleBuilder::addEpilogue() {
  auto indexVariable = newForBlock->getArgument(0);
  builder.setInsertionPointAfter(newForOp);

  Value lastValueToStore;

  for (auto epilogueIteration : schedule.getEpilogue()) {

    for (auto cycle : epilogueIteration.getCycles()) {

      for (auto scheduledOperation : cycle.getOperations()) {

        auto operands = scheduledOperation.operation->getOperands();

        // Check if some of the operands depend on the iteration variable
        // and if so add additional operations to define constant indexes
        for (auto operand : operands) {
          // If the mapping has already be done for this operand, skip other
          // checks
          if (originalIterationMappings[scheduledOperation.originalIteration]
                  ->contains(operand)) {
            continue;
          }

          // Create constant operation if the operand refers to the
          // iteration variable. The constant operation will have the value
          // of the original iteration from which current operation is
          // taken
          if (operand == indexVariable) {
            auto replacementOperation = createEpilogueIndexVariableReplacement(
                scheduledOperation.originalIteration);
            originalIterationMappings[scheduledOperation.originalIteration]
                ->map(operand, replacementOperation->getResult(0));
          }
        }

        // Clone the operation and position it
        Operation *cloned = builder.clone(
            *scheduledOperation.operation,
            *originalIterationMappings[scheduledOperation.originalIteration]);

        originalIterationMappings[scheduledOperation.originalIteration]->erase(
            scheduledOperation.operation->getResult(0));

        // Affine load and store don't accept loop result as an index
//        if (isa<AffineLoadOp>(cloned)) {
//          cloned = replaceAffineLoad(cloned);
//        }
//
//        if (isa<AffineStoreOp>(cloned)) {
//          cloned = replaceAffineStore(cloned);
//        }

        cloned->dropAllUses();
        cloned->setAttr(createdByScheduling, builder.getStringAttr("true"));
        mapResults(scheduledOperation.originalIteration,
                   scheduledOperation.operation, cloned);
        builder.setInsertionPointAfter(cloned);

        if (forwardedLoop &&
            originalYieldOperand ==
                scheduledOperation.operation->getResult(0)) {
          if (originalIterationMappings.size() > scheduledOperation.originalIteration + 1 ) {
            originalIterationMappings[scheduledOperation.originalIteration + 1]
                ->map(originalIterationArgument, cloned->getResult(0));
          } else {
            lastValueToStore = cloned->getResult(0);
          }
        }
      }
    }
  }

  if (forwardedLoop) {
    originalForOp->getResult(0).replaceAllUsesWith(lastValueToStore);
  }
}

void mlir::LoopScheduleBuilder::rebuildLoop() {
  // Initialize original iteration mappings map
  int numOfInitialIterationsInPrologue = schedule.getPrologue().size();
  for (int i = 0; i <= numOfInitialIterationsInPrologue; i++) {
    BlockAndValueMapping *mapping = new BlockAndValueMapping();
    originalIterationMappings.insert({i, mapping});
  }

  preprocessLoop();

  addPrologue();

  rebuildIteration();

  addEpilogue();

  int counter = 0;
  for (auto result: originalForOp->getResults()) {
    result.replaceAllUsesWith(newForOp->getResult(counter++));
  }

  for (Operation &op: newForBlock->getOperations()){
    auto toRemove = op.removeAttr(createdByScheduling);
  }

  for (Operation &op: newForBlock->getParent()->getParentRegion()->getOps()){
    auto toRemove = op.removeAttr(createdByScheduling);
  }

  originalForOp->erase();
}
