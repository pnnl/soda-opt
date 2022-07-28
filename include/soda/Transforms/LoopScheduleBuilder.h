#ifndef MLIR_LOOP_SCHEDULE_BUILDER_H
#define MLIR_LOOP_SCHEDULE_BUILDER_H

#include "soda/Transforms/LoopSchedule.h"

#include <fstream>
#include <iostream>
#include <map>
#include <vector>

namespace mlir {

/// This class rebuilds a loop following the received schedule.
class LoopScheduleBuilder {
public:
  LoopScheduleBuilder(AffineForOp *_forOp, LoopSchedule &_schedule)
      : originalForOp(_forOp), originalLoopBody(_forOp->getBody()),
        schedule(_schedule), builder(_forOp->getOperation()) {
    context = originalForOp->getOperation()->getContext();
  }

  /// Following the schedule obtained by an external tool, this method rebuilds the
  /// given loop in the following way:
  ///   1. Build the prologue
  ///   2. Create a new loop following the new loop iteration
  ///   3. Build the epilogue
  ///   4. Delete the original for operation along with the contained operations
  ///
  /// During steps 1-3, the operations from the original loop are cloned and
  /// their dependencies are fixed (results and arguments). In step 4, these
  /// operations are deleted, as they are not needed anymore.
  void rebuildLoop();

private:
  void addPrologue();

  void rebuildIteration();

  void addEpilogue();

  Operation *createPrologueIndexVariableReplacement(int difference);
  Operation *createEpilogueIndexVariableReplacement(int difference);

  void preprocessLoop();

  void updateLoopLowerBound();

  Operation *replaceAffineLoad(Operation *affineLoad);
  Operation *replaceAffineStore(Operation *affineStore);

  /// Map results of the original operation to the cloned one.
  void mapResults(int originalIteration, Operation *original,
                  Operation *cloned);

  AffineForOp *originalForOp = nullptr;
  Block *originalLoopBody = nullptr;

  Operation *newForOp = nullptr;
  Block *newForBlock = nullptr;

  // Information needed if loop already has an iteration argument as a result
  // of the forwarding pass.
  Value originalIterationArgument, originalYieldOperand;
  int originalIterationOperandIndex;
  bool forwardedLoop = false;

  /// Map with the original iteration numbers as keys, and operation mappings
  /// as values. The mappings are needed for cloning the original operation.
  std::map<int, BlockAndValueMapping *> originalIterationMappings;

  LoopSchedule &schedule;
  OpBuilder builder;
  MLIRContext *context;

  std::vector<Operation *> prologueOperations;
  std::vector<int> prologueOperationsOriginalIterations;
  std::vector<Operation *> prologueOriginalOperations;
};

} // namespace mlir

#endif // MLIR_LOOP_SCHEDUE_BUILDER_H