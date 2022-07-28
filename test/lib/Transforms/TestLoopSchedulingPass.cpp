#include "soda/Analysis/DataFlowGraph.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Pass/Pass.h"
#include "soda/Transforms/LoopSchedule.h"
#include "soda/Transforms/LoopScheduleBuilder.h"
#include "soda/Transforms/SchedulingUtils.h"

#include <iostream>
#include <map>

#define DEBUG_TYPE "test-loop-scheduling"

using namespace mlir;

namespace {

struct TestLoopSchedulingPass
    : public PassWrapper<TestLoopSchedulingPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestLoopSchedulingPass)

  void runOnOperation() override;

  TestLoopSchedulingPass() = default;

  TestLoopSchedulingPass(const TestLoopSchedulingPass &pass) {}

  StringRef getArgument() const final { return "test-loop-scheduling"; }
  StringRef getDescription() const final {
    return "Rotates operations to achieve given schedule.";
  }

  Option<std::string> scheduleFile{
      *this, "schedule", llvm::cl::desc("schedule in CSV format")};
  Option<int> loopNumber{
      *this, "loop", llvm ::cl::desc("loop sequence number in the function")};
};

} // end anonymous namespace

void TestLoopSchedulingPass::runOnOperation() {

  std::cout << scheduleFile << std::endl;
  // Build data flow graph
  DataFlowGraph graph;
  LoopDataFlowGraphBuilder builder(graph);
  func::FuncOp funcOp = getOperation();

  AffineForOp forOp = extractInnermostLoop(funcOp, loopNumber);

  builder.buildGraphFromLoop(&forOp);

  // Extract schedule
  std::ifstream infile(scheduleFile);
  ExternalLoopSchedule schedule(infile, graph, &forOp);

  if (forOp.hasConstantUpperBound() && forOp.hasConstantLowerBound() &&
      forOp.getConstantUpperBound() - forOp.getConstantLowerBound() <
          static_cast<long int>(schedule.getPrologue().size() + 1)) {
      llvm::errs() << "There are not enough iterations to pipeline this loop." << "\n";
      return;
  }

  if(!forOp.hasConstantUpperBound() || !forOp.hasConstantLowerBound()) {
    AffineMap mapL = forOp.getLowerBoundMap();
    AffineMap mapU = forOp.getUpperBoundMap();
    if(mapL.getNumDims() > 1 || mapU.getNumDims() > 1 || mapL.getNumSymbols() > 1 || mapU.getNumSymbols() > 1) {
      llvm::errs() << "Unsupported variable loop bound." << "\n";
      return;
    }
    else
    {
      conditionLoopExecution(&forOp, schedule);
    }
  }

  LoopScheduleBuilder scheduleBuilder(&forOp, schedule);
  scheduleBuilder.rebuildLoop();
}

namespace mlir {
namespace test {
  void registerTestLoopSchedulingPass() {
    PassRegistration<TestLoopSchedulingPass>();
  }
} // namespace test
} // namespace mlir
