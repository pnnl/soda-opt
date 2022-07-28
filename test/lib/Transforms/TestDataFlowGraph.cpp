#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/AffineStructures.h"
#include "soda/Analysis/DataFlowGraph.h"
#include "mlir/Dialect/Affine/Analysis/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/Bufferization/Transforms/Bufferize.h"
#include "soda/Transforms/SchedulingUtils.h"
#include "llvm/Support/Debug.h"

#include <iostream>

#define DEBUG_TYPE "test-data-flow-graph"

using namespace mlir;

namespace {

struct TestDataFlowGraphPass
    : public PassWrapper<TestDataFlowGraphPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestDataFlowGraphPass)
  
  void runOnOperation() override;

  TestDataFlowGraphPass() = default;

  TestDataFlowGraphPass(const TestDataFlowGraphPass &pass) {}

  StringRef getArgument() const final { return "test-data-flow-graph"; }
  StringRef getDescription() const final {
    return "Prints a data flow graph of the innermost loop.";
  }

  Option<int> loopNumber{
      *this, "loop", llvm ::cl::desc("loop sequence number in the function")};
};

} // end anonymous namespace

void TestDataFlowGraphPass::runOnOperation() {
  DataFlowGraph graph;
  LoopDataFlowGraphBuilder builder(graph);
  func::FuncOp funcOp = getOperation();

  AffineForOp forOp = extractInnermostLoop(funcOp, loopNumber);

  forOp.dump();

  builder.buildGraphFromLoop(&forOp);

  graph.printToGraphmlFile("dfg.graphml");
}

namespace mlir {
namespace test {
void registerTestDataFlowGraphPass() {
  PassRegistration<TestDataFlowGraphPass>();
}
} // namespace test
} // namespace mlir
