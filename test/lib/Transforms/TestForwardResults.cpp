#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/Bufferization/Transforms/Bufferize.h"
#include "soda/Transforms/ForwardResults.h"
#include "soda/Transforms/SchedulingUtils.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "test-forward-results"

using namespace mlir;

namespace {

struct TestForwardResultsPass
    : public PassWrapper<TestForwardResultsPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestForwardResultsPass)

  void runOnOperation() override;

  TestForwardResultsPass() = default;

  TestForwardResultsPass(const TestForwardResultsPass &pass) {}

  StringRef getArgument() const final { return "test-forward-results"; }
  StringRef getDescription() const final {
    return "Forwards results between loop iterations through affine.yield.";
  }

  Option<int> loopNumber{
      *this, "loop", llvm ::cl::desc("loop sequence number in the function")};
};

} // end anonymous namespace

void TestForwardResultsPass::runOnOperation() {
  func::FuncOp funcOp = getOperation();
  AffineForOp forOp = extractInnermostLoop(funcOp, loopNumber);
  forwardResults(&forOp);
}

namespace mlir {
namespace test {
void registerTestForwardResultsPass() {
  PassRegistration<TestForwardResultsPass>();
}
} // namespace test
} // namespace mlir
