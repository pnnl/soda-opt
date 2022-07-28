#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/Bufferization/Transforms/Bufferize.h"
#include "soda/Transforms/IfConversion.h"
#include "soda/Transforms/SchedulingUtils.h"
#include "mlir/IR/IntegerSet.h"

#define DEBUG_TYPE "test-if-conversion"

using namespace mlir;

namespace {

struct TestIfConversionPass
    : public PassWrapper<TestIfConversionPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestIfConversionPass)
  
  void runOnOperation() override;

  TestIfConversionPass() = default;

  TestIfConversionPass(const TestIfConversionPass &pass) {}

  StringRef getArgument() const final { return "test-if-conversion"; }
  StringRef getDescription() const final {
    return "If conversion.";
  }

  Option<int> loopNumber{
      *this, "loop", llvm ::cl::desc("loop sequence number in the function")};
};

} // end anonymous namespace

void TestIfConversionPass::runOnOperation() {
  func::FuncOp funcOp = getOperation();

  AffineForOp forOp = extractInnermostLoop(funcOp, loopNumber);
  forOp.walk([&](Operation *operation) {
    if (isa<AffineIfOp>(operation)) {
      AffineIfOp ifOperation = dyn_cast<AffineIfOp>(operation);
      if (ifOperation.getIntegerSet().getNumConstraints() == 1) {
        // We support only single constraint sets.
        convertAffineIf(&ifOperation);
      }
    }
  });
}

namespace mlir {
namespace test {
void registerTestIfConversionPass() {
  PassRegistration<TestIfConversionPass>();
}
} // namespace test
} // namespace mlir
