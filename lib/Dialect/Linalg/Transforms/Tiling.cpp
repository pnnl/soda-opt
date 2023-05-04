//===----------------------------------------------------------------------===//
//
// This file implements a linalg Tiling pass using the impl from dialect/linalg.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

#include "mlir/Dialect/Linalg/TransformOps/LinalgTransformOps.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/IR/TransformOps.h"

#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "soda/Dialect/Linalg/Passes.h"

#include "mlir/Parser/Parser.h"

#include "soda/Dialect/Transform/Transforms/Passes.h"

namespace mlir {
namespace soda {
#define GEN_PASS_DEF_LINALGTILING
#include "soda/Dialect/Linalg/Transforms/Passes.h.inc"
} // namespace soda
} // namespace mlir

#define DEBUG_TYPE "soda-linalg-tiling"

using namespace mlir;
using namespace mlir::soda;

namespace {

// Parse the string in the top of a module.
static LogicalResult parseSourceString(StringRef src, ModuleOp &module,
                                       MLIRContext *context) {
  ParserConfig config(context, /*verifyAfterParse=*/false);
  return parseSourceString(src, module.getBody(), config);
}

// Parse tiling string
// Update internal values with options.tileSizes
static LogicalResult
parseTilingString(ModuleOp &module, MLIRContext *context,
                  mlir::Pass::ListOption<int64_t> &tileSizes,
                  mlir::Pass::Option<std::string> &anchorOp) {
  // Parse the string
  std::string str = R"MLIR(
      transform.sequence failures(propagate) {
      ^bb0(%arg1: !pdl.operation):
        %0 = transform.structured.match ops{["<anchor-op>"]} in %arg1
        %1, %loops:<tileNDims> = transform.structured.tile %0 [<tileSizes>]
      }
    )MLIR";

  // replace <anchor-op> with anchorOp
  str = str.replace(str.find("<anchor-op>"), 11, anchorOp);

  // reaplce <tileSizes> with tileSizes
  std::string tileSizesStr = "";
  for (size_t i = 0; i < tileSizes.size(); i++) {
    tileSizesStr += std::to_string(tileSizes[i]);
    if (i != tileSizes.size() - 1) {
      tileSizesStr += ", ";
    }
  }
  // perform string replacement
  str = str.replace(str.find("<tileSizes>"), 11, tileSizesStr);
  // src = src.replace(src.find("<tileSizes>"), 11, tileSizesStr);

  // replace <tileNDims> with tile.size()
  std::string tileNDimsStr = std::to_string(tileSizes.size());
  str = str.replace(str.find("<tileNDims>"), 11, tileNDimsStr);

  // Parse the string
  return parseSourceString(str, module, context);
}

// Create tiling operation using builder
// TODO: This function is not implemented yet
// static void createTilingOp(ModuleOp &module, MLIRContext *context) {
//   // Create a builder
//   OpBuilder builder(context);

//   // Test we are building in the right module.
//   // Add a simple constant operation to the first function of the module
//   // module.walk([&](func::FuncOp funcOp) {
//   //   builder.setInsertionPointToStart(&funcOp.front());
//   //   builder.create<arith::ConstantOp>(builder.getUnknownLoc(),
//   //    builder.getI32IntegerAttr(42));
//   // });

//   builder.setInsertionPointToStart(module.getBody());

//   // Create IR for the following MLIR code:
//   // transform.sequence failures(propagate) {
//   // ^bb0(%arg1: !pdl.operation):
//   //   %0 = transform.structured.match ops{["linalg.matmul"]} in %arg1
//   //   %1, %loops:2 = transform.structured.tile %0 [4, 4]
//   // }
//   auto sequenceOp = builder.create<transform::SequenceOp>(
//       builder.getUnknownLoc(), /*resultTypes=*/TypeRange{},
//       /*propagation*/ transform::FailurePropagationMode::Propagate,
//       /*bbArgType=*/builder.getType<pdl::OperationType>(),
//       [&](OpBuilder &b, Location loc, Value bbArg) {
//         // create match op
//         auto matchOp = b.create<transform::MatchOp>(
//             loc, bbArg, linalg::MatmulOp::getOperationName());

//         // todo: create tile op
//         // SmallVector<int64_t, 2> tileSizes = {4, 4};
//         // auto tiletoScfForOp =
//         //     b.create<transform::TileOp>(loc, matchOp.getResult(), tileSizes);
//         // auto forLoops = tiletoScfForOp.getLoops();
//         // auto tiledOpH = tiletoScfForOp.getTiledLinalgOp();

//         b.create<transform::YieldOp>(loc);
//       });

//   sequenceOp.print(llvm::outs());

//   return;
// }

struct LinalgTilingPass
    : public mlir::soda::impl::LinalgTilingBase<LinalgTilingPass> {
  LinalgTilingPass() = default;
  LinalgTilingPass(const LinalgTilingOptions &options) {
    this->tileSizes = options.tileSizes;
    this->loopType = options.loopType;

    mlir::linalg::LinalgTilingLoopType type =
        llvm::StringSwitch<mlir::linalg::LinalgTilingLoopType>(loopType)
            .Case("for", mlir::linalg::LinalgTilingLoopType::Loops)
            .Case("affine", mlir::linalg::LinalgTilingLoopType::AffineLoops)
            .Case("parallel", mlir::linalg::LinalgTilingLoopType::ParallelLoops)
            .Default(loopTypeEnum);
    this->loopTypeEnum = type;
  }

  // TODO: tilling configuration is done by modifying a string. This is not
  // ideal. We should use a builder to create the transform tiling operation.
  void runOnOperation() override {
    MLIRContext *context = getOperation().getContext();
    ModuleOp m = getOperation();

    // Create tiling operation using builder
    // createTilingOp(m, context);

    // Create tiling operation using parser
    if (failed(parseTilingString(m, context, tileSizes, anchorOp))) {
      getOperation().emitWarning("Failed to parse source string");
      return signalPassFailure();
    }

    {
      // Using the pass manager
      OpPassManager pm("builtin.module");
      pm.addPass(mlir::soda::trans::createTransformDialectInterpreter());
      pm.addPass(mlir::soda::trans::createTransformDialectEraseSchedule());
      if (failed(runPipeline(pm, getOperation())))
        return signalPassFailure();
    }

    return;
  }

  mlir::linalg::LinalgTilingLoopType loopTypeEnum;
};

} // namespace
