//===- soda-opt.cpp ---------------------------------------------*- C++ -*-===//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"

#include "soda/InitUpstreamDialects.h"

#include "soda/Conversion/Passes.h"
// WIP: AXI4MLIR integration
// #include "soda/Dialect/Accel/IR/Accel.h"
#include "soda/Dialect/Linalg/Reports/Passes.h"
#include "soda/Dialect/Linalg/Transforms/Passes.h"
// WIP: Removing SNN dialect support
// #include "soda/Dialect/SNN/IR/SNN.h"
// #include "soda/Dialect/SNN/Transforms/Passes.h"
#include "soda/Dialect/SODA/Passes.h"
#include "soda/Dialect/SODA/SODADialect.h"
#include "soda/Dialect/Transform/TransformOps/AffineTransformOps.h"
#include "soda/Dialect/Transform/Transforms/Passes.h"
#include "soda/Misc/Passes.h"
#include "soda/Misc/Pipelines.h"

#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/Func/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"

#include "mlir/Dialect/Linalg/Transforms/AllInterfaces.h"

#include "mlir/Dialect/Affine/TransformOps/AffineTransformOps.h"
#include "mlir/Dialect/Bufferization/TransformOps/BufferizationTransformOps.h"
#include "mlir/Dialect/Func/TransformOps/FuncTransformOps.h"
#include "mlir/Dialect/Linalg/TransformOps/DialectExtension.h"
#include "mlir/Dialect/SCF/TransformOps/SCFTransformOps.h"
#include "mlir/Dialect/Transform/DebugExtension/DebugExtension.h"
#include "mlir/Dialect/Transform/LoopExtension/LoopExtension.h"

// Defined in the test directory, no public header.
namespace mlir {
void registerTestLoopPermutationPass();
namespace test {
void registerTestLinalgTransforms();
} // namespace test
} // namespace mlir

// Register important linalg passes
inline void registerLinalgPassesForSoda() {
  mlir::registerLinalgPasses();
  mlir::test::registerTestLinalgTransforms();
  mlir::soda::registerLinalgTiling();
}

// Register important affine passes
inline void registerAffinePassesForSoda() {
  mlir::affine::registerAffineDataCopyGenerationPass();
  mlir::affine::registerAffineLoopInvariantCodeMotionPass();
  mlir::affine::registerAffineLoopTilingPass();
  mlir::affine::registerAffineLoopFusionPass();
  mlir::affine::registerAffineLoopUnrollPass();
  mlir::affine::registerAffineScalarReplacementPass();

  // Test passes
  mlir::registerTestLoopPermutationPass();
}

int main(int argc, char **argv) {
  // mlir::registerAllDialects();
  // mlir::registerAllPasses();
  mlir::DialectRegistry registry;

  //===--------------------------------------------------------------------===//
  // Register mlir dialects and passes
  //===--------------------------------------------------------------------===//

  transform::registerTransformPasses();

  mlir::registerInlinerPass();
  mlir::registerCanonicalizerPass();
  mlir::registerCSEPass();

  registerLinalgPassesForSoda();
  registerAffinePassesForSoda();
  mlir::bufferization::registerPromoteBuffersToStackPass();

  mlir::registerConvertLinalgToStandardPass();
  // mlir::registerConvertLinalgToLLVMPass(); // This pass maps linalg to blas
  mlir::registerConvertLinalgToAffineLoopsPass();
  mlir::registerConvertFuncToLLVMPass();
  mlir::registerFinalizeMemRefToLLVMConversionPass();
  mlir::registerSCFToControlFlowPass();
  mlir::registerConvertAffineToStandardPass();
  mlir::registerConvertMathToLLVMPass();
  mlir::registerConvertMathToLibmPass();
  mlir::registerArithToLLVMConversionPass();
  mlir::arith::registerArithExpandOpsPass();
  mlir::memref::registerExpandOpsPass();
  mlir::registerReconcileUnrealizedCastsPass();

  // Add the following to selectively include the necessary dialects. You only
  // need to register dialects that will be *parsed* by the tool, not the one
  // generated

  mlir::registerUpstreamDialects(registry);

  // Interfaces
  linalg::registerAllDialectInterfaceImplementations(registry);

  // Register dialect extensions
  // linalg::registerTransformDialectExtension(registry);
  func::registerTransformDialectExtension(registry);
  linalg::registerTransformDialectExtension(registry);
  bufferization::registerTransformDialectExtension(registry);
  scf::registerTransformDialectExtension(registry);
  transform::registerDebugExtension(registry);
  transform::registerLoopExtension(registry);
  cf::registerConvertControlFlowToLLVMInterface(registry);

  // SODA
  soda::transform::registerTransformDialectExtension(registry);

  // Register external models
  // linalg::registerTilingInterfaceExternalModels(registry);
  memref::registerAllocationOpInterfaceExternalModels(registry);

  //===--------------------------------------------------------------------===//
  // Register SODA dialects and passes
  //===--------------------------------------------------------------------===//

  // Dialects
  registry.insert<mlir::soda::SODADialect>();
  // WIP: AXI4MLIR integration
  // registry.insert<mlir::soda::SODADialect, mlir::snn::SNNDialect,
  //                 mlir::accel::AccelDialect>();

  // ----- SODA -----
  // Misc passes
  mlir::soda::registerTestPrintOpNestingPass();
  mlir::soda::registerTestArgumentsToXMLPass();
  mlir::soda::registerTestArgumentsToCTestbenchPass();
  mlir::soda::registerEraseMemrefDeallocPass();
  mlir::soda::registerForwardMemrefAllocPass();
  mlir::soda::registerForwardLinalgFillPass();
  mlir::soda::registerForwardMemrefCopyPass();
  mlir::soda::linalg::reports::registerGenerateLinalgSummaryPass();

  // Temporary passes to trigger transformations using the transform dialect
  mlir::soda::trans::registerTransformDialectEraseSchedule();

  // SODA Passes
  mlir::soda::registerSodaKernelOutliningPass();
  mlir::soda::registerSodaKernelGenerationPass();
  mlir::soda::registerSodaHostGenerationPass();
  mlir::soda::registerSodaAsyncRegionPassPass();

  // Outlining passes
  mlir::soda::registerConvertAllToSODAPass();
  mlir::soda::registerConvertOperationToSODAPass();
  mlir::soda::registerConvertAffineForToSODAPass();
  mlir::soda::registerConvertSCFForToSODAPass();
  mlir::soda::registerConvertLinalgDotToSODAPass();
  mlir::soda::registerConvertLinalgMatmulToSODAPass();
  mlir::soda::registerConvertLinalgConvToSODAPass();
  mlir::soda::registerConvertLinalgGenericToSODAPass();

  // Optimization passes
  mlir::soda::registerPassManagerMiscPass();
  mlir::soda::registerSimpleLoweringPass();
  mlir::soda::registerOptimizedForBambuPass();
  mlir::soda::registerOptimizedForVitisHLSPass();

  // Conversion passes
  // WIP: AXI4MLIR integration
  // mlir::soda::registerConvertAccelToAXI4MLIR();

  // ----- SNN -----
  // WIP: Remove SNN dialect
  // mlir::snn::registerSNNPrintPass();

  return failed(
      mlir::MlirOptMain(argc, argv, "SODA optimizer driver\n", registry));
}
