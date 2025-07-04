//===- AffineTransformOps.cpp - Implementation of Affine transform ops ----===//
//
// Part of the SODA-OPT project.
//
//===----------------------------------------------------------------------===//

#include "soda/Dialect/Transform/TransformOps/AffineTransformOps.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/IR/TransformOps.h"
#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"

using namespace mlir;
using namespace mlir::affine;

//===----------------------------------------------------------------------===//
// FullLoopUnrollOp
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure transform::FullLoopUnrollOp::applyToOne(
    transform::TransformRewriter &rewriter, Operation *op,
    transform::ApplyToEachResultList &results,
    transform::TransformState &state) {
  LogicalResult result(failure());
  if (AffineForOp affineFor = dyn_cast<AffineForOp>(op))
    result = loopUnrollFull(affineFor);
  else if (scf::ForOp scfFor = dyn_cast<scf::ForOp>(op))
    return emitSilenceableError()
           << "failed to unroll, incorrect type of payload";
  else
    return emitSilenceableError()
           << "failed to unroll, incorrect type of payload";

  if (failed(result))
    return emitSilenceableError() << "failed to unroll";

  return DiagnosedSilenceableFailure::success();
}

//===----------------------------------------------------------------------===//
// Transform op registration
//===----------------------------------------------------------------------===//

namespace {
class SODAAffineTransformDialectExtension
    : public transform::TransformDialectExtension<
          SODAAffineTransformDialectExtension> {
public:
  using Base::Base;

  void init() {
    declareGeneratedDialect<affine::AffineDialect>();
    declareGeneratedDialect<func::FuncDialect>();

    registerTransformOps<
#define GET_OP_LIST
#include "soda/Dialect/Transform/TransformOps/AffineTransformOps.cpp.inc"
        >();
  }
};
} // namespace

#define GET_OP_CLASSES
#include "soda/Dialect/Transform/TransformOps/AffineTransformOps.cpp.inc"

void mlir::soda::transform::registerTransformDialectExtension(
    DialectRegistry &registry) {
  registry.addExtensions<SODAAffineTransformDialectExtension>();
}
