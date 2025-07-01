//===- AffineTransformOps.td - Affine (loop) transformation ops --*- tablegen -*-===//
//
// Part of the SODA-OPT Project
//
//===----------------------------------------------------------------------===//

#ifndef SODAAFFINE_TRANSFORM_OPS
#define SODAAFFINE_TRANSFORM_OPS

include "mlir/Dialect/Transform/IR/TransformDialect.td"
include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.td"
include "mlir/Dialect/Transform/IR/TransformTypes.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpBase.td"

def FullLoopUnrollOp : Op<Transform_Dialect, "loop.fullunroll",
    [FunctionalStyleTransformOpTrait, MemoryEffectsOpInterface,
     TransformOpInterface, TransformEachOpTrait]> {
  let summary = "Full unrolls the given loop";
  let description = [{
    Unrolls loop associated with the given handle to have up to the given
    number of loop body copies per iteration.

    #### Return modes

    This operation ignores non-`affine.for` ops and drops them
    in the return. If all the operations referred to by the `target` operand
    unroll properly, the transform succeeds. Otherwise the transform produces a
    silenceable failure.

    Does not return handles as the operation may result in the loop being
    removed after a full unrolling.
  }];

  let arguments = (ins TransformHandleTypeInterface:$target);

  let assemblyFormat = "$target attr-dict `:` type($target)";

  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
        ::mlir::transform::TransformRewriter &rewriter,
        ::mlir::Operation *target,
        ::mlir::transform::ApplyToEachResultList &results,
        ::mlir::transform::TransformState &state);
  }];
}

#endif // SODAAFFINE_TRANSFORM_OPS