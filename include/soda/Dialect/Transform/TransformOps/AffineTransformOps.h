//===- AffineTransformOps.h - Affine transformation ops ---------*- C++ -*-===//
//
// Part of the SODA-OPT Project.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_AFFINE_TRANSFORMOPS_AFFINETRANSFORMOPS_H
#define MLIR_DIALECT_AFFINE_TRANSFORMOPS_AFFINETRANSFORMOPS_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Dialect/Transform/IR/TransformTypes.h"
#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/LoopLikeInterface.h"

namespace mlir {
namespace func {
class FuncOp;
} // namespace func
namespace scf {
class ForOp;
} // namespace scf
} // namespace mlir

#define GET_OP_CLASSES
#include "soda/Dialect/Transform/TransformOps/AffineTransformOps.h.inc"

namespace mlir {
class DialectRegistry;

namespace soda {
namespace transform {
void registerTransformDialectExtension(DialectRegistry &registry);
} // namespace transform
} // namespace soda
} // namespace mlir

#endif // MLIR_DIALECT_AFFINE_TRANSFORMOPS_AFFINETRANSFORMOPS_H
