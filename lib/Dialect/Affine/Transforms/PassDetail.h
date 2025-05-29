//===- PassDetail.h - SODA Affine Pass class details --------*- C++ ////-*-===//
//===----------------------------------------------------------------------===//

#ifndef DIALECT_SODA_AFFINE_PASSDETAIL_H_
#define DIALECT_SODA_AFFINE_PASSDETAIL_H_

#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
// Forward declaration from Dialect.h
template <typename ConcreteDialect>
void registerDialect(DialectRegistry &registry);

namespace arith {
class ArithDialect;
} // namespace arith

namespace memref {
class MemRefDialect;
} // end namespace memref

namespace vector {
class VectorDialect;
} // end namespace vector

} // end namespace mlir

#endif // DIALECT_SODA_AFFINE_PASSDETAIL_H_
