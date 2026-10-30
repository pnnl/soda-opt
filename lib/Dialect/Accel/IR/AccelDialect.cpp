//===- AccelDialect.cpp - MLIR dialect for Accel implementation -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "soda/Dialect/Accel/IR/Accel.h"

#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/Transforms/InliningUtils.h"

using namespace mlir;
using namespace mlir::accel;

#include "soda/Dialect/Accel/IR/AccelOpsDialect.cpp.inc"

namespace {
/// This class defines the interface for handling inlining with accel
/// operations.
struct AccelInlinerInterface : public DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;

  /// All operations within accel ops can be inlined.
  bool isLegalToInline(Operation *, Region *, bool, IRMapping &) const final {
    return true;
  }
};
} // namespace

void mlir::accel::AccelDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "soda/Dialect/Accel/IR/AccelOps.cpp.inc"
      >();
  addInterfaces<AccelInlinerInterface>();
  registerAttributes();
}

//===----------------------------------------------------------------------===//
// Verification of accel.* attributes
//===----------------------------------------------------------------------===//

/// Returns true if output `initIdx` of `op` is computed as
/// `out = add(out, x)` (arith.addi or arith.addf).
static bool isAddReduction(linalg::LinalgOp op, unsigned initIdx) {
  SmallVector<Operation *, 1> combinerOps;
  Value reduced =
      matchReduction(op.getRegionOutputArgs(), initIdx, combinerOps);
  return reduced && combinerOps.size() == 1 &&
         isa<arith::AddIOp, arith::AddFOp>(combinerOps.front());
}

static LogicalResult verifyOpcodeMapOnLinalg(linalg::LinalgOp op,
                                             OpcodeMapAttr map) {
  unsigned numOperands = op->getNumOperands();
  for (const OpcodeMapEntry &entry : map.getEntries()) {
    for (Attribute attr : entry.actions) {
      auto action = cast<OpcodeActionAttr>(attr);
      if (!action.hasOperand())
        continue;
      auto emitActionError = [&]() {
        InFlightDiagnostic diag = op->emitOpError();
        diag << "opcode '" << entry.name.getValue() << "' action "
             << stringifyOpcodeActionKind(action.getKind()) << "("
             << action.getOperand() << ") ";
        return diag;
      };
      if (action.getOperand() >= numOperands)
        return emitActionError()
               << "refers to operand " << action.getOperand()
               << ", but the op has " << numOperands << " operands";
      OpOperand &operand = op->getOpOperand(action.getOperand());
      auto shapedType = dyn_cast<ShapedType>(operand.get().getType());
      if (!shapedType)
        return emitActionError() << "needs a shaped operand, got "
                                 << operand.get().getType();
      switch (action.getKind()) {
      case OpcodeActionKind::send_dim:
      case OpcodeActionKind::send_idx:
        if (action.getDim() >= shapedType.getRank())
          return emitActionError()
                 << "uses dimension " << action.getDim()
                 << ", but the operand has rank " << shapedType.getRank();
        break;
      case OpcodeActionKind::recv:
        if (!op.isDpsInit(&operand))
          return emitActionError() << "must receive into an output operand";
        if (action.getMode() == RecvMode::accumulate &&
            !isAddReduction(op, operand.getOperandNumber() -
                                    op.getNumDpsInputs()))
          return emitActionError()
                 << "uses accumulate mode, but the op does not compute this "
                    "output as an add reduction";
        break;
      case OpcodeActionKind::send:
      case OpcodeActionKind::send_literal:
        break;
      }
    }
  }
  return success();
}

/// Number of loops tiled for the accelerator according to `accel.accel_dim`,
/// or std::nullopt if the op has no (valid) `accel.accel_dim`.
static std::optional<unsigned> getNumTiledLoops(Operation *op) {
  auto accelDim =
      op->getAttrOfType<AffineMapAttr>(AccelDialect::kAccelDimAttrName);
  if (!accelDim)
    return std::nullopt;
  unsigned count = 0;
  for (AffineExpr expr : accelDim.getValue().getResults()) {
    auto cst = dyn_cast<AffineConstantExpr>(expr);
    if (!cst)
      return std::nullopt;
    count += cst.getValue() != 0;
  }
  return count;
}

static LogicalResult verifyFlow(Operation *op, StringRef attrName,
                                OpcodeFlowAttr flow) {
  auto map = op->getAttrOfType<OpcodeMapAttr>(AccelDialect::kOpcodeMapAttrName);
  if (!map)
    return op->emitOpError() << "has '" << attrName << "' but no '"
                             << AccelDialect::kOpcodeMapAttrName
                             << "' that defines its opcodes";
  LogicalResult result = success();
  flow.walkOpcodes([&](StringAttr name) {
    if (succeeded(result) && !map.lookup(name.getValue()))
      result = op->emitOpError()
               << "'" << attrName << "' uses opcode '" << name.getValue()
               << "', which is not defined in '"
               << AccelDialect::kOpcodeMapAttrName << "'";
  });
  if (failed(result))
    return failure();

  if (attrName == AccelDialect::kInitOpcodesAttrName) {
    if (flow.getDepth() != 1)
      return op->emitOpError() << "'" << attrName
                               << "' must be a single group without nesting";
    return success();
  }
  if (std::optional<unsigned> numTiled = getNumTiledLoops(op)) {
    if (flow.getDepth() > *numTiled + 1)
      return op->emitOpError()
             << "'" << attrName << "' has " << flow.getDepth()
             << " nested groups, but only " << *numTiled
             << " loops are tiled for the accelerator (at most "
             << *numTiled + 1 << " groups)";
  }
  return success();
}

static LogicalResult verifyAccelDim(Operation *op, AffineMap map) {
  for (AffineExpr expr : map.getResults()) {
    auto cst = dyn_cast<AffineConstantExpr>(expr);
    if (!cst || cst.getValue() < 0)
      return op->emitOpError()
             << "'" << AccelDialect::kAccelDimAttrName
             << "' results must be non-negative constants, got "
             << AffineMapAttr::get(map);
  }
  auto linalgOp = dyn_cast<linalg::LinalgOp>(op);
  if (!linalgOp)
    return success();
  unsigned numLoops = linalgOp.getNumLoops();
  if (map.getNumDims() != numLoops || map.getNumResults() != numLoops)
    return op->emitOpError() << "'" << AccelDialect::kAccelDimAttrName
                             << "' must map the op's " << numLoops
                             << " loops to " << numLoops
                             << " tile sizes, got "
                             << AffineMapAttr::get(map);
  return success();
}

static LogicalResult verifyPermutationMap(Operation *op, AffineMap map) {
  if (!map.isPermutation())
    return op->emitOpError() << "'" << AccelDialect::kPermutationMapAttrName
                             << "' must be a permutation, got "
                             << AffineMapAttr::get(map);
  auto linalgOp = dyn_cast<linalg::LinalgOp>(op);
  if (linalgOp && map.getNumDims() != linalgOp.getNumLoops())
    return op->emitOpError() << "'" << AccelDialect::kPermutationMapAttrName
                             << "' must permute the op's "
                             << linalgOp.getNumLoops() << " loops, got "
                             << AffineMapAttr::get(map);
  return success();
}

LogicalResult AccelDialect::verifyOperationAttribute(Operation *op,
                                                     NamedAttribute attr) {
  StringRef name = attr.getName().strref();
  Attribute value = attr.getValue();
  auto expectedKind = [&](StringRef kind) {
    return op->emitOpError()
           << "'" << name << "' must be " << kind << ", got " << value;
  };

  if (name == kOpcodeMapAttrName) {
    auto map = dyn_cast<OpcodeMapAttr>(value);
    if (!map)
      return expectedKind("an #accel.opcode_map");
    if (auto linalgOp = dyn_cast<linalg::LinalgOp>(op))
      return verifyOpcodeMapOnLinalg(linalgOp, map);
    return success();
  }
  if (name == kOpcodeFlowAttrName || name == kInitOpcodesAttrName) {
    auto flow = dyn_cast<OpcodeFlowAttr>(value);
    if (!flow)
      return expectedKind("an #accel.opcode_flow");
    return verifyFlow(op, name, flow);
  }
  if (name == kDmaInitConfigAttrName) {
    if (!isa<DmaConfigAttr>(value))
      return expectedKind("an #accel.dma_config");
    return success();
  }
  if (name == kAccelDimAttrName) {
    auto map = dyn_cast<AffineMapAttr>(value);
    if (!map)
      return expectedKind("an affine_map");
    return verifyAccelDim(op, map.getValue());
  }
  if (name == kPermutationMapAttrName) {
    auto map = dyn_cast<AffineMapAttr>(value);
    if (!map)
      return expectedKind("an affine_map");
    return verifyPermutationMap(op, map.getValue());
  }
  return op->emitOpError() << "has unknown accel attribute '" << name << "'";
}
