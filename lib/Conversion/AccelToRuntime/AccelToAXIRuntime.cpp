//===- AccelToAXIRuntime.cpp - Lower accel ops to AXI runtime calls -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Lowers the accel dialect to calls into the AXI DMA runtime library declared
// in soda/ExecutionEngine/axi/AxiRuntime.h.
//
//===----------------------------------------------------------------------===//

#include "soda/Conversion/AccelPasses.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "soda/Dialect/Accel/IR/Accel.h"
#include "llvm/ADT/TypeSwitch.h"

namespace mlir {
namespace accel {
#define GEN_PASS_DEF_CONVERTACCELTOAXIRUNTIME
#include "soda/Conversion/AccelPasses.h.inc"
} // namespace accel
} // namespace mlir

using namespace mlir;
using namespace mlir::accel;

namespace {

/// Declares the runtime functions on demand and creates calls to them.
class RuntimeCallBuilder {
public:
  explicit RuntimeCallBuilder(ModuleOp module) : module(module) {}

  /// Calls runtime function `name`, declaring it on first use. Fails if the
  /// module already has a function `name` of a different type.
  FailureOr<func::CallOp> call(OpBuilder &b, Location loc, StringRef name,
                               TypeRange results, ValueRange args) {
    auto type = b.getFunctionType(args.getTypes(), results);
    auto func = module.lookupSymbol<func::FuncOp>(name);
    if (!func) {
      // Keep the declarations at the top of the module, in creation order.
      OpBuilder::InsertionGuard guard(b);
      if (lastDecl)
        b.setInsertionPointAfter(lastDecl);
      else
        b.setInsertionPointToStart(module.getBody());
      func = b.create<func::FuncOp>(module.getLoc(), name, type);
      func.setPrivate();
      lastDecl = func;
    } else if (func.getFunctionType() != type) {
      return emitError(loc) << "runtime function '" << name << "' is used with "
                            << "type " << type << ", but it is declared with "
                            << "type " << func.getFunctionType();
    }
    return b.create<func::CallOp>(loc, func, args);
  }

private:
  ModuleOp module;
  func::FuncOp lastDecl;
};

/// Name suffix of the runtime functions that handle `elementType`.
static std::optional<StringRef> getRuntimeTypeSuffix(Type elementType) {
  if (elementType.isSignlessInteger(32))
    return StringRef("i32");
  if (elementType.isF32())
    return StringRef("f32");
  return std::nullopt;
}

static Value createI32Constant(OpBuilder &b, Location loc, int64_t value) {
  return b.create<arith::ConstantIntOp>(loc, value, 32);
}

/// Converts an index or integer value to i32.
static Value castToI32(OpBuilder &b, Location loc, Value value) {
  Type i32 = b.getI32Type();
  if (value.getType().isIndex())
    return b.create<arith::IndexCastOp>(loc, i32, value);
  unsigned width = value.getType().getIntOrFloatBitWidth();
  if (width > 32)
    return b.create<arith::TruncIOp>(loc, i32, value);
  if (width < 32)
    return b.create<arith::ExtUIOp>(loc, i32, value);
  return value;
}

/// Number of elements of `memref` as an i32.
static Value getNumElements(OpBuilder &b, Location loc, Value memref) {
  auto type = cast<MemRefType>(memref.getType());
  if (type.hasStaticShape())
    return createI32Constant(b, loc, type.getNumElements());
  Value count = b.create<arith::ConstantIndexOp>(loc, 1);
  for (int64_t dim = 0, rank = type.getRank(); dim < rank; ++dim) {
    Value size = b.createOrFold<memref::DimOp>(loc, memref, dim);
    count = b.createOrFold<arith::MulIOp>(loc, count, size);
  }
  return castToI32(b, loc, count);
}

static Value getOffsetOrZero(OpBuilder &b, Location loc, Value offset) {
  return offset ? offset : createI32Constant(b, loc, 0);
}

/// Value of `accel.send_idx %tile[dim]`: the index of `tile` along `dim`, its
/// offset in the source memref divided by its size.
static Value getTileIndex(OpBuilder &b, Location loc, Value tile,
                          int64_t dim) {
  while (auto cast = tile.getDefiningOp<memref::CastOp>())
    tile = cast.getSource();
  auto subview = tile.getDefiningOp<memref::SubViewOp>();
  if (!subview)
    return createI32Constant(b, loc, 0);
  Value offset = getValueOrCreateConstantIndexOp(
      b, loc, subview.getMixedOffsets()[dim]);
  Value size =
      getValueOrCreateConstantIndexOp(b, loc, subview.getMixedSizes()[dim]);
  return castToI32(b, loc, b.createOrFold<arith::DivUIOp>(loc, offset, size));
}

/// A run of send-like ops sent to the accelerator as one DMA transaction.
struct SendTransaction {
  SmallVector<Operation *> ops;
};

static bool isSendLike(Operation *op) {
  return isa<SendOp, SendLiteralOp, SendDimOp, SendIdxOp>(op);
}

static Value getOffsetOperand(Operation *op) {
  return TypeSwitch<Operation *, Value>(op)
      .Case<SendOp, SendLiteralOp, SendDimOp, SendIdxOp, RecvOp>(
          [](auto op) { return op.getOffsetValue(); })
      .Default([](Operation *) { return Value(); });
}

/// Groups the send-like ops of `block` into transactions. With `coalesce`, an
/// op whose offset is the result of the previous send-like op joins its
/// transaction. A transaction ends at any op that has side effects or
/// regions (a recv, a loop, a call, ...), so the order of transfers and
/// receives is preserved.
static void collectTransactions(Block &block, bool coalesce,
                                SmallVectorImpl<SendTransaction> &result) {
  SendTransaction current;
  auto flush = [&]() {
    if (!current.ops.empty())
      result.push_back(std::move(current));
    current = SendTransaction();
  };
  for (Operation &op : block) {
    if (isSendLike(&op)) {
      Value offset = getOffsetOperand(&op);
      bool chained = coalesce && !current.ops.empty() && offset &&
                     offset == current.ops.back()->getResult(0);
      if (!chained)
        flush();
      current.ops.push_back(&op);
      continue;
    }
    if (op.getNumRegions() != 0 || !isMemoryEffectFree(&op))
      flush();
  }
  flush();
}

class AccelLowering {
public:
  AccelLowering(ModuleOp module) : rewriter(module.getContext()),
                                   runtime(module) {}

  LogicalResult lowerInitDma(InitDMAOp op);
  LogicalResult lowerTransaction(const SendTransaction &transaction);
  LogicalResult lowerRecv(RecvOp op);
  LogicalResult insertDmaFree(func::FuncOp func);

private:
  /// Lowers one send-like op into a copy into the DMA input buffer and
  /// returns the offset after it.
  FailureOr<Value> lowerSendLike(Operation *op, Value offset);

  IRRewriter rewriter;
  RuntimeCallBuilder runtime;
};

LogicalResult AccelLowering::lowerInitDma(InitDMAOp op) {
  rewriter.setInsertionPoint(op);
  Location loc = op.getLoc();
  SmallVector<Value> args;
  for (Value operand : op->getOperands())
    args.push_back(castToI32(rewriter, loc, operand));
  if (failed(runtime.call(rewriter, loc, "dma_init", {}, args)))
    return failure();
  rewriter.eraseOp(op);
  return success();
}

LogicalResult AccelLowering::insertDmaFree(func::FuncOp func) {
  WalkResult result = func.walk([&](func::ReturnOp ret) {
    rewriter.setInsertionPoint(ret);
    if (failed(runtime.call(rewriter, ret.getLoc(), "dma_free", {}, {})))
      return WalkResult::interrupt();
    return WalkResult::advance();
  });
  return failure(result.wasInterrupted());
}

FailureOr<Value> AccelLowering::lowerSendLike(Operation *op, Value offset) {
  Location loc = op->getLoc();
  Type i32 = rewriter.getI32Type();
  rewriter.setInsertionPoint(op);

  auto sendWord = [&](Value word) -> FailureOr<Value> {
    if (failed(runtime.call(rewriter, loc, "copy_literal_to_inbuffer_i32",
                            {i32}, {word, offset})))
      return failure();
    return rewriter.create<arith::AddIOp>(loc, offset,
                                          createI32Constant(rewriter, loc, 1))
        .getResult();
  };

  if (auto send = dyn_cast<SendOp>(op)) {
    auto type = cast<MemRefType>(send.getInput().getType());
    std::optional<StringRef> suffix =
        getRuntimeTypeSuffix(type.getElementType());
    if (!suffix)
      return op->emitOpError() << "has unsupported element type "
                               << type.getElementType()
                               << " (supported: i32, f32)";
    auto unranked = UnrankedMemRefType::get(type.getElementType(),
                                            type.getMemorySpace());
    Value cast = rewriter.create<memref::CastOp>(loc, unranked, send.getInput());
    if (failed(runtime.call(rewriter, loc,
                            ("copy_to_inbuffer_" + *suffix).str(), {i32},
                            {cast, offset})))
      return failure();
    Value count = getNumElements(rewriter, loc, send.getInput());
    return rewriter.create<arith::AddIOp>(loc, offset, count).getResult();
  }
  if (auto literal = dyn_cast<SendLiteralOp>(op))
    return sendWord(literal.getLiteral());
  if (auto sendDim = dyn_cast<SendDimOp>(op)) {
    Value size = rewriter.createOrFold<memref::DimOp>(loc, sendDim.getTile(),
                                                      sendDim.getDim());
    return sendWord(castToI32(rewriter, loc, size));
  }
  auto sendIdx = cast<SendIdxOp>(op);
  return sendWord(
      getTileIndex(rewriter, loc, sendIdx.getTile(), sendIdx.getDim()));
}

LogicalResult
AccelLowering::lowerTransaction(const SendTransaction &transaction) {
  Operation *head = transaction.ops.front();
  rewriter.setInsertionPoint(head);
  Value start =
      getOffsetOrZero(rewriter, head->getLoc(), getOffsetOperand(head));

  Value offset = start;
  for (Operation *op : transaction.ops) {
    // The offset of a chained op is the result of the previous op, which was
    // already replaced with `offset`.
    if (op != head)
      assert(getOffsetOperand(op) == offset && "expected a chained send");
    FailureOr<Value> end = lowerSendLike(op, offset);
    if (failed(end))
      return failure();
    offset = *end;
    if (op == transaction.ops.back()) {
      Location loc = op->getLoc();
      Type i32 = rewriter.getI32Type();
      Value length = rewriter.create<arith::SubIOp>(loc, offset, start);
      if (failed(runtime.call(rewriter, loc, "dma_start_send", {i32},
                              {length, start})) ||
          failed(runtime.call(rewriter, loc, "dma_wait_send", {}, {})))
        return failure();
    }
    rewriter.replaceOp(op, offset);
  }
  return success();
}

LogicalResult AccelLowering::lowerRecv(RecvOp op) {
  Location loc = op.getLoc();
  Type i32 = rewriter.getI32Type();
  rewriter.setInsertionPoint(op);

  auto type = cast<MemRefType>(op.getDst().getType());
  std::optional<StringRef> suffix = getRuntimeTypeSuffix(type.getElementType());
  if (!suffix)
    return op.emitOpError() << "has unsupported element type "
                            << type.getElementType()
                            << " (supported: i32, f32)";

  Value offset = getOffsetOrZero(rewriter, loc, op.getOffsetValue());
  Value count = getNumElements(rewriter, loc, op.getDst());
  if (failed(runtime.call(rewriter, loc, "dma_start_recv", {i32},
                          {count, offset})) ||
      failed(runtime.call(rewriter, loc, "dma_wait_recv", {}, {})))
    return failure();

  auto unranked =
      UnrankedMemRefType::get(type.getElementType(), type.getMemorySpace());
  Value cast = rewriter.create<memref::CastOp>(loc, unranked, op.getDst());
  std::string copyFn = op.isAccumulate()
                           ? ("copy_from_outbuffer_accumulate_" + *suffix).str()
                           : ("copy_from_outbuffer_" + *suffix).str();
  if (failed(runtime.call(rewriter, loc, copyFn, {i32}, {cast, offset})))
    return failure();
  rewriter.replaceOp(op, rewriter.create<arith::AddIOp>(loc, offset, count));
  return success();
}

struct ConvertAccelToAXIRuntimePass
    : public accel::impl::ConvertAccelToAXIRuntimeBase<
          ConvertAccelToAXIRuntimePass> {
  using ConvertAccelToAXIRuntimeBase::ConvertAccelToAXIRuntimeBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    AccelLowering lowering(module);

    // Collect everything first: lowering changes the blocks.
    SmallVector<func::FuncOp> funcsWithDma;
    SmallVector<InitDMAOp> initOps;
    SmallVector<SendTransaction> transactions;
    SmallVector<RecvOp> recvOps;
    module.walk([&](Operation *op) {
      if (auto init = dyn_cast<InitDMAOp>(op)) {
        initOps.push_back(init);
        if (auto func = op->getParentOfType<func::FuncOp>())
          if (funcsWithDma.empty() || funcsWithDma.back() != func)
            funcsWithDma.push_back(func);
      } else if (auto recv = dyn_cast<RecvOp>(op)) {
        recvOps.push_back(recv);
      }
      for (Region &region : op->getRegions())
        for (Block &block : region)
          collectTransactions(block, coalesceSends, transactions);
    });

    for (InitDMAOp op : initOps)
      if (failed(lowering.lowerInitDma(op)))
        return signalPassFailure();
    for (func::FuncOp func : llvm::make_range(funcsWithDma.begin(),
                                              std::unique(funcsWithDma.begin(),
                                                          funcsWithDma.end())))
      if (failed(lowering.insertDmaFree(func)))
        return signalPassFailure();
    for (const SendTransaction &transaction : transactions)
      if (failed(lowering.lowerTransaction(transaction)))
        return signalPassFailure();
    for (RecvOp op : recvOps)
      if (failed(lowering.lowerRecv(op)))
        return signalPassFailure();
  }
};

} // namespace
