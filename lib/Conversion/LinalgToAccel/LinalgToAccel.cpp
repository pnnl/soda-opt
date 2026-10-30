//===- LinalgToAccel.cpp - Lower linalg ops to accel ops ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implements the AXI4MLIR lowering of linalg ops annotated with accel.*
// attributes into loop nests of accel ops.
//
// The tiled loops (non-zero accel.accel_dim, in accel.permutation_map order)
// form a single nest of depth L. Depth j is the body of the j-th loop (depth
// 0 is outside the nest, depth L the innermost body). A flow with D groups
// places group i at depth L - D + 1 + i: the names before its nested group
// run before the loop of depth j + 1, the names after it run after that loop.
//
// The slice of an operand moved at depth j uses the loops bound at depth j
// (offset = induction variable, size = tile size) and the whole range of every
// other loop. That keeps a tile stationary across the loops nested inside.
//
//===----------------------------------------------------------------------===//

#include "soda/Conversion/AccelPasses.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/AsmParser/AsmParser.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Verifier.h"
#include "soda/Dialect/Accel/IR/Accel.h"
#include "llvm/ADT/DenseMap.h"

namespace mlir {
namespace accel {
#define GEN_PASS_DEF_CONVERTLINALGTOACCEL
#include "soda/Conversion/AccelPasses.h.inc"
} // namespace accel
} // namespace mlir

using namespace mlir;
using namespace mlir::accel;

namespace {

/// The accel.* attributes of an op, with defaults filled in.
struct AccelConfig {
  OpcodeMapAttr map;
  OpcodeFlowAttr flow;
  OpcodeFlowAttr init;     // May be null.
  DmaConfigAttr dma;
  SmallVector<int64_t> tileSizes; // Per loop; 0 = not tiled.
  SmallVector<unsigned> loopOrder; // Loops, outermost first.
};

/// Generates the accelerator code of one linalg op.
class AccelCodegen {
public:
  AccelCodegen(RewriterBase &rewriter, linalg::LinalgOp op,
               const AccelConfig &config, bool strictPlacement)
      : rewriter(rewriter), op(op), config(config),
        strictPlacement(strictPlacement), loc(op.getLoc()) {}

  /// Checks the op and computes the loop nest.
  LogicalResult plan();
  /// Emits the code before `op`.
  LogicalResult emit();

private:
  unsigned getNestDepth() const { return tiledLoops.size(); }
  /// Depth of the outermost flow group.
  unsigned getFirstGroupDepth() const {
    return getNestDepth() + 1 - config.flow.getDepth();
  }

  LogicalResult emitDepth(unsigned depth);
  LogicalResult emitOpcodes(ArrayAttr names, unsigned depth);
  LogicalResult emitOpcode(StringRef name, unsigned depth);
  FailureOr<Value> getSlice(unsigned operand, unsigned depth,
                            StringRef opcode);
  OpFoldResult getClosedSize(unsigned loop, unsigned depth);
  void warnOverwrittenReduction(unsigned operand, unsigned depth,
                                StringRef opcode);

  RewriterBase &rewriter;
  linalg::LinalgOp op;
  const AccelConfig &config;
  bool strictPlacement;
  Location loc;

  SmallVector<Range> loopRanges;
  SmallVector<unsigned> tiledLoops;   // Loop dims of the nest, outermost first.
  SmallVector<int64_t> nestPosition;  // Loop dim -> position in nest, or -1.
  SmallVector<Value> ivs;             // Induction variables of the nest.
  DenseMap<std::pair<unsigned, unsigned>, Value> sliceCache;
};

LogicalResult AccelCodegen::plan() {
  if (!op.hasPureBufferSemantics())
    return op.emitOpError()
           << "must have memref operands to be converted to accel ops "
              "(bufferize it first)";

  SmallVector<int64_t> staticRanges = op.getStaticLoopRanges();
  nestPosition.assign(op.getNumLoops(), -1);
  for (unsigned loop : config.loopOrder) {
    int64_t tile = config.tileSizes[loop];
    if (tile == 0)
      continue;
    int64_t range = staticRanges[loop];
    if (ShapedType::isDynamic(range))
      return op.emitOpError() << "loop " << loop
                              << " is tiled for the accelerator but has a "
                                 "dynamic size";
    if (range % tile != 0)
      return op.emitOpError()
             << "loop " << loop << " of size " << range
             << " is not divisible by its accelerator tile size " << tile;
    nestPosition[loop] = tiledLoops.size();
    tiledLoops.push_back(loop);
  }

  unsigned depth = config.flow.getDepth();
  if (depth > getNestDepth() + 1)
    return op.emitOpError()
           << "'" << AccelDialect::kOpcodeFlowAttrName << "' has " << depth
           << " nested groups, but only " << getNestDepth()
           << " loops are tiled for the accelerator (at most "
           << getNestDepth() + 1 << " groups)";
  return success();
}

LogicalResult AccelCodegen::emit() {
  rewriter.setInsertionPoint(op);
  loopRanges = op.createLoopRanges(rewriter, loc);
  if (config.init)
    if (failed(emitOpcodes(config.init.getLevels().front().pre, 0)))
      return failure();
  return emitDepth(0);
}

LogicalResult AccelCodegen::emitDepth(unsigned depth) {
  ArrayRef<OpcodeFlowLevel> levels = config.flow.getLevels();
  int64_t group = static_cast<int64_t>(depth) - getFirstGroupDepth();
  bool hasGroup = group >= 0 && group < static_cast<int64_t>(levels.size());

  if (hasGroup && failed(emitOpcodes(levels[group].pre, depth)))
    return failure();

  if (depth < getNestDepth()) {
    unsigned loop = tiledLoops[depth];
    const Range &range = loopRanges[loop];
    Value lb = getValueOrCreateConstantIndexOp(rewriter, loc, range.offset);
    Value ub = getValueOrCreateConstantIndexOp(
        rewriter, loc,
        affine::makeComposedFoldedAffineApply(
            rewriter, loc,
            rewriter.getAffineSymbolExpr(0) + rewriter.getAffineSymbolExpr(1),
            {range.offset, range.size}));
    Value step = rewriter.create<arith::ConstantIndexOp>(
        loc, config.tileSizes[loop]);
    auto forOp = rewriter.create<scf::ForOp>(loc, lb, ub, step);
    ivs.push_back(forOp.getInductionVar());
    rewriter.setInsertionPoint(forOp.getBody()->getTerminator());
    if (failed(emitDepth(depth + 1)))
      return failure();
    ivs.pop_back();
    rewriter.setInsertionPointAfter(forOp);
  }

  if (hasGroup && failed(emitOpcodes(levels[group].post, depth)))
    return failure();
  return success();
}

LogicalResult AccelCodegen::emitOpcodes(ArrayAttr names, unsigned depth) {
  for (Attribute name : names)
    if (failed(emitOpcode(cast<StringAttr>(name).getValue(), depth)))
      return failure();
  return success();
}

/// Closed size (size - 1) of `loop` in the slices of depth `depth`.
OpFoldResult AccelCodegen::getClosedSize(unsigned loop, unsigned depth) {
  int64_t position = nestPosition[loop];
  if (position >= 0 && position < static_cast<int64_t>(depth))
    return rewriter.getIndexAttr(config.tileSizes[loop] - 1);
  return affine::makeComposedFoldedAffineApply(
      rewriter, loc, rewriter.getAffineSymbolExpr(0) - 1,
      {loopRanges[loop].size});
}

FailureOr<Value> AccelCodegen::getSlice(unsigned operand, unsigned depth,
                                        StringRef opcode) {
  auto key = std::make_pair(operand, depth);
  if (Value slice = sliceCache.lookup(key))
    return slice;

  OpOperand &opOperand = op->getOpOperand(operand);
  AffineMap map = op.getMatchingIndexingMap(&opOperand);

  if (strictPlacement) {
    for (unsigned loop : tiledLoops) {
      if (nestPosition[loop] < static_cast<int64_t>(depth) ||
          !map.isFunctionOfDim(loop))
        continue;
      return op.emitOpError()
             << "opcode '" << opcode << "' moves operand " << operand
             << " outside of loop " << loop
             << ", which the operand depends on; change '"
             << AccelDialect::kPermutationMapAttrName
             << "' to make that loop enclose the opcode";
    }
  }

  SmallVector<OpFoldResult> lbs, closedSizes;
  SmallVector<OpFoldResult> ones(op.getNumLoops(), rewriter.getIndexAttr(1));
  SmallVector<OpFoldResult> ubs;
  for (unsigned loop = 0, e = op.getNumLoops(); loop < e; ++loop) {
    int64_t position = nestPosition[loop];
    if (position >= 0 && position < static_cast<int64_t>(depth))
      lbs.push_back(ivs[position]);
    else
      lbs.push_back(loopRanges[loop].offset);
    closedSizes.push_back(getClosedSize(loop, depth));
    ubs.push_back(loopRanges[loop].size);
  }

  Value source = opOperand.get();
  linalg::SliceParameters params = linalg::computeSliceParameters(
      rewriter, loc, source, ones, map, lbs, ubs, closedSizes,
      /*omitPartialTileCheck=*/true);

  // A slice that covers the whole operand is the operand itself.
  auto type = cast<MemRefType>(source.getType());
  bool isWhole = true;
  for (auto [dim, offset, size] :
       llvm::enumerate(params.offsets, params.sizes)) {
    std::optional<int64_t> cstOffset = getConstantIntValue(offset);
    std::optional<int64_t> cstSize = getConstantIntValue(size);
    if (!cstOffset || *cstOffset != 0 || !cstSize ||
        type.isDynamicDim(dim) || *cstSize != type.getDimSize(dim)) {
      isWhole = false;
      break;
    }
  }
  Value slice = source;
  if (!isWhole)
    slice = rewriter.create<memref::SubViewOp>(loc, source, params.offsets,
                                               params.sizes, params.strides);
  sliceCache[key] = slice;
  return slice;
}

void AccelCodegen::warnOverwrittenReduction(unsigned operand, unsigned depth,
                                            StringRef opcode) {
  AffineMap map = op.getMatchingIndexingMap(&op->getOpOperand(operand));
  SmallVector<utils::IteratorType> iterators = op.getIteratorTypesArray();
  for (unsigned position = 0; position < depth; ++position) {
    unsigned loop = tiledLoops[position];
    if (iterators[loop] != utils::IteratorType::reduction ||
        map.isFunctionOfDim(loop))
      continue;
    op.emitWarning() << "opcode '" << opcode << "' receives operand "
                     << operand << " inside reduction loop " << loop
                     << ", so every iteration overwrites the previous partial "
                        "result; use recv("
                     << operand << ", accumulate) to add them";
    return;
  }
}

LogicalResult AccelCodegen::emitOpcode(StringRef name, unsigned depth) {
  ArrayAttr actions = config.map.lookup(name);
  assert(actions && "opcodes are verified to exist");
  Type i32 = rewriter.getI32Type();

  // Each opcode is one DMA transaction: its sends are chained through their
  // offsets, and so are its receives.
  Value sendOffset, recvOffset;
  int64_t sendWords = 0, recvWords = 0;
  auto countWords = [](Value memref, int64_t &total) {
    auto type = cast<MemRefType>(memref.getType());
    if (type.hasStaticShape() && total >= 0)
      total += type.getNumElements();
    else
      total = -1;
  };

  for (Attribute attr : actions) {
    auto action = cast<OpcodeActionAttr>(attr);
    switch (action.getKind()) {
    case OpcodeActionKind::send_literal: {
      int32_t word =
          static_cast<int32_t>(static_cast<uint32_t>(action.getLiteral()));
      Value literal = rewriter.create<arith::ConstantIntOp>(loc, word, 32);
      sendOffset =
          rewriter.create<SendLiteralOp>(loc, i32, literal, sendOffset);
      if (sendWords >= 0)
        ++sendWords;
      break;
    }
    case OpcodeActionKind::send: {
      FailureOr<Value> slice = getSlice(action.getOperand(), depth, name);
      if (failed(slice))
        return failure();
      sendOffset = rewriter.create<SendOp>(loc, i32, *slice, sendOffset);
      countWords(*slice, sendWords);
      break;
    }
    case OpcodeActionKind::send_dim:
    case OpcodeActionKind::send_idx: {
      FailureOr<Value> slice = getSlice(action.getOperand(), depth, name);
      if (failed(slice))
        return failure();
      auto dim = rewriter.getI64IntegerAttr(action.getDim());
      if (action.getKind() == OpcodeActionKind::send_dim)
        sendOffset =
            rewriter.create<SendDimOp>(loc, i32, *slice, dim, sendOffset);
      else
        sendOffset =
            rewriter.create<SendIdxOp>(loc, i32, *slice, dim, sendOffset);
      if (sendWords >= 0)
        ++sendWords;
      break;
    }
    case OpcodeActionKind::recv: {
      FailureOr<Value> slice = getSlice(action.getOperand(), depth, name);
      if (failed(slice))
        return failure();
      RecvModeAttr mode;
      if (action.getMode() != RecvMode::replace)
        mode = RecvModeAttr::get(rewriter.getContext(), action.getMode());
      else
        warnOverwrittenReduction(action.getOperand(), depth, name);
      recvOffset =
          rewriter.create<RecvOp>(loc, i32, *slice, recvOffset, mode);
      countWords(*slice, recvWords);
      break;
    }
    }
  }

  if (sendWords > config.dma.getInputSize())
    return op.emitOpError() << "opcode '" << name << "' sends " << sendWords
                            << " words, but the DMA input buffer holds "
                            << config.dma.getInputSize();
  if (recvWords > config.dma.getOutputSize())
    return op.emitOpError() << "opcode '" << name << "' receives "
                            << recvWords
                            << " words, but the DMA output buffer holds "
                            << config.dma.getOutputSize();
  return success();
}

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//

/// Parses an attribute given as a pass option. The `#accel.` prefix is
/// optional.
template <typename AttrT>
static FailureOr<AttrT> parseOptionAttr(MLIRContext *ctx, StringRef option,
                                        StringRef text) {
  std::string source = text.trim().str();
  if (source.empty())
    return AttrT();
  if (!StringRef(source).starts_with("#"))
    source = "#accel." + source;
  Attribute attr = parseAttribute(source, ctx);
  auto typed = dyn_cast_or_null<AttrT>(attr);
  if (!typed) {
    emitError(UnknownLoc::get(ctx))
        << "invalid value for option '" << option << "': " << text;
    return failure();
  }
  return typed;
}

struct ConvertLinalgToAccelPass
    : public accel::impl::ConvertLinalgToAccelBase<ConvertLinalgToAccelPass> {
  using ConvertLinalgToAccelBase::ConvertLinalgToAccelBase;

  void runOnOperation() override;

private:
  /// Attaches the attributes given by options to `op`, if it is anchored.
  LogicalResult attachOptionAttrs(linalg::LinalgOp op);
  FailureOr<AccelConfig> getConfig(linalg::LinalgOp op);
  LogicalResult createInitDma(func::FuncOp func, DmaConfigAttr dma,
                              linalg::LinalgOp op);

  OpcodeMapAttr optionMap;
  OpcodeFlowAttr optionFlow, optionInit;
  DmaConfigAttr functionDma;
};

LogicalResult ConvertLinalgToAccelPass::attachOptionAttrs(linalg::LinalgOp op) {
  if (anchorOpName.empty() || op->getName().getStringRef() != anchorOpName)
    return success();
  MLIRContext *ctx = op->getContext();
  auto setIfMissing = [&](StringRef name, Attribute value) {
    if (value && !op->hasAttr(name))
      op->setAttr(name, value);
  };
  setIfMissing(AccelDialect::kOpcodeMapAttrName, optionMap);
  setIfMissing(AccelDialect::kOpcodeFlowAttrName, optionFlow);
  setIfMissing(AccelDialect::kInitOpcodesAttrName, optionInit);
  unsigned numLoops = op.getNumLoops();
  if (!accelDim.empty()) {
    SmallVector<AffineExpr> sizes;
    for (int64_t size : accelDim)
      sizes.push_back(getAffineConstantExpr(size, ctx));
    setIfMissing(AccelDialect::kAccelDimAttrName,
                 AffineMapAttr::get(AffineMap::get(numLoops, 0, sizes, ctx)));
  }
  if (!permutation.empty()) {
    setIfMissing(AccelDialect::kPermutationMapAttrName,
                 AffineMapAttr::get(AffineMap::getPermutationMap(
                     SmallVector<unsigned>(permutation.begin(),
                                           permutation.end()),
                     ctx)));
  }
  // The dialect checks the attributes against the op.
  return verify(op, /*verifyRecursively=*/false);
}

FailureOr<AccelConfig>
ConvertLinalgToAccelPass::getConfig(linalg::LinalgOp op) {
  AccelConfig config;
  config.flow =
      op->getAttrOfType<OpcodeFlowAttr>(AccelDialect::kOpcodeFlowAttrName);
  config.map =
      op->getAttrOfType<OpcodeMapAttr>(AccelDialect::kOpcodeMapAttrName);
  config.init =
      op->getAttrOfType<OpcodeFlowAttr>(AccelDialect::kInitOpcodesAttrName);
  if (!config.map)
    return op.emitOpError() << "has '" << AccelDialect::kOpcodeFlowAttrName
                            << "' but no '" << AccelDialect::kOpcodeMapAttrName
                            << "'";

  config.dma =
      op->getAttrOfType<DmaConfigAttr>(AccelDialect::kDmaInitConfigAttrName);
  if (!config.dma)
    config.dma = DmaConfigAttr::get(op->getContext(), dmaAddress,
                                    dmaInputAddress, dmaInputSize,
                                    dmaOutputAddress, dmaOutputSize);

  unsigned numLoops = op.getNumLoops();
  config.tileSizes.assign(numLoops, 0);
  if (auto accelDimAttr =
          op->getAttrOfType<AffineMapAttr>(AccelDialect::kAccelDimAttrName)) {
    for (auto [loop, expr] :
         llvm::enumerate(accelDimAttr.getValue().getResults()))
      config.tileSizes[loop] = cast<AffineConstantExpr>(expr).getValue();
  }

  for (unsigned loop = 0; loop < numLoops; ++loop)
    config.loopOrder.push_back(loop);
  if (auto permutationAttr = op->getAttrOfType<AffineMapAttr>(
          AccelDialect::kPermutationMapAttrName)) {
    for (auto [position, expr] :
         llvm::enumerate(permutationAttr.getValue().getResults()))
      config.loopOrder[position] = cast<AffineDimExpr>(expr).getPosition();
  }
  return config;
}

LogicalResult ConvertLinalgToAccelPass::createInitDma(func::FuncOp func,
                                                      DmaConfigAttr dma,
                                                      linalg::LinalgOp op) {
  if (functionDma) {
    if (functionDma != dma)
      return op.emitOpError()
             << "uses DMA configuration " << dma << ", but an earlier op in '"
             << func.getSymName() << "' uses " << functionDma
             << "; a function can only use one DMA configuration";
    return success();
  }
  functionDma = dma;
  OpBuilder b = OpBuilder::atBlockBegin(&func.getBody().front());
  Location loc = func.getLoc();
  auto cst = [&](uint32_t value) -> Value {
    return b.create<arith::ConstantIntOp>(loc, static_cast<int32_t>(value), 32);
  };
  b.create<InitDMAOp>(loc, cst(dma.getAddress()), cst(dma.getInputAddress()),
                      cst(dma.getInputSize()), cst(dma.getOutputAddress()),
                      cst(dma.getOutputSize()));
  return success();
}

void ConvertLinalgToAccelPass::runOnOperation() {
  func::FuncOp func = getOperation();
  if (!anchorFuncName.empty() && func.getSymName() != anchorFuncName)
    return;
  MLIRContext *ctx = &getContext();

  FailureOr<OpcodeMapAttr> map =
      parseOptionAttr<OpcodeMapAttr>(ctx, "opcode-map", opcodeMap);
  FailureOr<OpcodeFlowAttr> flow =
      parseOptionAttr<OpcodeFlowAttr>(ctx, "opcode-flow", opcodeFlow);
  FailureOr<OpcodeFlowAttr> init =
      parseOptionAttr<OpcodeFlowAttr>(ctx, "init-opcodes", initOpcodes);
  if (failed(map) || failed(flow) || failed(init))
    return signalPassFailure();
  optionMap = *map;
  optionFlow = *flow;
  optionInit = *init;
  functionDma = nullptr;

  SmallVector<linalg::LinalgOp> candidates;
  func.walk([&](linalg::LinalgOp op) { candidates.push_back(op); });

  IRRewriter rewriter(ctx);
  for (linalg::LinalgOp op : candidates) {
    if (failed(attachOptionAttrs(op)))
      return signalPassFailure();
    if (!op->hasAttr(AccelDialect::kOpcodeFlowAttrName))
      continue;
    FailureOr<AccelConfig> config = getConfig(op);
    if (failed(config))
      return signalPassFailure();
    AccelCodegen codegen(rewriter, op, *config, strictPlacement);
    if (failed(codegen.plan()) ||
        failed(createInitDma(func, config->dma, op)) ||
        failed(codegen.emit()))
      return signalPassFailure();
    rewriter.eraseOp(op);
  }
}

} // namespace
