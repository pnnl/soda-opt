//===- LinalgGenericToAccel.cpp - Generic to accel conversions --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements from linalg generic to accel calls
//
//===----------------------------------------------------------------------===//

#include "soda/Conversion/LinalgToAccel/LinalgGenericToAccel.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"

#include "../PassDetail.h"

#include "soda/Dialect/Accel/IR/Accel.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

#include "mlir/IR/OpcodeExpr.h"

#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

const StringLiteral kLinalgTransformMarker = "__internal_linalg_transform__";
const StringLiteral kAccelTransformMarker = "__accel_transform__";
const StringLiteral kAccel_dmaAddress = "accel_dmaAddress";
const StringLiteral kAccel_dmaInputAddress = "accel_dmaInputAddress";
const StringLiteral kAccel_dmaInputBufferSize = "accel_dmaInputBufferSize";
const StringLiteral kAccel_dmaOuputAddress = "accel_dmaOutputAddress";
const StringLiteral kAccel_dmaOuputBufferSize = "accel_dmaOutputBufferSize";
const StringLiteral kAccel_acc_on_cpu = "accel_acc_on_cpu";
const StringLiteral kAccel_accumulate_on_cpu = "accel_accumulate_on_cpu";
const StringLiteral kAccel_opcode_map = "accel_opcode_map";
const StringLiteral kAccel_opcode_map_str = "accel_opcode_map_str";
const StringLiteral kAccel_opcode_flow = "accel_opcode_flow";
const StringLiteral kAccel_opcode_flow_str = "accel_opcode_flow_str";
const StringLiteral kAccel_loop_permutation = "accel_loop_permutation";
const StringLiteral kAccel_accel_tile_size = "accel_accel_tile_size";
const StringLiteral kAccel_accel_tile_sizes = "accel_accel_tile_sizes";
const StringLiteral kAccel_tile_sizes = "accel_tile_sizes";
const StringLiteral kAccel_init_flow = "accel_init_flow";
const StringLiteral kAccel_init_flow_str = "accel_init_flow_str";

IntegerAttr getU32IntegerAttr(PatternRewriter &rewriter, unsigned value) {
  return rewriter.getIntegerAttr(rewriter.getIntegerType(32, false), value);
}

/// Remove quotes from string to prevent parser from treating it as string.
static StringRef prepStringOption(std::string &s, const char delim = '\"') {
  // NOTE: There is an inconsistent bug with
  //       StringRef::drop_front(),drop_back(),consume_front(),consume_back()
  //       It likely does not update the size every time.
  // NOTE: Input &s must be live after this function call. Passing by copy
  //       also does not work.
  // return StringRef(s).consume_front(delim).consume_back(delim);

  if (s[s.length() - 1] == delim)
    s.erase(s.end() - ((s.length() > 0) ? 1 : 0), s.end());
  if (s[0] == delim)
    s.erase(s.begin());

  return StringRef(s);
}

/// Sets operation Attrs used in generic to accel conversion
class GenericAttrAnnotationPattern
    : public OpRewritePattern<linalg::GenericOp> {
public:
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  /// Construct a generic pattern applied to all GenericOp that verify `filter`.
  /// If attributes are already annotated, skip the replacement.
  GenericAttrAnnotationPattern(
      MLIRContext *context,
      linalg::LinalgTransformationFilter f =
          linalg::LinalgTransformationFilter(),
      AccelTransformationOptions options = AccelTransformationOptions(),
      PatternBenefit benefit = 1)
      : OpRewritePattern<linalg::GenericOp>(context, benefit), filter(f),
        options(std::move(options)) {}

  LogicalResult matchAndRewrite(linalg::GenericOp op,
                                PatternRewriter &rewriter) const override {
    return returningMatchAndRewrite(op, rewriter);
  }

  /// Check if the attribute attrName is already set, if not, use a lambda
  /// function to set it.
  template <typename Func>
  static void setAttrIfEmpty(Operation *op, StringRef attrName, Func lambda) {
    if (!op->getAttr(attrName)) {
      lambda();
    }
  }

  LogicalResult returningMatchAndRewrite(linalg::GenericOp op,
                                         PatternRewriter &rewriter) const {
    if (failed(filter.checkAndNotify(rewriter, op)))
      return failure();
    rewriter.startRootUpdate(op);

    // DMA Attributes
    setAttrIfEmpty(op, kAccel_dmaAddress, [&]() {
      op->setAttr(kAccel_dmaAddress,
                  rewriter.getI32IntegerAttr(options.dmaAddress));
    });
    setAttrIfEmpty(op, kAccel_dmaInputAddress, [&]() {
      op->setAttr(kAccel_dmaInputAddress,
                  rewriter.getI32IntegerAttr(options.dmaInputAddress));
    });
    setAttrIfEmpty(op, kAccel_dmaInputBufferSize, [&]() {
      op->setAttr(kAccel_dmaInputBufferSize,
                  rewriter.getI32IntegerAttr(options.dmaInputBufferSize));
    });
    setAttrIfEmpty(op, kAccel_dmaOuputAddress, [&]() {
      op->setAttr(kAccel_dmaOuputAddress,
                  rewriter.getI32IntegerAttr(options.dmaOutputAddress));
    });
    setAttrIfEmpty(op, kAccel_dmaOuputBufferSize, [&]() {
      op->setAttr(kAccel_dmaOuputBufferSize,
                  rewriter.getI32IntegerAttr(options.dmaOutputBufferSize));
    });
    setAttrIfEmpty(op, kAccel_acc_on_cpu, [&]() {
      op->setAttr(kAccel_acc_on_cpu, rewriter.getBoolAttr(options.flowCpuAcc));
    });

    // OpcodeMap Attribute
    // as string
    std::string s0 = options.opcodeMap;
    StringRef opcodeMapStr = prepStringOption(s0);
    if (opcodeMapStr == "" && !op->getAttr(kAccel_opcode_map_str)) {
      op->emitWarning("No opcode map attribute found, skipping");
      filter.replaceLinalgTransformationFilter(rewriter, op);
      rewriter.finalizeRootUpdate(op);
      return success();
    }
    setAttrIfEmpty(op, kAccel_opcode_map_str, [&]() {
      op->setAttr(kAccel_opcode_map_str, rewriter.getStringAttr(opcodeMapStr));
    });
    // as attribute
    setAttrIfEmpty(op, kAccel_opcode_map, [&]() {
      OpcodeMapAttr opcodeMapAttr =
          parseAttribute(
              op->getAttrOfType<StringAttr>(kAccel_opcode_map_str).getValue(),
              rewriter.getContext())
              .dyn_cast<OpcodeMapAttr>();
      op->setAttr(kAccel_opcode_map, opcodeMapAttr);
    });

    // OpcodeFlow Attribute
    // as string
    std::string s1 = options.opcodeFlow;
    StringRef opcodeFlowStr = prepStringOption(s1);
    setAttrIfEmpty(op, kAccel_opcode_flow_str, [&]() {
      op->setAttr(kAccel_opcode_flow_str,
                  rewriter.getStringAttr(opcodeFlowStr));
    });
    // as attribute
    // TODO: handle kAccel_opcode_flow, parse string to validate identifiers

    // InitFlow Attribute
    // as string
    std::string s2 = options.initFlow;
    StringRef initFlowStr = prepStringOption(s2);
    setAttrIfEmpty(op, kAccel_init_flow_str, [&]() {
      op->setAttr(kAccel_init_flow_str, rewriter.getStringAttr(initFlowStr));
    });
    // as attribute
    // TODO: handle kAccel_init_flow, parse string to validate identifiers

    // Create a lambda function for ArrayRef<unsigned> options
    auto getArrayAttr = [&](const ArrayRef<unsigned> &inArray) -> ArrayAttr {
      SmallVector<Attribute> tmpArray;
      for (auto v : inArray)
        tmpArray.push_back(rewriter.getI32IntegerAttr(v));
      return rewriter.getArrayAttr(tmpArray);
    };

    // Attributes for tilling and permutation
    // TODO: currently the attribute is set correctly but the rewriter pass uses
    // what is inside the command line options

    // LoopPermutation Attribute
    setAttrIfEmpty(op, kAccel_loop_permutation, [&]() {
      op->setAttr(kAccel_loop_permutation,
                  getArrayAttr(options.loopPermutation));
    });

    // AccelSizes Attribute
    setAttrIfEmpty(op, kAccel_accel_tile_sizes, [&]() {
      op->setAttr(kAccel_accel_tile_sizes, getArrayAttr(options.accelSizes));
    });

    // LoopTiling Attribute
    setAttrIfEmpty(op, kAccel_tile_sizes, [&]() {
      op->setAttr(kAccel_tile_sizes, getArrayAttr(options.tileSizes));
    });

    // Accelerator Tile Size Attribute
    setAttrIfEmpty(op, kAccel_accel_tile_size, [&]() {
      op->setAttr(kAccel_accel_tile_size,
                  rewriter.getI32IntegerAttr(options.accelSize));
    });

    // List of operand ids to accumulate on cpu
    setAttrIfEmpty(op, kAccel_accumulate_on_cpu, [&]() {
      op->setAttr(kAccel_accumulate_on_cpu, getArrayAttr(options.accOnCpu));
    });

    filter.replaceLinalgTransformationFilter(rewriter, op);
    rewriter.finalizeRootUpdate(op);
    return success();
  }

private:
  /// LinalgTransformMarker handles special attribute manipulations.
  linalg::LinalgTransformationFilter filter;
  /// Options for accel transformation
  AccelTransformationOptions options;
};

/// Function to materialize DMA attributes as constants
static void materializeDMAConstants(PatternRewriter &rewriter, Operation *op,
                                    Location loc,
                                    SmallVector<Value, 5> &values) {
  values.push_back(rewriter.create<arith::ConstantOp>(
      loc, op->getAttrOfType<IntegerAttr>(kAccel_dmaAddress)));
  values.push_back(rewriter.create<arith::ConstantOp>(
      loc, op->getAttrOfType<IntegerAttr>(kAccel_dmaInputAddress)));
  values.push_back(rewriter.create<arith::ConstantOp>(
      loc, op->getAttrOfType<IntegerAttr>(kAccel_dmaInputBufferSize)));
  values.push_back(rewriter.create<arith::ConstantOp>(
      loc, op->getAttrOfType<IntegerAttr>(kAccel_dmaOuputAddress)));
  values.push_back(rewriter.create<arith::ConstantOp>(
      loc, op->getAttrOfType<IntegerAttr>(kAccel_dmaOuputBufferSize)));
}

/// Rewrites GenericOp as a series of of accel.<operations>
/// Expects the correct attributes to be already set as it
/// does not use options flags and instead, reads the op attributes.
/// TODO: Let this be the case for accelerators with no OPCODES
class LinalgGenericToAccel : public OpRewritePattern<linalg::GenericOp> {
public:
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  // Create a function that depending on an integer, adds a value to the
  // correct loop body in a nested loop structure.
  // ex: if loop_offset = 0,
  //        then add to the innermost loop body, before `op`
  //     if loop_offset = 1,
  //        then add to the second innermost loop body, before terminator the
  //        `op`
  //     if loop_offset = 2,
  //        then add to the third innermost loop body, after the `op`
  //     if loop_offset = -1,
  //        then add to the second innermost loop body, before `op`
  //
  template <typename Func>
  static void addOperationToLoopBody(PatternRewriter &rewriter, Location loc,
                                     Operation *op, int loop_offset,
                                     Func lambda) {

    // if loop_offset = 0, then add to the innermost loop body
    if (loop_offset == 0) {
      // Set insertion point before the operation
      // op->emitWarning() << "Offset is 0, calling lambda";
      rewriter.setInsertionPoint(op);
      lambda();
      return;
    }

    // Get the parent loop operation
    scf::ForOp parent_loop_op = op->getParentOfType<scf::ForOp>();
    assert(
        parent_loop_op &&
        "Accessing parent scf::ForOp, but a parent scf::ForOp was not found.");

    switch (loop_offset) {
    case -1: {
      // op->emitWarning() << "Offset is -1, calling lambda";
      if (parent_loop_op) {
        // Set insertion point right before the scf::ForOp
        rewriter.setInsertionPoint(parent_loop_op);
      }
      lambda();
      break;
    }
    case 1: {
      if (parent_loop_op) {
        // op->emitWarning() << "Offset is 1, calling lambda";
        // Set insertion point before the terminator of parent loop operation
        rewriter.setInsertionPoint(parent_loop_op->getBlock()->getTerminator());
      }
      lambda();
      break;
    }
    default: {
      // if not -1, 0, 1, we have to recursively call this function with parent
      // loop operation as the operation and loop_offset -1 if positive, or +1
      // if negative
      addOperationToLoopBody(
          rewriter, loc, parent_loop_op,
          loop_offset > 0 ? loop_offset - 1 : loop_offset + 1, lambda);
    }
    }
    return;
  }

  // Function to parse accel_opcode_flow_str and generate a vector of where each
  // operation should be placed
  // The attribute opcode flow string has the following format:
  // opcode_flow ::=  opcode_flow_expr
  // opcode_flow_expr ::= `(` opcode_flow_expr `)`
  //                    | `(` opcode_flow_expr opcode_id `)`
  //                    | `(` opcode_id `opcode_flow_expr )`
  //                    | opcode_id
  //

  // Examples and outputs:
  // accel_opcode_flow_str = "(s0 (s1 s2 r2))"
  // [(-1,[s0]), (0,[s1,s2,r2])]
  //
  // accel_opcode_flow_str = "(s0 (s1 s2) r2)"
  // [(-1,[s0]), (0,[s1,s2]), (1,[r2])]
  //
  // accel_opcode_flow_str = "((s0 s1 s2) r2)"
  // [(0,[s0,s1,s2 ]), (1,[r2])]
  static LogicalResult parseOpcodeFlowStr(
      Operation *op, SmallVectorImpl<int> &loop_offsets,
      SmallVectorImpl<std::string> &opcodes_strs,
      SmallVectorImpl<SmallVector<StringRef, 3>> &lists_of_opcode_ids) {
    // op->emitWarning() << "Parsing opcode flow str";
    std::string opcode_flow_str =
        op->getAttrOfType<StringAttr>(kAccel_opcode_flow_str).str();
    // op->emitWarning() << opcode_flow_str;

    assert(!opcode_flow_str.empty() &&
           "accel_opcode_flow_str is empty, but it should not be.");

    int n_left_paren = 0;
    int n_right_paren = 0;
    for (char c : opcode_flow_str) {
      if (c == '(')
        n_left_paren++;
      if (c == ')')
        n_right_paren++;
    }
    assert(n_left_paren == n_right_paren &&
           "accel_opcode_flow_str has mismatched parentheses");

    // get substring between parentheses
    int c_paren = 0;
    for (size_t i = 0; i < opcode_flow_str.size(); i++) {
      if (opcode_flow_str[i] == '(' || opcode_flow_str[i] == ')') {
        size_t j = i + 1;
        while ((opcode_flow_str[j] != ')') && (opcode_flow_str[j] != '(')) {
          j++;
        }
        if (opcode_flow_str[i] == '(' || opcode_flow_str[i] == ')') {
          c_paren++;

          // Only print if still inside parentheses
          if (c_paren < n_left_paren + n_right_paren) {
            // if (j != opcode_flow_str.size()) {
            std::string substring = opcode_flow_str.substr(i + 1, j - i - 1);

            // Only push back if the substring is not empty
            if (!substring.empty()) {
              loop_offsets.push_back(c_paren - n_left_paren);
              opcodes_strs.push_back(substring);
              // op->emitWarning() << substring << " " << c_paren -
              // n_left_paren;
            }
          }
        }
      }
    }

    // The strings in opcodes_strs represent a string of multiple opcodes
    // separated by spaces. We need to split them into individual opcodes.
    for (auto &&opcode_str : opcodes_strs) {
      SmallVector<StringRef, 3> splitted_opcodes;
      StringRef opcode_id_sr = opcode_str;
      // First trim leading and trailing spaces
      opcode_id_sr = opcode_id_sr.trim();
      // Finally split the string into individual opcodes
      opcode_id_sr.split(splitted_opcodes, " ");

      // push back the vector of opcode ids
      lists_of_opcode_ids.push_back(splitted_opcodes);

      // print the opcodes
      // for (auto &&opcode_id_split : splitted_opcodes) {
      //   op->emitWarning() << "Opcode id: " << opcode_id_split<< "!";
      // }
    }

    assert(loop_offsets.size() == lists_of_opcode_ids.size() &&
           "loop_offsets and lists_of_opcode_ids have different sizes");

    // Print the loop offsets and opcode ids
    // op->emitWarning() << "Opcode flow str parsed successfully!"
    //                   << "\n\tloop_offsets: " << loop_offsets
    //                   << "\n\topcodes_strs: " << opcodes_strs
    //                   << "\n\tlists_of_opcode_ids_size: " <<
    //                   lists_of_opcode_ids.size()
    //                   << "\n\tlists_of_opcode_ids: " << lists_of_opcode_ids;

    return success();
  }

  static void printOpcodesInMap(
      Operation *op, SmallVectorImpl<int> &loop_offsets,
      SmallVectorImpl<std::string> &opcodes_strs,
      SmallVectorImpl<SmallVector<StringRef, 3>> &lists_of_opcode_ids) {

    // Get the opcodeMap from operation
    auto opcodeMap =
        op->getAttrOfType<OpcodeMapAttr>(kAccel_opcode_map).getValue();
    llvm::errs() << "OpcodeMap: " << opcodeMap << "\n";
    op->emitWarning() << "Number of opcodes in the map: "
                      << opcodeMap.getNumOpcodes() << "!";

    // Print value associated with opcode in the opcodeMap attribute
    // Use OpcodeList OpcodeMap::getOpcodeList(StringRef key)
    for (auto &&list_of_opcode_ids : lists_of_opcode_ids) {
      for (auto &&opcode_id : list_of_opcode_ids) {
        // Print id and position of opcode in the map
        op->emitWarning() << "Opcode id: " << opcode_id << " at position "
                          << opcodeMap.getOpcodeListPosition(opcode_id) << "!";
        assert(opcodeMap.getOpcodeListPosition(opcode_id) != -1 &&
               "Opcode id not found in the map!");
        OpcodeList opcodeList = opcodeMap.getOpcodeList(opcode_id);
        // Print number of opcodes in the list
        op->emitWarning() << "Number of opcodes in the list: "
                          << opcodeList.getNumActions() << "!";
        // Print id and dump of each opcode in the list
        llvm::errs() << "Opcode id: " << opcode_id << " "
                     << "OpcodeListDump: " << opcodeList << "\n";

        for (auto &&action : opcodeList.getActions()) {
          // Switch case on the kind of action
          switch (action.getKind()) {
          case OpcodeExprKind::Send: {
            auto id = action.cast<OpcodeSendIdExpr>().getId();
            llvm::errs() << "Send action. "
                         << "id: " << id << "\n";
            break;
          }
          case OpcodeExprKind::Recv: {
            llvm::errs() << "Recv action. ";
            break;
          }
          case OpcodeExprKind::SendLiteral: {
            llvm::errs() << "SendLiteral action. ";
            break;
          }
          case OpcodeExprKind::SendDim: {
            llvm::errs() << "SendDim action. ";
            break;
          }
          case OpcodeExprKind::SendIdx: {
            llvm::errs() << "SendIdx action. ";
            break;
          }
          default: {
            llvm_unreachable("Unknown action.");
          }
          }
          llvm::errs() << "action dump: " << action << "\n";
        }
      }
    }
  }

  /// Add accel.send and accel.recv operations to the function based on the
  /// loop_offsets and lists_of_opcode_ids paired with the opcodeMap attribute.
  static void
  addAccelOps(Operation *op, PatternRewriter &rewriter,
              SmallVectorImpl<int> &loop_offsets,
              SmallVectorImpl<SmallVector<StringRef, 3>> &lists_of_opcode_ids) {

    Location loc = op->getLoc();
    // op->emitWarning() << "Adding accel.send and accel.recv operations...";
    auto opcodeMap =
        op->getAttrOfType<OpcodeMapAttr>(kAccel_opcode_map).getValue();
    // llvm::errs() << "OpcodeMap: " << opcodeMap << "\n";
    // op->emitWarning() << "Number of opcodes in the map: "
    //                   << opcodeMap.getNumOpcodes() << "!";

    std::vector<std::pair<int, SmallVector<StringRef, 3>>> zipped;
    std::transform(loop_offsets.begin(), loop_offsets.end(),
                   lists_of_opcode_ids.begin(), std::back_inserter(zipped),
                   [](int a, SmallVector<StringRef, 3> b) {
                     return std::make_pair(a, b);
                   });

    for (auto &&pair : zipped) {
      int loop_offset = pair.first;
      SmallVector<StringRef, 3> list_of_opcode_ids = pair.second;
      for (auto &&opcode_id : list_of_opcode_ids) {
        // Print id and position of opcode in the map
        // op->emitWarning() << "Opcode id: " << opcode_id
        //                   << " in map at position "
        //                   << opcodeMap.getOpcodeListPosition(opcode_id) <<
        //                   "!";
        assert(opcodeMap.getOpcodeListPosition(opcode_id) != -1 &&
               "Opcode id not found in the map!");
        OpcodeList opcodeList = opcodeMap.getOpcodeList(opcode_id);
        // Print number of opcodes in the list
        // op->emitWarning() << "Number of opcodes in the list: "
        //                   << opcodeList.getNumActions() << "!";
        // Print id and dump of each opcode in the list
        // llvm::errs() << "Opcode id: " << opcode_id << " "
        //              << "OpcodeListDump: " << opcodeList << "\n";

        Value initialOffset = nullptr;

        addOperationToLoopBody(rewriter, op->getLoc(), op, loop_offset, [&]() {
          // Create the value to track the offset of the data
          Value cteZero = rewriter.create<arith::ConstantOp>(
              loc, IntegerAttr::get(rewriter.getI32Type(), 0));
          initialOffset = cteZero;
        });

        // Insert the actions in the IR
        for (auto &&action : opcodeList.getActions()) {
          // Switch case on the kind of action
          switch (action.getKind()) {
          case OpcodeExprKind::Send: {
            auto id = action.cast<OpcodeSendIdExpr>().getId();

            Value operand = op->getOperands()[id];
            addOperationToLoopBody(
                rewriter, op->getLoc(), op, loop_offset, [&]() {
                  // Operand is a subview of the original memref, we need to
                  // move this subview to correct loop_offset. We do this by
                  // creating a new memref.subview with the same input
                  // parameters. And replacing the operand with this new
                  // subview.
                  auto subViewOp = operand.getDefiningOp<memref::SubViewOp>();
                  if (!subViewOp) {
                    // Simply create a send operation with the operand
                    initialOffset = rewriter.create<accel::SendOp>(
                        loc, rewriter.getI32Type(), operand, initialOffset);
                    return;
                  }

                  // // TODO: Check if subview has been replaced
                  // // Only create the replacement if the subview has not been
                  // // moved yet. To verify this, check if the parent of the
                  // // subview is the same as the parent of op.
                  // if (subViewOp->getParentOp() == op->getParentOp()) {
                  //   op->emitWarning() << "Subview has already been moved!";
                  //   initialOffset = rewriter.create<accel::SendOp>(
                  //       loc, rewriter.getI32Type(), subViewOp,
                  //       initialOffset);
                  // } else {
                  //   op->emitError() << "Subview has not been moved yet!";
                  //   return;
                  // }

                  // Value newSubView = rewriter.create<memref::SubViewOp>(
                  //     loc, subViewOp.getType(), subViewOp.source(),
                  //     subViewOp.static_offsets(), subViewOp.static_sizes(),
                  //     subViewOp.static_strides());
                  Value newSubView = rewriter.create<memref::SubViewOp>(
                      loc, subViewOp.getType(), subViewOp.source(),
                      subViewOp.offsets(), subViewOp.sizes(),
                      subViewOp.strides(), subViewOp.static_offsets(),
                      subViewOp.static_sizes(), subViewOp.static_strides());

                  // Iterate on the operands, get defining op, if it is a
                  // constantop then move it before the newsubview
                  for (auto &&operand : subViewOp.getOperands()) {
                    Operation *defOp = operand.getDefiningOp();
                    if (defOp && isa<arith::ConstantOp>(defOp)) {
                      defOp->moveBefore(newSubView.getDefiningOp());
                    }
                  }
                  rewriter.replaceOp(subViewOp, newSubView);

                  initialOffset = rewriter.create<accel::SendOp>(
                      loc, rewriter.getI32Type(), newSubView, initialOffset);
                });
            break;
          }
          case OpcodeExprKind::Recv: {
            auto id = action.cast<OpcodeRecvIdExpr>().getId();

            Value operand = op->getOperands()[id];
            addOperationToLoopBody(
                rewriter, op->getLoc(), op, loop_offset, [&]() {
                  auto subViewOp = operand.getDefiningOp<memref::SubViewOp>();
                  if (!subViewOp) {
                    // Simply create a Recv operation with the operand
                    initialOffset = rewriter.create<accel::RecvOp>(
                        loc, rewriter.getI32Type(), operand, initialOffset);
                    return;
                  }

                  // TODO: Check if subview has been replaced

                  Value newSubView = rewriter.create<memref::SubViewOp>(
                      loc, subViewOp.getType(), subViewOp.source(),
                      subViewOp.offsets(), subViewOp.sizes(),
                      subViewOp.strides(), subViewOp.static_offsets(),
                      subViewOp.static_sizes(), subViewOp.static_strides());

                  for (auto &&operand : subViewOp.getOperands()) {
                    Operation *defOp = operand.getDefiningOp();
                    if (defOp && isa<arith::ConstantOp>(defOp)) {
                      defOp->moveBefore(newSubView.getDefiningOp());
                    }
                  }
                  rewriter.replaceOp(subViewOp, newSubView);

                  // Generate accumulation on CPU if needed.
                  bool acc_on_cpu = false;
                  if (op->getAttrOfType<BoolAttr>(kAccel_acc_on_cpu).getValue())
                    acc_on_cpu = true;
                  else {
                    // Set acc_on_cpu true if the operand is in the list of
                    // operands to be accumulated.
                    for (auto &&operand : op->getAttrOfType<ArrayAttr>(
                             kAccel_accumulate_on_cpu)) {
                      if (operand.cast<IntegerAttr>().getInt() == id) {
                        acc_on_cpu = true;
                        break;
                      }
                    }
                  }
                  if (acc_on_cpu) {
                    MemRefType sVmrType =
                        newSubView.getType().cast<MemRefType>();

                    SmallVector<int64_t, 2> shape;
                    auto accelSizes = op->getAttrOfType<ArrayAttr>(
                        kAccel_accel_tile_sizes);

                    // TODO: get shape from SubViewOp creating the subview
                    auto loopPerm = op->getAttrOfType<ArrayAttr>(
                                          kAccel_loop_permutation);
                    int index[3];
                    for (unsigned i = 0; i < 3; i++) {
                      index[loopPerm[i].cast<IntegerAttr>().getInt()]=i;
                    }   
                      // SmallVector<int64_t> rootTileSizes(options.tileSizes.begin(),
                      //                options.tileSizes.begin() +
                      //                    rootOp.getNumLoops());
                    // if access sizes bigger than 0, use them
                    if (accelSizes.size() > 0) {
                      // TODO use begin and end iterator
                      for (unsigned i = 0; i < sVmrType.getRank(); i++) {
                        shape.push_back(accelSizes[index[i]].cast<IntegerAttr>().getInt());
                      }
                    } else {
                      for (unsigned i = 0; i < sVmrType.getRank(); i++) {
                        auto accelSize = op->getAttrOfType<IntegerAttr>(
                            kAccel_accel_tile_size);

                        // TODO: Support multi-dimensions
                        shape.push_back(accelSize.getInt());
                      }
                    }
                    // Transform SmallVector in ArrayRef
                    ArrayRef<int64_t> shapeRef(shape);
                    MemRefType mrType =
                        MemRefType::get(shapeRef, sVmrType.getElementType());
                    Value tMr = rewriter.create<memref::AllocaOp>(loc, mrType);
                    initialOffset = rewriter.create<accel::RecvOp>(
                        loc, rewriter.getI32Type(), tMr, initialOffset);

                    // Create affine maps and attributes for CPU accumulation
                    MemRefType tmpMrType = tMr.getType().cast<MemRefType>();
                    unsigned rank = tmpMrType.getRank();
                    SmallVector<AffineMap, 3> indexingMaps(
                        /*1 inputs, 1 (inplace) output*/ 2,
                        rewriter.getMultiDimIdentityMap(rank));
                    auto loopsAttr = SmallVector<StringRef>(
                        rank, getParallelIteratorTypeName());

                    rewriter.create<linalg::GenericOp>(
                        loc,
                        /*resultTypes=*/TypeRange(),
                        /*inputs=*/tMr,
                        /*outputs=*/newSubView,
                        /*indexingMaps=*/indexingMaps,
                        /*iteratorTypes=*/loopsAttr,
                        /*bodyBuilder=*/
                        [&](OpBuilder &nestedBuilder, Location nestedLoc,
                            ValueRange args) {
                          Value added = nestedBuilder.create<arith::AddIOp>(
                              loc, args[0], args[1]);
                          nestedBuilder.create<linalg::YieldOp>(nestedLoc,
                                                                added);
                        });
                  } else {
                    //     initialOffset = rewriter.create<accel::RecvOp>(
                    //         loc, rewriter.getI32Type(), operand,
                    //         initialOffset);
                    initialOffset = rewriter.create<accel::RecvOp>(
                        loc, rewriter.getI32Type(), newSubView, initialOffset);
                  }
                });
            break;
          }
          case OpcodeExprKind::SendLiteral: {
            auto value = action.cast<OpcodeSendLiteralExpr>().getValue();

            Value literal = rewriter.create<arith::ConstantOp>(
                loc, IntegerAttr::get(rewriter.getI32Type(), value));
            addOperationToLoopBody(
                rewriter, op->getLoc(), op, loop_offset, [&]() {
                  initialOffset = rewriter.create<accel::SendLiteralOp>(
                      loc, rewriter.getI32Type(), literal, initialOffset);
                });
            break;
          }
          case OpcodeExprKind::SendDim: {
            llvm::errs() << "SendDim action. ";
            llvm_unreachable("No support for SendDim yet.");
            break;
          }
          case OpcodeExprKind::SendIdx: {
            llvm::errs() << "No support for SendIdx yet. ";
            break;
          }
          default:
            llvm_unreachable("Unknown action.");
          }
        }
      }
    }
  }

  LogicalResult matchAndRewrite(linalg::GenericOp op,
                                PatternRewriter &rewriter) const override {

    Location loc = op->getLoc();

    // Get location before first operation inside funcOp
    FuncOp funcOp = op->getParentOfType<FuncOp>();
    // Location funcFrontLoc = funcOp.front().front().getLoc();

    rewriter.setInsertionPointToStart(&funcOp.front());
    Location funcFrontLoc = rewriter.getInsertionPoint()->getLoc();

    SmallVector<Value, 5> valuesForInitDMA;
    materializeDMAConstants(rewriter, op, funcFrontLoc, valuesForInitDMA);

    // TODO check if such operation already exists for the same DMA address
    // Create the accel.init_dma operation
    rewriter.create<accel::InitDMAOp>(funcFrontLoc, valuesForInitDMA[0],
                                      valuesForInitDMA[1], valuesForInitDMA[2],
                                      valuesForInitDMA[3], valuesForInitDMA[4]);

    SmallVector<int, 5> loop_offsets;
    SmallVector<std::string, 4> opcodes_strs;
    SmallVector<SmallVector<StringRef, 3>, 4> lists_of_opcode_ids;
    parseOpcodeFlowStr(op, loop_offsets, opcodes_strs, lists_of_opcode_ids);

    // printOpcodesInMap(op, loop_offsets, opcodes_strs, lists_of_opcode_ids);
    addAccelOps(op, rewriter, loop_offsets, lists_of_opcode_ids);

    // for (auto && l: loop_offsets) {
    //   addOperationToLoopBody(rewriter, loc, op, l, [&]() {
    //     op->emitWarning() << "Creating testCte";
    //     // TODO: Create correct accel operation
    //     Value testCte = rewriter.create<arith::ConstantOp>(
    //         loc, IntegerAttr::get(rewriter.getI32Type(), 7777+l));
    //   });
    // }

    // rewriter.setInsertionPoint(op);

    // Value cteZero = rewriter.create<arith::ConstantOp>(
    //     loc, IntegerAttr::get(rewriter.getI32Type(), 0));
    // Value initialOffset = cteZero;

    // for (Value operand : op.inputs()) {
    //   initialOffset = rewriter.create<accel::SendOp>(loc,
    //   rewriter.getI32Type(),
    //                                                  operand, initialOffset);
    // }

    // initialOffset = cteZero;
    // for (Value operand : op.outputs()) {
    //   if (op->getAttrOfType<BoolAttr>(kAccel_acc_on_cpu).getValue()) {
    //     MemRefType mrType = operand.getType().cast<MemRefType>();
    //     Value tMr = rewriter.create<memref::AllocaOp>(loc, mrType);
    //     rewriter.create<accel::RecvOp>(
    //         loc, rewriter.getI32Type(), tMr,
    //         initialOffset); // TODO: Initial offset? Multiple outputs?

    //     // Create affine maps and attributes for CPU accumulation
    //     MemRefType tmpMrType = tMr.getType().cast<MemRefType>();
    //     unsigned rank = tmpMrType.getRank();
    //     SmallVector<AffineMap, 3> indexingMaps(
    //         /*1 inputs, 1 (inplace) output*/ 2,
    //         rewriter.getMultiDimIdentityMap(rank));
    //     auto loopsAttr =
    //         SmallVector<StringRef>(rank, getParallelIteratorTypeName());

    //     rewriter.create<linalg::GenericOp>(
    //         loc,
    //         /*resultTypes=*/TypeRange(),
    //         /*inputs=*/tMr,
    //         /*outputs=*/operand,
    //         /*indexingMaps=*/indexingMaps,
    //         /*iteratorTypes=*/loopsAttr,
    //         /*bodyBuilder=*/
    //         [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange
    //         args) {
    //           Value added =
    //               nestedBuilder.create<arith::AddIOp>(loc, args[0], args[1]);
    //           nestedBuilder.create<linalg::YieldOp>(nestedLoc, added);
    //         });
    //   } else {
    //     initialOffset = rewriter.create<accel::RecvOp>(
    //         loc, rewriter.getI32Type(), operand, initialOffset);
    //   }
    // }
    rewriter.eraseOp(op);

    return success();
  }
};

void mlir::populateLinalgGenericToAccelConversionPatternsWithOptions(
    RewritePatternSet &patterns, const AccelTransformationOptions &options) {
  MLIRContext *ctx = patterns.getContext();
  // This populate patterns that implement the following FSM modifying
  // kLinalgTransformMarker GENERALIZE -> ANNOTATE -> INTERCHANGE -> MEM(TILE)
  // L3(TILE) -> L2(TILE) -> L1(TILE) -> ACCEL
  patterns.add<GenericAttrAnnotationPattern>(
      ctx,
      linalg::LinalgTransformationFilter(StringAttr::get(ctx, "ANNOTATE"),
                                         StringAttr::get(ctx, "INTERCHANGE")),
      options);
  populateCommonLinalgTransformationPatterns(patterns, options);
}

void mlir::populateLinalgGenericToAccelConversionPatterns(
    RewritePatternSet &patterns) {
  patterns.add<LinalgGenericToAccel>(patterns.getContext());
}

namespace {
struct ConvertLinalgGenericToAccelPass
    : public ConvertLinalgGenericToAccelBase<ConvertLinalgGenericToAccelPass> {
  ConvertLinalgGenericToAccelPass() = default;

  /// Constructor to build this pass using user defined options
  /// Not used when the pass is created from commandline, helpful for creating
  /// this pass in code
  ConvertLinalgGenericToAccelPass(const AccelTransformationOptions &options) {
    this->accelSize = options.accelSize;
    this->accelSizes = options.accelSizes;
    this->dmaAddress = options.dmaAddress;
    this->dmaInputAddress = options.dmaInputAddress;
    this->dmaInputBufferSize = options.dmaInputBufferSize;
    this->dmaOutputAddress = options.dmaOutputAddress;
    this->dmaOutputBufferSize = options.dmaOutputAddress;
    this->accOnCpu = options.accOnCpu;
    this->flowCpuAcc = options.flowCpuAcc; // TODO: will be deprecated
    this->numberOfCaches = options.numberOfCaches;
    this->cacheSizes = options.cacheSizes;
    this->tileSizes = options.tileSizes;
    this->elementSize = options.elementSize;
    this->loopPermutation = options.loopPermutation;
    this->anchorFuncName = options.anchorFuncName;
    this->anchorOpName = options.anchorOpName;
    this->anchorFilterName = options.anchorFilterName;
    this->opcodeMap = options.opcodeMap;
    this->initFlow = options.initFlow;
    this->opcodeFlow = options.opcodeFlow;
  }

  void runOnOperation() override;

  void setOptions(AccelTransformationOptions &options) {
    options.accelSize = this->accelSize;
    options.accelSizes = this->accelSizes;
    options.dmaAddress = this->dmaAddress;
    options.dmaInputAddress = this->dmaInputAddress;
    options.dmaInputBufferSize = this->dmaInputBufferSize;
    options.dmaOutputAddress = this->dmaOutputAddress;
    options.dmaOutputBufferSize = this->dmaOutputBufferSize;
    options.accOnCpu = this->accOnCpu;
    options.flowCpuAcc = this->flowCpuAcc; // TODO: will be deprecated
    options.numberOfCaches = this->numberOfCaches;
    options.cacheSizes = this->cacheSizes;
    options.tileSizes = this->tileSizes;
    options.elementSize = this->elementSize;
    options.loopPermutation = this->loopPermutation;
    options.anchorFuncName = this->anchorFuncName;
    options.anchorOpName = this->anchorOpName;
    options.anchorFilterName = this->anchorFilterName;
    options.opcodeMap = this->opcodeMap;
    options.initFlow = this->initFlow;
    options.opcodeFlow = this->opcodeFlow;
  }
};
} // namespace

/// The conversion takes the following steps:
///   1. Marks anchor ops with the "generalize" attribute
///   2. Generalizes the marked ops, marking the Ops with the "ACCEL" attribute
///   3. Annotate attributes to the marked ops
///   4. Convert the marked ops to the accel dialect
void ConvertLinalgGenericToAccelPass::runOnOperation() {

  AccelTransformationOptions options;
  setOptions(options);

  auto module = getOperation();
  MLIRContext *ctx = &getContext();

  // 1. Marks anchor ops with the "GENERALIZE" or "ANNOTATE" attribute
  module.walk([&](FuncOp functionOp) {
    if (!anchorFuncName.empty() && anchorFuncName != functionOp.getName())
      return;

    functionOp.walk([&](linalg::LinalgOp op) {
      if (!anchorFilterName.empty()) {
        // Skip this op if the LinalgOp has kAccelTransformMarker that is not
        // equal to anchorFilterName
        if (op->getAttr(kAccelTransformMarker) !=
            StringAttr::get(ctx, anchorFilterName)) {
          return;
        }
      }

      if ((op->getAttr(kLinalgTransformMarker) !=
           StringAttr::get(ctx, "ACCELERATE"))) {
        if ((anchorOpName != op->getName().getStringRef()))
          return;
      }

      if (isa<linalg::GenericOp>(op)) {
        op->setAttr(kLinalgTransformMarker,
                    StringAttr::get(&getContext(), "ANNOTATE"));
      } else {
        op->setAttr(kLinalgTransformMarker,
                    StringAttr::get(&getContext(), "GENERALIZE"));
      }
    });
  });

  // 2. Generalizes the marked ops, marking the Ops with the next attribute in
  // the FSM. Uses a nested pass manager.
  PassManager pm(module.getContext());
  linalg::LinalgTransformationFilter f(StringAttr::get(ctx, "GENERALIZE"),
                                       StringAttr::get(ctx, "ANNOTATE"));
  pm.addNestedPass<FuncOp>(
      mlir::createLinalgStrategyGeneralizePass(anchorOpName, f));

  if (failed(pm.run(module)))
    signalPassFailure();

  // Using rewrite patterns
  // 3. Annotate attributes to the marked ops
  // 4. Convert the marked ops to the accel dialect
  RewritePatternSet patterns(&getContext());
  populateLinalgGenericToAccelConversionPatternsWithOptions(patterns, options);

  ConversionTarget target(getContext());
  // clang-format off
  target.addLegalDialect<linalg::LinalgDialect,
                         AffineDialect,
                         scf::SCFDialect,
                         memref::MemRefDialect, 
                         accel::AccelDialect, 
                         arith::ArithmeticDialect, 
                         BuiltinDialect,
                         StandardOpsDialect>();
  // clang-format on
  target.addDynamicallyLegalOp<linalg::GenericOp>(
      [&](linalg::GenericOp op) -> bool {
        MLIRContext *ctx = &getContext();
        SmallVector<StringRef, 8> markers = {
            "GENERALIZE", "ANNOTATE", "INTERCHANGE", "MEM", "L3", "L2", "L1"};

        auto aMarkerMatchesAttr = [&](const Attribute &attr) -> bool {
          // Acts like an OR operation, returns true in the first match
          for (auto marker : markers) {
            // TODO: Could be made more efficient by casting attr to StringAttr
            if (StringAttr::get(ctx, marker) == attr)
              return true;
          }
          return false;
        };

        return !(aMarkerMatchesAttr(op->getAttr(kLinalgTransformMarker)));
      });
  if (failed(applyPartialConversion(module, target, std::move(patterns))))
    signalPassFailure();

  RewritePatternSet patterns2(&getContext());
  populateLinalgGenericToAccelConversionPatterns(patterns2);
  target.addDynamicallyLegalOp<linalg::GenericOp>(
      [&](linalg::GenericOp op) -> bool {
        auto marker = StringAttr::get(&getContext(), "GENACCEL");
        return !((op->getAttr(kLinalgTransformMarker) == marker));
      });
  if (failed(applyPartialConversion(module, target, std::move(patterns2))))
    signalPassFailure();
}

std::unique_ptr<OperationPass<ModuleOp>>
mlir::createConvertLinalgGenericToAccelPass() {
  return std::make_unique<ConvertLinalgGenericToAccelPass>();
}

// std::unique_ptr<OperationPass<ModuleOp>>
// mlir::createConvertLinalgGenericToAccelPass(
//     const AccelTransformationOptions &options) {
//   return std::make_unique<ConvertLinalgGenericToAccelPass>(options);
// }
