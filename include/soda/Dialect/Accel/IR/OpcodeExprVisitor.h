//===- OpcodeExprVisitor.h - MLIR OpcodeExpr Visitor Class ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the OpcodeExpr visitor class.
// It was derived from the affine expression visitor.
// TODO: Some of the functionality can be removed.
//
//===----------------------------------------------------------------------===//

#ifndef SODA_DIALECT_ACCEL_IR_OPCODEEXPRVISITOR_H
#define SODA_DIALECT_ACCEL_IR_OPCODEEXPRVISITOR_H


#include "soda/Dialect/Accel/IR/OpcodeExpr.h"
#include "llvm/ADT/ArrayRef.h"

namespace mlir {

/// Base class for OpcodeExpr visitors/walkers.
///
/// OpcodeExpr visitors are used when you want to perform different actions
/// for different kinds of OpcodeExprs without having to use lots of casts
/// and a big switch instruction.
///
/// To define your own visitor, inherit from this class, specifying your
/// new type for the 'SubClass' template parameter, and "override" visitXXX
/// functions in your class. This class is defined in terms of statically
/// resolved overloading, not virtual functions.
///
/// For example, here is a visitor that counts the number of for OpcodeDimExprs
/// in an OpcodeExpr.
///
///  /// Declare the class.  Note that we derive from OpcodeExprVisitor
///  /// instantiated with our new subclasses_ type.
///
///  struct DimExprCounter : public OpcodeExprVisitor<DimExprCounter> {
///    unsigned numDimExprs;
///    DimExprCounter() : numDimExprs(0) {}
///    void visitDimExpr(OpcodeDimExpr expr) { ++numDimExprs; }
///  };
///
///  And this class would be used like this:
///    DimExprCounter dec;
///    dec.visit(affineExpr);
///    numDimExprs = dec.numDimExprs;
///
/// OpcodeExprVisitor provides visit methods for the following binary affine
/// op expressions:
/// OpcodeBinaryAddOpExpr, OpcodeBinaryMulOpExpr,
/// OpcodeBinaryModOpExpr, OpcodeBinaryFloorDivOpExpr,
/// OpcodeBinaryCeilDivOpExpr. Note that default implementations of these
/// methods will call the general OpcodeBinaryOpExpr method.
///
/// In addition, visit methods are provided for the following affine
//  expressions: OpcodeConstantExpr, OpcodeDimExpr, and
//  OpcodeSymbolExpr.
///
/// Note that if you don't implement visitXXX for some affine expression type,
/// the visitXXX method for Instruction superclass will be invoked.
///
/// Note that this class is specifically designed as a template to avoid
/// virtual function call overhead. Defining and using a OpcodeExprVisitor is
/// just as efficient as having your own switch instruction over the instruction
/// opcode.

template <typename SubClass, typename RetTy = void> class OpcodeExprVisitor {
  //===--------------------------------------------------------------------===//
  // Interface code - This is the public interface of the OpcodeExprVisitor
  // that you use to visit affine expressions...
public:
  // Function to walk an OpcodeExpr (in post order).
  RetTy walkPostOrder(OpcodeExpr expr) {
    static_assert(std::is_base_of<OpcodeExprVisitor, SubClass>::value,
                  "Must instantiate with a derived type of OpcodeExprVisitor");
    switch (expr.getKind()) {
    case OpcodeExprKind::Add: {
      auto binOpExpr = expr.cast<OpcodeBinaryOpExpr>();
      walkOperandsPostOrder(binOpExpr);
      return static_cast<SubClass *>(this)->visitAddExpr(binOpExpr);
    }
    case OpcodeExprKind::Mul: {
      auto binOpExpr = expr.cast<OpcodeBinaryOpExpr>();
      walkOperandsPostOrder(binOpExpr);
      return static_cast<SubClass *>(this)->visitMulExpr(binOpExpr);
    }
    case OpcodeExprKind::Mod: {
      auto binOpExpr = expr.cast<OpcodeBinaryOpExpr>();
      walkOperandsPostOrder(binOpExpr);
      return static_cast<SubClass *>(this)->visitModExpr(binOpExpr);
    }
    case OpcodeExprKind::FloorDiv: {
      auto binOpExpr = expr.cast<OpcodeBinaryOpExpr>();
      walkOperandsPostOrder(binOpExpr);
      return static_cast<SubClass *>(this)->visitFloorDivExpr(binOpExpr);
    }
    case OpcodeExprKind::CeilDiv: {
      auto binOpExpr = expr.cast<OpcodeBinaryOpExpr>();
      walkOperandsPostOrder(binOpExpr);
      return static_cast<SubClass *>(this)->visitCeilDivExpr(binOpExpr);
    }
    case OpcodeExprKind::Constant:
      return static_cast<SubClass *>(this)->visitConstantExpr(
          expr.cast<OpcodeConstantExpr>());
    case OpcodeExprKind::DimId:
      return static_cast<SubClass *>(this)->visitDimExpr(
          expr.cast<OpcodeDimExpr>());
    case OpcodeExprKind::SymbolId:
      return static_cast<SubClass *>(this)->visitSymbolExpr(
          expr.cast<OpcodeSymbolExpr>());
    }
  }

  // Function to visit an OpcodeExpr.
  RetTy visit(OpcodeExpr expr) {
    static_assert(std::is_base_of<OpcodeExprVisitor, SubClass>::value,
                  "Must instantiate with a derived type of OpcodeExprVisitor");
    switch (expr.getKind()) {
    case OpcodeExprKind::Add: {
      auto binOpExpr = expr.cast<OpcodeBinaryOpExpr>();
      return static_cast<SubClass *>(this)->visitAddExpr(binOpExpr);
    }
    case OpcodeExprKind::Mul: {
      auto binOpExpr = expr.cast<OpcodeBinaryOpExpr>();
      return static_cast<SubClass *>(this)->visitMulExpr(binOpExpr);
    }
    case OpcodeExprKind::Mod: {
      auto binOpExpr = expr.cast<OpcodeBinaryOpExpr>();
      return static_cast<SubClass *>(this)->visitModExpr(binOpExpr);
    }
    case OpcodeExprKind::FloorDiv: {
      auto binOpExpr = expr.cast<OpcodeBinaryOpExpr>();
      return static_cast<SubClass *>(this)->visitFloorDivExpr(binOpExpr);
    }
    case OpcodeExprKind::CeilDiv: {
      auto binOpExpr = expr.cast<OpcodeBinaryOpExpr>();
      return static_cast<SubClass *>(this)->visitCeilDivExpr(binOpExpr);
    }
    case OpcodeExprKind::Constant:
      return static_cast<SubClass *>(this)->visitConstantExpr(
          expr.cast<OpcodeConstantExpr>());
    case OpcodeExprKind::DimId:
      return static_cast<SubClass *>(this)->visitDimExpr(
          expr.cast<OpcodeDimExpr>());
    case OpcodeExprKind::SymbolId:
      return static_cast<SubClass *>(this)->visitSymbolExpr(
          expr.cast<OpcodeSymbolExpr>());
    }
    llvm_unreachable("Unknown OpcodeExpr");
  }

  //===--------------------------------------------------------------------===//
  // Visitation functions... these functions provide default fallbacks in case
  // the user does not specify what to do for a particular instruction type.
  // The default behavior is to generalize the instruction type to its subtype
  // and try visiting the subtype.  All of this should be inlined perfectly,
  // because there are no virtual functions to get in the way.
  //

  // Default visit methods. Note that the default op-specific binary op visit
  // methods call the general visitOpcodeBinaryOpExpr visit method.
  RetTy visitOpcodeBinaryOpExpr(OpcodeBinaryOpExpr expr) { return RetTy(); }
  RetTy visitAddExpr(OpcodeBinaryOpExpr expr) {
    return static_cast<SubClass *>(this)->visitOpcodeBinaryOpExpr(expr);
  }
  RetTy visitMulExpr(OpcodeBinaryOpExpr expr) {
    return static_cast<SubClass *>(this)->visitOpcodeBinaryOpExpr(expr);
  }
  RetTy visitModExpr(OpcodeBinaryOpExpr expr) {
    return static_cast<SubClass *>(this)->visitOpcodeBinaryOpExpr(expr);
  }
  RetTy visitFloorDivExpr(OpcodeBinaryOpExpr expr) {
    return static_cast<SubClass *>(this)->visitOpcodeBinaryOpExpr(expr);
  }
  RetTy visitCeilDivExpr(OpcodeBinaryOpExpr expr) {
    return static_cast<SubClass *>(this)->visitOpcodeBinaryOpExpr(expr);
  }
  RetTy visitConstantExpr(OpcodeConstantExpr expr) { return RetTy(); }
  RetTy visitDimExpr(OpcodeDimExpr expr) { return RetTy(); }
  RetTy visitSymbolExpr(OpcodeSymbolExpr expr) { return RetTy(); }

private:
  // Walk the operands - each operand is itself walked in post order.
  RetTy walkOperandsPostOrder(OpcodeBinaryOpExpr expr) {
    walkPostOrder(expr.getLHS());
    walkPostOrder(expr.getRHS());
  }
};

// This class is used to flatten a pure affine expression (OpcodeExpr,
// which is in a tree form) into a sum of products (w.r.t constants) when
// possible, and in that process simplifying the expression. For a modulo,
// floordiv, or a ceildiv expression, an additional identifier, called a local
// identifier, is introduced to rewrite the expression as a sum of product
// affine expression. Each local identifier is always and by construction a
// floordiv of a pure add/mul affine function of dimensional, symbolic, and
// other local identifiers, in a non-mutually recursive way. Hence, every local
// identifier can ultimately always be recovered as an affine function of
// dimensional and symbolic identifiers (involving floordiv's); note however
// that by OpcodeExpr construction, some floordiv combinations are converted to
// mod's. The result of the flattening is a flattened expression and a set of
// constraints involving just the local variables.
//
// d2 + (d0 + d1) floordiv 4  is flattened to d2 + q where 'q' is the local
// variable introduced, with localVarCst containing 4*q <= d0 + d1 <= 4*q + 3.
//
// The simplification performed includes the accumulation of contributions for
// each dimensional and symbolic identifier together, the simplification of
// floordiv/ceildiv/mod expressions and other simplifications that in turn
// happen as a result. A simplification that this flattening naturally performs
// is of simplifying the numerator and denominator of floordiv/ceildiv, and
// folding a modulo expression to a zero, if possible. Three examples are below:
//
// (d0 + 3 * d1) + d0) - 2 * d1) - d0    simplified to     d0 + d1
// (d0 - d0 mod 4 + 4) mod 4             simplified to     0
// (3*d0 + 2*d1 + d0) floordiv 2 + d1    simplified to     2*d0 + 2*d1
//
// The way the flattening works for the second example is as follows: d0 % 4 is
// replaced by d0 - 4*q with q being introduced: the expression then simplifies
// to: (d0 - (d0 - 4q) + 4) = 4q + 4, modulo of which w.r.t 4 simplifies to
// zero. Note that an affine expression may not always be expressible purely as
// a sum of products involving just the original dimensional and symbolic
// identifiers due to the presence of modulo/floordiv/ceildiv expressions that
// may not be eliminated after simplification; in such cases, the final
// expression can be reconstructed by replacing the local identifiers with their
// corresponding explicit form stored in 'localExprs' (note that each of the
// explicit forms itself would have been simplified).
//
// The expression walk method here performs a linear time post order walk that
// performs the above simplifications through visit methods, with partial
// results being stored in 'operandExprStack'. When a parent expr is visited,
// the flattened expressions corresponding to its two operands would already be
// on the stack - the parent expression looks at the two flattened expressions
// and combines the two. It pops off the operand expressions and pushes the
// combined result (although this is done in-place on its LHS operand expr).
// When the walk is completed, the flattened form of the top-level expression
// would be left on the stack.
//
// A flattener can be repeatedly used for multiple affine expressions that bind
// to the same operands, for example, for all result expressions of an
// OpcodeMap or OpcodeValueMap. In such cases, using it for multiple expressions
// is more efficient than creating a new flattener for each expression since
// common identical div and mod expressions appearing across different
// expressions are mapped to the same local identifier (same column position in
// 'localVarCst').
class SimpleOpcodeExprFlattener
    : public OpcodeExprVisitor<SimpleOpcodeExprFlattener> {
public:
  // Flattend expression layout: [dims, symbols, locals, constant]
  // Stack that holds the LHS and RHS operands while visiting a binary op expr.
  // In future, consider adding a prepass to determine how big the SmallVector's
  // will be, and linearize this to std::vector<int64_t> to prevent
  // SmallVector moves on re-allocation.
  std::vector<SmallVector<int64_t, 8>> operandExprStack;

  unsigned numDims;
  unsigned numSymbols;

  // Number of newly introduced identifiers to flatten mod/floordiv/ceildiv's.
  unsigned numLocals;

  // OpcodeExpr's corresponding to the floordiv/ceildiv/mod expressions for
  // which new identifiers were introduced; if the latter do not get canceled
  // out, these expressions can be readily used to reconstruct the OpcodeExpr
  // (tree) form. Note that these expressions themselves would have been
  // simplified (recursively) by this pass. Eg. d0 + (d0 + 2*d1 + d0) ceildiv 4
  // will be simplified to d0 + q, where q = (d0 + d1) ceildiv 2. (d0 + d1)
  // ceildiv 2 would be the local expression stored for q.
  SmallVector<OpcodeExpr, 4> localExprs;

  SimpleOpcodeExprFlattener(unsigned numDims, unsigned numSymbols);

  virtual ~SimpleOpcodeExprFlattener() = default;

  // Visitor method overrides.
  void visitMulExpr(OpcodeBinaryOpExpr expr);
  void visitAddExpr(OpcodeBinaryOpExpr expr);
  void visitDimExpr(OpcodeDimExpr expr);
  void visitSymbolExpr(OpcodeSymbolExpr expr);
  void visitConstantExpr(OpcodeConstantExpr expr);
  void visitCeilDivExpr(OpcodeBinaryOpExpr expr);
  void visitFloorDivExpr(OpcodeBinaryOpExpr expr);

  //
  // t = expr mod c   <=>  t = expr - c*q and c*q <= expr <= c*q + c - 1
  //
  // A mod expression "expr mod c" is thus flattened by introducing a new local
  // variable q (= expr floordiv c), such that expr mod c is replaced with
  // 'expr - c * q' and c * q <= expr <= c * q + c - 1 are added to localVarCst.
  void visitModExpr(OpcodeBinaryOpExpr expr);

protected:
  // Add a local identifier (needed to flatten a mod, floordiv, ceildiv expr).
  // The local identifier added is always a floordiv of a pure add/mul affine
  // function of other identifiers, coefficients of which are specified in
  // dividend and with respect to a positive constant divisor. localExpr is the
  // simplified tree expression (OpcodeExpr) corresponding to the quantifier.
  virtual void addLocalFloorDivId(ArrayRef<int64_t> dividend, int64_t divisor,
                                  OpcodeExpr localExpr);

  /// Add a local identifier (needed to flatten a mod, floordiv, ceildiv, mul
  /// expr) when the rhs is a symbolic expression. The local identifier added
  /// may be a floordiv, ceildiv, mul or mod of a pure affine/semi-affine
  /// function of other identifiers, coefficients of which are specified in the
  /// lhs of the mod, floordiv, ceildiv or mul expression and with respect to a
  /// symbolic rhs expression. `localExpr` is the simplified tree expression
  /// (OpcodeExpr) corresponding to the quantifier.
  virtual void addLocalIdSemiOpcode(OpcodeExpr localExpr);

private:
  /// Adds `expr`, which may be mod, ceildiv, floordiv or mod expression
  /// representing the affine expression corresponding to the quantifier
  /// introduced as the local variable corresponding to `expr`. If the
  /// quantifier is already present, we put the coefficient in the proper index
  /// of `result`, otherwise we add a new local variable and put the coefficient
  /// there.
  void addLocalVariableSemiOpcode(OpcodeExpr expr,
                                  SmallVectorImpl<int64_t> &result,
                                  unsigned long resultSize);

  // t = expr floordiv c   <=> t = q, c * q <= expr <= c * q + c - 1
  // A floordiv is thus flattened by introducing a new local variable q, and
  // replacing that expression with 'q' while adding the constraints
  // c * q <= expr <= c * q + c - 1 to localVarCst (done by
  // FlatOpcodeConstraints::addLocalFloorDiv).
  //
  // A ceildiv is similarly flattened:
  // t = expr ceildiv c   <=> t =  (expr + c - 1) floordiv c
  void visitDivExpr(OpcodeBinaryOpExpr expr, bool isCeil);

  int findLocalId(OpcodeExpr localExpr);

  inline unsigned getNumCols() const {
    return numDims + numSymbols + numLocals + 1;
  }
  inline unsigned getConstantIndex() const { return getNumCols() - 1; }
  inline unsigned getLocalVarStartIndex() const { return numDims + numSymbols; }
  inline unsigned getSymbolStartIndex() const { return numDims; }
  inline unsigned getDimStartIndex() const { return 0; }
};

} // namespace mlir

#endif // SODA_DIALECT_ACCEL_IR_OPCODEEXPRVISITOR_H
