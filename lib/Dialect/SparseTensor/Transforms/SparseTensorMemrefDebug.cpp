//===- SparseTensorMemrefDebug.cpp - Implementation of memref debugging instrumentation for sparse tensors. --------===//
//===----------------------------------------------------------------------===//
//
// This file implements debugging instrumentation for memref objects that represent
// sparse tensor comopnents (e.g. sparse_tensor.pointers, sparse_tensor.indices, and
// sparse_tensor.values).
//
// It matches .pointers, .indices, and .values  and inserts a call to
// the appropriate runtime library function that prints the memref object.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"

#include "soda/Dialect/SparseTensor/Transforms/Passes.h"

#include <vector>
#include <utility>

namespace mlir {
namespace soda {
#define GEN_PASS_DEF_SPARSETENSORMEMREFDEBUG
#include "soda/Dialect/SparseTensor/Transforms/Passes.h.inc"
} // namespace soda
} // namespace mlir

using namespace mlir;
using namespace mlir::soda;
using namespace mlir::sparse_tensor;

#define DEBUG_TYPE "soda-sparse-compiler-memref-debug"


namespace {

//===--------------------------------------------------------------------------------------------------===//

/// ======= Helpers ========

/// Shorthand aliases for the `emitCInterface` argument to `getFunc()`,
/// `createFuncCallHelper()`, and `replaceOpWithFuncCall()`.
///
/// Pulled from mlir/lib/Dialect/SparseTensor/Transforms/CodegenUtils.cpp from mainline MLIR
enum class EmitCInterfaceHelper : bool { Off = false, On = true };

/// Returns a function reference (first hit also inserts into module). Sets
/// the "_emit_c_interface" on the function declaration when requested,
/// so that LLVM lowering generates a wrapper function that takes care
/// of ABI complications with passing in and returning MemRefs to C functions.
///
/// Pulled from mlir/lib/Dialect/SparseTensor/Transforms/CodegenUtils.cpp from mainline MLIR
static FlatSymbolRefAttr getFuncHelper(ModuleOp module, StringRef name,
                                               TypeRange resultType,
                                               ValueRange operands,
                                               EmitCInterfaceHelper emitCInterface) {
  MLIRContext *context = module.getContext();
  auto result = SymbolRefAttr::get(context, name);
  auto func = module.lookupSymbol<func::FuncOp>(result.getAttr());
  if (!func) {
    OpBuilder moduleBuilder(module.getBodyRegion());
    func = moduleBuilder.create<func::FuncOp>(
        module.getLoc(), name,
        FunctionType::get(context, operands.getTypes(), resultType));
    func.setPrivate();
    if (static_cast<bool>(emitCInterface))
      func->setAttr(LLVM::LLVMDialect::getEmitCWrapperAttrName(),
                    UnitAttr::get(context));
  }
  return result;
}

/// Creates a `CallOp` to the function reference returned by `getFunc()` in
/// the builder's module.
///
/// Pulled from mlir/lib/Dialect/SparseTensor/Transforms/CodegenUtils.cpp from mainline MLIR
static func::CallOp createFuncCallHelper(
    OpBuilder &builder, Location loc, StringRef name, TypeRange resultType,
    ValueRange operands, EmitCInterfaceHelper emitCInterface) {
  auto module = builder.getBlock()->getParentOp()->getParentOfType<ModuleOp>();
  FlatSymbolRefAttr fn =
      getFuncHelper(module, name, resultType, operands, emitCInterface);
  return builder.create<func::CallOp>(loc, resultType, fn, operands);
}

/// Generates MLIR runtime calls to print memref values 
/// that represent (dense and sparse) tensor values
///
/// Modified from mlir/lib/Dialect/SparseTensor/Transforms/CodegenUtils.cpp from sgh185/mlir/sparse branch
static void createMemRefDebugPrint(
  OpBuilder &builder, 
  Location loc,
  Value tensor,
  Type internalTp,
  Value memRefOp,
  unsigned lvl,
  int64_t componentNum
) {
  /*
   * TOP
   *
   * Makes a call to the corresponding debug function in the 
   * mlir-cpu-runner library. *Assumes* that the memref type
   * is dynamic, and only checks the internal type @internalTp
   * 
   * The runtime library takes unranked memrefs, so we must
   * cast to lose the rank information.
   */

  /*
   * Insert calls only for inputs -> check if @tensor
   * is a function argument to @main or the corresponding
   * entry point (not implemented). Otherwise, don't print.
   * 
   * The reasoning here -> some tensors might be intermediate
   * tensors that are deallocated via the bufferization process.
   * We don't print these, since it's possible they are invalid
   * pointers when printing at runtime.
   */
  func::FuncOp func = dyn_cast<func::FuncOp>(builder.getBlock()->getParentOp());
  if (!func) {
    return;
  } 

  bool tensorIsArg = false;
  unsigned tid = 0;
  for (auto arg : func.getArguments()) {
    if (arg == tensor) {
      tensorIsArg = true;
      break;
    };
    tid++;
  }

  if (!tensorIsArg) {
    return;
  }

  /*
   * Determine the debug function name from @internalTp
   *
   * TODO: Simplify
   */
  std::string suffix = "";
  if (internalTp.isF32()) suffix = "F32";
  else if (internalTp.isF64()) suffix = "F64";
  else if (internalTp.isInteger(32)) suffix = "I32";
  else if (internalTp.isInteger(64)) suffix = "I64";
  else if (internalTp.isIndex()) suffix = "Ind";
  else llvm_unreachable("Unknown internal type");

  std::string funcName = "printMemref" + suffix;

  /*
   * Cast the memref to an unranked memref
   */
  UnrankedMemRefType castMemrefType = UnrankedMemRefType::get(internalTp, /*memorySpace=*/0);
  Value castOp = builder.create<memref::CastOp>(loc, castMemrefType, memRefOp);

  /*
   * Dump which sparse tensor component (pointers=0, indices=1,
   * values=2) or dense tensor (dense=3), level, and tensor ID
   * (if available) that we're printing.
   */
  Value inputComponentNum = builder.create<arith::ConstantIntOp>(loc, componentNum, 64);
  Value levelNum = builder.create<arith::ConstantIntOp>(loc, lvl, 64);
  Value tidNum = builder.create<arith::ConstantIntOp>(loc, tid, 64); 
  createFuncCallHelper(builder, loc, "printTensorComponent", {}, {inputComponentNum, levelNum, tidNum}, EmitCInterfaceHelper::Off);

  /*
   * Bulid the memref debug function call, add newline
   */
  createFuncCallHelper(builder, loc, funcName, {}, {castOp}, EmitCInterfaceHelper::On);
  createFuncCallHelper(builder, loc, "printNewline", {}, {}, EmitCInterfaceHelper::Off);
  return;
}

//===--------------------------------------------------------------------------------------------------===//

/// ======= Pass ========

class SparseTensorMemrefDebugPass
    : public mlir::soda::impl::SparseTensorMemrefDebugBase<SparseTensorMemrefDebugPass> {
public:
  void runOnOperation() override {

    // Setup
    #define MEMREF_DEBUG_PRINT(LVL, NUM) \
      OpBuilder builder(op); \
      builder.setInsertionPointAfter(op); \
      auto result = op.getResult(); \
      auto internalTp = result.getType().cast<MemRefType>().getElementType(); \
      createMemRefDebugPrint( \
        builder, op.getLoc(), op.getTensor(), internalTp, result, \
        LVL, /*componentNum*/ NUM \
      );

    // Walk through all operations and collect all sparse 
    // tensor components (pointers, indices, values)
    getOperation().walk([&](sparse_tensor::ToPointersOp op) {
      MEMREF_DEBUG_PRINT((op.getDimension().getZExtValue()), 0);
    });

    getOperation().walk([&](sparse_tensor::ToIndicesOp op) {
      MEMREF_DEBUG_PRINT((op.getDimension().getZExtValue()), 1);
    });

    getOperation().walk([&](sparse_tensor::ToPointersOp op) {
      MEMREF_DEBUG_PRINT(0, 2);
    });

  }
};

} // namespace

std::unique_ptr<Pass> mlir::soda::createSparseTensorMemrefDebugPass() {
  return std::make_unique<SparseTensorMemrefDebugPass>();
}