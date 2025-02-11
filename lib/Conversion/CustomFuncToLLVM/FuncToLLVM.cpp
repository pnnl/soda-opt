//===- FuncToLLVM.cpp - Standard to LLVM dialect conversion -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// This file was modified based on:
//  llvm-project/mlir/lib/Conversion/FuncToLLVM/FuncToLLVM.cpp
//
// This file implements a pass to convert MLIR standard and builtin dialects
// into the LLVM IR dialect.
//
// Upstream implementation requires a MLIR context during pass creation, which
// is not available during pass pipeline creation. Hence this file is needed.
//
//===----------------------------------------------------------------------===//

#include "soda/Conversion/CustomFuncToLLVM/ConvertCustomFuncToLLVMPass.h"

#include "mlir/Analysis/DataLayoutAnalysis.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/Passes.h"


namespace mlir {
namespace soda { // Namespace is necessary here, because we also include the
                // upstream implementation of the conversion pass.
#define GEN_PASS_DEF_CONVERTFUNCTOLLVMPASS
#include "mlir/Conversion/Passes.h.inc"
} // namespace soda
} // namespace mlir

using namespace mlir;

#define PASS_NAME "soda-convert-func-to-llvm"

namespace {
/// A pass converting MLIR operations into the LLVM IR dialect.
struct LLVMLoweringPass : public soda::impl::ConvertFuncToLLVMPassBase<LLVMLoweringPass> {
  using Base::Base;
  
  LLVMLoweringPass() = default;

  LLVMLoweringPass(bool useBarePtrCallConv) {
    this->useBarePtrCallConv = useBarePtrCallConv;
  }

  /// Run the dialect converter on the module.
  void runOnOperation() override {
    if (useBarePtrCallConv &&
        getOperation()->getAttrOfType<UnitAttr>(
            LLVM::LLVMDialect::getEmitCWrapperAttrName())) {
      getOperation().emitError()
          << "incompatible conversion options: bare-pointer calling convention "
             "and C wrapper emission";
      signalPassFailure();
      return;
    }
    
    ModuleOp m = getOperation();
    StringRef dataLayout;
    auto dataLayoutAttr = dyn_cast_or_null<StringAttr>(
        m->getAttr(LLVM::LLVMDialect::getDataLayoutAttrName()));
    if (dataLayoutAttr)
      dataLayout = dataLayoutAttr.getValue();

    if (failed(LLVM::LLVMDialect::verifyDataLayoutString(
            dataLayout, [this](const Twine &message) {
              getOperation().emitError() << message.str();
            }))) {
      signalPassFailure();
      return;
    }

    const auto &dataLayoutAnalysis = getAnalysis<DataLayoutAnalysis>();

    LowerToLLVMOptions options(&getContext(),
                               dataLayoutAnalysis.getAtOrAbove(m));
    options.useBarePtrCallConv = useBarePtrCallConv;
    if (indexBitwidth != kDeriveIndexBitwidthFromDataLayout)
      options.overrideIndexBitwidth(indexBitwidth);
    options.dataLayout = llvm::DataLayout(dataLayout);

    LLVMTypeConverter typeConverter(&getContext(), options,
                                    &dataLayoutAnalysis);

    std::optional<SymbolTable> optSymbolTable = std::nullopt;
    const SymbolTable *symbolTable = nullptr;
    if (!options.useBarePtrCallConv) {
      optSymbolTable.emplace(m);
      symbolTable = &optSymbolTable.value();
    }

    RewritePatternSet patterns(&getContext());
    populateFuncToLLVMConversionPatterns(typeConverter, patterns, symbolTable);

    // TODO(https://github.com/llvm/llvm-project/issues/70982): Remove these in
    // favor of their dedicated conversion passes.
    arith::populateArithToLLVMConversionPatterns(typeConverter, patterns);
    cf::populateControlFlowToLLVMConversionPatterns(typeConverter, patterns);

    LLVMConversionTarget target(getContext());
    if (failed(applyPartialConversion(m, target, std::move(patterns))))
      signalPassFailure();
  }
};
} // end namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::soda::createCustomFuncToLLVMPass(bool useBarePtrCallConv) {
  return std::make_unique<LLVMLoweringPass>(useBarePtrCallConv);
}