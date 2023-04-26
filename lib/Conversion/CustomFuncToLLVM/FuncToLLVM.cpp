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
#define GEN_PASS_DEF_CONVERTFUNCTOLLVM
#include "mlir/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;

#define PASS_NAME "soda-convert-func-to-llvm"

namespace {
/// A pass converting MLIR operations into the LLVM IR dialect.
struct LLVMLoweringPass : public impl::ConvertFuncToLLVMBase<LLVMLoweringPass> {
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
    if (failed(LLVM::LLVMDialect::verifyDataLayoutString(
            this->dataLayout, [this](const Twine &message) {
              getOperation().emitError() << message.str();
            }))) {
      signalPassFailure();
      return;
    }

    ModuleOp m = getOperation();
    const auto &dataLayoutAnalysis = getAnalysis<DataLayoutAnalysis>();

    LowerToLLVMOptions options(&getContext(),
                               dataLayoutAnalysis.getAtOrAbove(m));
    options.useBarePtrCallConv = useBarePtrCallConv;
    if (indexBitwidth != kDeriveIndexBitwidthFromDataLayout)
      options.overrideIndexBitwidth(indexBitwidth);
    options.dataLayout = llvm::DataLayout(this->dataLayout);

    LLVMTypeConverter typeConverter(&getContext(), options,
                                    &dataLayoutAnalysis);

    RewritePatternSet patterns(&getContext());
    populateFuncToLLVMConversionPatterns(typeConverter, patterns);
    arith::populateArithToLLVMConversionPatterns(typeConverter, patterns);
    cf::populateControlFlowToLLVMConversionPatterns(typeConverter, patterns);

    LLVMConversionTarget target(getContext());
    if (failed(applyPartialConversion(m, target, std::move(patterns))))
      signalPassFailure();

    m->setAttr(LLVM::LLVMDialect::getDataLayoutAttrName(),
               StringAttr::get(m.getContext(), this->dataLayout));
  }
};
} // end namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::soda::createCustomFuncToLLVMPass(bool useBarePtrCallConv) {
  return std::make_unique<LLVMLoweringPass>(useBarePtrCallConv);
}