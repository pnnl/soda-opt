//===- ConvertCustomFuncToLLVMPass.h - Pass entrypoint ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CONVERSION_CUSTOMFUNCTOLLVM_CONVERTFUNCTOLLVMPASS_H_
#define MLIR_CONVERSION_CUSTOMFUNCTOLLVM_CONVERTFUNCTOLLVMPASS_H_

#include <memory>
#include <string>

namespace mlir {
class ModuleOp;
template <typename T>
class OperationPass;
class Pass;

namespace soda {

//===----------------------------------------------------------------------===//
// Lowerings
//===----------------------------------------------------------------------===//

/// Perform lowering from std operations to LLVM dialect.
/// Exposing the options of barePtrCallConv or emitCWrappers without the need
/// to know the mlir context during pass (pipeline) creation. MLIR context is
/// obtained at runtime.
///
/// This pass is based on:
///    llvm-project/mlir/lib/Conversion/FuncToLLVM/FuncToLLVM.cpp
std::unique_ptr<OperationPass<ModuleOp>>
createCustomFuncToLLVMPass(bool useBarePtrCallConv);

} // namespace soda
} // namespace mlir

#endif // MLIR_CONVERSION_CUSTOMFUNCTOLLVM_CONVERTFUNCTOLLVMPASS_H_
