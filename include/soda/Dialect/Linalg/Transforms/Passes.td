//===-- Passes.td - Linalg pass definition file ------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SODA_DIALECT_LINALG_PASSES
#define SODA_DIALECT_LINALG_PASSES

include "mlir/Pass/PassBase.td"

def LinalgTiling : Pass<"soda-linalg-tile", "ModuleOp"> {
  let summary = "Tile operations in the linalg dialect. Default to tile all "
                "linalg.generic operations.";
  let dependentDialects = [
    "affine::AffineDialect",
    "linalg::LinalgDialect",
    "memref::MemRefDialect",
    "scf::SCFDialect",
    "transform::TransformDialect"
  ];
  let options = [
    ListOption<"tileSizes", "tile-sizes", "int64_t", "Tile sizes">,
    Option<"loopType", "loop-type", "std::string", /*default=*/"\"for\"",
           "Specify the type of loops to generate: for, parallel">,
    Option<"anchorOp", "anchor-op", "std::string", /*default=*/"\"linalg.generic\"",
           "Specify the op to anchor the tiling to, ex: linalg.generic or matmul">
  ];
}

#endif // SODA_DIALECT_LINALG_PASSES
