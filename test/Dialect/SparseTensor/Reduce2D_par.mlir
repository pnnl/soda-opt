// XFAIL: *
// RUN: soda-opt %s -soda-sparse-compiler="enable-runtime-library=false parallelization-strategy=any-storage-any-loop enable-openmp" | FileCheck %s

// This is expected to fail as the baseline sparse compiler can't gracefully 
// decide not to parallelize a 2D reduction. It's assumed this isn't supported
// by MLIR and the sparse_tensor dialect now.

#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> ()>
module {
  func.func @Reduce2D(%arg0: tensor<16x16xf32, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ] }>>) -> tensor<f32> {
    %0 = tensor.empty() : tensor<f32>
    %cst = arith.constant 0.000000e+00 : f32
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<f32>) -> tensor<f32>
    %2 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction", "reduction"]} ins(%arg0 : tensor<16x16xf32, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ] }>>) outs(%1 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %3 = arith.addf %out, %in : f32
      linalg.yield %3 : f32
    } -> tensor<f32>
    return %2 : tensor<f32>
  }
}

