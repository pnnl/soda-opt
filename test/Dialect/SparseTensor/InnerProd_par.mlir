// XFAIL: *
// RUN: soda-opt %s -soda-sparse-compiler="enable-runtime-library=false parallelization-strategy=any-storage-any-loop enable-openmp" | FileCheck %s

// This is expected to fail as the baseline sparse compiler can't gracefully 
// decide not to parallelize a 3D reduction. It's assumed this isn't supported
// by MLIR and the sparse_tensor dialect now.

#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2) -> ()>
module {
  func.func @InnerProd(%arg0: tensor<16x16x16xf32, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed", "compressed" ] }>>, %arg1: tensor<16x16x16xf32, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed", "compressed" ] }>>) -> tensor<f32> {
    %0 = tensor.empty() : tensor<f32>
    %cst = arith.constant 0.000000e+00 : f32
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<f32>) -> tensor<f32>
    %2 = linalg.generic {indexing_maps = [#map, #map, #map1], iterator_types = ["reduction", "reduction", "reduction"]} ins(%arg0, %arg1 : tensor<16x16x16xf32, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed", "compressed" ] }>>, tensor<16x16x16xf32, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed", "compressed" ] }>>) outs(%1 : tensor<f32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %3 = arith.mulf %in, %in_0 : f32
      %4 = arith.addf %out, %3 : f32
      linalg.yield %4 : f32
    } -> tensor<f32>
    return %2 : tensor<f32>
  }
}
