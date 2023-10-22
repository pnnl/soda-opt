// RUN: soda-opt %s -soda-sparse-compiler="enable-runtime-library=false parallelization-strategy=any-storage-any-loop enable-openmp" | FileCheck %s

// CHECK: omp.parallel
// CHECK: omp.wsloop reduction
// CHECK: omp.yield
// CHECK: omp.terminator
// CHECK-NOT: omp.wsloop for

#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0) -> ()>
module {
  func.func @Reduce1D(%arg0: tensor<16xf32, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ] }>>) -> tensor<f32> {
    %0 = tensor.empty() : tensor<f32>
    %cst = arith.constant 0.000000e+00 : f32
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<f32>) -> tensor<f32>
    %2 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg0 : tensor<16xf32, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ] }>>) outs(%1 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %3 = arith.addf %out, %in : f32
      linalg.yield %3 : f32
    } -> tensor<f32>
    return %2 : tensor<f32>
  }
}