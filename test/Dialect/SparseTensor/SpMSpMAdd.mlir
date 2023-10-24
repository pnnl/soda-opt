// RUN: soda-opt %s -soda-sparse-compiler="enable-runtime-library=false parallelization-strategy=any-storage-any-loop enable-openmp" | FileCheck %s

// CHECK: omp.parallel
// CHECK: omp.wsloop for
// CHECK: omp.parallel
// CHECK: omp.wsloop for
// CHECK: omp.yield
// CHECK: omp.terminator
// CHECK: omp.parallel
// CHECK: omp.wsloop for
// CHECK: omp.yield
// CHECK: omp.terminator
// CHECK: omp.yield
// CHECK: omp.terminator
// CHECK-NOT: omp.wsloop reduction

#map = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @SpMSpMAdd(%arg0: tensor<16x16xf32, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ] }>>, %arg1: tensor<16x16xf32, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ] }>>) -> tensor<16x16xf32, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "dense" ] }>> {
    %0 = bufferization.alloc_tensor() : tensor<16x16xf32, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "dense" ] }>>
    %1 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0, %arg1 : tensor<16x16xf32, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ] }>>, tensor<16x16xf32, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ] }>>) outs(%0 : tensor<16x16xf32, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "dense" ] }>>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %2 = arith.addf %in, %in_0 : f32
      linalg.yield %2 : f32
    } -> tensor<16x16xf32, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "dense" ] }>>
    return %1 : tensor<16x16xf32, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "dense" ] }>>
  }
}