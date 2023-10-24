// RUN: soda-opt %s -soda-sparse-compiler="enable-runtime-library=false parallelization-strategy=any-storage-any-loop enable-openmp" | FileCheck %s

// CHECK: omp.parallel
// CHECK: omp.wsloop for
// CHECK: omp.parallel
// CHECK: omp.wsloop for
// CHECK: omp.parallel
// CHECK: omp.wsloop for
// CHECK: omp.yield
// CHECK: omp.terminator
// CHECK: omp.yield
// CHECK: omp.terminator
// CHECK: omp.yield
// CHECK: omp.terminator
// CHECK-NOT: omp.wsloop reduction

#map = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d3, d1)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1)>
#map3 = affine_map<(d0, d1, d2) -> (d0, d2, d1)>
#map4 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map5 = affine_map<(d0, d1, d2) -> (d0, d1)>
module {
  func.func @MTTKRP(%arg0: tensor<16x16x16xf32, #sparse_tensor.encoding<{ dimLevelType = [ "compressed", "compressed", "compressed" ] }>>, %arg1: tensor<16x16xf32, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "dense" ] }>>, %arg2: tensor<16x16xf32, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "dense" ] }>>) -> tensor<16x16xf32, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "dense" ] }>> {
    %0 = bufferization.alloc_tensor() : tensor<16x16x16xf32, #sparse_tensor.encoding<{ dimLevelType = [ "compressed", "compressed", "dense" ] }>>
    %1 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%arg0, %arg1 : tensor<16x16x16xf32, #sparse_tensor.encoding<{ dimLevelType = [ "compressed", "compressed", "compressed" ] }>>, tensor<16x16xf32, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "dense" ] }>>) outs(%0 : tensor<16x16x16xf32, #sparse_tensor.encoding<{ dimLevelType = [ "compressed", "compressed", "dense" ] }>>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %4 = arith.mulf %in, %in_0 : f32
      %5 = arith.addf %out, %4 : f32
      linalg.yield %5 : f32
    } -> tensor<16x16x16xf32, #sparse_tensor.encoding<{ dimLevelType = [ "compressed", "compressed", "dense" ] }>>
    %2 = bufferization.alloc_tensor() : tensor<16x16xf32, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "dense" ] }>>
    %3 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "parallel", "reduction"]} ins(%1, %arg2 : tensor<16x16x16xf32, #sparse_tensor.encoding<{ dimLevelType = [ "compressed", "compressed", "dense" ] }>>, tensor<16x16xf32, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "dense" ] }>>) outs(%2 : tensor<16x16xf32, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "dense" ] }>>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %4 = arith.mulf %in, %in_0 : f32
      %5 = arith.addf %out, %4 : f32
      linalg.yield %5 : f32
    } -> tensor<16x16xf32, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "dense" ] }>>
    return %3 : tensor<16x16xf32, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "dense" ] }>>
  }
}