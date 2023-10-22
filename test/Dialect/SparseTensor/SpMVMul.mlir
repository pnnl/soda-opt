// RUN: soda-opt %s -soda-sparse-compiler="enable-runtime-library=false parallelization-strategy=any-storage-any-loop enable-openmp" | FileCheck %s

// CHECK: omp.parallel
// CHECK: omp.wsloop for
// CHECK: omp.parallel
// CHECK: omp.wsloop reduction
// CHECK: omp.yield
// CHECK: omp.terminator
// CHECK: omp.yield
// CHECK: omp.terminator

#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d1)>
#map2 = affine_map<(d0, d1) -> (d0)>
module {
  func.func @SpMVMul(%arg0: tensor<16x16xf32, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ] }>>, %arg1: tensor<16xf32, #sparse_tensor.encoding<{ dimLevelType = [ "dense" ] }>>) -> tensor<16xf32, #sparse_tensor.encoding<{ dimLevelType = [ "dense" ] }>> {
    %0 = bufferization.alloc_tensor() : tensor<16xf32, #sparse_tensor.encoding<{ dimLevelType = [ "dense" ] }>>
    %1 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "reduction"]} ins(%arg0, %arg1 : tensor<16x16xf32, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ] }>>, tensor<16xf32, #sparse_tensor.encoding<{ dimLevelType = [ "dense" ] }>>) outs(%0 : tensor<16xf32, #sparse_tensor.encoding<{ dimLevelType = [ "dense" ] }>>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %2 = arith.mulf %in, %in_0 : f32
      %3 = arith.addf %out, %2 : f32
      linalg.yield %3 : f32
    } -> tensor<16xf32, #sparse_tensor.encoding<{ dimLevelType = [ "dense" ] }>>
    return %1 : tensor<16xf32, #sparse_tensor.encoding<{ dimLevelType = [ "dense" ] }>>
  }
}

