// RUN: soda-opt -pass-pipeline="func.func(test-forward-results {loop=1})" %s | FileCheck  %s

func.func @kernel(%arg0: f64, %arg1: memref<16x18xf64>, %arg2: memref<16x22xf64>, %arg3: memref<22x18xf64>) {
  affine.for %arg4 = 0 to 16 {
    affine.for %arg5 = 0 to 18 {
      affine.for %arg6 = 0 to 22 {
        %0 = affine.load %arg2[%arg4, %arg6] : memref<16x22xf64>
        %1 = arith.mulf %arg0, %0 : f64
        %2 = affine.load %arg3[%arg6, %arg5] : memref<22x18xf64>
        %3 = arith.mulf %1, %2 : f64
        %4 = affine.load %arg1[%arg4, %arg5] : memref<16x18xf64>
        %5 = arith.addf %4, %3 : f64
        affine.store %5, %arg1[%arg4, %arg5] : memref<16x18xf64>
      }
    }
  }
  return
}

// CHECK: %0 = affine.load %arg1[%arg4, %arg5] : memref<16x18xf64>
// CHECK: %1 = affine.for %arg6 = 0 to 22 iter_args(%arg7 = %0) -> (f64) {
// CHECK: affine.yield %6 : f64
// CHECK: affine.store %1, %arg1[%arg4, %arg5] : memref<16x18xf64>    