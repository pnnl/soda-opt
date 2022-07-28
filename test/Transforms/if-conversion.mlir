// RUN: soda-opt -pass-pipeline="func.func(test-if-conversion {loop=1})" %s | FileCheck  %s

#set0 = affine_set<(d0) : (d0 - 9 >= 0)>

func.func @kernel(%arg0: f64, %arg1: memref<16x18xf64>, %arg2: memref<16x22xf64>, %arg3: memref<22x18xf64>) {
  affine.for %arg4 = 0 to 16 {
    affine.for %arg5 = 0 to 18 {
      affine.for %arg6 = 0 to 22 {
        %0 = affine.load %arg2[%arg4, %arg6] : memref<16x22xf64>
        %1 = arith.mulf %arg0, %0 : f64
        %2 = affine.load %arg3[%arg6, %arg5] : memref<22x18xf64>
        %3 = arith.mulf %1, %2 : f64
        %4 = affine.load %arg1[%arg4, %arg5] : memref<16x18xf64>
        %5 = affine.if #set0(%arg6) -> f64 {
          %6 = arith.mulf %4, %3 : f64
          affine.yield %6 : f64
        } else {
          %6 = arith.addf %4, %3 : f64
          affine.yield %6 : f64
        }
        affine.store %5, %arg1[%arg4, %arg5] : memref<16x18xf64>
        }
      }
  }
  return
}

// CHECK: #map = affine_map<(d0) -> (d0 - 9)>
// CHECK: %5 = arith.mulf %4, %3 : f64
// CHECK: %6 = arith.addf %4, %3 : f64
// CHECK: %7 = affine.apply #map(%arg6)
// CHECK: %c0 = arith.constant 0 : index
// CHECK: %8 = arith.cmpi sge, %7, %c0 : index
// CHECK: %9 = arith.select %8, %5, %6 : f64