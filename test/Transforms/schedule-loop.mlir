// RUN: soda-opt -pass-pipeline="func.func(test-loop-scheduling {schedule=%S/schedule.csv loop=1})" %s | FileCheck  %s

func.func @example(%arg0: memref<10xi32>, %arg1: memref<10xi32>) {
	affine.for %arg2 = 0 to 10 {
		%0 = affine.load %arg0[%arg2] : memref<10xi32>
		%1 = arith.muli %0, %0 : i32
		affine.store %1, %arg1[%arg2] : memref<10xi32>
	}
	return
}

// CHECK: #map = affine_map<(d0) -> (d0 - 2)>
// CHECK: module {
// CHECK:   func.func @example(%arg0: memref<10xi32>, %arg1: memref<10xi32>) {
// CHECK:     %c0 = arith.constant 0 : index
// CHECK:     %0 = affine.load %arg0[%c0] : memref<10xi32>
// CHECK:     %c1 = arith.constant 1 : index
// CHECK:     %1 = affine.load %arg0[%c1] : memref<10xi32>
// CHECK:     %2 = arith.muli %0, %0 : i32
// CHECK:     %3:2 = affine.for %arg2 = 2 to 10 iter_args(%arg3 = %1, %arg4 = %2) -> (i32, i32) {
// CHECK:       %5 = affine.load %arg0[%arg2] : memref<10xi32>
// CHECK:       %6 = arith.muli %arg3, %arg3 : i32
// CHECK:       %7 = affine.apply #map(%arg2)
// CHECK:       affine.store %arg4, %arg1[%7] : memref<10xi32>
// CHECK:       affine.yield %5, %6 : i32, i32
// CHECK:     }
// CHECK:     %4 = arith.muli %3#0, %3#0 : i32
// CHECK:     %c8 = arith.constant 8 : index
// CHECK:     affine.store %3#1, %arg1[%c8] : memref<10xi32>
// CHECK:     %c9 = arith.constant 9 : index
// CHECK:     affine.store %4, %arg1[%c9] : memref<10xi32>
// CHECK:     return
// CHECK:   }
// CHECK: }