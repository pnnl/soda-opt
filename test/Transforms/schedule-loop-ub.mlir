// RUN: soda-opt -pass-pipeline="func.func(test-loop-scheduling {schedule=%S/schedule.csv loop=1})" %s | FileCheck  %s

func.func @example(%v0: index, %arg0: memref<10xi32>, %arg1: memref<10xi32>) {
	affine.for %arg2 = 0 to %v0 {
		%0 = affine.load %arg0[%arg2] : memref<10xi32>
		%1 = arith.muli %0, %0 : i32
		affine.store %1, %arg1[%arg2] : memref<10xi32>
	}
	return
}

// CHECK: #map0 = affine_map<(d0) -> (d0 - 2)>
// CHECK: #map1 = affine_map<()[s0] -> (s0)>
// CHECK: #map2 = affine_map<(d0) -> (d0 - 1)>
// CHECK: #set = affine_set<()[s0] : (s0 - 3 >= 0)>
// CHECK: module {
// CHECK:   func.func @example(%arg0: index, %arg1: memref<10xi32>, %arg2: memref<10xi32>) {
// CHECK:     affine.if #set()[%arg0] {
// CHECK:       %c0 = arith.constant 0 : index
// CHECK:       %0 = affine.load %arg1[%c0] : memref<10xi32>
// CHECK:       %c1 = arith.constant 1 : index
// CHECK:       %1 = affine.load %arg1[%c1] : memref<10xi32>
// CHECK:       %2 = arith.muli %0, %0 : i32
// CHECK:       %3:2 = affine.for %arg3 = 2 to %arg0 iter_args(%arg4 = %1, %arg5 = %2) -> (i32, i32) {
// CHECK:         %9 = affine.load %arg1[%arg3] : memref<10xi32>
// CHECK:         %10 = arith.muli %arg4, %arg4 : i32
// CHECK:         %11 = affine.apply #map0(%arg3)
// CHECK:         affine.store %arg5, %arg2[%11] : memref<10xi32>
// CHECK:         affine.yield %9, %10 : i32, i32
// CHECK:       }
// CHECK:       %4 = arith.muli %3#0, %3#0 : i32
// CHECK:       %5 = affine.apply #map1()[%arg0]
// CHECK:       %6 = affine.apply #map0(%5)
// CHECK:       affine.store %3#1, %arg2[%6] : memref<10xi32>
// CHECK:       %7 = affine.apply #map1()[%arg0]
// CHECK:       %8 = affine.apply #map2(%7)
// CHECK:       affine.store %4, %arg2[%8] : memref<10xi32>
// CHECK:     } else {
// CHECK:       affine.for %arg3 = 0 to %arg0 {
// CHECK:         %0 = affine.load %arg1[%arg3] : memref<10xi32>
// CHECK:         %1 = arith.muli %0, %0 : i32
// CHECK:         affine.store %1, %arg2[%arg3] : memref<10xi32>
// CHECK:       }
// CHECK:     }
// CHECK:     return
// CHECK:   }
// CHECK: }