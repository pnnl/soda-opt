// RUN: soda-opt -pass-pipeline="func.func(test-loop-scheduling {schedule=%S/schedule.csv loop=1})" %s | FileCheck  %s

func.func @example(%v0: index, %arg0: memref<10xi32>, %arg1: memref<10xi32>) {
	affine.for %arg2 = %v0 to 10 {
		%0 = affine.load %arg0[%arg2] : memref<10xi32>
		%1 = arith.muli %0, %0 : i32
		affine.store %1, %arg1[%arg2] : memref<10xi32>
	}
	return
}

// CHECK: #map0 = affine_map<()[s0] -> (s0)>
// CHECK: #map1 = affine_map<(d0) -> (d0 + 1)>
// CHECK: #map2 = affine_map<()[s0] -> (s0 + 2)>
// CHECK: #map3 = affine_map<(d0) -> (d0 - 2)>
// CHECK: #set = affine_set<()[s0] : (-s0 + 7 >= 0)>
// CHECK: module {
// CHECK:   func.func @example(%arg0: index, %arg1: memref<10xi32>, %arg2: memref<10xi32>) {
// CHECK:     affine.if #set()[%arg0] {
// CHECK:       %0 = affine.apply #map0()[%arg0]
// CHECK:       %1 = affine.load %arg1[%0] : memref<10xi32>
// CHECK:       %2 = affine.apply #map0()[%arg0]
// CHECK:       %3 = affine.apply #map1(%2)
// CHECK:       %4 = affine.load %arg1[%3] : memref<10xi32>
// CHECK:       %5 = arith.muli %1, %1 : i32
// CHECK:       %6:2 = affine.for %arg3 = #map2()[%arg0] to 10 iter_args(%arg4 = %4, %arg5 = %5) -> (i32, i32) {
// CHECK:         %8 = affine.load %arg1[%arg3] : memref<10xi32>
// CHECK:         %9 = arith.muli %arg4, %arg4 : i32
// CHECK:         %10 = affine.apply #map3(%arg3)
// CHECK:         affine.store %arg5, %arg2[%10] : memref<10xi32>
// CHECK:         affine.yield %8, %9 : i32, i32
// CHECK:       }
// CHECK:       %7 = arith.muli %6#0, %6#0 : i32
// CHECK:       %c8 = arith.constant 8 : index
// CHECK:       affine.store %6#1, %arg2[%c8] : memref<10xi32>
// CHECK:       %c9 = arith.constant 9 : index
// CHECK:       affine.store %7, %arg2[%c9] : memref<10xi32>
// CHECK:     } else {
// CHECK:       affine.for %arg3 = %arg0 to 10 {
// CHECK:         %0 = affine.load %arg1[%arg3] : memref<10xi32>
// CHECK:         %1 = arith.muli %0, %0 : i32
// CHECK:         affine.store %1, %arg2[%arg3] : memref<10xi32>
// CHECK:       }
// CHECK:     }
// CHECK:     return
// CHECK:   }
// CHECK: }