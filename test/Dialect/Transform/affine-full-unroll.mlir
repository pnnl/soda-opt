// RUN: soda-opt %s -transform-interpreter --canonicalize| FileCheck %s

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["arith.mulf"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %innermost = transform.get_parent_op %0 {op_name = "affine.for"} : (!transform.any_op) -> !transform.op<"affine.for"> 
    %second = transform.get_parent_op %innermost {op_name = "affine.for"} : (!transform.op<"affine.for"> ) -> !transform.op<"affine.for"> 
    transform.loop.fullunroll %innermost : !transform.op<"affine.for"> 
    transform.loop.fullunroll %second : !transform.op<"affine.for"> 
    transform.yield
  }
}


#map = affine_map<(d0) -> (d0 * 2)>
#map1 = affine_map<(d0, d1) -> (d0 * 2 + d1)>
module {
func.func @conv(%arg0: memref<4x36x36x1xf32>, %arg1: memref<1x5x5x1xf32>, %arg2: memref<4x16x16x1xf32>) {
    %c16 = arith.constant 16 : index
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c2 = arith.constant 2 : index
    scf.for %arg3 = %c0 to %c4 step %c2 {
      scf.for %arg4 = %c0 to %c16 step %c4 {
        scf.for %arg5 = %c0 to %c16 step %c4 {
          %0 = affine.apply #map(%arg4)
          %1 = affine.apply #map(%arg5)
          %subview = memref.subview %arg0[%arg3, %0, %1, 0] [2, 11, 11, 1] [1, 1, 1, 1] : memref<4x36x36x1xf32> to memref<2x11x11x1xf32, strided<[1296, 36, 1, 1], offset: ?>>
          %subview_0 = memref.subview %arg1[0, 0, 0, 0] [1, 5, 5, 1] [1, 1, 1, 1] : memref<1x5x5x1xf32> to memref<1x5x5x1xf32, strided<[25, 5, 1, 1]>>
          %subview_1 = memref.subview %arg2[%arg3, %arg4, %arg5, 0] [2, 4, 4, 1] [1, 1, 1, 1] : memref<4x16x16x1xf32> to memref<2x4x4x1xf32, strided<[256, 16, 1, 1], offset: ?>>
          affine.for %arg6 = 0 to 2 {
            affine.for %arg7 = 0 to 4 {
              affine.for %arg8 = 0 to 4 {
                affine.for %arg9 = 0 to 1 {
                  affine.for %arg10 = 0 to 5 {
                    // This loop will be full unrolled and will disapear.
                    affine.for %arg11 = 0 to 5 {
                      // This loop will be full unrolled and will disapear.
                      affine.for %arg12 = 0 to 1 {
                        %2 = affine.apply #map1(%arg7, %arg10)
                        %3 = affine.apply #map1(%arg8, %arg11)
                        %4 = affine.load %subview[%arg6, %2, %3, %arg12] : memref<2x11x11x1xf32, strided<[1296, 36, 1, 1], offset: ?>>
                        %5 = affine.load %subview_0[%arg9, %arg10, %arg11, %arg12] : memref<1x5x5x1xf32, strided<[25, 5, 1, 1]>>
                        %6 = affine.load %subview_1[%arg6, %arg7, %arg8, %arg9] : memref<2x4x4x1xf32, strided<[256, 16, 1, 1], offset: ?>>
                        %7 = arith.mulf %4, %5 : f32
                        %8 = arith.addf %6, %7 : f32
                        affine.store %8, %subview_1[%arg6, %arg7, %arg8, %arg9] : memref<2x4x4x1xf32, strided<[256, 16, 1, 1], offset: ?>>
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
    return
  }
}

// Should find 5 loops instead of 7
// CHECK-LABEL: func.func @conv
// CHECK: affine.for 
// CHECK: affine.for 
// CHECK: affine.for 
// CHECK: affine.for 
// CHECK: affine.for 
// CHECK-NOT: affine.for 