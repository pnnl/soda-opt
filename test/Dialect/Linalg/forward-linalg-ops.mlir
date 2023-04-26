// RUN: soda-opt %s -forward-linalg-fill | FileCheck %s --check-prefix=FFILL
// RUN: soda-opt %s -forward-memref-allocations -forward-linalg-fill | FileCheck %s --check-prefix=F_A_F
// RUN: soda-opt %s -forward-memref-allocations -forward-linalg-fill -forward-memref-copy -forward-memref-allocations | FileCheck %s --check-prefix=F_A_F_C_A

func.func private @do_something1(memref<4x1xf32>) -> ()
func.func private @do_something2(memref<4x2xf32>) -> ()
func.func private @do_something3(memref<4x3xf32>) -> ()
func.func private @do_something4(memref<4x4xf32>) -> ()

func.func @forward_allocations() {
  %0 = memref.alloc() : memref<4x1xf32>
  call @do_something1(%0) : (memref<4x1xf32>) -> ()
  %1 = memref.alloc() : memref<4x2xf32>
  call @do_something2(%1) : (memref<4x2xf32>) -> ()
  %2 = memref.alloca() : memref<4x4xf32>
  call @do_something4(%2) : (memref<4x4xf32>) -> ()
  %3 = memref.alloc() : memref<4x3xf32>
  call @do_something3(%3) : (memref<4x3xf32>) -> ()
  %4 = memref.alloc() : memref<1x7x7x32xf32>
  %5 = memref.alloc() : memref<1x7x7x32xf32>
  %cst = arith.constant 1.000000e+00 : f32
  linalg.fill ins(%cst : f32) outs(%4: memref<1x7x7x32xf32> )
  memref.copy %4, %5  : memref<1x7x7x32xf32> to  memref<1x7x7x32xf32> 
  return
}

// FFILL-LABEL: func.func @forward_allocations 
// FFILL-NEXT:  %[[v0:.*]] = memref.alloc() : memref<4x1xf32>
// FFILL-NEXT:  %[[vcst:.*]] = arith.constant 1.000000e+00 : f32
// FFILL-NEXT:  call @do_something1(%[[v0]]) : (memref<4x1xf32>) -> ()
// FFILL-NEXT:  %[[v1:.*]] = memref.alloc() : memref<4x2xf32>
// FFILL-NEXT:  call @do_something2(%[[v1]]) : (memref<4x2xf32>) -> ()
// FFILL-NEXT:  %[[v2:.*]] = memref.alloca() : memref<4x4xf32>
// FFILL-NEXT:  call @do_something4(%[[v2]]) : (memref<4x4xf32>) -> ()
// FFILL-NEXT:  %[[v3:.*]] = memref.alloc() : memref<4x3xf32>
// FFILL-NEXT:  call @do_something3(%[[v3]]) : (memref<4x3xf32>) -> ()
// FFILL-NEXT:  %[[v4:.*]] = memref.alloc() : memref<1x7x7x32xf32>
// FFILL-NEXT:  linalg.fill ins(%[[vcst]] : f32) outs(%[[v4]] : memref<1x7x7x32xf32>) 
// FFILL-NEXT:  %[[v5:.*]] = memref.alloc() : memref<1x7x7x32xf32>
// FFILL-NEXT:  memref.copy %[[v4]], %[[v5]] : memref<1x7x7x32xf32> to memref<1x7x7x32xf32> 


// F_A_F-LABEL: func.func @forward_allocations 
// F_A_F-NEXT:  %[[v0:.*]] = memref.alloc() : memref<4x1xf32>
// F_A_F-NEXT:  %[[vcst:.*]] = arith.constant 1.000000e+00 : f32
// F_A_F-NEXT:  %[[v1:.*]] = memref.alloc() : memref<4x2xf32>
// F_A_F-NEXT:  %[[v2:.*]] = memref.alloc() : memref<4x3xf32>
// F_A_F-NEXT:  %[[v3:.*]] = memref.alloc() : memref<1x7x7x32xf32>
// F_A_F-NEXT:  linalg.fill ins(%[[vcst]] : f32) outs(%[[v3]] : memref<1x7x7x32xf32>) 
// F_A_F-NEXT:  %[[v4:.*]] = memref.alloc() : memref<1x7x7x32xf32>
// F_A_F-NEXT:  %[[v5:.*]] = memref.alloca() : memref<4x4xf32>
// F_A_F-NEXT:  call @do_something1(%[[v0]]) : (memref<4x1xf32>) -> ()
// F_A_F-NEXT:  call @do_something2(%[[v1]]) : (memref<4x2xf32>) -> ()
// F_A_F-NEXT:  call @do_something4(%[[v5]]) : (memref<4x4xf32>) -> ()
// F_A_F-NEXT:  call @do_something3(%[[v2]]) : (memref<4x3xf32>) -> ()
// F_A_F-NEXT:  memref.copy %[[v3]], %[[v4]]  : memref<1x7x7x32xf32> to memref<1x7x7x32xf32> 

// F_A_F_C_A-LABEL: func.func @forward_allocations 
// F_A_F_C_A-NEXT:    %[[v0:.*]] = memref.alloc() : memref<4x1xf32>
// F_A_F_C_A-NEXT:    %[[v1:.*]] = memref.alloc() : memref<4x2xf32>
// F_A_F_C_A-NEXT:    %[[v2:.*]] = memref.alloc() : memref<4x3xf32>
// F_A_F_C_A-NEXT:    %[[v3:.*]] = memref.alloc() : memref<1x7x7x32xf32>
// F_A_F_C_A-NEXT:    %[[v4:.*]] = memref.alloc() : memref<1x7x7x32xf32>
// F_A_F_C_A-NEXT:    %[[v5:.*]] = memref.alloca() : memref<4x4xf32>
// F_A_F_C_A-NEXT:    %[[vcst:.*]] = arith.constant 1.000000e+00 : f32
// F_A_F_C_A-NEXT:    linalg.fill ins(%[[vcst]] : f32) outs(%[[v3]] : memref<1x7x7x32xf32>) 
// F_A_F_C_A-NEXT:    memref.copy %[[v3]], %[[v4]]  : memref<1x7x7x32xf32> to memref<1x7x7x32xf32> 
// F_A_F_C_A-NEXT:    call @do_something1(%[[v0]]) : (memref<4x1xf32>) -> ()
// F_A_F_C_A-NEXT:    call @do_something2(%[[v1]]) : (memref<4x2xf32>) -> ()
// F_A_F_C_A-NEXT:    call @do_something4(%[[v5]]) : (memref<4x4xf32>) -> ()
// CHECK-NEXT:    call @do_something3(%[[v2]]) : (memref<4x3xf32>) -> ()