// RUN: soda-opt %s --convert-accel-to-axi-runtime --canonicalize | FileCheck %s

// A is split into tiles of size (10, 20). The tile at offset (10, 80) is tile
// (1, 4): send_idx sends 1 and 4, send_dim sends 10 and 20.

// CHECK-LABEL: func.func @static_tile(
// CHECK-DAG:     %[[C1:.*]] = arith.constant 1 : i32
// CHECK-DAG:     %[[C4:.*]] = arith.constant 4 : i32
// CHECK-DAG:     %[[C10:.*]] = arith.constant 10 : i32
// CHECK-DAG:     %[[C20:.*]] = arith.constant 20 : i32
// CHECK:         call @copy_literal_to_inbuffer_i32(%[[C1]], %c0_i32)
// CHECK-NEXT:    call @copy_literal_to_inbuffer_i32(%[[C4]], %c1_i32)
// CHECK-NEXT:    call @copy_literal_to_inbuffer_i32(%[[C10]], %c2_i32)
// CHECK-NEXT:    call @copy_literal_to_inbuffer_i32(%[[C20]], %c3_i32)
// CHECK-NEXT:    call @dma_start_send(%c4_i32, %c0_i32)
func.func @static_tile(%A: memref<40x80xi32>) -> i32 {
  %tA = memref.subview %A[10, 80] [10, 20] [1, 1]
      : memref<40x80xi32> to memref<10x20xi32, strided<[80, 1], offset: 880>>
  %0 = accel.send_idx %tA[0]
      : (memref<10x20xi32, strided<[80, 1], offset: 880>>) -> i32
  %1 = accel.send_idx %tA[1], %0
      : (memref<10x20xi32, strided<[80, 1], offset: 880>>, i32) -> i32
  %2 = accel.send_dim %tA[0], %1
      : (memref<10x20xi32, strided<[80, 1], offset: 880>>, i32) -> i32
  %3 = accel.send_dim %tA[1], %2
      : (memref<10x20xi32, strided<[80, 1], offset: 880>>, i32) -> i32
  return %3 : i32
}

// In a loop the tile index is computed from the loop-dependent offset.
// CHECK-LABEL: func.func @loop_tile(
// CHECK:         scf.for %[[I:.*]] = %c0 to %c40 step %c10
// CHECK:           %[[IDX:.*]] = arith.divui %[[I]], %c10 : index
// CHECK:           %[[IDX32:.*]] = arith.index_cast %[[IDX]] : index to i32
// CHECK:           call @copy_literal_to_inbuffer_i32(%[[IDX32]], %c0_i32)
func.func @loop_tile(%A: memref<40x80xi32>) {
  %c0 = arith.constant 0 : index
  %c10 = arith.constant 10 : index
  %c40 = arith.constant 40 : index
  scf.for %i = %c0 to %c40 step %c10 {
    %tA = memref.subview %A[%i, 0] [10, 20] [1, 1]
        : memref<40x80xi32> to memref<10x20xi32, strided<[80, 1], offset: ?>>
    %0 = accel.send_idx %tA[0]
        : (memref<10x20xi32, strided<[80, 1], offset: ?>>) -> i32
  }
  return
}

// A whole buffer is tile 0, and dynamic sizes are read with memref.dim.
// CHECK-LABEL: func.func @whole_buffer(
// CHECK-SAME:      %[[A:.*]]: memref<?x8xi32>)
// CHECK:         call @copy_literal_to_inbuffer_i32(%c0_i32, %c0_i32)
// CHECK:         %[[D:.*]] = memref.dim %[[A]], %c0
// CHECK:         %[[D32:.*]] = arith.index_cast %[[D]] : index to i32
// CHECK:         call @copy_literal_to_inbuffer_i32(%[[D32]], %c1_i32)
func.func @whole_buffer(%A: memref<?x8xi32>) -> i32 {
  %0 = accel.send_idx %A[0] : (memref<?x8xi32>) -> i32
  %1 = accel.send_dim %A[0], %0 : (memref<?x8xi32>, i32) -> i32
  return %1 : i32
}
