// RUN: soda-opt %s --convert-accel-to-axi-runtime --canonicalize | FileCheck %s
// RUN: soda-opt %s --convert-accel-to-axi-runtime=coalesce-sends=false --canonicalize | FileCheck %s --check-prefix=SPLIT

// An opcode followed by two 4x4 tiles, chained through their offsets, is
// sent as a single 33-word transaction.

// CHECK-LABEL: func.func @chain(
// CHECK:         call @copy_literal_to_inbuffer_i32(%{{.*}}, %c0_i32)
// CHECK:         call @copy_to_inbuffer_i32(%{{.*}}, %c1_i32)
// CHECK:         call @copy_to_inbuffer_i32(%{{.*}}, %c17_i32)
// CHECK-NEXT:    call @dma_start_send(%c33_i32, %c0_i32)
// CHECK-NEXT:    call @dma_wait_send()
// CHECK-NOT:     dma_start_send

// SPLIT-LABEL: func.func @chain(
// SPLIT:         call @copy_literal_to_inbuffer_i32(%{{.*}}, %c0_i32)
// SPLIT-NEXT:    call @dma_start_send(%c1_i32, %c0_i32)
// SPLIT-NEXT:    call @dma_wait_send()
// SPLIT:         call @copy_to_inbuffer_i32(%{{.*}}, %c1_i32)
// SPLIT-NEXT:    call @dma_start_send(%c16_i32, %c1_i32)
// SPLIT-NEXT:    call @dma_wait_send()
// SPLIT:         call @copy_to_inbuffer_i32(%{{.*}}, %c17_i32)
// SPLIT-NEXT:    call @dma_start_send(%c16_i32, %c17_i32)
// SPLIT-NEXT:    call @dma_wait_send()
func.func @chain(%A: memref<4x4xi32>, %B: memref<4x4xi32>) -> i32 {
  %opcode = arith.constant 7 : i32
  %0 = accel.send_literal %opcode : (i32) -> i32
  %1 = accel.send %A, %0 : (memref<4x4xi32>, i32) -> i32
  %2 = accel.send %B, %1 : (memref<4x4xi32>, i32) -> i32
  return %2 : i32
}

// Sends that are not chained are separate transactions.
// CHECK-LABEL: func.func @unchained(
// CHECK:         call @copy_to_inbuffer_i32(%{{.*}}, %c0_i32)
// CHECK-NEXT:    call @dma_start_send(%c16_i32, %c0_i32)
// CHECK-NEXT:    call @dma_wait_send()
// CHECK:         call @copy_to_inbuffer_i32(%{{.*}}, %c0_i32)
// CHECK-NEXT:    call @dma_start_send(%c16_i32, %c0_i32)
// CHECK-NEXT:    call @dma_wait_send()
func.func @unchained(%A: memref<4x4xi32>, %B: memref<4x4xi32>) {
  %0 = accel.send %A : (memref<4x4xi32>) -> i32
  %1 = accel.send %B : (memref<4x4xi32>) -> i32
  return
}

// Pending sends are started before a receive, even when the chain goes on
// after it, so the accelerator gets its inputs before the host waits.
// CHECK-LABEL: func.func @flush_before_recv(
// CHECK:         call @copy_literal_to_inbuffer_i32(%{{.*}}, %c0_i32)
// CHECK:         call @copy_to_inbuffer_i32(%{{.*}}, %c1_i32)
// CHECK-NEXT:    call @dma_start_send(%c17_i32, %c0_i32)
// CHECK-NEXT:    call @dma_wait_send()
// CHECK-NEXT:    call @dma_start_recv(%c16_i32, %c0_i32)
// CHECK-NEXT:    call @dma_wait_recv()
// CHECK:         call @copy_from_outbuffer_i32
// CHECK:         call @copy_to_inbuffer_i32(%{{.*}}, %c17_i32)
// CHECK-NEXT:    call @dma_start_send(%c16_i32, %c17_i32)
func.func @flush_before_recv(%A: memref<4x4xi32>, %C: memref<4x4xi32>) {
  %opcode = arith.constant 5 : i32
  %0 = accel.send_literal %opcode : (i32) -> i32
  %1 = accel.send %A, %0 : (memref<4x4xi32>, i32) -> i32
  %2 = accel.recv %C : (memref<4x4xi32>) -> i32
  %3 = accel.send %A, %1 : (memref<4x4xi32>, i32) -> i32
  return
}

// Pure ops between chained sends do not split the transaction, but a loop
// does.
// CHECK-LABEL: func.func @subviews_and_loops(
// CHECK:         call @copy_literal_to_inbuffer_i32(%{{.*}}, %c0_i32)
// CHECK:         memref.subview
// CHECK:         call @copy_to_inbuffer_i32(%{{.*}}, %c1_i32)
// CHECK-NEXT:    call @dma_start_send(%c17_i32, %c0_i32)
// CHECK:         scf.for
// CHECK:           func.call @copy_to_inbuffer_i32(%{{.*}}, %c0_i32)
// CHECK-NEXT:      func.call @dma_start_send(%c16_i32, %c0_i32)
func.func @subviews_and_loops(%A: memref<8x8xi32>, %i: index) {
  %c0 = arith.constant 0 : index
  %c2 = arith.constant 2 : index
  %c1 = arith.constant 1 : index
  %opcode = arith.constant 1 : i32
  %0 = accel.send_literal %opcode : (i32) -> i32
  %sv = memref.subview %A[%i, 0] [4, 4] [1, 1]
      : memref<8x8xi32> to memref<4x4xi32, strided<[8, 1], offset: ?>>
  %1 = accel.send %sv, %0
      : (memref<4x4xi32, strided<[8, 1], offset: ?>>, i32) -> i32
  scf.for %iv = %c0 to %c2 step %c1 {
    %2 = accel.send %sv
        : (memref<4x4xi32, strided<[8, 1], offset: ?>>) -> i32
  }
  return
}
