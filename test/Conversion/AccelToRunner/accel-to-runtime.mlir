// RUN: soda-opt %s --convert-accel-to-axi-runtime --canonicalize | FileCheck %s

// The runtime functions are declared once, in the order they are first used.
// CHECK-LABEL: module
// CHECK:       func.func private @dma_init(i32, i32, i32, i32, i32)
// CHECK-NEXT:  func.func private @dma_free()
// CHECK-NOT:   func.func private @dma_init
// CHECK-NOT:   func.func private @dma_free

// CHECK-LABEL: func.func @init_dma(
// CHECK-SAME:      %[[ADDR:.*]]: i32, %[[IN:.*]]: i32, %[[IN_SIZE:.*]]: i32, %[[OUT:.*]]: i32, %[[OUT_SIZE:.*]]: i32)
// CHECK:         call @dma_init(%[[ADDR]], %[[IN]], %[[IN_SIZE]], %[[OUT]], %[[OUT_SIZE]])
// CHECK-NEXT:    call @dma_free()
// CHECK-NEXT:    return
func.func @init_dma(%addr: i32, %in: i32, %in_size: i32, %out: i32,
                    %out_size: i32) {
  accel.init_dma %addr, %in, %in_size, %out, %out_size
      : (i32, i32, i32, i32, i32)
  return
}

// dma_free is called before every return of a function with an init_dma.
// CHECK-LABEL: func.func @init_dma_two_returns(
// CHECK:         call @dma_init(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : (i32, i32, i32, i32, i32) -> ()
// CHECK:         cf.cond_br
// CHECK:         call @dma_free()
// CHECK-NEXT:    return
// CHECK:         call @dma_free()
// CHECK-NEXT:    return
func.func @init_dma_two_returns(%a: index, %cond: i1) {
  accel.init_dma %a, %a, %a, %a, %a : (index, index, index, index, index)
  cf.cond_br %cond, ^bb1, ^bb2
^bb1:
  return
^bb2:
  return
}

// CHECK-LABEL: func.func @send(
// CHECK-SAME:      %[[A:.*]]: memref<60x80xi32>)
// CHECK-DAG:     %[[C0:.*]] = arith.constant 0 : i32
// CHECK-DAG:     %[[N:.*]] = arith.constant 4800 : i32
// CHECK:         %[[CAST:.*]] = memref.cast %[[A]] : memref<60x80xi32> to memref<*xi32>
// CHECK:         call @copy_to_inbuffer_i32(%[[CAST]], %[[C0]]) : (memref<*xi32>, i32) -> i32
// CHECK:         call @dma_start_send(%[[N]], %[[C0]]) : (i32, i32) -> i32
// CHECK:         call @dma_wait_send() : () -> ()
// CHECK:         return %[[N]] : i32
func.func @send(%A: memref<60x80xi32>) -> i32 {
  %offset = accel.send %A : (memref<60x80xi32>) -> i32
  return %offset : i32
}

// The result is the offset after the transfer.
// CHECK-LABEL: func.func @send_with_offset(
// CHECK-SAME:      %[[A:.*]]: memref<60x80xf32>, %[[OFF:.*]]: i32)
// CHECK-DAG:     %[[N:.*]] = arith.constant 4800 : i32
// CHECK:         %[[CAST:.*]] = memref.cast %[[A]] : memref<60x80xf32> to memref<*xf32>
// CHECK:         call @copy_to_inbuffer_f32(%[[CAST]], %[[OFF]]) : (memref<*xf32>, i32) -> i32
// CHECK:         %[[END:.*]] = arith.addi %[[OFF]], %[[N]] : i32
// CHECK:         call @dma_start_send(%[[N]], %[[OFF]])
// CHECK:         call @dma_wait_send()
// CHECK:         return %[[END]] : i32
func.func @send_with_offset(%A: memref<60x80xf32>, %offset0: i32) -> i32 {
  %offset = accel.send %A, %offset0 : (memref<60x80xf32>, i32) -> i32
  return %offset : i32
}

// CHECK-LABEL: func.func @send_subview(
// CHECK:         %[[SV:.*]] = memref.subview
// CHECK:         %[[CAST:.*]] = memref.cast %[[SV]] : memref<2x256xi32, strided<[1024, 1], offset: ?>> to memref<*xi32>
// CHECK:         call @copy_to_inbuffer_i32(%[[CAST]], %{{.*}})
// CHECK:         return %c512_i32
func.func @send_subview(%input: memref<4x1024xi32>, %i: index) -> i32 {
  %0 = memref.subview %input[%i, 256] [2, 256] [1, 1]
      : memref<4x1024xi32> to memref<2x256xi32, strided<[1024, 1], offset: ?>>
  %offset = accel.send %0
      : (memref<2x256xi32, strided<[1024, 1], offset: ?>>) -> i32
  return %offset : i32
}

// The number of elements of a dynamic memref is computed at runtime.
// CHECK-LABEL: func.func @send_dynamic(
// CHECK-SAME:      %[[A:.*]]: memref<?x8xi32>)
// CHECK:         %[[D0:.*]] = memref.dim %[[A]], %c0
// CHECK:         %[[N:.*]] = arith.muli %[[D0]], %c8
// CHECK:         %[[N32:.*]] = arith.index_cast %[[N]] : index to i32
// CHECK:         call @dma_start_send(%[[N32]], %c0_i32)
// CHECK:         return %[[N32]]
func.func @send_dynamic(%A: memref<?x8xi32>) -> i32 {
  %offset = accel.send %A : (memref<?x8xi32>) -> i32
  return %offset : i32
}

// Literals are written straight into the DMA buffer.
// CHECK-LABEL: func.func @send_literal(
// CHECK-SAME:      %[[V:.*]]: i32)
// CHECK-NOT:     memref.alloc
// CHECK:         call @copy_literal_to_inbuffer_i32(%[[V]], %c0_i32) : (i32, i32) -> i32
// CHECK:         call @dma_start_send(%c1_i32, %c0_i32)
// CHECK:         call @dma_wait_send()
// CHECK:         return %c1_i32
func.func @send_literal(%v: i32) -> i32 {
  %offset = accel.send_literal %v : (i32) -> i32
  return %offset : i32
}

// CHECK-LABEL: func.func @recv(
// CHECK-SAME:      %[[C:.*]]: memref<60x80xi32>, %[[OFF:.*]]: i32)
// CHECK-DAG:     %[[N:.*]] = arith.constant 4800 : i32
// CHECK:         call @dma_start_recv(%[[N]], %[[OFF]]) : (i32, i32) -> i32
// CHECK:         call @dma_wait_recv() : () -> ()
// CHECK:         %[[CAST:.*]] = memref.cast %[[C]] : memref<60x80xi32> to memref<*xi32>
// CHECK:         call @copy_from_outbuffer_i32(%[[CAST]], %[[OFF]]) : (memref<*xi32>, i32) -> i32
// CHECK:         %[[END:.*]] = arith.addi %[[OFF]], %[[N]] : i32
// CHECK:         return %[[END]]
func.func @recv(%C: memref<60x80xi32>, %offset0: i32) -> i32 {
  %offset = accel.recv %C, %offset0 : (memref<60x80xi32>, i32) -> i32
  return %offset : i32
}

// CHECK-LABEL: func.func @recv_accumulate(
// CHECK:         call @dma_start_recv(%c16_i32, %c0_i32)
// CHECK:         call @dma_wait_recv()
// CHECK:         call @copy_from_outbuffer_accumulate_f32(%{{.*}}, %c0_i32) : (memref<*xf32>, i32) -> i32
func.func @recv_accumulate(%C: memref<4x4xf32>) -> i32 {
  %offset = accel.recv %C mode = accumulate : (memref<4x4xf32>) -> i32
  return %offset : i32
}
