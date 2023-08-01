// RUN: soda-opt %s --test-accel-to-axi4mlir | FileCheck %s


// CHECK: func.func private @dma_init
// CHECK-NOT: func.func private @dma_init

// CHECK: func.func private @dma_free
// CHECK-NOT: func.func private @dma_free

// CHECK-LABEL: test_init_dma
// CHECK:         call @dma_init(%arg0
// CHECK:         call @dma_free
func.func @test_init_dma(
  %dmaAddress : i32,
  %dmaInputAddress : i32,
  %dmaInputBufferSize : i32,
  %dmaOutputAddress : i32,
  %dmaOutputBufferSize : i32) {
  accel.init_dma  %dmaAddress,
                  %dmaInputAddress, %dmaInputBufferSize,
                  %dmaOutputAddress, %dmaOutputBufferSize
  : (i32, i32, i32, i32, i32)
  return
}

// CHECK-LABEL: test_init_dma2
// CHECK:         call @dma_init(%arg0
// CHECK-NEXT:    call @dma_init(%arg1
// CHECK-NEXT:    call @dma_init(%arg2
// CHECK:         call @dma_free
func.func @test_init_dma2(
  %dmaAddress : i32,
  %dmaAddress1 : i32,
  %dmaAddress2 : i32,
  %dmaInputAddress : i32,
  %dmaInputBufferSize : i32,
  %dmaOutputAddress : i32,
  %dmaOutputBufferSize : i32) {
  accel.init_dma  %dmaAddress,
                  %dmaInputAddress, %dmaInputBufferSize,
                  %dmaOutputAddress, %dmaOutputBufferSize
  : (i32, i32, i32, i32, i32)
  accel.init_dma  %dmaAddress1,
                  %dmaInputAddress, %dmaInputBufferSize,
                  %dmaOutputAddress, %dmaOutputBufferSize
  : (i32, i32, i32, i32, i32)
  accel.init_dma  %dmaAddress2,
                  %dmaInputAddress, %dmaInputBufferSize,
                  %dmaOutputAddress, %dmaOutputBufferSize
  : (i32, i32, i32, i32, i32)
  return
}

// CHECK-LABEL: test_send
// CHECK:   %[[C0:.*]] = arith.constant 0
// CHECK:   %[[CASTED:.*]] = memref.cast
// CHECK:   call @copy_to_inbuffer_i32(%[[CASTED]], %[[C0]]) : (memref<*xi32>, i32) -> i32
// CHECk:   call @dma_start_send
// CHECK:   call @dma_wait_send
func.func @test_send(%A: memref<60x80xi32>) -> i32 {
  %offset = accel.send %A  : ( memref<60x80xi32> ) -> i32
  return %offset : i32
}

// CHECK-LABEL: test_send_with_offset
// CHECK:   %[[CASTED:.*]] = memref.cast
// CHECK:   call @copy_to_inbuffer_i32(%[[CASTED]], %{{.*}}) : (memref<*xi32>, i32) -> i32
// CHECK:   return %c4800
func.func @test_send_with_offset(%A: memref<60x80xi32>, %offset0: i32) -> i32 {
  %offset = accel.send %A, %offset0  : (memref<60x80xi32> , i32) -> i32
  return %offset : i32
}

// CHECK-LABEL: test_send_with_subview
// CHECK:   %[[CASTED:.*]] = memref.cast
// CHECK:   call @copy_to_inbuffer_i32(%[[CASTED]], %{{.*}}) : (memref<*xi32>, i32) -> i32
// CHECK:   return %c512
#map = affine_map<(d0, d1)[s0] -> (d0 * 1024 + s0 + d1)>
func.func @test_send_with_subview(%input: memref<4x1024xi32>) -> i32 {
  %cst_2 = arith.constant 2 : index
  %0 = memref.subview %input[%cst_2, 256] [2, 256] [1, 1] : memref<4x1024xi32> to memref<2x256xi32, #map>
  %offset = accel.send %0  : ( memref<2x256xi32, #map> ) -> i32
  return %offset : i32
}

// CHECK-LABEL: test_sendLiteral
// CHECK:   %[[INPUT:.*]]: i32)
// CHECK:   %[[TMP:.*]] = memref.alloc() : memref<i32>
// CHECK:   memref.store %[[INPUT]], %[[TMP]][] : memref<i32>
// CHECK:   %[[CASTED:.*]] = memref.cast
// CHECK:   call @copy_to_inbuffer_i32(%[[CASTED]], %{{.*}}) : (memref<*xi32>, i32) -> i32
// CHECK:   memref.dealloc %[[TMP]] : memref<i32>
// CHECK:   return %c1
func.func @test_sendLiteral(%input: i32) -> i32 {
  %offset = accel.sendLiteral %input  : ( i32 ) -> i32
  return %offset : i32
}

// CHECK-LABEL: test_recv_with_offset
// CHECK:   %[[CASTED:.*]] = memref.cast
// CHECk:   call @dma_start_recv
// CHECK:   call @dma_wait_recv
// CHECK:   call @copy_from_outbuffer_i32(%[[CASTED]], %{{.*}}) : (memref<*xi32>, i32) -> i32
func.func @test_recv_with_offset(%A: memref<60x80xi32>, %offset0: i32) -> i32 {
  %offset = accel.recv %A, %offset0  : (memref<60x80xi32> , i32) -> i32
  return %offset : i32
}
