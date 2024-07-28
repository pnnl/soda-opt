// RUN: soda-opt %s | FileCheck %s

// CHECK-LABEL: test_init_dma
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
  func.return
}

// CHECK-LABEL: test_send
func.func @test_send(%A: memref<60x80xf32>) -> i32 {
  %offset = accel.send %A  : (memref<60x80xf32>) -> i32
  func.return %offset : i32
}

// CHECK-LABEL: test_send_with_offset
func.func @test_send_with_offset(%A: memref<60x80xf32>, %offset0: i32) -> i32 {
  %offset = accel.send %A, %offset0  : (memref<60x80xf32> , i32) -> i32
  func.return %offset : i32
}

// CHECK-LABEL: test_recv_with_offset
func.func @test_recv_with_offset(%A: memref<60x80xf32>, %offset0: i32) -> i32 {
  %offset = accel.recv %A, %offset0  : (memref<60x80xf32> , i32) -> i32
  func.return %offset : i32
}
