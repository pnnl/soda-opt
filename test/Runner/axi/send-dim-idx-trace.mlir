// RUN: soda-opt %s %axi_lower_to_llvm \
// RUN: | env SODA_AXI_TRACE=1 mlir-runner -e main -entry-point-result=void \
// RUN:   -shared-libs=%axi_mock_lib %axi_runner_libs \
// RUN: | FileCheck %s

// A 40x80 matrix is split into tiles of size (10, 20). For every tile, send
// its index and its sizes: tile (i, j) sends i, j, 10, 20.

// CHECK: [axi] dma_init mock: address=0x40400000 input=0x16000000[1024] output=0x16400000[1024]
// CHECK: [axi] send 4 words @0: 0 0 10 20
// CHECK: [axi] send 4 words @0: 0 1 10 20
// CHECK: [axi] send 4 words @0: 0 2 10 20
// CHECK: [axi] send 4 words @0: 0 3 10 20
// CHECK: [axi] send 4 words @0: 1 0 10 20
// CHECK: [axi] send 4 words @0: 1 1 10 20
// CHECK: [axi] send 4 words @0: 1 2 10 20
// CHECK: [axi] send 4 words @0: 1 3 10 20
// CHECK: [axi] send 4 words @0: 1 4 10 20
// CHECK-NOT: [axi] send
// CHECK: [axi] dma_free

func.func @main() {
  %addr = arith.constant 0x40400000 : i32
  %in = arith.constant 0x16000000 : i32
  %out = arith.constant 0x16400000 : i32
  %size = arith.constant 1024 : i32
  accel.init_dma %addr, %in, %size, %out, %size : (i32, i32, i32, i32, i32)

  %A = memref.alloc() : memref<40x80xi32>
  %c0 = arith.constant 0 : index
  %c10 = arith.constant 10 : index
  %c20 = arith.constant 20 : index
  %c40 = arith.constant 40 : index
  %c80 = arith.constant 80 : index
  // Rows of tiles 0 and 1, all columns of tiles 0..3; then tile (1, 4) of a
  // wider view.
  %c11 = arith.constant 11 : index
  scf.for %i = %c0 to %c11 step %c10 {
    scf.for %j = %c0 to %c80 step %c20 {
      %t = memref.subview %A[%i, %j] [10, 20] [1, 1]
          : memref<40x80xi32> to memref<10x20xi32, strided<[80, 1], offset: ?>>
      %0 = accel.send_idx %t[0]
          : (memref<10x20xi32, strided<[80, 1], offset: ?>>) -> i32
      %1 = accel.send_idx %t[1], %0
          : (memref<10x20xi32, strided<[80, 1], offset: ?>>, i32) -> i32
      %2 = accel.send_dim %t[0], %1
          : (memref<10x20xi32, strided<[80, 1], offset: ?>>, i32) -> i32
      %3 = accel.send_dim %t[1], %2
          : (memref<10x20xi32, strided<[80, 1], offset: ?>>, i32) -> i32
    }
  }
  %W = memref.alloc() : memref<40x100xi32>
  %t = memref.subview %W[10, 80] [10, 20] [1, 1]
      : memref<40x100xi32> to memref<10x20xi32, strided<[100, 1], offset: 1080>>
  %0 = accel.send_idx %t[0]
      : (memref<10x20xi32, strided<[100, 1], offset: 1080>>) -> i32
  %1 = accel.send_idx %t[1], %0
      : (memref<10x20xi32, strided<[100, 1], offset: 1080>>, i32) -> i32
  %2 = accel.send_dim %t[0], %1
      : (memref<10x20xi32, strided<[100, 1], offset: 1080>>, i32) -> i32
  %3 = accel.send_dim %t[1], %2
      : (memref<10x20xi32, strided<[100, 1], offset: 1080>>, i32) -> i32
  memref.dealloc %A : memref<40x80xi32>
  memref.dealloc %W : memref<40x100xi32>
  return
}
