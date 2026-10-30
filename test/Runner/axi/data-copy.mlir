// RUN: soda-opt %s %axi_lower_to_llvm \
// RUN: | mlir-runner -e main -entry-point-result=void \
// RUN:   -shared-libs=%axi_mock_lib %axi_runner_libs \
// RUN: | FileCheck %s

// Moves data through the mock DMA, which streams sent words back on receive.
// This checks the memref copies of the runtime: strided tiles, chained
// offsets, f32 data and accumulate mode.

func.func private @printMemrefI32(memref<*xi32>) attributes {llvm.emit_c_interface}
func.func private @printMemrefF32(memref<*xf32>) attributes {llvm.emit_c_interface}

func.func @iota(%m: memref<8x8xi32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  scf.for %i = %c0 to %c8 step %c1 {
    scf.for %j = %c0 to %c8 step %c1 {
      %row = arith.muli %i, %c8 : index
      %idx = arith.addi %row, %j : index
      %v = arith.index_cast %idx : index to i32
      memref.store %v, %m[%i, %j] : memref<8x8xi32>
    }
  }
  return
}

func.func @main() {
  %c0_i32 = arith.constant 0 : i32
  %c64 = arith.constant 64 : i32
  accel.init_dma %c0_i32, %c0_i32, %c64, %c0_i32, %c64
      : (i32, i32, i32, i32, i32)

  %A = memref.alloc() : memref<8x8xi32>
  call @iota(%A) : (memref<8x8xi32>) -> ()
  %B = memref.alloc() : memref<4x4xi32>
  %zero = arith.constant 0 : i32
  linalg.fill ins(%zero : i32) outs(%B : memref<4x4xi32>)

  // A strided 4x4 tile of A (rows 2-5, columns 4-7) goes out and comes back.
  %tile = memref.subview %A[2, 4] [4, 4] [1, 1]
      : memref<8x8xi32> to memref<4x4xi32, strided<[8, 1], offset: 20>>
  %0 = accel.send %tile
      : (memref<4x4xi32, strided<[8, 1], offset: 20>>) -> i32
  %1 = accel.recv %B : (memref<4x4xi32>) -> i32
  %Bc = memref.cast %B : memref<4x4xi32> to memref<*xi32>
  call @printMemrefI32(%Bc) : (memref<*xi32>) -> ()
  // CHECK:      [20,   21,   22,   23],
  // CHECK-NEXT: [28,   29,   30,   31],
  // CHECK-NEXT: [36,   37,   38,   39],
  // CHECK-NEXT: [44,   45,   46,   47]]

  // A column with a non-unit stride, chained after an opcode word, is
  // received with accumulate mode into a row of B.
  %col = memref.subview %A[0, 1] [4, 1] [2, 1]
      : memref<8x8xi32> to memref<4x1xi32, strided<[16, 1], offset: 1>>
  %row = memref.subview %B[0, 0] [1, 4] [1, 1]
      : memref<4x4xi32> to memref<1x4xi32, strided<[4, 1]>>
  %opcode = arith.constant 100 : i32
  %2 = accel.send_literal %opcode : (i32) -> i32
  %3 = accel.send %col, %2
      : (memref<4x1xi32, strided<[16, 1], offset: 1>>, i32) -> i32
  %4 = accel.recv %row : (memref<1x4xi32, strided<[4, 1]>>) -> i32
  %5 = accel.recv %row mode = accumulate
      : (memref<1x4xi32, strided<[4, 1]>>) -> i32
  call @printMemrefI32(%Bc) : (memref<*xi32>) -> ()
  // The row first receives [100, 1, 17, 33], then adds [49, 0, 0, 0] (the
  // last word of the stream, then zeros once it is empty).
  // CHECK:      [149,   1,   17,   33],
  // CHECK-NEXT: [28,   29,   30,   31],

  // f32 data.
  %F = memref.alloc() : memref<2x3xf32>
  %G = memref.alloc() : memref<2x3xf32>
  %half = arith.constant 0.5 : f32
  %one = arith.constant 1.0 : f32
  linalg.fill ins(%half : f32) outs(%F : memref<2x3xf32>)
  linalg.fill ins(%one : f32) outs(%G : memref<2x3xf32>)
  %6 = accel.send %F : (memref<2x3xf32>) -> i32
  %7 = accel.recv %G mode = accumulate : (memref<2x3xf32>) -> i32
  %Gc = memref.cast %G : memref<2x3xf32> to memref<*xf32>
  call @printMemrefF32(%Gc) : (memref<*xf32>) -> ()
  // CHECK:      [1.5,   1.5,   1.5],
  // CHECK-NEXT: [1.5,   1.5,   1.5]]

  memref.dealloc %A : memref<8x8xi32>
  memref.dealloc %B : memref<4x4xi32>
  memref.dealloc %F : memref<2x3xf32>
  memref.dealloc %G : memref<2x3xf32>
  return
}
