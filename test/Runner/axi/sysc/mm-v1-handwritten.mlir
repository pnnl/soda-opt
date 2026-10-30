// REQUIRES: systemc
// RUN: soda-opt %s %axi_lower_to_llvm \
// RUN: | mlir-runner -e main -entry-point-result=void \
// RUN:   -shared-libs=%axi_sysc_lib_dir/libsoda_axi_runtime_sysc_mm_4x4_v1%shlibext \
// RUN:   %axi_runner_libs \
// RUN: | FileCheck %s

// An 8x8x8 matmul written directly with accel ops, driving the SystemC model
// of mm_4x4_v1. That model has no opcodes: it reads a 4x4 tile of A and of
// B, then sends back their product. The host accumulates the products into C.

func.func private @verifyMemRefI32(memref<*xi32>, memref<*xi32>) -> i64 attributes {llvm.emit_c_interface}
func.func private @printMemrefI32(memref<*xi32>) attributes {llvm.emit_c_interface}
func.func private @printI64(i64)
func.func private @printNewline()

// Fills `m` with (i * 3 + j * 5) % 7 - 3 + `seed`.
func.func @fill(%m: memref<8x8xi32>, %seed: i32) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %c3 = arith.constant 3 : index
  %c5 = arith.constant 5 : index
  %c7 = arith.constant 7 : index
  scf.for %i = %c0 to %c8 step %c1 {
    scf.for %j = %c0 to %c8 step %c1 {
      %a = arith.muli %i, %c3 : index
      %b = arith.muli %j, %c5 : index
      %s = arith.addi %a, %b : index
      %r = arith.remui %s, %c7 : index
      %v = arith.index_cast %r : index to i32
      %c3_i32 = arith.constant 3 : i32
      %w = arith.subi %v, %c3_i32 : i32
      %x = arith.addi %w, %seed : i32
      memref.store %x, %m[%i, %j] : memref<8x8xi32>
    }
  }
  return
}

func.func @matmul_accel(%A: memref<8x8xi32>, %B: memref<8x8xi32>,
                        %C: memref<8x8xi32>) {
  %addr = arith.constant 0x40400000 : i32
  %in = arith.constant 0x16000000 : i32
  %out = arith.constant 0x16400000 : i32
  %size = arith.constant 1024 : i32
  accel.init_dma %addr, %in, %size, %out, %size : (i32, i32, i32, i32, i32)
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  scf.for %m = %c0 to %c8 step %c4 {
    scf.for %n = %c0 to %c8 step %c4 {
      scf.for %k = %c0 to %c8 step %c4 {
        %tA = memref.subview %A[%m, %k] [4, 4] [1, 1]
            : memref<8x8xi32> to memref<4x4xi32, strided<[8, 1], offset: ?>>
        %tB = memref.subview %B[%k, %n] [4, 4] [1, 1]
            : memref<8x8xi32> to memref<4x4xi32, strided<[8, 1], offset: ?>>
        %tC = memref.subview %C[%m, %n] [4, 4] [1, 1]
            : memref<8x8xi32> to memref<4x4xi32, strided<[8, 1], offset: ?>>
        %0 = accel.send %tA
            : (memref<4x4xi32, strided<[8, 1], offset: ?>>) -> i32
        %1 = accel.send %tB, %0
            : (memref<4x4xi32, strided<[8, 1], offset: ?>>, i32) -> i32
        %2 = accel.recv %tC mode = accumulate
            : (memref<4x4xi32, strided<[8, 1], offset: ?>>) -> i32
      }
    }
  }
  return
}

func.func @main() {
  %A = memref.alloc() : memref<8x8xi32>
  %B = memref.alloc() : memref<8x8xi32>
  %C = memref.alloc() : memref<8x8xi32>
  %Cref = memref.alloc() : memref<8x8xi32>
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  call @fill(%A, %c0_i32) : (memref<8x8xi32>, i32) -> ()
  call @fill(%B, %c1_i32) : (memref<8x8xi32>, i32) -> ()
  linalg.fill ins(%c0_i32 : i32) outs(%C : memref<8x8xi32>)
  linalg.fill ins(%c0_i32 : i32) outs(%Cref : memref<8x8xi32>)

  call @matmul_accel(%A, %B, %C)
      : (memref<8x8xi32>, memref<8x8xi32>, memref<8x8xi32>) -> ()
  linalg.matmul ins(%A, %B : memref<8x8xi32>, memref<8x8xi32>)
                outs(%Cref : memref<8x8xi32>)

  %Cu = memref.cast %C : memref<8x8xi32> to memref<*xi32>
  %Crefu = memref.cast %Cref : memref<8x8xi32> to memref<*xi32>
  %errors = call @verifyMemRefI32(%Cu, %Crefu)
      : (memref<*xi32>, memref<*xi32>) -> i64
  // CHECK: {{^0$}}
  call @printI64(%errors) : (i64) -> ()
  call @printNewline() : () -> ()
  // CHECK: [20,   -9,   -3,   17,   -19,   -20,   -7,   20],
  call @printMemrefI32(%Cu) : (memref<*xi32>) -> ()

  memref.dealloc %A : memref<8x8xi32>
  memref.dealloc %B : memref<8x8xi32>
  memref.dealloc %C : memref<8x8xi32>
  memref.dealloc %Cref : memref<8x8xi32>
  return
}
