// REQUIRES: systemc
// RUN: soda-opt %s --convert-linalg-to-accel %axi_lower_to_llvm \
// RUN: | mlir-runner -e main -entry-point-result=void \
// RUN:   -shared-libs=%axi_sysc_lib_dir/libsoda_axi_runtime_sysc_mm_4x4_v4%shlibext \
// RUN:   %axi_runner_libs \
// RUN: | FileCheck %s

// mm_4x4_v4, A stationary (As), 32x16x16 tiles
//
// Loops (m, k, n): a 32x16 tile of A is loaded once and reused for every
// 16x16 tile of B.
//
// C starts non-zero and is checked against linalg.matmul on the CPU.

// CHECK: {{^0$}}

func.func @matmul_accel(%A: memref<64x48xi32>, %B: memref<48x32xi32>,
                        %C: memref<64x32xi32>) {
  linalg.matmul {
      accel.accel_dim = affine_map<(m, n, k) -> (32, 16, 16)>,
      accel.permutation_map = affine_map<(m, n, k) -> (m, k, n)>,
      accel.opcode_map = #accel.opcode_map<
          sA = [send_literal(1), send_literal(0x1004020), send(0)],
          sBc = [send_literal(14), send_literal(0x1004020), send(1)],
          r = [recv(2, accumulate)]>,
      accel.opcode_flow = #accel.opcode_flow<(sA (sBc r))>}
      ins(%A, %B : memref<64x48xi32>, memref<48x32xi32>)
      outs(%C : memref<64x32xi32>)
  return
}

func.func private @verifyMemRefI32(memref<*xi32>, memref<*xi32>) -> i64 attributes {llvm.emit_c_interface}
func.func private @printI64(i64)
func.func private @printNewline()

// Fills `m` with (i * 7 + j * 3 + seed) % 11 - 5.
func.func @fill(%m: memref<?x?xi32>, %seed: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %c5 = arith.constant 5 : i32
  %c7 = arith.constant 7 : index
  %c11 = arith.constant 11 : index
  %rows = memref.dim %m, %c0 : memref<?x?xi32>
  %cols = memref.dim %m, %c1 : memref<?x?xi32>
  scf.for %i = %c0 to %rows step %c1 {
    scf.for %j = %c0 to %cols step %c1 {
      %a = arith.muli %i, %c7 : index
      %b = arith.muli %j, %c3 : index
      %s0 = arith.addi %a, %b : index
      %s1 = arith.addi %s0, %seed : index
      %r = arith.remui %s1, %c11 : index
      %v = arith.index_cast %r : index to i32
      %w = arith.subi %v, %c5 : i32
      memref.store %w, %m[%i, %j] : memref<?x?xi32>
    }
  }
  return
}

func.func @main() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %A = memref.alloc() : memref<64x48xi32>
  %B = memref.alloc() : memref<48x32xi32>
  %C = memref.alloc() : memref<64x32xi32>
  %Cref = memref.alloc() : memref<64x32xi32>
  %Ad = memref.cast %A : memref<64x48xi32> to memref<?x?xi32>
  %Bd = memref.cast %B : memref<48x32xi32> to memref<?x?xi32>
  %Cd = memref.cast %C : memref<64x32xi32> to memref<?x?xi32>
  %Crefd = memref.cast %Cref : memref<64x32xi32> to memref<?x?xi32>
  call @fill(%Ad, %c0) : (memref<?x?xi32>, index) -> ()
  call @fill(%Bd, %c1) : (memref<?x?xi32>, index) -> ()
  call @fill(%Cd, %c2) : (memref<?x?xi32>, index) -> ()
  call @fill(%Crefd, %c2) : (memref<?x?xi32>, index) -> ()

  call @matmul_accel(%A, %B, %C)
      : (memref<64x48xi32>, memref<48x32xi32>, memref<64x32xi32>) -> ()
  linalg.matmul ins(%A, %B : memref<64x48xi32>, memref<48x32xi32>)
                outs(%Cref : memref<64x32xi32>)

  %Cu = memref.cast %C : memref<64x32xi32> to memref<*xi32>
  %Crefu = memref.cast %Cref : memref<64x32xi32> to memref<*xi32>
  %errors = call @verifyMemRefI32(%Cu, %Crefu)
      : (memref<*xi32>, memref<*xi32>) -> i64
  call @printI64(%errors) : (i64) -> ()
  call @printNewline() : () -> ()

  memref.dealloc %A : memref<64x48xi32>
  memref.dealloc %B : memref<48x32xi32>
  memref.dealloc %C : memref<64x32xi32>
  memref.dealloc %Cref : memref<64x32xi32>
  return
}
