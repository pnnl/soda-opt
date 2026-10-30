// REQUIRES: systemc
// RUN: soda-opt %s %axi_lower_to_llvm \
// RUN: | env SODA_AXI_SYSC_MAX_NS=10000 not mlir-runner -e main -entry-point-result=void \
// RUN:   -shared-libs=%axi_sysc_lib_dir/libsoda_axi_runtime_sysc_mm_4x4_v3%shlibext \
// RUN:   %axi_runner_libs 2>&1 \
// RUN: | FileCheck %s

// Opcode 3 only loads A and B, so mm_4x4_v3 never sends C. Instead of
// hanging, the receive stops the program once the simulated-time budget is
// spent.

// CHECK: soda axi runtime error: the sysc mm_4x4_v3 simulated accelerator did not complete a receive within 10000 ns of simulated time

func.func @main() {
  %c0 = arith.constant 0 : i32
  %size = arith.constant 256 : i32
  accel.init_dma %c0, %c0, %size, %c0, %size : (i32, i32, i32, i32, i32)
  %A = memref.alloc() : memref<4x4xi32>
  %C = memref.alloc() : memref<4x4xi32>
  %opcode = arith.constant 3 : i32
  %0 = accel.send_literal %opcode : (i32) -> i32
  %1 = accel.send %A, %0 : (memref<4x4xi32>, i32) -> i32
  %2 = accel.send %A, %1 : (memref<4x4xi32>, i32) -> i32
  %3 = accel.recv %C : (memref<4x4xi32>) -> i32
  memref.dealloc %A : memref<4x4xi32>
  memref.dealloc %C : memref<4x4xi32>
  return
}
