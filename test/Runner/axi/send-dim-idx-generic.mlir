// RUN: soda-opt %s --convert-linalg-to-accel %axi_lower_to_llvm \
// RUN: | env SODA_AXI_TRACE=1 mlir-runner -e main -entry-point-result=void \
// RUN:   -shared-libs=%axi_mock_lib %axi_runner_libs \
// RUN: | FileCheck %s

// From linalg to the runtime: a 40x100 op tiled by (10, 20) sends, for every
// tile of A, its tile index and sizes as one 4-word transaction. Tile (1, 4)
// is at offset (10, 80): send_idx sends 1 and 4, send_dim sends 10 and 20.

// CHECK:      [axi] dma_init mock
// CHECK-NEXT: [axi] send 4 words @0: 0 0 10 20
// CHECK-NEXT: [axi] send 4 words @0: 0 1 10 20
// CHECK-NEXT: [axi] send 4 words @0: 0 2 10 20
// CHECK-NEXT: [axi] send 4 words @0: 0 3 10 20
// CHECK-NEXT: [axi] send 4 words @0: 0 4 10 20
// CHECK-NEXT: [axi] send 4 words @0: 1 0 10 20
// CHECK-NEXT: [axi] send 4 words @0: 1 1 10 20
// CHECK-NEXT: [axi] send 4 words @0: 1 2 10 20
// CHECK-NEXT: [axi] send 4 words @0: 1 3 10 20
// CHECK-NEXT: [axi] send 4 words @0: 1 4 10 20
// CHECK-NEXT: [axi] send 4 words @0: 2 0 10 20
// CHECK:      [axi] send 4 words @0: 3 4 10 20
// CHECK-NEXT: [axi] dma_free

func.func @tile_info(%A: memref<40x100xi32>, %B: memref<40x100xi32>) {
  linalg.generic {
      indexing_maps = [affine_map<(i, j) -> (i, j)>, affine_map<(i, j) -> (i, j)>],
      iterator_types = ["parallel", "parallel"],
      accel.dma_init_config = #accel.dma_config<address = 0, input_address = 0,
          input_size = 256, output_address = 0, output_size = 256>,
      accel.accel_dim = affine_map<(i, j) -> (10, 20)>,
      accel.opcode_map = #accel.opcode_map<
          t = [send_idx(0, 0), send_idx(0, 1), send_dim(0, 0), send_dim(0, 1)]>,
      accel.opcode_flow = #accel.opcode_flow<(t)>}
      ins(%A : memref<40x100xi32>) outs(%B : memref<40x100xi32>) {
  ^bb0(%a: i32, %b: i32):
    linalg.yield %a : i32
  }
  return
}

func.func @main() {
  %A = memref.alloc() : memref<40x100xi32>
  %B = memref.alloc() : memref<40x100xi32>
  call @tile_info(%A, %B) : (memref<40x100xi32>, memref<40x100xi32>) -> ()
  memref.dealloc %A : memref<40x100xi32>
  memref.dealloc %B : memref<40x100xi32>
  return
}
