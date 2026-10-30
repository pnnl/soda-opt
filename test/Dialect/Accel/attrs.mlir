// RUN: soda-opt %s --allow-unregistered-dialect | FileCheck %s
// RUN: soda-opt %s --allow-unregistered-dialect | soda-opt --allow-unregistered-dialect | FileCheck %s

// Standalone attributes.

// CHECK-LABEL: "test.actions"
// CHECK-SAME: a = #accel.action<send(0)>
// CHECK-SAME: b = #accel.action<send_literal(0x22)>
// CHECK-SAME: c = #accel.action<send_dim(1, 3)>
// CHECK-SAME: d = #accel.action<send_idx(0, 1)>
// CHECK-SAME: e = #accel.action<recv(2)>
// CHECK-SAME: f = #accel.action<recv(2, accumulate)>
// CHECK-SAME: g = #accel.action<recv(2, accumulate)>
// CHECK-SAME: h = #accel.action<recv(2)>
// CHECK-SAME: i = #accel.action<send_literal(0x7)>
"test.actions"() {a = #accel.action<send(0)>,
                  b = #accel.action<send_literal(34)>,
                  c = #accel.action<send_dim(1, 3)>,
                  d = #accel.action<send_idx(0, 1)>,
                  e = #accel.action<recv(2)>,
                  f = #accel.action<recv(2, accumulate)>,
                  g = #accel.action<recv(2, "accumulate")>,
                  h = #accel.action<recv(2, replace)>,
                  i = #accel.action<send_literal(0x7)>} : () -> ()

// The flows of the AXI4MLIR paper.
// CHECK-LABEL: "test.flows"
// CHECK-SAME: as = #accel.opcode_flow<(sA (sBcCrC))>
// CHECK-SAME: conv = #accel.opcode_flow<(sF (sIcO) rO)>
// CHECK-SAME: conv_skip = #accel.opcode_flow<(sF ((sIcO)) rO)>
// CHECK-SAME: cs = #accel.opcode_flow<((sA sB cC) rC)>
// CHECK-SAME: ns = #accel.opcode_flow<(sB sA cC rC)>
// CHECK-SAME: quoted = #accel.opcode_flow<("send A" (s1))>
"test.flows"() {ns = #accel.opcode_flow<(sB sA cC rC)>,
                as = #accel.opcode_flow<(sA (sBcCrC))>,
                cs = #accel.opcode_flow<((sA sB cC) rC)>,
                conv = #accel.opcode_flow<(sF (sIcO) rO)>,
                conv_skip = #accel.opcode_flow<( sF ( ( sIcO ) ) rO )>,
                quoted = #accel.opcode_flow<("send A" ("s1"))>} : () -> ()

// CHECK-LABEL: "test.map"
// CHECK-SAME: #accel.opcode_map<sA = [send_literal(0x22), send(0)], sB = [send_literal(0x23), send(1)], cC = [send_literal(0xF0)], rC = [send_literal(0x24), recv(2)], sBcCrC = [send_literal(0x25), send(1), recv(2)], reset = [send_literal(0xFF)], "a b" = [send_dim(1, 3)]>
"test.map"() {map = #accel.opcode_map<
    sA       = [send_literal(0x22), send(0)],
    sB       = [send_literal(0x23), send(1)],
    cC       = [send_literal(0xF0)],
    rC       = [send_literal(0x24), recv(2)],
    sBcCrC   = [send_literal(0x25), send(1), recv(2)],
    reset    = [send_literal(0xFF)],
    "a b"    = [send_dim(1, 3)]>} : () -> ()

// CHECK-LABEL: "test.dma"
// CHECK-SAME: #accel.dma_config<address = 1077936128, input_address = 369098752, input_size = 65536, output_address = 373293056, output_size = 65536>
"test.dma"() {cfg = #accel.dma_config<address = 0x40400000,
    input_address = 0x16000000, input_size = 65536,
    output_address = 0x16400000, output_size = 65536>} : () -> ()

// The attributes on a linalg.generic, as in Fig. 6a of the AXI4MLIR paper.

#map_a = affine_map<(m, n, k) -> (m, k)>
#map_b = affine_map<(m, n, k) -> (k, n)>
#map_c = affine_map<(m, n, k) -> (m, n)>
#matmul_accel_trait = {
  indexing_maps = [#map_a, #map_b, #map_c],
  iterator_types = ["parallel", "parallel", "reduction"],
  accel.dma_init_config = #accel.dma_config<address = 0, input_address = 0x42,
      input_size = 0xFF00, output_address = 0xFF42, output_size = 0xFF00>,
  accel.init_opcodes = #accel.opcode_flow<(reset)>,
  accel.accel_dim = affine_map<(m, n, k) -> (4, 4, 4)>,
  accel.permutation_map = affine_map<(m, n, k) -> (m, k, n)>,
  accel.opcode_map = #accel.opcode_map<
    sA     = [send_literal(0x22), send(0)],
    sB     = [send_literal(0x23), send(1)],
    cC     = [send_literal(0xF0)],
    rC     = [send_literal(0x24), recv(2)],
    sBcCrC = [send_literal(0x25), send(1), recv(2, accumulate)],
    reset  = [send_literal(0xFF)]>,
  accel.opcode_flow = #accel.opcode_flow<(sA (sBcCrC))>
}

// CHECK-LABEL: func.func @matmul_as
// CHECK: linalg.generic
// CHECK-SAME: accel.accel_dim = #map{{[0-9]*}},
// CHECK-SAME: accel.dma_init_config = #accel.dma_config<address = 0, input_address = 66, input_size = 65280, output_address = 65346, output_size = 65280>
// CHECK-SAME: accel.init_opcodes = #accel.opcode_flow<(reset)>
// CHECK-SAME: accel.opcode_flow = #accel.opcode_flow<(sA (sBcCrC))>
// CHECK-SAME: accel.opcode_map = #accel.opcode_map<sA = [send_literal(0x22), send(0)], sB = [send_literal(0x23), send(1)], cC = [send_literal(0xF0)], rC = [send_literal(0x24), recv(2)], sBcCrC = [send_literal(0x25), send(1), recv(2, accumulate)], reset = [send_literal(0xFF)]>
// CHECK-SAME: accel.permutation_map = #map{{[0-9]*}}}
func.func @matmul_as(%A: memref<16x8xi32>, %B: memref<8x32xi32>,
                     %C: memref<16x32xi32>) {
  linalg.generic #matmul_accel_trait
      ins(%A, %B : memref<16x8xi32>, memref<8x32xi32>)
      outs(%C : memref<16x32xi32>) {
  ^bb0(%a: i32, %b: i32, %c: i32):
    %0 = arith.muli %a, %b : i32
    %1 = arith.addi %c, %0 : i32
    linalg.yield %1 : i32
  }
  return
}

// Named ops carry the attributes too.
// CHECK-LABEL: func.func @named_matmul
// CHECK: linalg.matmul
// CHECK-SAME: accel.opcode_flow = #accel.opcode_flow<(s r)>
func.func @named_matmul(%A: memref<16x8xf32>, %B: memref<8x32xf32>,
                        %C: memref<16x32xf32>) {
  linalg.matmul {accel.opcode_map = #accel.opcode_map<s = [send(0), send(1)],
                                                      r = [recv(2, accumulate)]>,
                 accel.opcode_flow = #accel.opcode_flow<(s r)>}
      ins(%A, %B : memref<16x8xf32>, memref<8x32xf32>)
      outs(%C : memref<16x32xf32>)
  return
}
