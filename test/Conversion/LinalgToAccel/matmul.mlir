// RUN: soda-opt %s --convert-linalg-to-accel --canonicalize --cse -split-input-file -verify-diagnostics | FileCheck %s

// The four dataflows of the AXI4MLIR paper on a 16x32x8 matmul with 4x4x4
// accelerator tiles, using the opcodes of its Fig. 6a.

#map_a = affine_map<(m, n, k) -> (m, k)>
#map_b = affine_map<(m, n, k) -> (k, n)>
#map_c = affine_map<(m, n, k) -> (m, n)>
#opcodes = #accel.opcode_map<
  sA     = [send_literal(0x22), send(0)],
  sB     = [send_literal(0x23), send(1)],
  cC     = [send_literal(0xF0)],
  rC     = [send_literal(0x24), recv(2)],
  rCacc  = [send_literal(0x24), recv(2, accumulate)],
  sBcCrC = [send_literal(0x25), send(1), recv(2, accumulate)],
  reset  = [send_literal(0xFF)]>

// Nothing stationary: every opcode runs in the innermost loop.
// CHECK-LABEL: func.func @ns(
// CHECK-SAME:      %[[A:.*]]: memref<16x8xi32>, %[[B:.*]]: memref<8x32xi32>, %[[C:.*]]: memref<16x32xi32>)
// CHECK:         accel.init_dma
// CHECK:         scf.for %[[M:.*]] = %c0 to %c16 step %c4 {
// CHECK-NEXT:      scf.for %[[N:.*]] = %c0 to %c32 step %c4 {
// CHECK-NEXT:        scf.for %[[K:.*]] = %c0 to %c8 step %c4 {
// CHECK-NEXT:          %[[O0:.*]] = accel.send_literal %c35_i32
// CHECK-NEXT:          %[[TB:.*]] = memref.subview %[[B]][%[[K]], %[[N]]] [4, 4] [1, 1]
// CHECK-NEXT:          accel.send %[[TB]], %[[O0]]
// CHECK-NEXT:          %[[O1:.*]] = accel.send_literal %c34_i32
// CHECK-NEXT:          %[[TA:.*]] = memref.subview %[[A]][%[[M]], %[[K]]] [4, 4] [1, 1]
// CHECK-NEXT:          accel.send %[[TA]], %[[O1]]
// CHECK-NEXT:          accel.send_literal %c240_i32
// CHECK-NEXT:          accel.send_literal %c36_i32
// CHECK-NEXT:          %[[TC:.*]] = memref.subview %[[C]][%[[M]], %[[N]]] [4, 4] [1, 1]
// CHECK-NEXT:          accel.recv %[[TC]] mode = accumulate
// CHECK-NEXT:        }
// CHECK-NEXT:      }
// CHECK-NEXT:    }
// CHECK-NOT:     linalg.generic
func.func @ns(%A: memref<16x8xi32>, %B: memref<8x32xi32>, %C: memref<16x32xi32>) {
  linalg.generic {indexing_maps = [#map_a, #map_b, #map_c],
                  iterator_types = ["parallel", "parallel", "reduction"],
                  accel.accel_dim = affine_map<(m, n, k) -> (4, 4, 4)>,
                  accel.opcode_map = #opcodes,
                  accel.opcode_flow = #accel.opcode_flow<(sB sA cC rCacc)>}
      ins(%A, %B : memref<16x8xi32>, memref<8x32xi32>)
      outs(%C : memref<16x32xi32>) {
  ^bb0(%a: i32, %b: i32, %c: i32):
    %0 = arith.muli %a, %b : i32
    %1 = arith.addi %c, %0 : i32
    linalg.yield %1 : i32
  }
  return
}

// -----

#opcodes = #accel.opcode_map<
  sA     = [send_literal(0x22), send(0)],
  sBcCrC = [send_literal(0x25), send(1), recv(2, accumulate)],
  reset  = [send_literal(0xFF)]>

// A stationary: loops (m, k, n); A is sent before the n loop, and the reset
// opcode once before the nest.
// CHECK-LABEL: func.func @as(
// CHECK-SAME:      %[[A:.*]]: memref<16x8xi32>, %[[B:.*]]: memref<8x32xi32>, %[[C:.*]]: memref<16x32xi32>)
// CHECK:         accel.init_dma
// CHECK-NEXT:    accel.send_literal %c255_i32
// CHECK-NEXT:    scf.for %[[M:.*]] = %c0 to %c16 step %c4 {
// CHECK-NEXT:      scf.for %[[K:.*]] = %c0 to %c8 step %c4 {
// CHECK-NEXT:        %[[O0:.*]] = accel.send_literal %c34_i32
// CHECK-NEXT:        %[[TA:.*]] = memref.subview %[[A]][%[[M]], %[[K]]] [4, 4] [1, 1]
// CHECK-NEXT:        accel.send %[[TA]], %[[O0]]
// CHECK-NEXT:        scf.for %[[N:.*]] = %c0 to %c32 step %c4 {
// CHECK-NEXT:          %[[O1:.*]] = accel.send_literal %c37_i32
// CHECK-NEXT:          %[[TB:.*]] = memref.subview %[[B]][%[[K]], %[[N]]] [4, 4] [1, 1]
// CHECK-NEXT:          accel.send %[[TB]], %[[O1]]
// CHECK-NEXT:          %[[TC:.*]] = memref.subview %[[C]][%[[M]], %[[N]]] [4, 4] [1, 1]
// CHECK-NEXT:          accel.recv %[[TC]] mode = accumulate
// CHECK-NEXT:        }
// CHECK-NEXT:      }
// CHECK-NEXT:    }
func.func @as(%A: memref<16x8xi32>, %B: memref<8x32xi32>, %C: memref<16x32xi32>) {
  linalg.matmul {accel.accel_dim = affine_map<(m, n, k) -> (4, 4, 4)>,
                 accel.permutation_map = affine_map<(m, n, k) -> (m, k, n)>,
                 accel.opcode_map = #opcodes,
                 accel.init_opcodes = #accel.opcode_flow<(reset)>,
                 accel.opcode_flow = #accel.opcode_flow<(sA (sBcCrC))>}
      ins(%A, %B : memref<16x8xi32>, memref<8x32xi32>)
      outs(%C : memref<16x32xi32>)
  return
}

// -----

#opcodes = #accel.opcode_map<
  sB     = [send_literal(0x23), send(1)],
  sAcCrC = [send_literal(0x26), send(0), recv(2, accumulate)]>

// B stationary: loops (n, k, m).
// CHECK-LABEL: func.func @bs(
// CHECK-SAME:      %[[A:.*]]: memref<16x8xi32>, %[[B:.*]]: memref<8x32xi32>, %[[C:.*]]: memref<16x32xi32>)
// CHECK:         scf.for %[[N:.*]] = %c0 to %c32 step %c4 {
// CHECK-NEXT:      scf.for %[[K:.*]] = %c0 to %c8 step %c4 {
// CHECK-NEXT:        accel.send_literal %c35_i32
// CHECK-NEXT:        %[[TB:.*]] = memref.subview %[[B]][%[[K]], %[[N]]] [4, 4] [1, 1]
// CHECK-NEXT:        accel.send %[[TB]]
// CHECK-NEXT:        scf.for %[[M:.*]] = %c0 to %c16 step %c4 {
// CHECK-NEXT:          accel.send_literal %c38_i32
// CHECK-NEXT:          %[[TA:.*]] = memref.subview %[[A]][%[[M]], %[[K]]] [4, 4] [1, 1]
// CHECK-NEXT:          accel.send %[[TA]]
// CHECK-NEXT:          %[[TC:.*]] = memref.subview %[[C]][%[[M]], %[[N]]] [4, 4] [1, 1]
// CHECK-NEXT:          accel.recv %[[TC]] mode = accumulate
func.func @bs(%A: memref<16x8xi32>, %B: memref<8x32xi32>, %C: memref<16x32xi32>) {
  linalg.matmul {accel.accel_dim = affine_map<(m, n, k) -> (4, 4, 4)>,
                 accel.permutation_map = affine_map<(m, n, k) -> (n, k, m)>,
                 accel.opcode_map = #opcodes,
                 accel.opcode_flow = #accel.opcode_flow<(sB (sAcCrC))>}
      ins(%A, %B : memref<16x8xi32>, memref<8x32xi32>)
      outs(%C : memref<16x32xi32>)
  return
}

// -----

#opcodes = #accel.opcode_map<
  sA = [send_literal(0x22), send(0)],
  sB = [send_literal(0x23), send(1)],
  cC = [send_literal(0xF0)],
  rC = [send_literal(0x24), recv(2, accumulate)]>

// C stationary: C is received after the k loop.
// CHECK-LABEL: func.func @cs(
// CHECK-SAME:      %[[A:.*]]: memref<16x8xi32>, %[[B:.*]]: memref<8x32xi32>, %[[C:.*]]: memref<16x32xi32>)
// CHECK:         scf.for %[[M:.*]] = %c0 to %c16 step %c4 {
// CHECK-NEXT:      scf.for %[[N:.*]] = %c0 to %c32 step %c4 {
// CHECK-NEXT:        scf.for %[[K:.*]] = %c0 to %c8 step %c4 {
// CHECK-NEXT:          accel.send_literal %c34_i32
// CHECK-NEXT:          memref.subview %[[A]][%[[M]], %[[K]]] [4, 4] [1, 1]
// CHECK-NEXT:          accel.send
// CHECK-NEXT:          accel.send_literal %c35_i32
// CHECK-NEXT:          memref.subview %[[B]][%[[K]], %[[N]]] [4, 4] [1, 1]
// CHECK-NEXT:          accel.send
// CHECK-NEXT:          accel.send_literal %c240_i32
// CHECK-NEXT:        }
// CHECK-NEXT:        %[[O:.*]] = accel.send_literal %c36_i32
// CHECK-NEXT:        %[[TC:.*]] = memref.subview %[[C]][%[[M]], %[[N]]] [4, 4] [1, 1]
// CHECK-NEXT:        accel.recv %[[TC]] mode = accumulate
// CHECK-NEXT:      }
// CHECK-NEXT:    }
func.func @cs(%A: memref<16x8xi32>, %B: memref<8x32xi32>, %C: memref<16x32xi32>) {
  linalg.matmul {accel.accel_dim = affine_map<(m, n, k) -> (4, 4, 4)>,
                 accel.opcode_map = #opcodes,
                 accel.opcode_flow = #accel.opcode_flow<((sA sB cC) rC)>}
      ins(%A, %B : memref<16x8xi32>, memref<8x32xi32>)
      outs(%C : memref<16x32xi32>)
  return
}

// -----

// A replace-mode receive inside the reduction loop loses partial results.
func.func @ns_replace(%A: memref<8x8xi32>, %B: memref<8x8xi32>, %C: memref<8x8xi32>) {
  // expected-warning @below {{opcode 'r' receives operand 2 inside reduction loop 2, so every iteration overwrites the previous partial result; use recv(2, accumulate) to add them}}
  linalg.matmul {accel.accel_dim = affine_map<(m, n, k) -> (4, 4, 4)>,
                 accel.opcode_map = #accel.opcode_map<s = [send(0), send(1)], r = [recv(2)]>,
                 accel.opcode_flow = #accel.opcode_flow<(s r)>}
      ins(%A, %B : memref<8x8xi32>, memref<8x8xi32>)
      outs(%C : memref<8x8xi32>)
  return
}

// -----

// Without accel_dim nothing is tiled: the accelerator gets whole operands,
// once. Two ops in a function share one init_dma.
// CHECK-LABEL: func.func @untiled_twice(
// CHECK-SAME:      %[[A:.*]]: memref<4x4xf32>, %[[B:.*]]: memref<4x4xf32>, %[[C:.*]]: memref<4x4xf32>)
// CHECK-COUNT-1: accel.init_dma
// CHECK-NOT:     accel.init_dma
// CHECK-NOT:     scf.for
// CHECK:         %[[O:.*]] = accel.send %[[A]]
// CHECK-NEXT:    accel.send %[[B]], %[[O]]
// CHECK-NEXT:    accel.recv %[[C]] mode = accumulate
// CHECK:         %[[O2:.*]] = accel.send %[[A]]
// CHECK-NEXT:    accel.send %[[B]], %[[O2]]
// CHECK-NEXT:    accel.recv %[[C]] mode = accumulate
func.func @untiled_twice(%A: memref<4x4xf32>, %B: memref<4x4xf32>, %C: memref<4x4xf32>) {
  linalg.matmul {accel.opcode_map = #accel.opcode_map<s = [send(0), send(1)], r = [recv(2, accumulate)]>,
                 accel.opcode_flow = #accel.opcode_flow<(s r)>}
      ins(%A, %B : memref<4x4xf32>, memref<4x4xf32>)
      outs(%C : memref<4x4xf32>)
  linalg.matmul {accel.opcode_map = #accel.opcode_map<s = [send(0), send(1)], r = [recv(2, accumulate)]>,
                 accel.opcode_flow = #accel.opcode_flow<(s r)>}
      ins(%A, %B : memref<4x4xf32>, memref<4x4xf32>)
      outs(%C : memref<4x4xf32>)
  return
}

// -----

// Ops without accel.opcode_flow are left alone.
// CHECK-LABEL: func.func @no_flow(
// CHECK-NOT:     accel.
// CHECK:         linalg.matmul
func.func @no_flow(%A: memref<4x4xf32>, %B: memref<4x4xf32>, %C: memref<4x4xf32>) {
  linalg.matmul ins(%A, %B : memref<4x4xf32>, memref<4x4xf32>)
                outs(%C : memref<4x4xf32>)
  return
}
