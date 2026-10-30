// The attributes can come from pass options, as in the AXI4MLIR experiment
// scripts. The #accel. prefix of attribute options is optional.
// RUN: soda-opt %s --convert-linalg-to-accel="anchor-op=linalg.matmul \
// RUN:   opcode-map=opcode_map<sA=[send_literal(1),send(0)],sBcCrC=[send_literal(6),send(1),recv(2,accumulate)],reset=[send_literal(0xFF)]> \
// RUN:   opcode-flow=opcode_flow<(sA(sBcCrC))> init-opcodes=#accel.opcode_flow<(reset)> \
// RUN:   accel-dim=4,4,4 permutation=0,2,1 \
// RUN:   dma-address=41 dma-input-address=42 dma-input-size=1024 dma-output-address=43 dma-output-size=512" \
// RUN:   --canonicalize --cse | FileCheck %s

// Options also select the function.
// RUN: soda-opt %s --convert-linalg-to-accel="anchor-func=other anchor-op=linalg.matmul \
// RUN:   opcode-map=opcode_map<s=[send(0)]> opcode-flow=opcode_flow<(s)>" | FileCheck %s --check-prefix=FUNC

// CHECK-LABEL: func.func @matmul(
// CHECK-SAME:      %[[A:.*]]: memref<8x8xi32>, %[[B:.*]]: memref<8x8xi32>, %[[C:.*]]: memref<8x8xi32>)
// CHECK-DAG:     %[[C41:.*]] = arith.constant 41 : i32
// CHECK-DAG:     %[[C42:.*]] = arith.constant 42 : i32
// CHECK-DAG:     %[[C1024:.*]] = arith.constant 1024 : i32
// CHECK-DAG:     %[[C43:.*]] = arith.constant 43 : i32
// CHECK-DAG:     %[[C512:.*]] = arith.constant 512 : i32
// CHECK:         accel.init_dma %[[C41]], %[[C42]], %[[C1024]], %[[C43]], %[[C512]]
// CHECK-NEXT:    accel.send_literal %c255_i32
// CHECK-NEXT:    scf.for %[[M:.*]] = %c0 to %c8 step %c4 {
// CHECK-NEXT:      scf.for %[[K:.*]] = %c0 to %c8 step %c4 {
// CHECK-NEXT:        accel.send_literal %c1_i32
// CHECK-NEXT:        memref.subview %[[A]][%[[M]], %[[K]]] [4, 4] [1, 1]
// CHECK-NEXT:        accel.send
// CHECK-NEXT:        scf.for %[[N:.*]] = %c0 to %c8 step %c4 {
// CHECK-NEXT:          accel.send_literal %c6_i32
// CHECK-NEXT:          memref.subview %[[B]][%[[K]], %[[N]]] [4, 4] [1, 1]
// CHECK-NEXT:          accel.send
// CHECK-NEXT:          memref.subview %[[C]][%[[M]], %[[N]]] [4, 4] [1, 1]
// CHECK-NEXT:          accel.recv {{.*}} mode = accumulate
// CHECK-NOT:     linalg.matmul

// FUNC-LABEL: func.func @matmul(
// FUNC:         linalg.matmul
// FUNC-NOT:     accel.opcode_flow
// FUNC-LABEL: func.func @other(
// FUNC-NOT:     linalg.matmul
// FUNC:         accel.send
func.func @matmul(%A: memref<8x8xi32>, %B: memref<8x8xi32>, %C: memref<8x8xi32>) {
  linalg.matmul ins(%A, %B : memref<8x8xi32>, memref<8x8xi32>)
                outs(%C : memref<8x8xi32>)
  return
}

// Attributes already on the op take precedence over the options.
// CHECK-LABEL: func.func @other(
// CHECK-NOT:     scf.for
// CHECK:         accel.send_literal %c255_i32
// CHECK-NEXT:    accel.send_literal %c7_i32
// CHECK-NEXT:    accel.send
func.func @other(%A: memref<8x8xi32>, %B: memref<8x8xi32>, %C: memref<8x8xi32>) {
  linalg.matmul {accel.opcode_map = #accel.opcode_map<s = [send_literal(7), send(0)], reset = [send_literal(0xFF)]>,
                 accel.opcode_flow = #accel.opcode_flow<(s)>,
                 accel.accel_dim = affine_map<(m, n, k) -> (0, 0, 0)>}
      ins(%A, %B : memref<8x8xi32>, memref<8x8xi32>)
      outs(%C : memref<8x8xi32>)
  return
}
