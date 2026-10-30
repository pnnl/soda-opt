// RUN: soda-opt %s --transform-interpreter --convert-linalg-to-accel --canonicalize --cse | FileCheck %s

// Tiling for CPU caches composes with the accelerator lowering: tile the
// matmul by 8 with the transform dialect, then the 8x8x8 tiles, which keep
// their accel.* attributes, are lowered with 4x4x4 accelerator tiles.

// CHECK-LABEL: func.func @matmul(
// CHECK-SAME:      %[[A:[^:]*]]: memref<16x16xi32>
// CHECK:         accel.init_dma
// CHECK:         scf.for %[[M0:.*]] = %c0 to %c16 step %c8 {
// CHECK-NEXT:      scf.for %[[N0:.*]] = %c0 to %c16 step %c8 {
// CHECK-NEXT:        scf.for %[[K0:.*]] = %c0 to %c16 step %c8 {
// CHECK-NEXT:          %[[TA:.*]] = memref.subview %[[A]][%[[M0]], %[[K0]]] [8, 8] [1, 1]
// CHECK:               scf.for %[[M1:.*]] = %c0 to %c8 step %c4 {
// CHECK-NEXT:            scf.for %[[N1:.*]] = %c0 to %c8 step %c4 {
// CHECK-NEXT:              scf.for %[[K1:.*]] = %c0 to %c8 step %c4 {
// CHECK-NEXT:                accel.send_literal %c7_i32
// CHECK-NEXT:                memref.subview %[[TA]][%[[M1]], %[[K1]]] [4, 4] [1, 1]
// CHECK:                     accel.recv {{.*}} mode = accumulate
// CHECK-NOT:     {{^ *}}linalg.matmul
func.func @matmul(%A: memref<16x16xi32>, %B: memref<16x16xi32>,
                  %C: memref<16x16xi32>) {
  linalg.matmul {accel.accel_dim = affine_map<(m, n, k) -> (4, 4, 4)>,
                 accel.opcode_map = #accel.opcode_map<
                     s = [send_literal(7), send(0), send(1)],
                     r = [recv(2, accumulate)]>,
                 accel.opcode_flow = #accel.opcode_flow<(s r)>}
      ins(%A, %B : memref<16x16xi32>, memref<16x16xi32>)
      outs(%C : memref<16x16xi32>)
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %mm = transform.structured.match ops{["linalg.matmul"]} in %root
        : (!transform.any_op) -> !transform.any_op
    %tiled, %loops:3 = transform.structured.tile_using_for %mm tile_sizes [8, 8, 8]
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op,
                                  !transform.any_op, !transform.any_op)
    transform.yield
  }
}
