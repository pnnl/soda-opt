// RUN: soda-opt %s --convert-linalg-to-accel --canonicalize --cse | FileCheck %s
// RUN: soda-opt %s --convert-linalg-to-accel=strict-placement -verify-diagnostics

// sA runs before the k loop, but A depends on k. By default A is sent for the
// whole k range (a 4x8 slice) once per (m, n) tile. With strict-placement
// this is an error.

// CHECK-LABEL: func.func @hoisted_dependent_operand(
// CHECK-SAME:      %[[A:[^:]*]]: memref<8x8xi32>
// CHECK:         scf.for %[[M:.*]] = %c0 to %c8 step %c4 {
// CHECK-NEXT:      scf.for %[[N:.*]] = %c0 to %c8 step %c4 {
// CHECK-NEXT:        %[[TA:.*]] = memref.subview %[[A]][%[[M]], 0] [4, 8] [1, 1]
// CHECK-NEXT:        accel.send %[[TA]]
// CHECK-NEXT:        scf.for %[[K:.*]] = %c0 to %c8 step %c4 {
func.func @hoisted_dependent_operand(%A: memref<8x8xi32>, %B: memref<8x8xi32>,
                                     %C: memref<8x8xi32>) {
  // expected-error @below {{opcode 'sA' moves operand 0 outside of loop 2, which the operand depends on; change 'accel.permutation_map' to make that loop enclose the opcode}}
  linalg.matmul {accel.accel_dim = affine_map<(m, n, k) -> (4, 4, 4)>,
                 accel.opcode_map = #accel.opcode_map<sA = [send(0)], sB = [send(1)], r = [recv(2, accumulate)]>,
                 accel.opcode_flow = #accel.opcode_flow<(sA (sB r))>}
      ins(%A, %B : memref<8x8xi32>, memref<8x8xi32>) outs(%C : memref<8x8xi32>)
  return
}
