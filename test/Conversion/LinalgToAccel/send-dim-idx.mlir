// RUN: soda-opt %s --convert-linalg-to-accel --canonicalize --cse | FileCheck %s

// A 40x80 elementwise op tiled by (10, 20). For each tile of A, the
// accelerator gets its tile index and its sizes before the data. On the tile
// at offset (10, 80), send_idx sends 1 and 4 and send_dim sends 10 and 20
// (see test/Runner/axi/send-dim-idx-generic.mlir for the values at runtime).

// CHECK-LABEL: func.func @tile_info(
// CHECK-SAME:      %[[A:.*]]: memref<40x80xi32>, %[[B:.*]]: memref<40x80xi32>)
// CHECK:         scf.for %[[I:.*]] = %c0 to %c40 step %c10 {
// CHECK-NEXT:      scf.for %[[J:.*]] = %c0 to %c80 step %c20 {
// CHECK-NEXT:        %[[TA:.*]] = memref.subview %[[A]][%[[I]], %[[J]]] [10, 20] [1, 1]
// CHECK-NEXT:        %[[O0:.*]] = accel.send_idx %[[TA]][0] :
// CHECK-NEXT:        %[[O1:.*]] = accel.send_idx %[[TA]][1], %[[O0]] :
// CHECK-NEXT:        %[[O2:.*]] = accel.send_dim %[[TA]][0], %[[O1]] :
// CHECK-NEXT:        %[[O3:.*]] = accel.send_dim %[[TA]][1], %[[O2]] :
// CHECK-NEXT:        accel.send %[[TA]], %[[O3]] :
// CHECK-NEXT:        %[[TB:.*]] = memref.subview %[[B]][%[[I]], %[[J]]] [10, 20] [1, 1]
// CHECK-NEXT:        accel.recv %[[TB]] :
func.func @tile_info(%A: memref<40x80xi32>, %B: memref<40x80xi32>) {
  linalg.generic {
      indexing_maps = [affine_map<(i, j) -> (i, j)>, affine_map<(i, j) -> (i, j)>],
      iterator_types = ["parallel", "parallel"],
      accel.accel_dim = affine_map<(i, j) -> (10, 20)>,
      accel.opcode_map = #accel.opcode_map<
          t = [send_idx(0, 0), send_idx(0, 1), send_dim(0, 0), send_dim(0, 1),
               send(0), recv(1)]>,
      accel.opcode_flow = #accel.opcode_flow<(t)>}
      ins(%A : memref<40x80xi32>) outs(%B : memref<40x80xi32>) {
  ^bb0(%a: i32, %b: i32):
    linalg.yield %a : i32
  }
  return
}

// A send_dim of an untiled dimension sends its whole size; send_idx of it
// sends 0 (the op is not tiled along it).
// CHECK-LABEL: func.func @untiled_dim(
// CHECK-SAME:      %[[A:.*]]: memref<40x80xi32>, %[[B:.*]]: memref<40x80xi32>)
// CHECK:         scf.for %[[I:.*]] = %c0 to %c40 step %c10 {
// CHECK-NEXT:      %[[TA:.*]] = memref.subview %[[A]][%[[I]], 0] [10, 80] [1, 1]
// CHECK-NEXT:      accel.send_dim %[[TA]][1] :
func.func @untiled_dim(%A: memref<40x80xi32>, %B: memref<40x80xi32>) {
  linalg.generic {
      indexing_maps = [affine_map<(i, j) -> (i, j)>, affine_map<(i, j) -> (i, j)>],
      iterator_types = ["parallel", "parallel"],
      accel.accel_dim = affine_map<(i, j) -> (10, 0)>,
      accel.opcode_map = #accel.opcode_map<t = [send_dim(0, 1), send(0), recv(1)]>,
      accel.opcode_flow = #accel.opcode_flow<(t)>}
      ins(%A : memref<40x80xi32>) outs(%B : memref<40x80xi32>) {
  ^bb0(%a: i32, %b: i32):
    linalg.yield %a : i32
  }
  return
}
