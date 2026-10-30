// RUN: soda-opt %s --convert-linalg-to-accel --canonicalize --cse -split-input-file | FileCheck %s

// conv_v3 of AXI4MLIR on linalg.conv_2d_nchw_fchw, loops
// (n, f, oh, ow, c, kh, kw). The batch and output loops are tiled by 1; the
// accelerator handles the input channels and the filter window.
//
// init: set the filter size (32, FH) and the input channels (16, IC).
// sF: load the filters of one output channel (1).
// sIcO: send one input window and compute one output (2 + 4 + 64 = 70).
// rO: send back the outputs of the current filters (8).

#opcodes = #accel.opcode_map<
  cfgF = [send_literal(32), send_dim(1, 2)],
  cfgC = [send_literal(16), send_dim(0, 1)],
  sF   = [send_literal(1), send(1)],
  sIcO = [send_literal(70), send(0)],
  rO   = [send_literal(8), recv(2, accumulate)]>

// Filter and output stationary per output channel: the empty group skips the
// oh loop, so rO receives the whole output plane after the oh and ow loops.
// This is the structure of the hand-written conv_v3 code of AXI4MLIR.
// CHECK-LABEL: func.func @conv_f_stationary(
// CHECK-SAME:      %[[I:.*]]: memref<1x4x7x7xi32>, %[[F:.*]]: memref<8x4x3x3xi32>, %[[O:.*]]: memref<1x8x5x5xi32>)
// CHECK:         accel.init_dma
// CHECK-NEXT:    %[[I0:.*]] = accel.send_literal %c32_i32
// CHECK-NEXT:    accel.send_dim %[[F]][2], %[[I0]] : (memref<8x4x3x3xi32>, i32) -> i32
// CHECK-NEXT:    %[[I1:.*]] = accel.send_literal %c16_i32
// CHECK-NEXT:    accel.send_dim %[[I]][1], %[[I1]] : (memref<1x4x7x7xi32>, i32) -> i32
// CHECK-NEXT:    scf.for %[[OC:.*]] = %c0 to %c8 step %c1 {
// CHECK-NEXT:      %[[S0:.*]] = accel.send_literal %c1_i32
// CHECK-NEXT:      %[[TF:.*]] = memref.subview %[[F]][%[[OC]], 0, 0, 0] [1, 4, 3, 3] [1, 1, 1, 1]
// CHECK-NEXT:      accel.send %[[TF]], %[[S0]]
// CHECK-NEXT:      scf.for %[[OH:.*]] = %c0 to %c5 step %c1 {
// CHECK-NEXT:        scf.for %[[OW:.*]] = %c0 to %c5 step %c1 {
// CHECK-NEXT:          %[[S1:.*]] = accel.send_literal %c70_i32
// CHECK-NEXT:          %[[TI:.*]] = memref.subview %[[I]][0, 0, %[[OH]], %[[OW]]] [1, 4, 3, 3] [1, 1, 1, 1]
// CHECK-NEXT:          accel.send %[[TI]], %[[S1]]
// CHECK-NEXT:        }
// CHECK-NEXT:      }
// CHECK-NEXT:      accel.send_literal %c8_i32
// CHECK-NEXT:      %[[TO:.*]] = memref.subview %[[O]][0, %[[OC]], 0, 0] [1, 1, 5, 5] [1, 1, 1, 1]
// CHECK-NEXT:      accel.recv %[[TO]] mode = accumulate
// CHECK-NEXT:    }
func.func @conv_f_stationary(%I: memref<1x4x7x7xi32>, %F: memref<8x4x3x3xi32>,
                             %O: memref<1x8x5x5xi32>) {
  linalg.conv_2d_nchw_fchw {
      dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>,
      accel.opcode_map = #opcodes,
      accel.init_opcodes = #accel.opcode_flow<(cfgF cfgC)>,
      accel.opcode_flow = #accel.opcode_flow<(sF ((sIcO)) rO)>,
      accel.accel_dim = affine_map<(n, f, oh, ow, c, kh, kw) -> (1, 1, 1, 1, 0, 0, 0)>}
      ins(%I, %F : memref<1x4x7x7xi32>, memref<8x4x3x3xi32>)
      outs(%O : memref<1x8x5x5xi32>)
  return
}

// -----

#opcodes = #accel.opcode_map<
  sF   = [send_literal(1), send(1)],
  sIcO = [send_literal(70), send(0)],
  rO   = [send_literal(8), recv(2, accumulate)]>

// The flow of the paper, (sF (sIcO) rO), handles one output row per sF/rO.
// With stride 2, the input window moves by 2.
// CHECK-DAG:   #[[STRIDE2:.*]] = affine_map<(d0) -> (d0 * 2)>
// CHECK-LABEL: func.func @conv_paper_flow_stride2(
// CHECK-SAME:      %[[I:.*]]: memref<1x4x11x11xi32>, %[[F:.*]]: memref<8x4x3x3xi32>, %[[O:.*]]: memref<1x8x5x5xi32>)
// CHECK:         scf.for %[[OC:.*]] = %c0 to %c8 step %c1 {
// CHECK-NEXT:      scf.for %[[OH:.*]] = %c0 to %c5 step %c1 {
// CHECK-NEXT:        accel.send_literal %c1_i32
// CHECK-NEXT:        memref.subview %[[F]][%[[OC]], 0, 0, 0] [1, 4, 3, 3] [1, 1, 1, 1]
// CHECK-NEXT:        accel.send
// CHECK-NEXT:        scf.for %[[OW:.*]] = %c0 to %c5 step %c1 {
// CHECK-NEXT:          accel.send_literal %c70_i32
// CHECK-NEXT:          %[[H:.*]] = affine.apply #[[STRIDE2]](%[[OH]])
// CHECK-NEXT:          %[[W:.*]] = affine.apply #[[STRIDE2]](%[[OW]])
// CHECK-NEXT:          %[[TI:.*]] = memref.subview %[[I]][0, 0, %[[H]], %[[W]]] [1, 4, 3, 3] [1, 1, 1, 1]
// CHECK-NEXT:          accel.send %[[TI]]
// CHECK-NEXT:        }
// CHECK-NEXT:        accel.send_literal %c8_i32
// CHECK-NEXT:        %[[TO:.*]] = memref.subview %[[O]][0, %[[OC]], %[[OH]], 0] [1, 1, 1, 5] [1, 1, 1, 1]
// CHECK-NEXT:        accel.recv %[[TO]] mode = accumulate
func.func @conv_paper_flow_stride2(%I: memref<1x4x11x11xi32>,
                                   %F: memref<8x4x3x3xi32>,
                                   %O: memref<1x8x5x5xi32>) {
  linalg.conv_2d_nchw_fchw {
      dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>,
      accel.opcode_map = #opcodes,
      accel.opcode_flow = #accel.opcode_flow<(sF (sIcO) rO)>,
      accel.accel_dim = affine_map<(n, f, oh, ow, c, kh, kw) -> (1, 1, 1, 1, 0, 0, 0)>}
      ins(%I, %F : memref<1x4x11x11xi32>, memref<8x4x3x3xi32>)
      outs(%O : memref<1x8x5x5xi32>)
  return
}
