// REQUIRES: systemc
// RUN: soda-opt %s --convert-linalg-to-accel %axi_lower_to_llvm \
// RUN: | mlir-runner -e main -entry-point-result=void \
// RUN:   -shared-libs=%axi_sysc_lib_dir/libsoda_axi_runtime_sysc_conv_v3%shlibext \
// RUN:   %axi_runner_libs \
// RUN: | FileCheck %s

// conv_v3, filter and output stationary, 3x3 filter, stride 1
//
// linalg.conv_2d_nchw_fchw generated for the conv_v3 accelerator of AXI4MLIR,
// with loops (n, f, oh, ow, c, kh, kw). The batch, output channel and output
// pixel loops are tiled by 1; the accelerator reduces over the input channels
// and the filter window. The opcodes of conv_v3:
//   cfgF: set the filter size (32) to the filter height (send_dim(1, 2)).
//   cfgC: set the input channels (16) to dim 1 of the input (send_dim(0, 1)).
//   sF:   load the filters of one output channel (1).
//   sIcO: read an input window, compute, keep the output (2 + 4 + 64).
//   rO:   send back the kept outputs (8).
//
// The empty group skips the oh loop: the filters of an output channel are
// loaded once and its whole 5x5 output plane is received at once. This is
// the dataflow of the hand-written AXI4MLIR conv_v3 driver.
//
// O starts non-zero and is checked against linalg.conv_2d_nchw_fchw on the CPU.

// CHECK: {{^0$}}

func.func @conv_accel(%I: memref<1x4x7x7xi32>, %F: memref<8x4x3x3xi32>, %O: memref<1x8x5x5xi32>) {
  linalg.conv_2d_nchw_fchw {
      dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>,
      accel.opcode_map = #accel.opcode_map<
          cfgF = [send_literal(32), send_dim(1, 2)],
          cfgC = [send_literal(16), send_dim(0, 1)],
          sF = [send_literal(1), send(1)],
          sIcO = [send_literal(70), send(0)],
          rO = [send_literal(8), recv(2, accumulate)]>,
      accel.init_opcodes = #accel.opcode_flow<(cfgF cfgC)>,
      accel.opcode_flow = #accel.opcode_flow<(sF ((sIcO)) rO)>,
      accel.accel_dim = affine_map<(n, f, oh, ow, c, kh, kw) -> (1, 1, 1, 1, 0, 0, 0)>}
      ins(%I, %F : memref<1x4x7x7xi32>, memref<8x4x3x3xi32>) outs(%O : memref<1x8x5x5xi32>)
  return
}

func.func private @verifyMemRefI32(memref<*xi32>, memref<*xi32>) -> i64 attributes {llvm.emit_c_interface}
func.func private @printI64(i64)
func.func private @printNewline()

// Fills `m` with (a * 7 + b * 5 + c * 3 + d + seed) % 11 - 5.
func.func @fill(%m: memref<?x?x?x?xi32>, %seed: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c5 = arith.constant 5 : index
  %c7 = arith.constant 7 : index
  %c11 = arith.constant 11 : index
  %c5_i32 = arith.constant 5 : i32
  %d0 = memref.dim %m, %c0 : memref<?x?x?x?xi32>
  %d1 = memref.dim %m, %c1 : memref<?x?x?x?xi32>
  %d2 = memref.dim %m, %c2 : memref<?x?x?x?xi32>
  %d3 = memref.dim %m, %c3 : memref<?x?x?x?xi32>
  scf.for %a = %c0 to %d0 step %c1 {
    scf.for %b = %c0 to %d1 step %c1 {
      scf.for %c = %c0 to %d2 step %c1 {
        scf.for %d = %c0 to %d3 step %c1 {
          %x0 = arith.muli %a, %c7 : index
          %x1 = arith.muli %b, %c5 : index
          %x2 = arith.muli %c, %c3 : index
          %s0 = arith.addi %x0, %x1 : index
          %s1 = arith.addi %s0, %x2 : index
          %s2 = arith.addi %s1, %d : index
          %s3 = arith.addi %s2, %seed : index
          %r = arith.remui %s3, %c11 : index
          %v = arith.index_cast %r : index to i32
          %w = arith.subi %v, %c5_i32 : i32
          memref.store %w, %m[%a, %b, %c, %d] : memref<?x?x?x?xi32>
        }
      }
    }
  }
  return
}

func.func @main() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %I = memref.alloc() : memref<1x4x7x7xi32>
  %F = memref.alloc() : memref<8x4x3x3xi32>
  %O = memref.alloc() : memref<1x8x5x5xi32>
  %Oref = memref.alloc() : memref<1x8x5x5xi32>
  %Id = memref.cast %I : memref<1x4x7x7xi32> to memref<?x?x?x?xi32>
  %Fd = memref.cast %F : memref<8x4x3x3xi32> to memref<?x?x?x?xi32>
  %Od = memref.cast %O : memref<1x8x5x5xi32> to memref<?x?x?x?xi32>
  %Orefd = memref.cast %Oref : memref<1x8x5x5xi32> to memref<?x?x?x?xi32>
  call @fill(%Id, %c0) : (memref<?x?x?x?xi32>, index) -> ()
  call @fill(%Fd, %c1) : (memref<?x?x?x?xi32>, index) -> ()
  call @fill(%Od, %c2) : (memref<?x?x?x?xi32>, index) -> ()
  call @fill(%Orefd, %c2) : (memref<?x?x?x?xi32>, index) -> ()

  call @conv_accel(%I, %F, %O)
      : (memref<1x4x7x7xi32>, memref<8x4x3x3xi32>, memref<1x8x5x5xi32>) -> ()
  linalg.conv_2d_nchw_fchw {dilations = dense<1> : tensor<2xi64>,
                            strides = dense<1> : tensor<2xi64>}
      ins(%I, %F : memref<1x4x7x7xi32>, memref<8x4x3x3xi32>) outs(%Oref : memref<1x8x5x5xi32>)

  %Ou = memref.cast %O : memref<1x8x5x5xi32> to memref<*xi32>
  %Orefu = memref.cast %Oref : memref<1x8x5x5xi32> to memref<*xi32>
  %errors = call @verifyMemRefI32(%Ou, %Orefu)
      : (memref<*xi32>, memref<*xi32>) -> i64
  call @printI64(%errors) : (i64) -> ()
  call @printNewline() : () -> ()

  memref.dealloc %I : memref<1x4x7x7xi32>
  memref.dealloc %F : memref<8x4x3x3xi32>
  memref.dealloc %O : memref<1x8x5x5xi32>
  memref.dealloc %Oref : memref<1x8x5x5xi32>
  return
}
