// RUN: soda-opt %s --convert-linalg-to-accel -split-input-file -verify-diagnostics

#map = affine_map<(m, n, k) -> (4, 4, 4)>
#ops = #accel.opcode_map<s = [send(0), send(1)], r = [recv(2, accumulate)]>

func.func @not_divisible(%A: memref<6x8xi32>, %B: memref<8x8xi32>, %C: memref<6x8xi32>) {
  // expected-error @below {{loop 0 of size 6 is not divisible by its accelerator tile size 4}}
  linalg.matmul {accel.accel_dim = #map, accel.opcode_map = #ops,
                 accel.opcode_flow = #accel.opcode_flow<(s r)>}
      ins(%A, %B : memref<6x8xi32>, memref<8x8xi32>) outs(%C : memref<6x8xi32>)
  return
}

// -----

func.func @dynamic(%A: memref<?x8xi32>, %B: memref<8x8xi32>, %C: memref<?x8xi32>) {
  // expected-error @below {{loop 0 is tiled for the accelerator but has a dynamic size}}
  linalg.matmul {accel.accel_dim = affine_map<(m, n, k) -> (4, 4, 4)>,
                 accel.opcode_map = #accel.opcode_map<s = [send(0), send(1)], r = [recv(2, accumulate)]>,
                 accel.opcode_flow = #accel.opcode_flow<(s r)>}
      ins(%A, %B : memref<?x8xi32>, memref<8x8xi32>) outs(%C : memref<?x8xi32>)
  return
}

// -----

func.func @tensors(%A: tensor<8x8xi32>, %B: tensor<8x8xi32>, %C: tensor<8x8xi32>) -> tensor<8x8xi32> {
  // expected-error @below {{must have memref operands to be converted to accel ops (bufferize it first)}}
  %0 = linalg.matmul {accel.opcode_map = #accel.opcode_map<s = [send(0), send(1)], r = [recv(2, accumulate)]>,
                      accel.opcode_flow = #accel.opcode_flow<(s r)>}
      ins(%A, %B : tensor<8x8xi32>, tensor<8x8xi32>) outs(%C : tensor<8x8xi32>) -> tensor<8x8xi32>
  return %0 : tensor<8x8xi32>
}

// -----

// Without accel.accel_dim nothing is tiled, so only one flow group fits.
func.func @too_deep(%A: memref<8x8xi32>, %B: memref<8x8xi32>, %C: memref<8x8xi32>) {
  // expected-error @below {{'accel.opcode_flow' has 2 nested groups, but only 0 loops are tiled for the accelerator (at most 1 groups)}}
  linalg.matmul {accel.opcode_map = #accel.opcode_map<s = [send(0), send(1)], r = [recv(2, accumulate)]>,
                 accel.opcode_flow = #accel.opcode_flow<((s) r)>}
      ins(%A, %B : memref<8x8xi32>, memref<8x8xi32>) outs(%C : memref<8x8xi32>)
  return
}

// -----

func.func @two_dma_configs(%A: memref<4x4xi32>, %B: memref<4x4xi32>, %C: memref<4x4xi32>) {
  linalg.matmul {accel.opcode_map = #accel.opcode_map<s = [send(0)]>,
                 accel.opcode_flow = #accel.opcode_flow<(s)>,
                 accel.dma_init_config = #accel.dma_config<address = 1, input_address = 2, input_size = 64, output_address = 3, output_size = 64>}
      ins(%A, %B : memref<4x4xi32>, memref<4x4xi32>) outs(%C : memref<4x4xi32>)
  // expected-error @below {{a function can only use one DMA configuration}}
  linalg.matmul {accel.opcode_map = #accel.opcode_map<s = [send(0)]>,
                 accel.opcode_flow = #accel.opcode_flow<(s)>,
                 accel.dma_init_config = #accel.dma_config<address = 5, input_address = 2, input_size = 64, output_address = 3, output_size = 64>}
      ins(%A, %B : memref<4x4xi32>, memref<4x4xi32>) outs(%C : memref<4x4xi32>)
  return
}

// -----

func.func @dma_buffer_too_small(%A: memref<16x16xi32>, %B: memref<16x16xi32>, %C: memref<16x16xi32>) {
  // expected-error @below {{opcode 's' sends 512 words, but the DMA input buffer holds 256}}
  linalg.matmul {accel.opcode_map = #accel.opcode_map<s = [send(0), send(1)]>,
                 accel.opcode_flow = #accel.opcode_flow<(s)>,
                 accel.dma_init_config = #accel.dma_config<address = 1, input_address = 2, input_size = 256, output_address = 3, output_size = 256>}
      ins(%A, %B : memref<16x16xi32>, memref<16x16xi32>) outs(%C : memref<16x16xi32>)
  return
}
