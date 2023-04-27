// RUN: soda-opt %s --soda-generate-linalg-summary=write-to-terminal=true | FileCheck %s

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>

func.func @matmul(%A: memref<16x4xf32>, 
                   %B: memref<4x8xf32>, 
                   %C: memref<16x8xf32>) {
  linalg.generic {indexing_maps = [#map, #map1, #map2],
    iterator_types = ["parallel", "parallel", "reduction"]} 
    ins(%A, %B : memref<16x4xf32>, memref<4x8xf32>) 
    outs(%C : memref<16x8xf32>) {
      ^bb0(%in: f32, %in_0: f32, %out: f32):
        %0 = arith.mulf %in, %in_0 : f32
        %1 = arith.addf %out, %0 : f32
        linalg.yield %1 : f32
  }
  return
}
// CHECK: =========================
// CHECK: REPORT BEGIN                    
// CHECK: LinalgOpInfo: matmul/linalg.generic
// CHECK:   numInputs: 2
// CHECK:   numOutputs: 1
// CHECK:   inputSizes: 64 32                                                                               
// CHECK:   outputSizes: 128 
// CHECK:   inputTypes: memref<16x4xf32> memref<4x8xf32> 
// CHECK:   outputTypes: memref<16x8xf32> 
// CHECK:   inputTypesBitwidth: 32 32 
// CHECK:   outputTypesBitwidth: 32 
// CHECK:   numArithmeticOpsInKernel: 2
// CHECK:   numMemoryOpsInKernel: 4
// CHECK:   numArithmeticOpsEstimative: 1024
// CHECK:   numMemoryOpsEstimative: 2048
// CHECK: REPORT END
// CHECK: =========================


func.func @linalg_generic(%in0t: tensor<4x4xf32>, %out0t: tensor<4xf32>)-> tensor<4xf32> {
  linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                                   affine_map<(d0, d1) -> (d0)>],
   iterator_types = ["parallel", "reduction"]}
   ins(%in0t : tensor<4x4xf32>)
   outs(%out0t : tensor<4xf32>) {
    ^bb0(%in0: f32, %out0: f32):
      %cmp = arith.cmpf ogt, %in0, %out0 : f32
      %sel = arith.select %cmp, %in0, %out0 : f32
      linalg.yield %sel : f32
    } -> tensor<4xf32>
  return %out0t : tensor<4xf32>
}
// CHECK: =========================
// CHECK: REPORT BEGIN
// CHECK: LinalgOpInfo: linalg_generic/linalg.generic
// CHECK:   numInputs: 1
// CHECK:   numOutputs: 1
// CHECK:   inputSizes: 16 
// CHECK:   outputSizes: 4 
// CHECK:   inputTypes: tensor<4x4xf32> 
// CHECK:   outputTypes: tensor<4xf32> 
// CHECK:   inputTypesBitwidth: 32 
// CHECK:   outputTypesBitwidth: 32 
// CHECK:   numArithmeticOpsInKernel: 2
// CHECK:   numMemoryOpsInKernel: 3
// CHECK:   numArithmeticOpsEstimative: 32
// CHECK:   numMemoryOpsEstimative: 48
// CHECK: REPORT END
// CHECK: =========================
