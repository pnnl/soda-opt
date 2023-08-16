// RUN: soda-opt %s -soda-linalg-tile="tile-sizes=2,4,8 anchor-op=linalg.matmul" -cse| FileCheck %s --check-prefix=TILE
// RUN: soda-opt %s -soda-linalg-tile="tile-sizes=2,3 anchor-op=linalg.conv_2d" -cse| FileCheck %s --check-prefix=TILE_CONV


// transform.sequence failures(propagate) {
//   ^bb0(%arg1: !pdl.operation):
//     %0 = transform.structured.match ops{["linalg.conv_3d_ndhwc_dhwcf"]} in %arg1
//     %1, %loops:3 = transform.structured.tile %0 [0, 5, 5, 5]
// }

func.func @matmul(%A: memref<1024x1024xf32>, %B: memref<1024x1024xf32>, %C: memref<1024x1024xf32>) {
  linalg.matmul 
   ins(%A, %B: memref<1024x1024xf32>, memref<1024x1024xf32>)
   outs(%C: memref<1024x1024xf32>)
  return
}

// TILE-LABEL:  func.func @matmul
// TILE:         scf.for %{{.*}} = %{{.*}} to %{{.*}} step %c2 {
// TILE-NEXT:      scf.for %{{.*}} = %{{.*}} to %{{.*}} step %c4 {
// TILE-NEXT:        scf.for %{{.*}} = %{{.*}} to %{{.*}} step %c8 {
// TILE-NEXT:          memref.subview
// TILE-NEXT:          memref.subview
// TILE-NEXT:          memref.subview
// TILE-NEXT:          linalg.matmul


func.func @linalg_generic(%in0t: tensor<4x4xf32>, %out0t: tensor<4xf32>) {
  %red = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                                          affine_map<(d0, d1) -> (d0)>],
   iterator_types = ["parallel", "reduction"]}
   ins(%in0t : tensor<4x4xf32>)
   outs(%out0t : tensor<4xf32>) {
    ^bb0(%in0: f32, %out0: f32):
      %cmp = arith.cmpf ogt, %in0, %out0 : f32
      %sel = arith.select %cmp, %in0, %out0 : f32
      linalg.yield %sel : f32
    } -> tensor<4xf32>
  return
}

// TILE-LABEL:  func.func @linalg_generic
// TILE-NOT:    scf.for

func.func @conv(%arg0 : memref<?x?xf32>, %arg1 : memref<?x?xf32>, %arg2 : memref<?x?xf32>) {
  linalg.conv_2d ins(%arg0, %arg1 : memref<?x?xf32>, memref<?x?xf32>) outs(%arg2 : memref<?x?xf32>)
  return
}

// TILE_CONV: func @conv
// TILE_CONV:   scf.for %{{.*}} = %{{.*}} to %{{.*}} step
// TILE_CONV:     scf.for %{{.*}} = %{{.*}} to %{{.*}} step
// TILE_CONV:       linalg.conv_2d