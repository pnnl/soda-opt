// RUN: soda-opt %s -soda-transform-erase-schedule | FileCheck %s --check-prefixes=CHECK-ERASE-SCHEDULE
// RUN: soda-opt %s -soda-transform-interpreter -soda-transform-erase-schedule | FileCheck %s --check-prefixes=CHECK-TILE

// CHECK-ERASE-SCHEDULE-NOT: transform.sequence
// // CHECK-TILE-NOT: transform.sequence

transform.sequence failures(propagate) {
^bb0(%arg1: !pdl.operation):
  %0 = transform.structured.match ops{["linalg.matmul"]} in %arg1
  %1, %loops:3 = transform.structured.tile %0 [4, 4, 4]
}

// CHECK-TILE-LABEL: func @tile_linalg_matmul_on_tensors(
// CHECK-TILE-SAME:    %[[TA:[0-9a-z]+]]: tensor<128x128xf32>
// CHECK-TILE-SAME:    %[[TB:[0-9a-z]+]]: tensor<128x128xf32>
// CHECK-TILE-SAME:    %[[TC:[0-9a-z]+]]: tensor<128x128xf32>
// CHECK-TILE-SAME:  -> tensor<128x128xf32> {
func.func @tile_linalg_matmul_on_tensors(
  %arg0: tensor<128x128xf32>, %arg1: tensor<128x128xf32>, %arg2: tensor<128x128xf32>)
    -> tensor<128x128xf32> {
//      CHECK-TILE: %[[TD0:.*]] = scf.for {{.*}} to {{.*}} step {{.*}} iter_args(%[[TC0:.*]] = %[[TC]]) -> (tensor<128x128xf32>) {
//      CHECK-TILE:   %[[TD1:.*]] = scf.for {{.*}} to {{.*}} step {{.*}} iter_args(%[[TC1:.*]] = %[[TC0]]) -> (tensor<128x128xf32>) {
//      CHECK-TILE:     %[[TD2:.*]] = scf.for {{.*}} to {{.*}} step {{.*}} iter_args(%[[TC2:.*]] = %[[TC1]]) -> (tensor<128x128xf32>) {
//      CHECK-TILE:       %[[sTA:.*]] = tensor.extract_slice %[[TA]][{{.*}}] : tensor<128x128xf32> to tensor<4x4xf32>
//      CHECK-TILE:       %[[sTB:.*]] = tensor.extract_slice %[[TB]][{{.*}}] : tensor<128x128xf32> to tensor<4x4xf32>
//      CHECK-TILE:       %[[sTC:.*]] = tensor.extract_slice %[[TC2]][{{.*}}] : tensor<128x128xf32> to tensor<4x4xf32>
//      CHECK-TILE:       %[[sTD:.*]] = linalg.matmul ins(%[[sTA]], %[[sTB]] : tensor<4x4xf32>, tensor<4x4xf32>)
// CHECK-TILE-SAME:                                   outs(%[[sTC]] : tensor<4x4xf32>)  -> tensor<4x4xf32>
//      CHECK-TILE:       %[[TD:.*]] = tensor.insert_slice %[[sTD]] into %[[TC2]][{{.*}}]  : tensor<4x4xf32> into tensor<128x128xf32>
//      CHECK-TILE:       scf.yield %[[TD]] : tensor<128x128xf32>
//      CHECK-TILE:     scf.yield %[[TD2]] : tensor<128x128xf32>
//      CHECK-TILE:   scf.yield %[[TD1]] : tensor<128x128xf32>
  %0 = linalg.matmul  ins(%arg0, %arg1: tensor<128x128xf32>, tensor<128x128xf32>)
                     outs(%arg2: tensor<128x128xf32>) -> tensor<128x128xf32>
//      CHECK-TILE: return %[[TD0]] : tensor<128x128xf32>
  return %0 : tensor<128x128xf32>
}

// CHECK-TILE-LABEL: func @tile_linalg_matmul_on_memrefs(
// CHECK-TILE-SAME:    %[[TA:[0-9a-z]+]]: memref<128x128xf32>
// CHECK-TILE-SAME:    %[[TB:[0-9a-z]+]]: memref<128x128xf32>
// CHECK-TILE-SAME:    %[[TC:[0-9a-z]+]]: memref<128x128xf32>
func.func @tile_linalg_matmul_on_memrefs(
  %arg0: memref<128x128xf32>, %arg1: memref<128x128xf32>, %arg2: memref<128x128xf32>)
    -> memref<128x128xf32> {
//      CHECK-TILE: scf.for {{.*}} = {{.*}} to {{.*}} step {{.*}}
//      CHECK-TILE:   scf.for {{.*}} = {{.*}} to {{.*}} step {{.*}}
//      CHECK-TILE:     scf.for {{.*}} = {{.*}} to {{.*}} step {{.*}}
//      CHECK-TILE:       %[[sTA:.*]] = memref.subview %[[TA]][{{.*}}] : memref<128x128xf32> to memref<4x4xf32, {{.*}}>
//      CHECK-TILE:       %[[sTB:.*]] = memref.subview %[[TB]][{{.*}}] : memref<128x128xf32> to memref<4x4xf32, {{.*}}>
//      CHECK-TILE:       %[[sTC:.*]] = memref.subview %[[TC]][{{.*}}] : memref<128x128xf32> to memref<4x4xf32, {{.*}}>
//      CHECK-TILE:       linalg.matmul ins(%[[sTA]], %[[sTB]] : memref<4x4xf32, {{.*}}>, memref<4x4xf32, {{.*}}>) outs(%[[sTC]] : memref<4x4xf32, {{.*}}>)
  linalg.matmul  ins(%arg0, %arg1: memref<128x128xf32>, memref<128x128xf32>)
                     outs(%arg2: memref<128x128xf32>)

//      CHECK-TILE: return %[[TC]] : memref<128x128xf32>
  return %arg2 : memref<128x128xf32>
}