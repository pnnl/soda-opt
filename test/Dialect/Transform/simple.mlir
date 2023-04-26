// RUN: soda-opt %s --soda-transform-interpreter -allow-unregistered-dialect --split-input-file --verify-diagnostics

// CHECK-LABEL: @vectorize_all
func.func @vectorize_all(
  %arg0: tensor<128x128xf32>, %arg1: tensor<128x128xf32>, %arg2: tensor<128x128xf32>,
  %arg3: tensor<128x128xf32>)
    -> tensor<128x128xf32> {
  // CHECK: vector.contract
  %0 = linalg.matmul {test.attrA}
                     ins(%arg0, %arg1: tensor<128x128xf32>, tensor<128x128xf32>)
                     outs(%arg2: tensor<128x128xf32>)
    -> tensor<128x128xf32>
  // CHECK: vector.contract
  %1 = linalg.matmul ins(%arg0, %0: tensor<128x128xf32>, tensor<128x128xf32>)
                     outs(%arg3: tensor<128x128xf32>)
    -> tensor<128x128xf32>
  return %1 : tensor<128x128xf32>
}

transform.sequence failures(propagate) {
^bb0(%arg0: !pdl.operation):
  transform.structured.vectorize %arg0
}
