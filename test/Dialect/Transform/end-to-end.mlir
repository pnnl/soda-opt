// RUN: soda-opt %s -transform-interpreter | FileCheck %s

module attributes {torch.debug_module_name = "ThreeMM"} {
  func.func @forward(%arg0: memref<1x8x16xf32>, %arg1: memref<1x16x32xf32>, %arg2: memref<1x32x64xf32>, %arg3: memref<1x64x16xf32>, %arg4: memref<1x8x16xf32>) {
    %cst = arith.constant 0.000000e+00 : f32
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x8x32xf32>
    linalg.fill ins(%cst : f32) outs(%alloc : memref<1x8x32xf32>)
    linalg.batch_matmul ins(%arg0, %arg1 : memref<1x8x16xf32>, memref<1x16x32xf32>) outs(%alloc : memref<1x8x32xf32>)
    %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<1x8x64xf32>
    linalg.fill ins(%cst : f32) outs(%alloc_0 : memref<1x8x64xf32>)
    linalg.batch_matmul ins(%alloc, %arg2 : memref<1x8x32xf32>, memref<1x32x64xf32>) outs(%alloc_0 : memref<1x8x64xf32>)
    %alloc_1 = memref.alloc() {alignment = 64 : i64} : memref<1x8x16xf32>
    linalg.fill ins(%cst : f32) outs(%alloc_1 : memref<1x8x16xf32>)
    linalg.batch_matmul ins(%alloc_0, %arg3 : memref<1x8x64xf32>, memref<1x64x16xf32>) outs(%alloc_1 : memref<1x8x16xf32>)
    memref.copy %alloc_1, %arg4 : memref<1x8x16xf32> to memref<1x8x16xf32>
    return
  }
}

module @transforms attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(
      %root: !transform.any_op {transform.readonly}) {

    %all = transform.structured.match ops{["func.func"]} in %root : (!transform.any_op) -> !transform.any_op

    %matmul = transform.collect_matching @match_matmul in %root : (!transform.any_op) -> !transform.any_op

    %matmul_tile = transform.collect_matching @match_tile_matmul in %root : (!transform.any_op) -> !transform.any_op

    transform.include @print_matmul_tile failures(propagate)  (%matmul_tile) : (!transform.any_op) -> ()

    transform.include @tiling failures(propagate) (%matmul_tile) : (!transform.any_op) -> ()

    transform.include @lowering failures(propagate)  (%all) : (!transform.any_op) -> ()
    transform.yield
  }

  transform.named_sequence @tiling (%entry: !transform.any_op {transform.consumed}){
    %L, %loops:3 = transform.structured.tile_using_for %entry tile_sizes [1, 4, 8] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
    %1 = transform.get_parent_op %L {op_name = "scf.for"} : (!transform.any_op) -> !transform.op<"scf.for">
    transform.loop.unroll %1 { factor = 2} : !transform.op<"scf.for"> 
    transform.yield  
  }

  transform.named_sequence @lowering (%entry: !transform.any_op {transform.consumed}){
    %1 = transform.apply_registered_pass "convert-linalg-to-affine-loops" to %entry : (!transform.any_op) -> !transform.any_op
    %2 = transform.apply_registered_pass "lower-affine" to %1 : (!transform.any_op) -> !transform.any_op    
    %func = transform.get_parent_op %2 {isolated_from_above} : (!transform.any_op) -> !transform.any_op
    // %3 = transform.apply_registered_pass "lower-all-to-llvm" to %func : (!transform.any_op) -> !transform.any_op
    %3 = transform.apply_registered_pass "lower-all-to-llvm" to %func {options = "use-bare-ptr-memref-call-conv"}: (!transform.any_op) -> !transform.any_op
    transform.yield  
  }

  transform.named_sequence @match_tile_matmul(
    %candidate: !transform.any_op {transform.readonly}) -> !transform.any_op {
  // Match a structured linear algebra operation.
  transform.match.structured %candidate : !transform.any_op {
  ^bb0(%c: !transform.any_op):
    transform.match.operation_name %candidate ["linalg.batch_matmul"] : !transform.any_op
    %dim1 = transform.match.structured.dim %c[0] : (!transform.any_op) -> !transform.param<i64>
    %dim2 = transform.match.structured.dim %c[1] : (!transform.any_op) -> !transform.param<i64>
    %dim3 = transform.match.structured.dim %c[2] : (!transform.any_op) -> !transform.param<i64>
    %c16 = transform.param.constant 16 : i64 -> !transform.param<i64>
    %c32 = transform.param.constant 32 : i64 -> !transform.param<i64>
    transform.match.param.cmpi gt %dim3, %c16 : !transform.param<i64>
    transform.match.param.cmpi le %dim3, %c32 : !transform.param<i64>
  }
  transform.yield %candidate : !transform.any_op
}

  transform.named_sequence @match_matmul(
      %entry: !transform.any_op {transform.readonly}) -> !transform.any_op {
    transform.match.operation_name %entry ["linalg.batch_matmul"] : !transform.any_op
    transform.yield %entry : !transform.any_op
  }

  transform.named_sequence @print_matmul(
      %matmul: !transform.any_op {transform.readonly}) {
    transform.debug.emit_remark_at %matmul, "matmul" : !transform.any_op
    transform.yield
  }

  transform.named_sequence @print_matmul_tile(
      %matmul_tile: !transform.any_op {transform.readonly}) {
    transform.debug.emit_remark_at %matmul_tile, "matmul_tile" : !transform.any_op
    transform.yield
  }
}

// CHECK: llvm.func @forward(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr, %arg3: !llvm.ptr, %arg4: !llvm.ptr)
// Example contains 3 matmuls, we ask to match one of them and unroll a dimension
// by a factor of 2, this results in 4 kernel executions, resulting in 4 fmul ops.
// CHECK: llvm.fmul 
// CHECK: llvm.fmul 
// CHECK: llvm.fmul 
// CHECK: llvm.fmul 
// CHECK-NOT: llvm.fmul
