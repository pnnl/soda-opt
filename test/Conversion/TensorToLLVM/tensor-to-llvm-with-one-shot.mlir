// IMPORTANT: Changes in this file should be reflected in other scripts that use
//            similar patterns for lowering to LLVM.

// RUN: mlir-opt %s \
// RUN:  -pass-pipeline="builtin.module(func.func(tosa-to-arith, tosa-to-tensor, tosa-to-linalg-named, tosa-to-linalg))" | mlir-opt \
// RUN:  -tosa-to-arith="include-apply-rescale=true" \
// RUN:  -convert-tensor-to-linalg \
// RUN:  -eliminate-empty-tensors \
// RUN:  -empty-tensor-to-alloc-tensor \
// RUN:  -one-shot-bufferize="bufferize-function-boundaries allow-return-allocs-from-loops" \
// RUN:  -func-bufferize \
// RUN:  -finalizing-bufferize -buffer-deallocation \
// RUN:  -buffer-deallocation-simplification \
// RUN:  -bufferization-lower-deallocations \
// RUN:  --buffer-results-to-out-params \
// RUN:  --canonicalize -cse \
// RUN: \
// RUN:  -convert-linalg-to-affine-loops \
// RUN:  -expand-strided-metadata \
// RUN:  -lower-affine \
// RUN:  -convert-scf-to-cf \
// RUN:  -convert-complex-to-standard \
// RUN:  -convert-vector-to-llvm \
// RUN:  --convert-math-to-llvm \
// RUNL  --convert-math-to-libm \
// RUN:  -arith-expand   \
// RUN:  -memref-expand  \
// RUN:  -convert-to-llvm="filter-dialects=memref" \
// RUN:  -finalize-memref-to-llvm \
// RUN:  -convert-arith-to-llvm \
// RUN:  -finalize-memref-to-llvm \
// RUN:  -convert-complex-to-llvm \
// RUN:  -convert-func-to-llvm='use-bare-ptr-memref-call-conv=0' \
// RUN:  --test-lower-to-llvm \
// RUN:  -convert-cf-to-llvm \
// RUN:  -reconcile-unrealized-casts \
// RUN:  -symbol-dce \
// RUN: | FileCheck %s -check-prefix=CHECK-DEFAULT

// RUN: mlir-opt %s \
// RUN:  -pass-pipeline="builtin.module(func.func(tosa-to-arith, tosa-to-tensor, tosa-to-linalg-named, tosa-to-linalg))" | mlir-opt \
// RUN:  -tosa-to-arith="include-apply-rescale=true" \
// RUN:  -convert-tensor-to-linalg \
// RUN:  -eliminate-empty-tensors \
// RUN:  -empty-tensor-to-alloc-tensor \
// RUN:  -one-shot-bufferize="bufferize-function-boundaries allow-return-allocs-from-loops" \
// RUN:  -func-bufferize \
// RUN:  -finalizing-bufferize -buffer-deallocation \
// RUN:  -buffer-deallocation-simplification \
// RUN:  -bufferization-lower-deallocations \
// RUN:  --buffer-results-to-out-params \
// RUN:  --canonicalize -cse \
// RUN:  \
// RUN:  -convert-linalg-to-affine-loops \
// RUN:  -expand-strided-metadata \
// RUN:  -lower-affine \
// RUN:  -convert-scf-to-cf \
// RUN:  -convert-complex-to-standard \
// RUN:  -convert-vector-to-llvm \
// RUN:  --convert-math-to-llvm \
// RUNL  --convert-math-to-libm \
// RUN:  -arith-expand   \
// RUN:  -memref-expand  \
// RUN:  -convert-to-llvm="filter-dialects=memref" \
// RUN:  -finalize-memref-to-llvm \
// RUN:  -convert-arith-to-llvm \
// RUN:  -finalize-memref-to-llvm \
// RUN:  -convert-func-to-llvm='use-bare-ptr-memref-call-conv=1' \
// RUN:  --test-lower-to-llvm \
// RUN:  -convert-cf-to-llvm \
// RUN:  -reconcile-unrealized-casts \
// RUN:  -symbol-dce \
// RUN: | FileCheck %s -check-prefix=CHECK-DEFAULT

func.func @main1(%arg0: tensor<1x16xf32>, %arg1: tensor<1x16xf32>) -> tensor<1x16xf32> {
  %0 = "tosa.add"(%arg0, %arg1) : (tensor<1x16xf32>, tensor<1x16xf32>) -> tensor<1x16xf32>
  return %0 : tensor<1x16xf32>
}

//                                   PTR               ALIGNED           offset      size_d0     size_d1     stride_d0   stride_d1     repeats...
// CHECK-DEFAULT:   llvm.func @main1(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: i64, %arg3
// CHECK-BARE:      llvm.func @main1(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr) {

// -----

func.func @main2(%arg0: tensor<1x16xf32>) -> tensor<1x16xf32> {
  return %arg0 : tensor<1x16xf32>
}

//                                   PTR               ALIGNED           offset      size_d0     size_d1     stride_d0   stride_d1     repeats...
// CHECK-DEFAULT:   llvm.func @main2(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: i64, %arg6: i64
// CHECK-BARE:      llvm.func @main2(%arg0: !llvm.ptr, %arg1: !llvm.ptr) {

// -----

#map0 = affine_map<(d0, d1)[s0, s1, s2] -> (d0 * s1 + s0 + d1 * s2)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
func.func @main3(%arg0: memref<1x16xf32, #map0>, %arg1: memref<1x16xf32, #map0>, %arg2: memref<1x16xf32>) {
  %0 = memref.alloc() {alignment = 128 : i64} : memref<1x16xf32>
  linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg0, %arg1 : memref<1x16xf32, #map0>, memref<1x16xf32, #map0>) outs(%0 : memref<1x16xf32>) {
  ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):
    %1 = arith.addf %arg3, %arg4 : f32
    linalg.yield %1 : f32
  }
  memref.copy %0, %arg2 : memref<1x16xf32> to memref<1x16xf32>
  return
}

//                                   PTR               ALIGNED           offset      size_d0     size_d1     stride_d0   stride_d1     repeats...
// CHECK-DEFAULT:   llvm.func @main3(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: i64, %arg6: i64
// CHECK-BARE:      llvm.func @main3(%arg0: !llvm.ptr, %arg1: !llvm.ptr) {