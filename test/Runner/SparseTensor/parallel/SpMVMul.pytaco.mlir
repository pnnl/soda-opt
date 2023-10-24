// DEFINE: %{option} = "enable-runtime-library=false enable-openmp parallelization-strategy=any-storage-any-loop"
// DEFINE: %{compile} = soda-opt %s --soda-sparse-compiler=%{option}
// DEFINE: %{run} = OMP_NUM_THREADS=4 TENSOR0="%soda_test_dir/Runner/SparseTensor/data/T16x16_0.tns" TENSOR1="%soda_test_dir/Runner/SparseTensor/data/T16_1.tns" \
// DEFINE: mlir-cpu-runner \
// DEFINE:  -e main -entry-point-result=void  \
// DEFINE:  -shared-libs=%mlir_lib_dir/libmlir_c_runner_utils%shlibext,%openmp_lib,%sodashlibdir/libsoda_runner_ext%shlibext 2>&1 | \
// DEFINE: FileCheck %s
// RUN: %{compile} | %{run}

// This MLIR was generated from soda-pytaco.py via SpMVMul.py

#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d1)>
#map2 = affine_map<(d0, d1) -> (d0)>
module {
  func.func private @getTensorFilename(index) -> !llvm.ptr<i8>
  func.func private @rtclock() -> f64
  func.func private @rtclock_interval(f64, f64)
  llvm.mlir.global internal constant @none("\00A") {addr_space = 0 : i32}
  func.func @getStdOut() -> !llvm.ptr<i8> {
    %0 = llvm.mlir.addressof @none : !llvm.ptr<array<2 x i8>>
    %1 = llvm.mlir.constant(0 : index) : i64
    %2 = llvm.getelementptr %0[%1, %1] : (!llvm.ptr<array<2 x i8>>, i64, i64) -> !llvm.ptr<i8>
    return %2 : !llvm.ptr<i8>
  }
  func.func @SpMVMul_z.main(%arg0: tensor<16x16xf32, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ] }>>, %arg1: tensor<16xf32, #sparse_tensor.encoding<{ dimLevelType = [ "dense" ] }>>) -> tensor<16xf32, #sparse_tensor.encoding<{ dimLevelType = [ "dense" ] }>> attributes {llvm.emit_c_interface} {
    %0 = bufferization.alloc_tensor() : tensor<16xf32, #sparse_tensor.encoding<{ dimLevelType = [ "dense" ] }>>
    %1 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "reduction"]} ins(%arg0, %arg1 : tensor<16x16xf32, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ] }>>, tensor<16xf32, #sparse_tensor.encoding<{ dimLevelType = [ "dense" ] }>>) outs(%0 : tensor<16xf32, #sparse_tensor.encoding<{ dimLevelType = [ "dense" ] }>>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %2 = arith.mulf %in, %in_0 : f32
      %3 = arith.addf %out, %2 : f32
      linalg.yield %3 : f32
    } -> tensor<16xf32, #sparse_tensor.encoding<{ dimLevelType = [ "dense" ] }>>
    return %1 : tensor<16xf32, #sparse_tensor.encoding<{ dimLevelType = [ "dense" ] }>>
  }
  func.func @main() {
    %c0 = arith.constant 0 : index
    %0 = call @getTensorFilename(%c0) : (index) -> !llvm.ptr<i8>
    %1 = sparse_tensor.new %0 : !llvm.ptr<i8> to tensor<16x16xf32, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ] }>>
    %c1 = arith.constant 1 : index
    %2 = call @getTensorFilename(%c1) : (index) -> !llvm.ptr<i8>
    %3 = sparse_tensor.new %2 : !llvm.ptr<i8> to tensor<16xf32, #sparse_tensor.encoding<{ dimLevelType = [ "dense" ] }>>
    call @wrapper(%1, %3) : (tensor<16x16xf32, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ] }>>, tensor<16xf32, #sparse_tensor.encoding<{ dimLevelType = [ "dense" ] }>>) -> ()
    return
  }
  func.func private @wrapper(%arg0: tensor<16x16xf32, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ] }>>, %arg1: tensor<16xf32, #sparse_tensor.encoding<{ dimLevelType = [ "dense" ] }>>) {
    %0 = call @rtclock() : () -> f64
    %1 = call @SpMVMul_z.main(%arg0, %arg1) : (tensor<16x16xf32, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ] }>>, tensor<16xf32, #sparse_tensor.encoding<{ dimLevelType = [ "dense" ] }>>) -> tensor<16xf32, #sparse_tensor.encoding<{ dimLevelType = [ "dense" ] }>>
    %2 = call @rtclock() : () -> f64
    call @rtclock_interval(%0, %2) : (f64, f64) -> ()
    %3 = call @getStdOut() : () -> !llvm.ptr<i8>
    sparse_tensor.out %1, %3 : tensor<16xf32, #sparse_tensor.encoding<{ dimLevelType = [ "dense" ] }>>, !llvm.ptr<i8>
    return
  }

  // CHECK: rtclock_interval
  // CHECK: 1 16
  // CHECK: 16
  // CHECK: 1 0.479979
  // CHECK: 2 0.184463
  // CHECK: 3 0
  // CHECK: 4 0.412375
  // CHECK: 5 0
  // CHECK: 6 0
  // CHECK: 7 0
  // CHECK: 8 0.381202
  // CHECK: 9 0
  // CHECK: 10 0
  // CHECK: 11 0
  // CHECK: 12 0
  // CHECK: 13 0
  // CHECK: 14 0
  // CHECK: 15 0
  // CHECK: 16 0

}

