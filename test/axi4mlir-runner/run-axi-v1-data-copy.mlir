// RUN: soda-opt %s \
// RUN: -convert-linalg-to-loops -convert-scf-to-cf \
// RUN: --canonicalize --cse  \
// RUN: --convert-memref-to-llvm \
// RUN: --convert-math-to-llvm --convert-math-to-libm \
// RUN: -arith-expand   \
// RUN: -memref-expand  \
// RUN: --convert-arith-to-llvm \
// RUN: --convert-func-to-llvm --reconcile-unrealized-casts | \
// RUN: mlir-cpu-runner \
// RUN:  -e main -entry-point-result=void \
// RUN:  -shared-libs=%sodashlibdir/libmlir_mockaxi_runner_utils%shlibext \
// RUN:  -shared-libs=%mlir_lib_dir/libmlir_runner_utils%shlibext | \
// RUN: FileCheck %s

// MLIR Runner
func.func private @printMemrefF32(memref<*xf32>)

// AXI4MLIR func.functions
func.func private @dma_init(index, index, index, index, index) -> ()
func.func private @dma_free() -> ()

func.func private @mlir_dma_copy_to_inbuffer(memref<*xf32>, i64, i64) -> (i64)
func.func private @mlir_dma_copy_from_outbuffer(memref<*xf32>, i64, i64) -> (i64)
func.func private @copy_to_inbuffer_f32(memref<*xf32>, i64) -> (i64)
func.func private @copy_from_outbuffer_f32(memref<*xf32>, i64) -> (i64)

func.func private @dma_start_send(i64, i64) -> (i64)
func.func private @dma_wait_send() -> ()

func.func private @dma_start_recv(i64, i64) -> (i64)
func.func private @dma_wait_recv() -> ()

// Performing these C opertaions
// dma1.dma_init(0,0,1000,0,1000);
// dma1.dma_copy_to_inbuffer(reinterpret_cast<unsigned int*>(inputs),rows*depth,0);
// dma1.dma_copy_to_inbuffer(reinterpret_cast<unsigned int*>(weightsT),depth*cols,rows*depth);
// dma1.dma_start_send(dma1.current_input_offset,0);
// dma1.dma_start_recv(rows*cols +1 ,0);
// dma1.dma_wait_send();
// dma1.dma_wait_recv();
// dma1.dma_copy_from_outbuffer(reinterpret_cast<unsigned int*>(accelerated_outputs),cols*rows,0);

func.func @alloc_2d_filled_f32(%s1 : index, %s2 : index, %f : f32) -> memref<?x?xf32> {
  %buf = memref.alloc(%s1, %s2) : memref<?x?xf32>
  linalg.fill ins(%f : f32) outs(%buf : memref<?x?xf32>)
  
  return %buf : memref<?x?xf32>
}

func.func @alloc_2d_filled_inc_f32(%arg0: index, %arg1: index, %arg2: f32) -> memref<?x?xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %cst = arith.constant 1.000000e+02 : f32
  %0 = memref.alloc(%arg0, %arg1) : memref<?x?xf32>
  linalg.fill ins(%arg2 : f32) outs(%0 : memref<?x?xf32>)
  scf.for %arg3 = %c0 to %arg0 step %c1 {
    scf.for %arg4 = %c0 to %arg1 step %c1 {
      %1 = arith.index_cast %arg3 : index to i32
      %2 = arith.index_cast %arg4 : index to i32
      %3 = arith.sitofp %1 : i32 to f32
      %4 = arith.sitofp %2 : i32 to f32
      %5 = arith.mulf %3, %cst : f32
      %6 = arith.addf %4, %5 : f32
      memref.store %6, %0[%arg3, %arg4] : memref<?x?xf32>
    }
  }
  return %0 : memref<?x?xf32>
}

func.func @main() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  %c16 = arith.constant 16 : index
  %c32 = arith.constant 32 : index
  %c1000 = arith.constant 1000 : index

  // Prepare tile sizes
  %ts_a1 = arith.constant 4 : i64
  %ts_a2 = arith.constant 4 : i64
  %ts_o1 = arith.constant 4 : i64
  %ts_o2 = arith.constant 4 : i64


  %c1_0 = arith.constant 1 : i64
  %cst_1 = arith.constant 1.000000e+00 : f32
  %cst_0 = arith.constant 0.000000e+00 : f32


  %A = call @alloc_2d_filled_inc_f32(%c4, %c4, %cst_1) : (index, index, f32) -> (memref<?x?xf32>)
  %B = call @alloc_2d_filled_f32(%c4, %c4, %cst_1) : (index, index, f32) -> (memref<?x?xf32>)
  %C = call @alloc_2d_filled_f32(%c4, %c4, %cst_0) : (index, index, f32) -> (memref<?x?xf32>)
  
  %A_typed = memref.cast %A: memref<?x?xf32> to memref<4x4xf32>
  %B_typed = memref.cast %B: memref<?x?xf32> to memref<4x4xf32>
  %C_typed = memref.cast %C: memref<?x?xf32> to memref<4x4xf32>

  %in1 = memref.cast %A_typed: memref<4x4xf32> to memref<*xf32>
  %in2 = memref.cast %B_typed: memref<4x4xf32> to memref<*xf32>
  %out1 = memref.cast %C_typed: memref<4x4xf32> to memref<*xf32>


  call @printMemrefF32(%in1) : (memref<*xf32>) -> ()
  call @printMemrefF32(%in2) : (memref<*xf32>) -> ()

  // Initializes the DMA
  call @dma_init(%c0, %c0, %c1000, %c0, %c1000) : (index,index,index,index,index ) -> ()
  
  // Sizes of in and out buffers
  %in1_lenght = arith.muli %ts_a1, %ts_a2 : i64
  %in2_lenght = arith.muli %ts_a1, %ts_a2 : i64
  %total_input_lenght = arith.addi %in1_lenght, %in2_lenght : i64
  %out_lenght = arith.muli %ts_o1, %ts_o2 : i64
  
  %in1_offset = arith.constant 0 : i64  // offset on the input buffer
  %in2_offset = arith.muli %c1_0, %in1_lenght : i64  // offset on the input buffer
  %out_offset = arith.constant 0 : i64 // offset on the output buffer

  // Copy data to be transfered and set the transfer size
  call @copy_to_inbuffer_f32(%in1, %in1_offset) : (memref<*xf32>, i64) -> (i64)
  call @copy_to_inbuffer_f32(%in2, %in2_offset) : (memref<*xf32>, i64) -> (i64)
  call @dma_start_send (%total_input_lenght, %in1_offset) : (i64, i64) -> (i64)
  call @dma_start_recv (%out_lenght, %out_offset) : (i64, i64) -> (i64)

  // Wait for operations to complete
  call @dma_wait_send () : () -> ()
  call @dma_wait_recv () : () -> ()
  

  // Copy C tile from DMA output buffer
  call @copy_from_outbuffer_f32 (%out1, %in2_offset) : (memref<*xf32>, i64) -> (i64)

  // Cleanup
  call @dma_free() : () -> ()

  // Print output
  call @printMemrefF32(%out1) : (memref<*xf32>) -> ()
  return
}

//CHECK: dma_init
//CHECK: dma_start_send
//CHECK: dma_start_recv
//CHECK: dma_wait_send
//CHECK: dma_wait_recv
//CHECK: dma_free