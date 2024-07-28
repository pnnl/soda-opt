//===- AxUtils.h - Utils for debugging MLIR execution -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares AXI4MLIR functions to be called by the host to communicate
// with AXI enabled accelerators.
//
//===----------------------------------------------------------------------===//

#ifndef EXECUTIONENGINE_AXIUTILS_H_
#define EXECUTIONENGINE_AXIUTILS_H_

#ifdef _WIN32
#ifndef MLIR_AXIRUNNERUTILS_EXPORT
#ifdef mlir_runner_utils_EXPORTS
// We are building this library
#define MLIR_AXIRUNNERUTILS_EXPORT __declspec(dllexport)
#else
// We are using this library
#define MLIR_AXIRUNNERUTILS_EXPORT __declspec(dllimport)
#endif // mlir_runner_utils_EXPORTS
#endif // MLIR_AXIRUNNERUTILS_EXPORT
#else
#define MLIR_AXIRUNNERUTILS_EXPORT
#endif // _WIN32

#include <iostream>
#include <mlir/ExecutionEngine/RunnerUtils.h>

// =============================================================================
// AXI_APIV1
// =============================================================================

//-----------------DMA Functions-----------------
/**
 * - dma_address is base address of dma
 * - dma_input_addr is starting memory location for the dma input buffer,
 * - dma_input_buffer_size is length of the buffer
 * - dma_output_addr is starting memory location for the dma output buffer,
 * - dma_output_buffer_size is length of the buffer
 *
 *
 * Runs starting controls signals and sets MMS2, S2MM address registers to start
 * memory locations of the input and output buffers
 */

extern "C" MLIR_AXIRUNNERUTILS_EXPORT void
dma_init(unsigned int dma_address, unsigned int dma_input_address,
         unsigned int dma_input_buffer_size, unsigned int dma_output_address,
         unsigned int dma_output_buffer_size);

// Memory unmaps DMA control_register_address and Input and output buffers
extern "C" MLIR_AXIRUNNERUTILS_EXPORT void dma_free();

//================================================================================================================

//-----------------BUFFER Functions-----------------
// Get the MMap address of the input buffer of the dma  *Needed to copy data to
// Input_Buffer*
extern "C" MLIR_AXIRUNNERUTILS_EXPORT unsigned int *dma_get_inbuffer();

// Get the MMap address of the output buffer of the dma *Needed to copy data
// from Output_Buffer*
extern "C" MLIR_AXIRUNNERUTILS_EXPORT unsigned int *dma_get_outbuffer();

//================================================================================================================

//-----------------BUFFER Functions-----------------
// Copy data into the Input Buffer (length to write, offset to write to) returns
// 0 if successful
extern "C" MLIR_AXIRUNNERUTILS_EXPORT int
dma_copy_to_inbuffer(unsigned int *host_src_address, int data_length,
                     int offset);

// Copy data from the Output Buffer (length to read, offset to read from)
// returns 0 if successful
extern "C" MLIR_AXIRUNNERUTILS_EXPORT int
dma_copy_from_outbuffer(unsigned int *host_dst_address, int data_length,
                        int offset);

//-----------------BUFFER Functions-----------------
// Copy data into the Input Buffer (length to write, offset to write to) returns
// 0 if successful
template <typename T>
int mlir_dma_copy_to_inbuffer(const DynamicMemRefType<T> &src, int data_length,
                              int offset);

// Copy data from the Output Buffer (length to read, offset to read from)
// returns 0 if successful
template <typename T>
int mlir_dma_copy_from_outbuffer(const DynamicMemRefType<T> &dst,
                                 int data_length, int offset);

extern "C" MLIR_RUNNERUTILS_EXPORT int
copy_to_inbuffer_f32(int64_t rank, void *ptr, int offset);

extern "C" MLIR_RUNNERUTILS_EXPORT int
copy_from_outbuffer_f32(int64_t rank, void *ptr, int offset);

extern "C" MLIR_RUNNERUTILS_EXPORT int
copy_to_inbuffer_i32(int64_t rank, void *ptr, int offset);

extern "C" MLIR_RUNNERUTILS_EXPORT int
copy_from_outbuffer_i32(int64_t rank, void *ptr, int offset);

//================================================================================================================

//-----------------DMA MMS2 Functions-----------------
/**
 * Checks if input buffer size is >= length
 * Sets DMA MMS2 transfer length to length
 * Starts transfers to the accelerator using dma associated with dma_id
 * Return 0 if successful, returns negative if error occurs
 */
extern "C" MLIR_AXIRUNNERUTILS_EXPORT int dma_start_send(int length,
                                                         int offset);

// Same as dma_send but thread does not block, returns if 0
extern "C" MLIR_AXIRUNNERUTILS_EXPORT int dma_check_send();

// Blocks thread until dma MMS2 transfer is complete
extern "C" MLIR_AXIRUNNERUTILS_EXPORT void dma_wait_send();

//-----------------DMA S2MM Functions-----------------
/**
 * Checks if buffer size is >= length
 * Sets 2SMM store length
 * Starts storing data recieved through dma associated with dma_id
 * Return 0 if successful, returns negative if error occurs
 */
extern "C" MLIR_AXIRUNNERUTILS_EXPORT int dma_start_recv(int length,
                                                         int offset);

// Blocks thread until dma S2MM transfer is complete (TLAST signal is seen)
extern "C" MLIR_AXIRUNNERUTILS_EXPORT void dma_wait_recv();

// Same as dma_recv but thread does not block, returns if 0
extern "C" MLIR_AXIRUNNERUTILS_EXPORT int dma_check_recv();

// Unexposed to MLIR
extern "C" MLIR_AXIRUNNERUTILS_EXPORT unsigned int
dma_set(unsigned int *dma_virtual_address, int offset, unsigned int value);

// Unexposed to MLIR
extern "C" MLIR_AXIRUNNERUTILS_EXPORT unsigned int
dma_get(unsigned int *dma_virtual_address, int offset);

//-----------------Util Functions-----------------

// Converts memref into llvm_array pointers
// extern "C" MLIR_AXIRUNNERUTILS_EXPORT unsigned int *
// memref_to_ptr(UnrankedMemRefType<char> * in_memref) {
//   return in_memref->descriptor;
// }

// // Converts pointers into memrefs
// extern "C" MLIR_AXIRUNNERUTILS_EXPORT UnrankedMemRefType<char>
// ptr_to_memref(unsigned int *bare_ptr) {

//   UnrankedMemRefType<char> my_memref;
//   return my_memref;
// }

#endif // EXECUTIONENGINE_AXIUTILS_H_
