//===- AxiRuntime.h - AXI DMA runtime library -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// C interface of the AXI DMA runtime library. The convert-accel-to-axi-runtime
// pass lowers accel ops to calls to these functions.
//
// The DMA engine has an input buffer (host -> accelerator) and an output
// buffer (accelerator -> host) of 32-bit words. Sizes, lengths and offsets
// are counted in words. Memrefs are passed with the unranked memref ABI
// (rank, pointer to the ranked descriptor) and copied in row-major order.
//
// The library is built against several backends with the same interface:
//   - soda_axi_runtime_mock: a loopback stream, for testing data movement.
//   - soda_axi_runtime_hw: a Xilinx AXI DMA mapped through /dev/mem.
//   - soda_axi_runtime_sysc_<accel>: a SystemC model of the DMA engine and an
//     accelerator.
//
// Set SODA_AXI_TRACE=1 in the environment to print every DMA transaction.
//
//===----------------------------------------------------------------------===//

#ifndef SODA_EXECUTIONENGINE_AXI_AXIRUNTIME_H
#define SODA_EXECUTIONENGINE_AXI_AXIRUNTIME_H

#include <cstdint>

#if defined(_WIN32)
#define SODA_AXI_EXPORT __declspec(dllexport)
#else
#define SODA_AXI_EXPORT __attribute__((visibility("default")))
#endif

extern "C" {

/// Configures the DMA engine at `dma_address`, with an input and an output
/// buffer given by their physical address and size in words.
SODA_AXI_EXPORT void dma_init(uint32_t dma_address, uint32_t input_address,
                              uint32_t input_size, uint32_t output_address,
                              uint32_t output_size);
/// Releases the DMA engine.
SODA_AXI_EXPORT void dma_free(void);

/// Copy a memref into the input buffer at `offset`. Return `offset + n`.
SODA_AXI_EXPORT int32_t copy_to_inbuffer_i32(int64_t rank, void *descriptor,
                                             int32_t offset);
SODA_AXI_EXPORT int32_t copy_to_inbuffer_f32(int64_t rank, void *descriptor,
                                             int32_t offset);
/// Write one word into the input buffer at `offset`. Return `offset + 1`.
SODA_AXI_EXPORT int32_t copy_literal_to_inbuffer_i32(int32_t value,
                                                     int32_t offset);

/// Copy the output buffer at `offset` into a memref. Return `offset + n`.
SODA_AXI_EXPORT int32_t copy_from_outbuffer_i32(int64_t rank, void *descriptor,
                                                int32_t offset);
SODA_AXI_EXPORT int32_t copy_from_outbuffer_f32(int64_t rank, void *descriptor,
                                                int32_t offset);
/// Add the output buffer at `offset` to a memref. Return `offset + n`.
SODA_AXI_EXPORT int32_t copy_from_outbuffer_accumulate_i32(int64_t rank,
                                                           void *descriptor,
                                                           int32_t offset);
SODA_AXI_EXPORT int32_t copy_from_outbuffer_accumulate_f32(int64_t rank,
                                                           void *descriptor,
                                                           int32_t offset);

/// Start sending `length` words of the input buffer from `offset`.
SODA_AXI_EXPORT int32_t dma_start_send(int32_t length, int32_t offset);
/// Wait until the current send completes.
SODA_AXI_EXPORT void dma_wait_send(void);
/// Start receiving up to `length` words into the output buffer at `offset`.
SODA_AXI_EXPORT int32_t dma_start_recv(int32_t length, int32_t offset);
/// Wait until the current receive completes.
SODA_AXI_EXPORT void dma_wait_recv(void);

/// Direct access to the DMA buffers.
SODA_AXI_EXPORT uint32_t *dma_get_inbuffer(void);
SODA_AXI_EXPORT uint32_t *dma_get_outbuffer(void);

} // extern "C"

#endif // SODA_EXECUTIONENGINE_AXI_AXIRUNTIME_H
