//===- AxiRuntime.cpp - AXI DMA runtime library ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implements the C interface of AxiRuntime.h on top of a DmaBackend: memref
// copies into and out of the DMA buffers, bounds checks and tracing.
//
//===----------------------------------------------------------------------===//

#include "soda/ExecutionEngine/axi/AxiRuntime.h"

#include "DmaBackend.h"
#include "mlir/ExecutionEngine/CRunnerUtils.h"

#include <cinttypes>
#include <cstring>
#include <vector>

using namespace soda::axi;

namespace {

bool isTraceEnabled() {
  static const bool enabled = [] {
    const char *value = std::getenv("SODA_AXI_TRACE");
    return value && *value && std::strcmp(value, "0") != 0;
  }();
  return enabled;
}

void traceWords(const char *what, const uint32_t *words, uint32_t length,
                uint32_t offset) {
  std::printf("[axi] %s %" PRIu32 " words @%" PRIu32 ":", what, length,
              offset);
  for (uint32_t i = 0; i < length; ++i)
    std::printf(" %" PRId32, static_cast<int32_t>(words[offset + i]));
  std::printf("\n");
  std::fflush(stdout);
}

DmaBackend &getInitializedBackend() {
  DmaBackend &backend = getDmaBackend();
  if (!backend.isInitialized())
    fatalError("the DMA is used before dma_init");
  return backend;
}

void checkRange(const char *what, int64_t offset, int64_t length,
                uint32_t bufferSize) {
  if (offset < 0 || length < 0 || offset + length > bufferSize) {
    char message[256];
    std::snprintf(message, sizeof(message),
                  "%s of %" PRId64 " words at offset %" PRId64
                  " is outside the %" PRIu32 "-word DMA buffer",
                  what, length, offset, bufferSize);
    fatalError(message);
  }
}

int64_t getNumElements(const DynamicMemRefType<int32_t> &memref) {
  int64_t count = 1;
  for (int64_t dim = 0; dim < memref.rank; ++dim)
    count *= memref.sizes[dim];
  return count;
}

/// Calls `fn(row, length, stride, index)` for every innermost row of `memref`
/// in row-major order, where `index` is the row-major index of the row's first
/// element.
template <typename T, typename Fn>
void forEachRow(const DynamicMemRefType<T> &memref, Fn fn) {
  T *base = memref.data + memref.offset;
  if (memref.rank == 0) {
    fn(base, 1, 1, 0);
    return;
  }
  for (int64_t dim = 0; dim < memref.rank; ++dim)
    if (memref.sizes[dim] == 0)
      return;
  int64_t outerRank = memref.rank - 1;
  int64_t rowLength = memref.sizes[outerRank];
  int64_t rowStride = memref.strides[outerRank];
  std::vector<int64_t> indices(outerRank, 0);
  int64_t index = 0;
  while (true) {
    T *row = base;
    for (int64_t dim = 0; dim < outerRank; ++dim)
      row += indices[dim] * memref.strides[dim];
    fn(row, rowLength, rowStride, index);
    index += rowLength;
    int64_t dim = outerRank - 1;
    for (; dim >= 0; --dim) {
      if (++indices[dim] < memref.sizes[dim])
        break;
      indices[dim] = 0;
    }
    if (dim < 0)
      return;
  }
}

template <typename T>
DynamicMemRefType<T> getMemRef(int64_t rank, void *descriptor) {
  static_assert(sizeof(T) == sizeof(uint32_t), "DMA words are 32 bits");
  return DynamicMemRefType<T>(UnrankedMemRefType<T>{rank, descriptor});
}

template <typename T>
int32_t copyToInputBuffer(int64_t rank, void *descriptor, int32_t offset) {
  DmaBackend &backend = getInitializedBackend();
  DynamicMemRefType<T> memref = getMemRef<T>(rank, descriptor);
  int64_t count = getNumElements(getMemRef<int32_t>(rank, descriptor));
  checkRange("copy into the input buffer", offset, count,
             backend.getInputSize());
  uint32_t *dst = backend.getInputBuffer() + offset;
  forEachRow(memref, [&](T *row, int64_t length, int64_t stride,
                         int64_t index) {
    if (stride == 1) {
      std::memcpy(dst + index, row, length * sizeof(T));
      return;
    }
    for (int64_t i = 0; i < length; ++i)
      std::memcpy(dst + index + i, row + i * stride, sizeof(T));
  });
  return offset + static_cast<int32_t>(count);
}

template <typename T>
int32_t copyFromOutputBuffer(int64_t rank, void *descriptor, int32_t offset) {
  DmaBackend &backend = getInitializedBackend();
  DynamicMemRefType<T> memref = getMemRef<T>(rank, descriptor);
  int64_t count = getNumElements(getMemRef<int32_t>(rank, descriptor));
  checkRange("copy from the output buffer", offset, count,
             backend.getOutputSize());
  const uint32_t *src = backend.getOutputBuffer() + offset;
  forEachRow(memref, [&](T *row, int64_t length, int64_t stride,
                         int64_t index) {
    if (stride == 1) {
      std::memcpy(row, src + index, length * sizeof(T));
      return;
    }
    for (int64_t i = 0; i < length; ++i)
      std::memcpy(row + i * stride, src + index + i, sizeof(T));
  });
  return offset + static_cast<int32_t>(count);
}

template <typename T>
T addWord(T value, uint32_t word) {
  T received;
  std::memcpy(&received, &word, sizeof(T));
  return value + received;
}

/// Integer accumulation wraps around like arith.addi.
template <>
int32_t addWord<int32_t>(int32_t value, uint32_t word) {
  return static_cast<int32_t>(static_cast<uint32_t>(value) + word);
}

template <typename T>
int32_t accumulateFromOutputBuffer(int64_t rank, void *descriptor,
                                   int32_t offset) {
  DmaBackend &backend = getInitializedBackend();
  DynamicMemRefType<T> memref = getMemRef<T>(rank, descriptor);
  int64_t count = getNumElements(getMemRef<int32_t>(rank, descriptor));
  checkRange("copy from the output buffer", offset, count,
             backend.getOutputSize());
  const uint32_t *src = backend.getOutputBuffer() + offset;
  forEachRow(memref, [&](T *row, int64_t length, int64_t stride,
                         int64_t index) {
    for (int64_t i = 0; i < length; ++i)
      row[i * stride] = addWord(row[i * stride], src[index + i]);
  });
  return offset + static_cast<int32_t>(count);
}

} // namespace

extern "C" {

void dma_init(uint32_t dma_address, uint32_t input_address,
              uint32_t input_size, uint32_t output_address,
              uint32_t output_size) {
  DmaBackend &backend = getDmaBackend();
  if (isTraceEnabled()) {
    std::printf("[axi] dma_init %s: address=0x%" PRIx32 " input=0x%" PRIx32
                "[%" PRIu32 "] output=0x%" PRIx32 "[%" PRIu32 "]\n",
                backend.getName(), dma_address, input_address, input_size,
                output_address, output_size);
    std::fflush(stdout);
  }
  if (input_size == 0 || output_size == 0)
    fatalError("dma_init needs non-empty input and output buffers");
  if (backend.isInitialized())
    backend.free();
  backend.init(
      {dma_address, input_address, input_size, output_address, output_size});
}

void dma_free(void) {
  DmaBackend &backend = getDmaBackend();
  if (!backend.isInitialized())
    return;
  backend.free();
  if (isTraceEnabled()) {
    std::printf("[axi] dma_free\n");
    std::fflush(stdout);
  }
}

int32_t copy_to_inbuffer_i32(int64_t rank, void *descriptor, int32_t offset) {
  return copyToInputBuffer<int32_t>(rank, descriptor, offset);
}

int32_t copy_to_inbuffer_f32(int64_t rank, void *descriptor, int32_t offset) {
  return copyToInputBuffer<float>(rank, descriptor, offset);
}

int32_t copy_literal_to_inbuffer_i32(int32_t value, int32_t offset) {
  DmaBackend &backend = getInitializedBackend();
  checkRange("copy into the input buffer", offset, 1, backend.getInputSize());
  backend.getInputBuffer()[offset] = static_cast<uint32_t>(value);
  return offset + 1;
}

int32_t copy_from_outbuffer_i32(int64_t rank, void *descriptor,
                                int32_t offset) {
  return copyFromOutputBuffer<int32_t>(rank, descriptor, offset);
}

int32_t copy_from_outbuffer_f32(int64_t rank, void *descriptor,
                                int32_t offset) {
  return copyFromOutputBuffer<float>(rank, descriptor, offset);
}

int32_t copy_from_outbuffer_accumulate_i32(int64_t rank, void *descriptor,
                                           int32_t offset) {
  return accumulateFromOutputBuffer<int32_t>(rank, descriptor, offset);
}

int32_t copy_from_outbuffer_accumulate_f32(int64_t rank, void *descriptor,
                                           int32_t offset) {
  return accumulateFromOutputBuffer<float>(rank, descriptor, offset);
}

int32_t dma_start_send(int32_t length, int32_t offset) {
  DmaBackend &backend = getInitializedBackend();
  checkRange("send", offset, length, backend.getInputSize());
  if (isTraceEnabled())
    traceWords("send", backend.getInputBuffer(), length, offset);
  backend.startSend(length, offset);
  return 0;
}

void dma_wait_send(void) { getInitializedBackend().waitSend(); }

namespace {
// The last receive, for tracing.
uint32_t pendingRecvOffset = 0;
} // namespace

int32_t dma_start_recv(int32_t length, int32_t offset) {
  DmaBackend &backend = getInitializedBackend();
  checkRange("receive", offset, length, backend.getOutputSize());
  pendingRecvOffset = offset;
  backend.startRecv(length, offset);
  return 0;
}

void dma_wait_recv(void) {
  DmaBackend &backend = getInitializedBackend();
  uint32_t received = backend.waitRecv();
  if (isTraceEnabled())
    traceWords("recv", backend.getOutputBuffer(), received,
               pendingRecvOffset);
}

uint32_t *dma_get_inbuffer(void) { return getDmaBackend().getInputBuffer(); }

uint32_t *dma_get_outbuffer(void) { return getDmaBackend().getOutputBuffer(); }

} // extern "C"
