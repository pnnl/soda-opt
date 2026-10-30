//===- DmaBackend.h - Interface of the AXI DMA backends ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The C interface in AxiRuntime.cpp implements memref copies and bounds checks
// once, and forwards the DMA transfers to a backend. Every runtime library
// links AxiRuntime.cpp with exactly one backend, which defines
// getDmaBackend().
//
//===----------------------------------------------------------------------===//

#ifndef SODA_LIB_EXECUTIONENGINE_AXI_DMABACKEND_H
#define SODA_LIB_EXECUTIONENGINE_AXI_DMABACKEND_H

#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace soda {
namespace axi {

struct DmaConfig {
  uint32_t address;
  uint32_t inputAddress;
  uint32_t inputSize; // words
  uint32_t outputAddress;
  uint32_t outputSize; // words
};

class DmaBackend {
public:
  virtual ~DmaBackend() = default;

  /// Short name used in diagnostics and traces.
  virtual const char *getName() const = 0;

  /// Sets up the engine and the input and output buffers.
  virtual void init(const DmaConfig &config) = 0;
  /// Releases the buffers.
  virtual void free() = 0;

  /// Transfers. Lengths and offsets are in words and already checked against
  /// the buffer sizes.
  virtual void startSend(uint32_t length, uint32_t offset) = 0;
  virtual void waitSend() = 0;
  virtual void startRecv(uint32_t length, uint32_t offset) = 0;
  /// Waits for the receive and returns the number of words received.
  virtual uint32_t waitRecv() = 0;

  uint32_t *getInputBuffer() const { return inputBuffer; }
  uint32_t *getOutputBuffer() const { return outputBuffer; }
  uint32_t getInputSize() const { return inputSize; }
  uint32_t getOutputSize() const { return outputSize; }
  bool isInitialized() const { return inputBuffer && outputBuffer; }

protected:
  uint32_t *inputBuffer = nullptr;
  uint32_t *outputBuffer = nullptr;
  uint32_t inputSize = 0;
  uint32_t outputSize = 0;
};

/// The backend of this library.
DmaBackend &getDmaBackend();

/// Prints an error and exits. It does not abort, so that the error is not
/// buried under the stack trace of the JIT's crash handler, and it skips
/// static destructors, which the SystemC models do not support mid-run.
[[noreturn]] inline void fatalError(const char *message) {
  std::fflush(stdout);
  std::fprintf(stderr, "soda axi runtime error: %s\n", message);
  std::fflush(stderr);
  std::_Exit(EXIT_FAILURE);
}

} // namespace axi
} // namespace soda

#endif // SODA_LIB_EXECUTIONENGINE_AXI_DMABACKEND_H
