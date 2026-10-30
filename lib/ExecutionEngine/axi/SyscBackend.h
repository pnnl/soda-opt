//===- SyscBackend.h - SystemC AXI DMA backend ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A backend that simulates the DMA engine and an accelerator with SystemC.
// Each runtime library includes this header, then one accelerator model, and
// then uses SODA_AXI_DEFINE_SYSC_BACKEND:
//
//   #include "SyscBackend.h"
//   #include "soda/ExecutionEngine/axi/accelerators/mm_4x4_v3/accelerator.sc.h"
//   SODA_AXI_DEFINE_SYSC_BACKEND("sysc mm_4x4_v3")
//
// The models are built once per process, the first time dma_init is called,
// because SystemC cannot create modules after the simulation started. Waiting
// for a transfer runs the simulation until the DMA engine completes it, for at
// most SODA_AXI_SYSC_MAX_NS nanoseconds of simulated time (default 1e9), so a
// wrong opcode stops the program with an error instead of hanging it.
//
//===----------------------------------------------------------------------===//

#ifndef SODA_LIB_EXECUTIONENGINE_AXI_SYSCBACKEND_H
#define SODA_LIB_EXECUTIONENGINE_AXI_SYSCBACKEND_H

#include "DmaBackend.h"

#include <systemc.h>

#include "soda/ExecutionEngine/axi/accelerators/dma_engine.sc.h"

#include <vector>

namespace soda {
namespace axi {

/// Simulated-time budget of one wait, in nanoseconds.
double getSyscBudgetNs();

/// Silences SystemC reports that the models trigger.
void configureSyscReports();

template <typename Accelerator>
class SyscBackend final : public DmaBackend {
public:
  using ConnectFn = void (*)(Accelerator *, DMA_DRIVER *, int, int);

  SyscBackend(const char *name, ConnectFn connect)
      : name(name), connect(connect) {}

  const char *getName() const override { return name; }

  void init(const DmaConfig &config) override {
    input.assign(config.inputSize, 0);
    output.assign(config.outputSize, 0);
    inputBuffer = input.data();
    outputBuffer = output.data();
    inputSize = config.inputSize;
    outputSize = config.outputSize;
    if (!accelerator) {
      configureSyscReports();
      // The modules live until the process exits, like the simulation.
      accelerator = new Accelerator("accelerator");
      dma = new DMA_DRIVER("dma");
      dma->send = false;
      dma->recv = false;
      dma->input_len = dma->input_offset = 0;
      dma->output_len = dma->output_offset = 0;
      connect(accelerator, dma, config.inputSize, config.outputSize);
    }
    dma->DMA_input_buffer = reinterpret_cast<int *>(inputBuffer);
    dma->DMA_output_buffer = reinterpret_cast<int *>(outputBuffer);
  }

  void free() override {
    input.clear();
    output.clear();
    inputBuffer = outputBuffer = nullptr;
    inputSize = outputSize = 0;
  }

  void startSend(uint32_t length, uint32_t offset) override {
    dma->input_len = length;
    dma->input_offset = offset;
    dma->send = true;
  }

  void waitSend() override {
    runWhile([&] { return dma->send; }, "send");
  }

  void startRecv(uint32_t length, uint32_t offset) override {
    dma->output_len = length;
    dma->output_offset = offset;
    dma->recv = true;
  }

  uint32_t waitRecv() override {
    runWhile([&] { return dma->recv; }, "receive");
    return dma->output_len;
  }

private:
  /// Runs the simulation while `busy()` holds. The DMA threads pause the
  /// simulation when they complete a transfer; a pause of the other thread
  /// only makes sc_start return early.
  template <typename Pred>
  void runWhile(Pred busy, const char *what) {
    sc_core::sc_time budget(getSyscBudgetNs(), sc_core::SC_NS);
    sc_core::sc_time start = sc_core::sc_time_stamp();
    while (busy()) {
      sc_core::sc_time elapsed = sc_core::sc_time_stamp() - start;
      if (elapsed >= budget) {
        char message[256];
        std::snprintf(message, sizeof(message),
                      "the %s simulated accelerator did not complete a %s "
                      "within %.0f ns of simulated time (check the opcodes it "
                      "receives, or raise SODA_AXI_SYSC_MAX_NS)",
                      name, what, getSyscBudgetNs());
        fatalError(message);
      }
      sc_core::sc_start(budget - elapsed);
    }
  }

  const char *name;
  ConnectFn connect;
  Accelerator *accelerator = nullptr;
  DMA_DRIVER *dma = nullptr;
  std::vector<uint32_t> input, output;
};

} // namespace axi
} // namespace soda

/// Defines the backend of a runtime library for the accelerator model included
/// last (named by the ACCNAME macro of the model headers).
#define SODA_AXI_DEFINE_SYSC_BACKEND(NAME)                                     \
  soda::axi::DmaBackend &soda::axi::getDmaBackend() {                          \
    static soda::axi::SyscBackend<ACCNAME> backend(                            \
        NAME, [](ACCNAME *accelerator, DMA_DRIVER *dma, int inputSize,         \
                 int outputSize) {                                             \
          accelerator_dma_connect<int>(accelerator, dma, inputSize,            \
                                       outputSize);                            \
        });                                                                    \
    return backend;                                                            \
  }

#endif // SODA_LIB_EXECUTIONENGINE_AXI_SYSCBACKEND_H
