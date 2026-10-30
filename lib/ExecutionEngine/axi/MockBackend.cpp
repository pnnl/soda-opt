//===- MockBackend.cpp - Loopback AXI DMA backend -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A backend without an accelerator: sent words go into a stream, and receives
// read them back in order (missing words read as 0). It is used to test data
// movement and, with SODA_AXI_TRACE=1, the transactions a program performs.
//
//===----------------------------------------------------------------------===//

#include "DmaBackend.h"

#include <deque>
#include <vector>

using namespace soda::axi;

namespace {

class MockBackend final : public DmaBackend {
public:
  const char *getName() const override { return "mock"; }

  void init(const DmaConfig &config) override {
    input.assign(config.inputSize, 0);
    output.assign(config.outputSize, 0);
    inputBuffer = input.data();
    outputBuffer = output.data();
    inputSize = config.inputSize;
    outputSize = config.outputSize;
    stream.clear();
  }

  void free() override {
    input.clear();
    output.clear();
    inputBuffer = outputBuffer = nullptr;
    inputSize = outputSize = 0;
  }

  void startSend(uint32_t length, uint32_t offset) override {
    stream.insert(stream.end(), inputBuffer + offset,
                  inputBuffer + offset + length);
  }

  void waitSend() override {}

  void startRecv(uint32_t length, uint32_t offset) override {
    recvLength = length;
    recvOffset = offset;
  }

  uint32_t waitRecv() override {
    for (uint32_t i = 0; i < recvLength; ++i) {
      uint32_t word = 0;
      if (!stream.empty()) {
        word = stream.front();
        stream.pop_front();
      }
      outputBuffer[recvOffset + i] = word;
    }
    return recvLength;
  }

private:
  std::vector<uint32_t> input, output;
  std::deque<uint32_t> stream;
  uint32_t recvLength = 0, recvOffset = 0;
};

} // namespace

DmaBackend &soda::axi::getDmaBackend() {
  static MockBackend backend;
  return backend;
}
