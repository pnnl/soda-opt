//===- HwBackend.cpp - Xilinx AXI DMA backend -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Drives a Xilinx AXI DMA (simple mode) whose registers and buffers are mapped
// through /dev/mem, as on the PYNQ boards used by AXI4MLIR. Ported from the
// AXI4MLIR api_v1 library.
//
//===----------------------------------------------------------------------===//

#include "DmaBackend.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

using namespace soda::axi;

namespace {

// Register offsets of the Xilinx AXI DMA.
constexpr unsigned kMM2SControl = 0x00;
constexpr unsigned kMM2SStatus = 0x04;
constexpr unsigned kMM2SStartAddress = 0x18;
constexpr unsigned kMM2SLength = 0x28;
constexpr unsigned kS2MMControl = 0x30;
constexpr unsigned kS2MMStatus = 0x34;
constexpr unsigned kS2MMDestinationAddress = 0x48;
constexpr unsigned kS2MMLength = 0x58;

// Status bits: idle (1) and IOC interrupt (12).
constexpr uint32_t kIdle = 1u << 1;
constexpr uint32_t kIOCIrq = 1u << 12;

constexpr size_t kWordBytes = sizeof(uint32_t);

class HwBackend final : public DmaBackend {
public:
  const char *getName() const override { return "hw"; }

  void init(const DmaConfig &config) override {
    int fd = open("/dev/mem", O_RDWR | O_SYNC);
    if (fd < 0)
      fatalError("cannot open /dev/mem to map the AXI DMA");
    pageSize = sysconf(_SC_PAGESIZE);
    registers = map(fd, pageSize, config.address, PROT_READ | PROT_WRITE);
    inputBuffer =
        static_cast<uint32_t *>(map(fd, config.inputSize * kWordBytes,
                                    config.inputAddress, PROT_READ | PROT_WRITE));
    outputBuffer = static_cast<uint32_t *>(map(
        fd, config.outputSize * kWordBytes, config.outputAddress, PROT_READ));
    close(fd);
    inputSize = config.inputSize;
    outputSize = config.outputSize;
    inputAddress = config.inputAddress;
    outputAddress = config.outputAddress;
    resetEngine();
  }

  void free() override {
    munmap(inputBuffer, inputSize * kWordBytes);
    munmap(outputBuffer, outputSize * kWordBytes);
    munmap(registers, pageSize);
    registers = nullptr;
    inputBuffer = outputBuffer = nullptr;
    inputSize = outputSize = 0;
  }

  void startSend(uint32_t length, uint32_t offset) override {
    set(kMM2SStartAddress, inputAddress + offset * kWordBytes);
    set(kMM2SLength, length * kWordBytes);
  }

  void waitSend() override { waitIdle(kMM2SStatus); }

  void startRecv(uint32_t length, uint32_t offset) override {
    set(kS2MMDestinationAddress, outputAddress + offset * kWordBytes);
    set(kS2MMLength, length * kWordBytes);
  }

  uint32_t waitRecv() override {
    waitIdle(kS2MMStatus);
    // The length register holds the number of bytes actually received.
    return get(kS2MMLength) / kWordBytes;
  }

private:
  void *map(int fd, size_t bytes, uint32_t address, int protection) {
    void *ptr = mmap(nullptr, bytes, protection, MAP_SHARED, fd, address);
    if (ptr == MAP_FAILED)
      fatalError("cannot map the AXI DMA through /dev/mem");
    return ptr;
  }

  void set(unsigned reg, uint32_t value) {
    *reinterpret_cast<volatile uint32_t *>(static_cast<char *>(registers) +
                                           reg) = value;
  }

  uint32_t get(unsigned reg) {
    return *reinterpret_cast<volatile uint32_t *>(
        static_cast<char *>(registers) + reg);
  }

  void waitIdle(unsigned statusReg) {
    while ((get(statusReg) & (kIOCIrq | kIdle)) != (kIOCIrq | kIdle))
      ;
  }

  void resetEngine() {
    set(kS2MMControl, 4);
    set(kMM2SControl, 4);
    set(kS2MMControl, 0);
    set(kMM2SControl, 0);
    set(kS2MMDestinationAddress, outputAddress);
    set(kMM2SStartAddress, inputAddress);
    set(kS2MMControl, 0xf001);
    set(kMM2SControl, 0xf001);
  }

  void *registers = nullptr;
  long pageSize = 0;
  uint32_t inputAddress = 0, outputAddress = 0;
};

} // namespace

DmaBackend &soda::axi::getDmaBackend() {
  static HwBackend backend;
  return backend;
}
