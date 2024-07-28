//===- AxiUtils.cpp - AXI4MLIR  implementation ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements wrapper AXI4MLIR library calls. These are the calls
// visible to the MLIR.
//
//===----------------------------------------------------------------------===//

#include "soda/ExecutionEngine/axi/AxiUtils.h"

#include "soda/ExecutionEngine/axi/api_v1.h"

// =============================================================================
// AXI_APIV1
// =============================================================================

struct dma myDMA;

extern "C" void dma_init(unsigned int dma_address,
                         unsigned int dma_input_address,
                         unsigned int dma_input_buffer_size,
                         unsigned int dma_output_address,
                         unsigned int dma_output_buffer_size) {
  // std::cout << "Called: " << __func__ << " sysc version" << std::endl;
  // std::cout << "\t" << dma_address << std::endl;
  // std::cout << "\t" << dma_input_address << std::endl;
  // std::cout << "\t" << dma_input_buffer_size << std::endl;
  // std::cout << "\t" << dma_output_address << std::endl;
  // std::cout << "\t" << dma_output_buffer_size << std::endl;
  LOG("Called: " << __func__ << " sysc version");
  LOG("\t" << dma_address);
  LOG("\t" << dma_input_address);
  LOG("\t" << dma_input_buffer_size);
  LOG("\t" << dma_output_address);
  LOG("\t" << dma_output_buffer_size);

  myDMA.dma_init(dma_address, dma_input_address, dma_input_buffer_size,
                 dma_output_address, dma_output_buffer_size);
  return;
}

extern "C" void dma_free() {
  // std::cout << "Called: " << __func__ << " sysc version" << std::endl;
  LOG("Called: " << __func__ << " sysc version");
  myDMA.dma_free();
}

extern "C" unsigned int *dma_get_inbuffer() {
  // std::cout << "Called: " << __func__ << " sysc version" << std::endl;
  LOG("Called: " << __func__ << " sysc version");
  return myDMA.dma_get_inbuffer();
}

extern "C" unsigned int *dma_get_outbuffer() {
  // std::cout << "Called: " << __func__ << " sysc version" << std::endl;
  LOG("Called: " << __func__ << " sysc version");
  return myDMA.dma_get_outbuffer();
}

extern "C" int dma_copy_to_inbuffer(unsigned int *host_src_address,
                                    int data_length, int offset) {
  // std::cout << "Called: " << __func__ << " sysc version" << std::endl;
  LOG("Called: " << __func__ << " sysc version");
  return myDMA.dma_copy_to_inbuffer(host_src_address, data_length, offset);
}

extern "C" int dma_copy_from_outbuffer(unsigned int *host_dst_address,
                                       int data_length, int offset) {
  // std::cout << "Called: " << __func__ << " sysc version" << std::endl;
  LOG("Called: " << __func__ << " sysc version");
  return myDMA.dma_copy_from_outbuffer(host_dst_address, data_length, offset);
}

template <typename T>
int mlir_dma_copy_to_inbuffer(const DynamicMemRefType<T> &src, int data_length,
                              int offset) {
  myDMA.mlir_dma_copy_to_inbuffer(src.data, src.rank, src.rank, src.offset,
                                  src.sizes, src.strides, offset);
  return 0;
}

extern "C" int _mlir_ciface_copy_to_inbuffer_f32(UnrankedMemRefType<float> *M,
                                                 int offset) {
  mlir_dma_copy_to_inbuffer(DynamicMemRefType<float>(*M), 0, offset);
  return 0;
}

extern "C" int copy_to_inbuffer_f32(int64_t rank, void *ptr, int offset) {
  UnrankedMemRefType<float> descriptor = {rank, ptr};
  return _mlir_ciface_copy_to_inbuffer_f32(&descriptor, offset);
}

extern "C" int _mlir_ciface_copy_to_inbuffer_i32(UnrankedMemRefType<int> *M,
                                                 int offset) {
  mlir_dma_copy_to_inbuffer(DynamicMemRefType<int>(*M), 0, offset);
  return 0;
}

extern "C" int copy_to_inbuffer_i32(int64_t rank, void *ptr, int offset) {
  UnrankedMemRefType<int> descriptor = {rank, ptr};
  return _mlir_ciface_copy_to_inbuffer_i32(&descriptor, offset);
}

extern "C" int
_mlir_ciface_copy_from_outbuffer_f32(UnrankedMemRefType<float> *M, int offset) {
  mlir_dma_copy_from_outbuffer(DynamicMemRefType<float>(*M), 0, offset);
  return 0;
}

extern "C" int copy_from_outbuffer_f32(int64_t rank, void *ptr, int offset) {
  UnrankedMemRefType<float> descriptor = {rank, ptr};
  return _mlir_ciface_copy_from_outbuffer_f32(&descriptor, offset);
}

extern "C" int _mlir_ciface_copy_from_outbuffer_i32(UnrankedMemRefType<int> *M,
                                                    int offset) {
  mlir_dma_copy_from_outbuffer(DynamicMemRefType<int>(*M), 0, offset);
  return 0;
}

extern "C" int copy_from_outbuffer_i32(int64_t rank, void *ptr, int offset) {
  UnrankedMemRefType<int> descriptor = {rank, ptr};
  return _mlir_ciface_copy_from_outbuffer_i32(&descriptor, offset);
}

template <typename T>
int mlir_dma_copy_from_outbuffer(const DynamicMemRefType<T> &dst,
                                 int data_length, int offset) {
  // std::cout << "Called: " << __func__ << " sysc version" << std::endl;
  LOG("Called: " << __func__ << " sysc version");
  myDMA.mlir_dma_copy_from_outbuffer(dst.data, dst.rank, dst.rank, dst.offset,
                                     dst.sizes, dst.strides, offset);
  return 0;
}

extern "C" int dma_start_send(int length, int offset) {
  // std::cout << "Called: " << __func__ << " sysc version" << std::endl;
  LOG("Called: " << __func__ << " sysc version");
  return myDMA.dma_start_send(length, offset);
}

extern "C" int dma_check_send() {
  // std::cout << "Called: " << __func__ << " sysc version" << std::endl;
  LOG("Called: " << __func__ << " sysc version");
  return 0;
}

extern "C" void dma_wait_send() {
  // std::cout << "Called: " << __func__ << " sysc version" << std::endl;
  LOG("Called: " << __func__ << " sysc version");
  myDMA.dma_wait_send();
}

extern "C" int dma_start_recv(int length, int offset) {
  // std::cout << "Called: " << __func__ << " sysc version" << std::endl;
  LOG("Called: " << __func__ << " sysc version");
  return myDMA.dma_start_recv(length, offset);
}

extern "C" void dma_wait_recv() {
  // std::cout << "Called: " << __func__ << " sysc version" << std::endl;
  LOG("Called: " << __func__ << " sysc version");
  myDMA.dma_wait_recv();
}

extern "C" int dma_check_recv() {
  // std::cout << "Called: " << __func__ << " sysc version" << std::endl;
  LOG("Called: " << __func__ << " sysc version");
  return myDMA.dma_check_recv();
}

extern "C" unsigned int dma_set(unsigned int *dma_virtual_address, int offset,
                                unsigned int value) {
  // std::cout << "Called: " << __func__ << " sysc version" << std::endl;
  LOG("Called: " << __func__ << " sysc version");
  myDMA.dma_set(dma_virtual_address, offset, value);
  return 0;
}

extern "C" unsigned int dma_get(unsigned int *dma_virtual_address, int offset) {
  // std::cout << "Called: " << __func__ << " sysc version" << std::endl;
  LOG("Called: " << __func__ << " sysc version");
  return myDMA.dma_get(dma_virtual_address, offset);
}
