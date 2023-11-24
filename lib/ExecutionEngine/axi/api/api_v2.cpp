//===- api_v2.cpp - AXI core API implementation ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the core functions to use the AXI DMA interface.
//
//===----------------------------------------------------------------------===//

#include "mlir/ExecutionEngine/axi/api_v2.h"

#ifdef __arm__
#include "arm_neon.h"
#endif

void dma::dma_init(unsigned int _dma_address, unsigned int _dma_input_address,
                   unsigned int _dma_input_buffer_size, unsigned int _isize,
                   unsigned int _dma_output_address,
                   unsigned int _dma_output_buffer_size, unsigned int _osize) {

  dma_input_buffer_size = _dma_input_buffer_size;
  dma_output_buffer_size = _dma_output_buffer_size;
  dma_input_paddress = _dma_input_address;
  dma_output_paddress = _dma_output_address;
  isize = _isize;
  osize = _osize;

  unsigned int in_size_bytes = dma_input_buffer_size * isize;
  unsigned int out_size_bytes = dma_output_buffer_size * osize;
  int dh = open("/dev/mem", O_RDWR | O_SYNC);
  void *dma_mm = mmap(NULL, PAGE_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, dh,
                      _dma_address); // Memory map AXI Lite register block
  void *dma_in_mm =
      mmap(NULL, in_size_bytes, PROT_READ | PROT_WRITE, MAP_SHARED, dh,
           _dma_input_address); // Memory map source address
  void *dma_out_mm =
      mmap(NULL, out_size_bytes, PROT_READ, MAP_SHARED, dh,
           _dma_output_address); // Memory map destination address

  dma_address = reinterpret_cast<unsigned int *>(dma_mm);
  dma_input_address = reinterpret_cast<char *>(dma_in_mm);
  dma_output_address = reinterpret_cast<char *>(dma_out_mm);

  close(dh);
  initDMAControls(); // Causes Segfault atm
  LOG("DMA Initialised");
}

void dma::dma_free() {
  unsigned int in_size_bytes = dma_input_buffer_size * isize;
  unsigned int out_size_bytes = dma_output_buffer_size * osize;
  munmap(dma_input_address, in_size_bytes);
  munmap(dma_output_address, out_size_bytes);
  munmap(dma_address, getpagesize());
}

// We could reduce to one set of the following calls
//==============================================================================

char *dma::dma_get_inbuffer() { return dma_input_address; }

char *dma::dma_get_outbuffer() { return dma_output_address; }
//==============================================================================

// Removing these functions for now
// int dma::dma_copy_to_inbuffer(unsigned int *src_address, int data_length,
//                               int offset) {
//   m_assert("data copy will overflow input buffer",
//            (unsigned int)(offset + data_length) <= dma_input_buffer_size);
//   std::memcpy(dma_input_address + offset, src_address, data_length * 4);
//   current_input_offset += data_length;
//   return 0;
// }

// int dma::dma_copy_from_outbuffer(unsigned int *dst_address, int data_length,
//                                  int offset) {
//   m_assert("tries to access data outwith the output buffer",
//            (unsigned int)(offset + data_length) <= dma_output_buffer_size);
//   std::memcpy(dst_address, dma_output_address + offset, data_length * 4);
//   return 0;
// }

template <typename T>
inline void copy_memref_to_array(T *mr_base, int64_t mr_dim, int64_t mr_rank,
                                 int64_t mr_offset, const int64_t *mr_sizes,
                                 const int64_t *mr_strides, char *dst_base,
                                 const int dst_offset) {
  int64_t rank = mr_rank;
  // Handle empty shapes -> nothing to copy.
  for (int rankp = 0; rankp < rank; ++rankp)
    if (mr_sizes[rankp] == 0)
      return;

  T *srcPtr;
  srcPtr = mr_base + mr_offset;

  T *dstPtr;
  dstPtr = reinterpret_cast<T *>(dst_base) + dst_offset;

  if (rank == 0) {
    // memcpy(dstPtr, srcPtr, elemSize); // broken
    *dstPtr = *srcPtr; // opt 1
    // *dstPtr = mr_base[mr_offset]; // opt 2
    // dst_base[dst_offset] = mr_base[mr_offset]; // opt 3
    return;
  }

  int64_t *indices = static_cast<int64_t *>(alloca(sizeof(int64_t) * rank));
  int64_t *srcStrides = static_cast<int64_t *>(alloca(sizeof(int64_t) * rank));
  int64_t *dstStrides = static_cast<int64_t *>(alloca(sizeof(int64_t) * rank));

  // Initialize index and scale strides.
  for (int rankp = 0; rankp < rank; ++rankp) {
    indices[rankp] = 0;
    srcStrides[rankp] = mr_strides[rankp];

    // dstStrides for the array is derived from the input mr_sizes
    // if the rank is 3, and the mr_sizes are 4x8x16, the dstStrides are
    // 128x16x1
    dstStrides[rankp] = 1;
    for (int rankp2 = rankp + 1; rankp2 < rank; ++rankp2) {
      dstStrides[rankp] *= mr_sizes[rankp2];
    }
  }

  // DEBUG:
  // std::cout << "INFO copy_memref_to_array: rank: " << rank << std::endl;
  // std::cout << "INFO copy_memref_to_array: offset: " << mr_offset <<
  // std::endl; std::cout << "INFO copy_memref_to_array: sizes: "; for (int
  // rankp = 0; rankp < rank; ++rankp) {
  //   std::cout << mr_sizes[rankp] << " ";
  // }
  // std::cout << std::endl;
  // std::cout << "INFO copy_memref_to_array: strides: ";
  // for (int rankp = 0; rankp < rank; ++rankp) {
  //   std::cout << mr_strides[rankp] << " ";
  // }
  // std::cout << std::endl;

  // create a special case for rank==2 and strides[rank-1]==1 using memcpy
  if (rank == 2 && mr_strides[rank - 1] == 1) {
    int64_t size = mr_sizes[rank - 1];        // number of elements
    int64_t count = mr_sizes[rank - 2];       // number of rows
    int64_t srcStride = mr_strides[rank - 2]; // stride between rows
    int64_t dstStride = dstStrides[rank - 2]; // stride between rows
    const int64_t elemSize = sizeof(T);
    for (int64_t i = 0; i < count; ++i) {
      // std::cout << "INFO copy_memref_to_array: memcpy: " << dstPtr << " " <<
      // srcPtr << " " << size * elemSize << std::endl;
      memcpy(dstPtr, srcPtr, size * elemSize); // broken
      srcPtr += srcStride;
      dstPtr += dstStride;
    }
    return;
  }

  int64_t volatile readIndex = 0;
  int64_t volatile writeIndex = 0;
  for (;;) {
    D(std::cout << __FILE__ << ": " << __LINE__ << " [" << __func__ << "offset]"
                << dst_offset << "\n";
      std::cout << __FILE__ << ": " << __LINE__ << " [" << __func__ << "SRC]"
                << srcPtr << "\n";
      std::cout << __FILE__ << ": " << __LINE__ << " [" << __func__ << "DST]"
                << dstPtr << "\n";
      std::cout << __FILE__ << ": " << __LINE__ << " [" << __func__
                << "load from]" << srcPtr + readIndex << "\n";
      std::cout << __FILE__ << ": " << __LINE__ << " [" << __func__
                << "store at]" << dstPtr + writeIndex << "\n";
      std::cout << __FILE__ << ": " << __LINE__ << " [" << __func__
                << "loaded val]" << *(srcPtr + readIndex) << "\n";
      std::cout << __FILE__ << ": " << __LINE__ << " [" << __func__
                << "stored val]" << *(dstPtr + writeIndex) << "\n";);

    // TODO: Try option 1 again
    // NOTE: broken memcpy could have been a result of implicit casting
    //       due to type mismatch

    // Copy over the element, byte by byte.
    // memcpy(dstPtr + writeIndex, srcPtr + readIndex, elemSize); // broken
    *(dstPtr + writeIndex) = *(srcPtr + readIndex); // opt 1
    // *(dstPtr +writeIndex) = mr_base[mr_offset +readIndex]; // opt 2
    // dst_base[dst_offset+writeIndex] = mr_base[mr_offset +readIndex]; // opt 3

    // Advance index and read position.
    for (int64_t axis = rank - 1; axis >= 0; --axis) {
      // Advance at current axis.
      auto newIndex = ++indices[axis];
      readIndex += srcStrides[axis];
      writeIndex += 1; // Always increment, it is a flattened dense array
      // If this is a valid index, we have our next index, so continue copying.
      if (mr_sizes[axis] != newIndex)
        break;
      // We reached the end of this axis. If this is axis 0, we are done.
      if (axis == 0)
        return;
      // Else, reset to 0 and undo the advancement of the linear index that
      // this axis had. Then continue with the axis one outer.
      indices[axis] = 0;
      readIndex -= mr_sizes[axis] * srcStrides[axis];
      // We arrived in the last element of the current axis, we must decrement
      // writeIndex by 1 to fix the additional inc without write of this
      // iteration`
      writeIndex -= 1;
    }
  }
}

// Implements the actual copy
template <typename T>
int dma::mlir_dma_copy_to_inbuffer(T *mr_base, int64_t mr_dim, int64_t mr_rank,
                                   int64_t mr_offset, const int64_t *mr_sizes,
                                   const int64_t *mr_strides, int dma_offset) {
  D(std::cout << __FILE__ << ": " << __LINE__ << " [" << __func__ << "]\n";);

  copy_memref_to_array(mr_base, mr_dim, mr_rank, mr_offset, mr_sizes,
                       mr_strides, dma_get_inbuffer(), dma_offset);

  return 0;
}

template <typename T>
inline void copy_array_to_memref(T *mr_base, int64_t mr_dim, int64_t mr_rank,
                                 int64_t mr_offset, const int64_t *mr_sizes,
                                 const int64_t *mr_strides, char *src_base,
                                 const int src_offset, int elebytes) {
  int64_t rank = mr_rank;
  // Handle empty shapes -> nothing to copy.
  for (int rankp = 0; rankp < rank; ++rankp)
    if (mr_sizes[rankp] == 0)
      return;

  T *dstPtr;
  dstPtr = mr_base + mr_offset;

  T *srcPtr;
  srcPtr = reinterpret_cast<T *>(src_base) + src_offset;

  if (rank == 0) {
    // memcpy(dstPtr, srcPtr, elemSize); // broken
    *dstPtr = *srcPtr; // opt 1
    // *dstPtr = mr_base[mr_offset]; // opt 2
    // dst_base[dst_offset] = mr_base[mr_offset]; // opt 3
    return;
  }

  int64_t *indices = static_cast<int64_t *>(alloca(sizeof(int64_t) * rank));
  int64_t *srcStrides = static_cast<int64_t *>(alloca(sizeof(int64_t) * rank));
  int64_t *dstStrides = static_cast<int64_t *>(alloca(sizeof(int64_t) * rank));

  // Initialize index and scale strides.
  for (int rankp = 0; rankp < rank; ++rankp) {
    indices[rankp] = 0;
    dstStrides[rankp] = mr_strides[rankp];

    // srcStrides for the array is derived from the output mr_sizes
    // if the rank is 3, and the mr_sizes are 4x8x16, the srcStrides are
    // 128x16x1
    srcStrides[rankp] = 1;
    for (int rankp2 = rankp + 1; rankp2 < rank; ++rankp2) {
      srcStrides[rankp] *= mr_sizes[rankp2];
    }
  }

  // DEBUG:
  // std::cout << "INFO copy_memref_to_array: rank: " << rank << std::endl;
  // std::cout << "INFO copy_memref_to_array: offset: " << mr_offset <<
  // std::endl; std::cout << "INFO copy_memref_to_array: sizes: "; for (int
  // rankp = 0; rankp < rank; ++rankp) {
  //   std::cout << mr_sizes[rankp] << " ";
  // }
  // std::cout << std::endl;
  // std::cout << "INFO copy_memref_to_array: strides: ";
  // for (int rankp = 0; rankp < rank; ++rankp) {
  //   std::cout << mr_strides[rankp] << " ";
  // }
  // std::cout << std::endl;

  // create a special case for rank==2 and mr_strides[rank-1]==1 using memcpy
  if (rank == 2 && mr_strides[rank - 1] == 1) {
    int64_t size = mr_sizes[rank - 1];  // number of elements in one row
    int64_t nRows = mr_sizes[rank - 2]; // number of rows
    int64_t dstStride =
        mr_strides[rank - 2]; // #elements to skip to access next row
    int64_t srcStride =
        srcStrides[rank - 2]; // #elements to skip to access next row
    const int64_t elemSize = sizeof(T);
    for (int64_t i = 0; i < nRows; ++i) {
      // std::cout << "INFO copy_memref_to_array: memcpy: " << dstPtr << " " <<
      // srcPtr << " " << size * elemSize << std::endl;
      memcpy(dstPtr, srcPtr, size * elemSize); // broken
      srcPtr += srcStride;
      dstPtr += dstStride;
    }
    return;
  }

  int64_t volatile readIndex = 0;
  int64_t volatile writeIndex = 0;
  for (;;) {
    D(std::cout << __FILE__ << ": " << __LINE__ << " [" << __func__ << "offset]"
                << src_offset << "\n";
      std::cout << __FILE__ << ": " << __LINE__ << " [" << __func__ << "SRC]"
                << srcPtr << "\n";
      std::cout << __FILE__ << ": " << __LINE__ << " [" << __func__ << "DST]"
                << dstPtr << "\n";
      std::cout << __FILE__ << ": " << __LINE__ << " [" << __func__
                << "load from]" << srcPtr + readIndex << "\n";
      std::cout << __FILE__ << ": " << __LINE__ << " [" << __func__
                << "store at]" << dstPtr + writeIndex << "\n";
      std::cout << __FILE__ << ": " << __LINE__ << " [" << __func__
                << "loaded val]" << *(srcPtr + readIndex) << "\n";
      std::cout << __FILE__ << ": " << __LINE__ << " [" << __func__
                << "stored val]" << *(dstPtr + writeIndex) << "\n";);

    // TODO: Try option 1 again
    // NOTE: broken memcpy could have been a result of implicit casting
    //       due to type mismatch

    // Copy over the element, byte by byte.
    // memcpy(dstPtr + writeIndex, srcPtr + readIndex, elemSize); // broken
    *(dstPtr + writeIndex) = *(srcPtr + readIndex); // opt 1
    // *(dstPtr +writeIndex) = mr_base[mr_offset +readIndex]; // opt 2
    // dst_base[dst_offset+writeIndex] = mr_base[mr_offset +readIndex]; // opt 3

    // Advance index and read position.
    for (int64_t axis = rank - 1; axis >= 0; --axis) {
      // Advance at current axis.
      auto newIndex = ++indices[axis];
      writeIndex += dstStrides[axis];
      readIndex += 1; // Always increment, it is a flattened dense array

      // If this is a valid index, we have our next index, so continue copying.
      if (mr_sizes[axis] != newIndex)
        break;
      // We reached the end of this axis. If this is axis 0, we are done.
      if (axis == 0)
        return;
      // Else, reset to 0 and undo the advancement of the linear index that
      // this axis had. Then continue with the axis one outer.
      indices[axis] = 0;
      writeIndex -= mr_sizes[axis] * dstStrides[axis];
      // We arrived in the last element of the current axis, we must decrement
      // writeIndex by 1 to fix the additional inc without write of this
      // iteration`
      readIndex -= 1;
    }
  }
}

template <typename T>
int dma::mlir_dma_copy_from_outbuffer(T *mr_base, int64_t mr_dim,
                                      int64_t mr_rank, int64_t mr_offset,
                                      const int64_t *mr_sizes,
                                      const int64_t *mr_strides,
                                      int dma_offset) {

  D(std::cout << __FILE__ << ": " << __LINE__ << " [" << __func__ << "]\n";);

  copy_array_to_memref(mr_base, mr_dim, mr_rank, mr_offset, mr_sizes,
                       mr_strides, dma_get_outbuffer(), dma_offset);

  return 0;
}

// Make templates concrete:
template int dma::mlir_dma_copy_to_inbuffer<float>(
    float *mr_base, int64_t mr_dim, int64_t mr_rank, int64_t mr_offset,
    const int64_t *mr_sizes, const int64_t *mr_strides, int dma_offset);

template int dma::mlir_dma_copy_to_inbuffer<int>(
    int *mr_base, int64_t mr_dim, int64_t mr_rank, int64_t mr_offset,
    const int64_t *mr_sizes, const int64_t *mr_strides, int dma_offset);

template int dma::mlir_dma_copy_from_outbuffer<float>(
    float *mr_base, int64_t mr_dim, int64_t mr_rank, int64_t mr_offset,
    const int64_t *mr_sizes, const int64_t *mr_strides, int dma_offset);

template int dma::mlir_dma_copy_from_outbuffer<int>(
    int *mr_base, int64_t mr_dim, int64_t mr_rank, int64_t mr_offset,
    const int64_t *mr_sizes, const int64_t *mr_strides, int dma_offset);

// DMA Functions
// Updated for char
int dma::dma_start_send(unsigned int length, unsigned int offset) {
  m_assert("trying to send data outside the input buffer",
           (offset + length) <= dma_input_buffer_size);
  unsigned int new_length = length * isize;
  unsigned int new_offset = offset * isize;
  dma_set(dma_address, MM2S_START_ADDRESS, dma_input_paddress + new_offset);
  msync(dma_address, PAGE_SIZE, MS_SYNC);
  dma_set(dma_address, MM2S_LENGTH, new_length);
  LOG("Transfer Started - " << new_length << " bytes");
  return 0;
}

void dma::dma_wait_send() {
  LOG("Data Transfer - Waiting");
  dma_mm2s_sync();
  LOG("Data Transfer - Done");
}

int dma::dma_check_send() {
  unsigned int mm2s_status = dma_get(dma_address, MM2S_STATUS_REGISTER);
  bool done = !((!(mm2s_status & 1 << 12)) || (!(mm2s_status & 1 << 1)));
  if (done) {
    LOG("Data Transfer - Done");
  } else {
    LOG("Data Transfer - Not Done");
  }
  return done ? 0 : -1;
}

// Updated for char
int dma::dma_start_recv(unsigned int length, unsigned int offset) {
  m_assert("trying receive data outside the output buffer",
           (offset + length) <= dma_output_buffer_size);
  unsigned int new_length = length * osize;
  unsigned int new_offset = offset * osize;
  dma_set(dma_address, S2MM_DESTINATION_ADDRESS,
          dma_output_paddress + new_offset);
  msync(dma_address, PAGE_SIZE, MS_SYNC);
  dma_set(dma_address, S2MM_LENGTH, new_length);
  LOG("Started Receiving " << new_length << " bytes");
  return 0;
}

void dma::dma_wait_recv() {
  LOG("Data Receive - Waiting");
  dma_s2mm_sync();
  LOG("Data Received - " << dma_get(dma_address, S2MM_LENGTH) << " bytes");
}

int dma::dma_check_recv() {
  unsigned int s2mm_status = dma_get(dma_address, S2MM_STATUS_REGISTER);
  bool done = !((!(s2mm_status & 1 << 12)) || (!(s2mm_status & 1 << 1)));
  if (done) {
    LOG("Data Receive - Done");
  } else {
    LOG("Data Receive - Not Done");
  }
  return done ? 0 : -1;
}

//********************************** Unexposed Functions
//**********************************

void dma::initDMAControls() {
  dma_set(dma_address, S2MM_CONTROL_REGISTER, 4);
  dma_set(dma_address, MM2S_CONTROL_REGISTER, 4);
  dma_set(dma_address, S2MM_CONTROL_REGISTER, 0);
  dma_set(dma_address, MM2S_CONTROL_REGISTER, 0);
  // dma_set(dma_address, S2MM_DESTINATION_ADDRESS,
  //         (unsigned long)dma_output_address); // Write destination address
  // dma_set(dma_address, MM2S_START_ADDRESS,
  //         (unsigned long)dma_input_address); // Write source address
  dma_set(dma_address, S2MM_DESTINATION_ADDRESS,
          dma_output_paddress); // Write destination address
  dma_set(dma_address, MM2S_START_ADDRESS,
          dma_input_paddress); // Write source address
  dma_set(dma_address, S2MM_CONTROL_REGISTER, 0xf001);
  dma_set(dma_address, MM2S_CONTROL_REGISTER, 0xf001);
}

void dma::dma_set(unsigned int *dma_address, int offset, unsigned int value) {
  *((volatile unsigned int *)(reinterpret_cast<char *>(dma_address) + offset)) =
      value;
  // dma_address[offset >> 2] = value;
}

unsigned int dma::dma_get(unsigned int *dma_address, int offset) {
  return *((volatile unsigned int *)(reinterpret_cast<char *>(dma_address) +
                                     offset));
  // return *((volatile unsigned int*) dma_address[offset >> 2]);
  // return dma_address[offset >> 2];
}

void dma::dma_mm2s_sync() {
  msync(dma_address, PAGE_SIZE, MS_SYNC);
  unsigned int mm2s_status = dma_get(dma_address, MM2S_STATUS_REGISTER);
  while (!(mm2s_status & 1 << 12) || !(mm2s_status & 1 << 1)) {
    msync(dma_address, PAGE_SIZE, MS_SYNC);
    mm2s_status = dma_get(dma_address, MM2S_STATUS_REGISTER);
  }
}

void dma::dma_s2mm_sync() {
  msync(dma_address, PAGE_SIZE, MS_SYNC);
  unsigned int s2mm_status = dma_get(dma_address, S2MM_STATUS_REGISTER);
  while (!(s2mm_status & 1 << 12) || !(s2mm_status & 1 << 1)) {
    msync(dma_address, PAGE_SIZE, MS_SYNC);
    s2mm_status = dma_get(dma_address, S2MM_STATUS_REGISTER);
  }
}

void dma::acc_init(unsigned int base_addr, int length) {
  int dh = open("/dev/mem", O_RDWR | O_SYNC);
  size_t virt_base = base_addr & ~(PAGE_SIZE - 1);
  size_t virt_offset = base_addr - virt_base;
  void *addr = mmap(NULL, length + virt_offset, PROT_READ | PROT_WRITE,
                    MAP_SHARED, dh, virt_base);
  close(dh);
  if (addr == (void *)-1)
    exit(EXIT_FAILURE);
  acc_address = reinterpret_cast<unsigned int *>(addr);
}

void dma::dump_acc_signals(int state) {
  msync(acc_address, PAGE_SIZE, MS_SYNC);
  std::ofstream file;
  file.open("dump_acc_signals.dat", std::ios_base::app);
  file << "====================================================" << std::endl;
  file << "State: " << state << std::endl;
  file << "====================================================" << std::endl;
  for (int i = 0; i < 16; i++)
    file << acc_address[i] << ",";
  file << "====================================================" << std::endl;
}