#ifndef AXI_APIv2
#define AXI_APIv2

#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <unistd.h>

#ifdef SYSC
// Easy way to switch between systemC accelerators --- there is probably a
// better way

#ifdef CONV_V1
#include "mlir/ExecutionEngine/axi/accelerators/conv_v1/accelerator.sc.h"
#elif ACC_V5
#include "mlir/ExecutionEngine/axi/accelerators/mm_4x4_v5/accelerator.sc.h"
#elif ACC_V4
#include "mlir/ExecutionEngine/axi/accelerators/mm_4x4_v4/accelerator.sc.h"
#elif ACC_V3
#include "mlir/ExecutionEngine/axi/accelerators/mm_4x4_v3/accelerator.sc.h"
#elif ACC_V2
#include "mlir/ExecutionEngine/axi/accelerators/mm_4x4_v2/accelerator.sc.h"
#else
#include "mlir/ExecutionEngine/axi/accelerators/mm_4x4_v1/accelerator.sc.h"
#endif
#endif

// API Model = One DMA is allocated with a single input and output buffer (Can
// have different size)

// clang-format off
// Simple view of DMA
/*
dma -> {
    control_register_address : unsigned int # Mapped address to the start of 
                                              the DMA control registers 
    Buffer input_buffer  (address,length,size_of_element) 
    Buffer output_buffer (address,length,size_of_element)
    All lengths are in elements
    All offsets are in elements
    The library will handle the conversion to bytes
    Pointer access to the buffer returns char*
    Once, within dma_init, MLIR/external user needs to specify size of the I/O elements
}
*/
// clang-format on

struct dma {
#define MM2S_CONTROL_REGISTER 0x00
#define MM2S_STATUS_REGISTER 0x04
#define MM2S_START_ADDRESS 0x18
#define MM2S_LENGTH 0x28
#define S2MM_CONTROL_REGISTER 0x30
#define S2MM_STATUS_REGISTER 0x34
#define S2MM_DESTINATION_ADDRESS 0x48
#define S2MM_LENGTH 0x58
#define PAGE_SIZE getpagesize()

#define m_assert(expr, msg) assert(((void)(msg), (expr)))

// Define this variable for additional profiling info (in api_v1_sysc.cpp)
#define PROFILE
#ifdef PROFILE
#define PLOG(x) std::cout << x << std::endl
#define PFUNC(x) x
#else
// Safer option that requires a semicolon, but relies on compiler to be removed
// #define PLOG(x) do { } while(0)
// #define PFUNC(x) do { } while(0)
#define PLOG(x)
#define PFUNC(x)
#endif

// Define this variable for additional debug info
// #define VERBOSE_AXI
#ifdef VERBOSE_AXI
#define D(x)                                                                   \
  do {                                                                         \
    x                                                                          \
  } while (0)
#define LOG(x) std::cout << x << std::endl
#else
// Safer option that requires a semicolon, but relies on compiler to be removed
// #define D(x) do { } while(0)
// #define LOG(x) do { } while(0)
#define D(x)
#define LOG(x)
#endif
  // I/O addresses are in type char or handled with type char size
  // I/O lengths are in elements
  unsigned int *dma_address;
  char *dma_input_address;
  char *dma_output_address;
  unsigned int dma_input_buffer_size;
  unsigned int dma_output_buffer_size;
  unsigned int isize;
  unsigned int osize;

  // These addresses are in physical memory
  unsigned int dma_input_paddress;
  unsigned int dma_output_paddress;

  // Maybe remove
  unsigned int *acc_address;
  // unsigned int current_input_offset;

  // Profiling Variables
  unsigned int dma_send_length = 0;
  unsigned int dma_recv_length = 0;
  unsigned int dma_send_count = 0;
  unsigned int dma_recv_count = 0;

  // temp --- need to remove later
  bool verbose = false;

#ifdef SYSC
  ACCNAME *acc;
  DMA_DRIVER *dmad;
#endif

  void dma_init(unsigned int dma_address, unsigned int dma_input_address,
                unsigned int dma_input_buffer_size, unsigned int isize,
                unsigned int dma_output_address,
                unsigned int dma_output_buffer_size, unsigned int osize);

  // Memory unmaps DMA control_register_address and Input and output buffers
  void dma_free();

  // We could reduce to one set of the following calls
  //================================================================================================================

  //-----------------BUFFER Functions-----------------
  // Get the MMap address of the input buffer of the dma  *Needed to copy data
  // to Input_Buffer*
  char *dma_get_inbuffer();

  // Get the MMap address of the output buffer of the dma *Needed to copy data
  // from Output_Buffer*

  char *dma_get_outbuffer();

  //================================================================================================================

  //-----------------BUFFER Functions-----------------
  // Copy data into the Input Buffer (length to write, offset to write to)
  // returns 0 if successful
  // int dma_copy_to_inbuffer(unsigned int *host_src_address, int data_length,
  //                          int offset);

  // Copy data from the Output Buffer (length to read, offset to read from)
  // returns 0 if successful
  // int dma_copy_from_outbuffer(unsigned int *host_dst_address, int
  // data_length,
  //                             int offset);

  template <typename T>
  int mlir_dma_copy_to_inbuffer(T *mr_base, int64_t mr_dim, int64_t mr_rank,
                                int64_t mr_offset, const int64_t *mr_sizes,
                                const int64_t *mr_strides, int dma_offset);

  template <typename T>
  int mlir_dma_copy_from_outbuffer(T *mr_base, int64_t mr_dim, int64_t mr_rank,
                                   int64_t mr_offset, const int64_t *mr_sizes,
                                   const int64_t *mr_strides, int dma_offset);

  //============================================================================

  //-----------------DMA MMS2 Functions-----------------
  /**
   * Checks if input buffer size is >= length
   * Sets DMA MMS2 transfer length to length
   * Starts transfers to the accelerator using dma associated with dma_id
   * Return 0 if successful, returns negative if error occurs
   */
  int dma_start_send(unsigned int length, unsigned int offset);

  // Blocks thread until dma MMS2 transfer is complete
  void dma_wait_send();

  // Same as dma_send but thread does not block, returns 0 if done
  int dma_check_send();

  //-----------------DMA S2MM Functions-----------------
  /**
   * Checks if buffer size is >= length
   * Sets 2SMM store length
   * Starts storing data recieved through dma associated with dma_id
   * Return 0 if successful, returns negative if error occurs
   */
  int dma_start_recv(unsigned int length, unsigned int offset);

  // Blocks thread until dma S2MM transfer is complete (TLAST signal is seen)
  void dma_wait_recv();

  // Same as dma_recv but thread does not block, returns 0 if done
  int dma_check_recv();

  //********************************** Unexposed Functions
  //**********************************
  void initDMAControls();
  void dma_set(unsigned int *dma_virtual_address, int offset,
               unsigned int value);
  unsigned int dma_get(unsigned int *dma_virtual_address, int offset);
  void dma_mm2s_sync();
  void dma_s2mm_sync();
  void acc_init(unsigned int base_addr, int length);
  void dump_acc_signals(int state);
};

#endif